"""
core/llm_service.py — Central LLM interface for all modules.

All Ollama calls go through this service. No module should call Ollama directly.
Supports: standard completion, streaming, model switching, and retry logic.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Generator, Optional

import httpx

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMService:
    """
    Wraps Ollama's HTTP API to provide a clean, reusable interface
    for all modules in the system.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model or settings.DEFAULT_MODEL
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.timeout = settings.OLLAMA_TIMEOUT
        self.max_retries = settings.OLLAMA_MAX_RETRIES

    # ─── Primary interface ────────────────────────────────────────────────────

    async def complete_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Async version of complete() that uses AsyncClient and supports retries/timeouts.
        """
        payload = self._build_payload(prompt, system_prompt, model, temperature)
        start_time = time.time()

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "LLM call attempt %d/%d | model=%s | prompt_len=%d",
                    attempt,
                    self.max_retries,
                    payload["model"],
                    len(prompt),
                )

                timeout = httpx.Timeout(self.timeout, connect=self.timeout, read=self.timeout)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                    )
                    response.raise_for_status()

                result = self._parse_response(response.text)
                elapsed = time.time() - start_time
                response_len = len(result)
                logger.debug(
                    "span_llm_service | model=%s attempt=%d prompt_len=%d response_len=%d latency_ms=%.2f",
                    payload["model"],
                    attempt,
                    len(prompt),
                    response_len,
                    elapsed * 1000,
                )
                logger.info(
                    "span_llm_service_agg | model=%s response_len=%d latency_ms=%.2f",
                    payload["model"],
                    response_len,
                    elapsed * 1000,
                )
                return result

            except httpx.ConnectError as e:
                logger.error("Ollama not reachable at %s — is it running?", self.base_url)
                raise LLMServiceError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Ensure Ollama is running: `ollama serve`"
                ) from e

            except httpx.HTTPStatusError as e:
                logger.error("Ollama HTTP error: %s", str(e))
                if attempt == self.max_retries:
                    raise LLMServiceError(f"Ollama returned HTTP error: {e.response.status_code}") from e
                await asyncio.sleep(2 ** attempt)

            except (httpx.ReadTimeout, httpx.TimeoutException) as e:
                logger.warning(
                    "LLM request timed out on attempt %d/%d after %ss (model=%s)",
                    attempt,
                    self.max_retries,
                    self.timeout,
                    payload["model"],
                )
                if attempt == self.max_retries:
                    raise LLMServiceError(
                        f"LLM timed out after {self.timeout}s. "
                        "Try a shorter prompt, smaller top_k, or increase OLLAMA_TIMEOUT."
                    ) from e
                await asyncio.sleep(2 ** attempt)

            except Exception as e:
                logger.error("Unexpected LLM error on attempt %d: %s", attempt, str(e))
                if attempt == self.max_retries:
                    raise LLMServiceError(f"LLM call failed after {self.max_retries} attempts: {str(e)}") from e
                await asyncio.sleep(1)

        raise LLMServiceError("LLM call exhausted all retries without a result.")

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Sync wrapper for complete_async for compatibility with non-async callers."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.complete_async(prompt, system_prompt, model, temperature))
        else:
            raise RuntimeError("LLMService.complete should not be called in async context; use complete_async instead.")

    async def stream_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async streaming version with retry logic.
        Yields tokens one at a time from Ollama.
        """
        payload = self._build_payload(prompt, system_prompt, model, stream=True)
        start_time = time.time()

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "LLM stream attempt %d/%d | model=%s | prompt_len=%d",
                    attempt,
                    self.max_retries,
                    payload["model"],
                    len(prompt),
                )

                timeout = httpx.Timeout(self.timeout, connect=self.timeout, read=self.timeout)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                        response.raise_for_status()
                        token_count = 0
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                                if token := chunk.get("response", ""):
                                    token_count += 1
                                    yield token
                                if chunk.get("done", False):
                                    elapsed = time.time() - start_time
                                    logger.debug(
                                        "span_llm_stream | model=%s attempt=%d tokens=%d latency_ms=%.2f",
                                        payload["model"],
                                        attempt,
                                        token_count,
                                        elapsed * 1000,
                                    )
                                    return
                            except json.JSONDecodeError:
                                continue
                        return

            except httpx.ConnectError as e:
                logger.error("Ollama not reachable at %s during streaming", self.base_url)
                if attempt == self.max_retries:
                    raise LLMServiceError(f"Cannot connect to Ollama: {str(e)}") from e
                await asyncio.sleep(2 ** attempt)

            except (httpx.ReadTimeout, httpx.TimeoutException) as e:
                logger.warning("Stream timeout on attempt %d/%d after %ss", attempt, self.max_retries, self.timeout)
                if attempt == self.max_retries:
                    raise LLMServiceError(f"Streaming timed out after {self.timeout}s") from e
                await asyncio.sleep(2 ** attempt)

            except Exception as e:
                logger.error("Unexpected error during streaming attempt %d: %s", attempt, str(e))
                if attempt == self.max_retries:
                    raise LLMServiceError(f"Streaming failed after {self.max_retries} attempts: {str(e)}") from e
                await asyncio.sleep(1)

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Blocking streaming compatibility wrapper."""
        try:
            return asyncio.run(self.stream_async(prompt, system_prompt, model))
        except RuntimeError:
            raise RuntimeError("LLMService.stream should not be called from a running event loop. Use stream_async instead.")

    # ─── Model utilities ──────────────────────────────────────────────────────

    async def list_models_async(self) -> list[str]:
        """Return names of all locally available Ollama models asynchronously."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning("Could not list Ollama models: %s", str(e))
            return []

    def list_models(self) -> list[str]:
        try:
            return asyncio.run(self.list_models_async())
        except RuntimeError:
            raise RuntimeError("LLMService.list_models should not be called in async context; use list_models_async.")

    async def is_available_async(self) -> bool:
        """Quick health check — returns True if Ollama is reachable asynchronously."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                await client.get(f"{self.base_url}/api/tags")
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        try:
            return asyncio.run(self.is_available_async())
        except RuntimeError:
            raise RuntimeError("LLMService.is_available should not be called in async context; use is_available_async.")

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
    ) -> dict:
        """Assemble the JSON payload for Ollama's /api/generate endpoint."""
        effective_model = model or self.model
        effective_temp = temperature if temperature is not None else self.temperature

        # Prepend system prompt inline if provided (Ollama models vary in system support)
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{prompt}"

        return {
            "model": effective_model,
            "prompt": full_prompt,
            "stream": stream,
            "options": {
                "temperature": effective_temp,
                "num_predict": self.max_tokens,
            },
        }

    def _parse_response(self, raw: str) -> str:
        """
        Parse Ollama's non-streaming response.
        Ollama returns NDJSON — we collect all response chunks.
        """
        full_text = []
        for line in raw.strip().splitlines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                if text := chunk.get("response", ""):
                    full_text.append(text)
            except json.JSONDecodeError:
                continue
        return "".join(full_text).strip()

    # ─── Tool calling support ─────────────────────────────────────────────────

    async def complete_with_tools_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Call LLM with tool definitions available (for systems that support it).

        For Ollama (local), tool calling is limited. This method documents
        the interface and can be extended for OpenAI/Claude APIs.

        Returns:
            {
                "content": "response text",
                "tool_calls": [
                    {"name": "...", "arguments": {...}},
                    ...
                ],
                "stop_reason": "end_turn" | "tool_calls"
            }
        """
        from config import settings

        if not settings.ENABLE_TOOL_USE:
            # Fall back to regular completion
            content = await self.complete_async(prompt, system_prompt, model, temperature)
            return {"content": content, "tool_calls": [], "stop_reason": "end_turn"}

        # For Ollama, we cannot actually invoke tools through the API.
        # This method demonstrates the contract. In production, use FastAPI endpoint
        # that dispatches tool calls to task_router.
        logger.debug("Tool calling requested but not supported by Ollama backend")

        content = await self.complete_async(prompt, system_prompt, model, temperature)
        # Attempt to extract tool calls from response text (heuristic)
        tool_calls = self._extract_tool_calls_from_text(content)

        return {
            "content": content,
            "tool_calls": tool_calls,
            "stop_reason": "tool_calls" if tool_calls else "end_turn",
        }

    def _extract_tool_calls_from_text(self, text: str) -> list[dict]:
        """
        Heuristically extract tool call attempts from LLM response text.

        Looks for patterns like: <tool_call name="..." args={...}>
        """
        import re

        tool_calls = []
        # Pattern: <tool_call name="tool_name" args={json}>...</tool_call>
        pattern = r'<tool_call\s+name="([^"]+)"\s+args=({[^}]+})\s*>'
        matches = re.finditer(pattern, text)

        for match in matches:
            tool_name = match.group(1)
            args_str = match.group(2)
            try:
                args = json.loads(args_str)
                tool_calls.append({"name": tool_name, "arguments": args})
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool call arguments: %s", args_str)

        return tool_calls

    def complete_with_tools(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Sync wrapper for complete_with_tools_async."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.complete_with_tools_async(prompt, system_prompt, model, temperature)
            )
        else:
            raise RuntimeError(
                "LLMService.complete_with_tools should not be called in async context; "
                "use complete_with_tools_async instead."
            )

    # ─── Prompt caching support (Claude-only) ─────────────────────────────────

    async def complete_with_cache_async(
        self,
        prompt: str,
        cached_context: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        Call LLM with prompt caching headers (Claude-only).

        For Ollama, this degrades gracefully to regular completion.
        For Claude, caches the `cached_context` to reduce token usage.

        Args:
            prompt: The user message
            cached_context: Dict of cacheable content (e.g., session metadata, task definitions)
            system_prompt: System message
            model: Optional model override
            temperature: Optional temperature override

        Returns:
            {
                "content": "response text",
                "cache_hit": bool,
                "cache_tokens_saved": int,
            }
        """
        if not settings.ENABLE_CACHING:
            content = await self.complete_async(prompt, system_prompt, model, temperature)
            return {"content": content, "cache_hit": False, "cache_tokens_saved": 0}

        # For Ollama, caching is not available; fall back to regular completion
        if "claude" not in (model or self.model).lower():
            logger.debug("Prompt caching not available for model %s; using standard completion", model or self.model)
            content = await self.complete_async(prompt, system_prompt, model, temperature)
            return {"content": content, "cache_hit": False, "cache_tokens_saved": 0}

        # For Claude: add cache_control headers to make context cacheable
        # This is a documented interface; actual caching is handled by Claude API
        logger.debug(
            "Prompt caching enabled for Claude model with %d bytes of cacheable context",
            len(json.dumps(cached_context or {})),
        )

        # Prepend cacheable context to the prompt for Claude's cache_control processing
        full_prompt = prompt
        if cached_context:
            context_str = json.dumps(cached_context, indent=2)
            full_prompt = f"[CACHED_CONTEXT]\n{context_str}\n\n[USER_QUERY]\n{prompt}"

        # For now, fall back to regular completion (actual caching handled by Claude API wrapper if deployed)
        content = await self.complete_async(full_prompt, system_prompt, model or "claude-3.5-sonnet", temperature)

        return {
            "content": content,
            "cache_hit": False,  # Without direct Claude API access, we can't track cache stats
            "cache_tokens_saved": 0,
        }

    def complete_with_cache(
        self,
        prompt: str,
        cached_context: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> dict:
        """Sync wrapper for complete_with_cache_async."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.complete_with_cache_async(prompt, cached_context, system_prompt, model, temperature)
            )
        else:
            raise RuntimeError(
                "LLMService.complete_with_cache should not be called in async context; "
                "use complete_with_cache_async instead."
            )


class LLMServiceError(Exception):
    """Raised when the LLM service encounters an unrecoverable error."""
    pass


# ─── Module-level singleton ───────────────────────────────────────────────────
# Modules import this directly: `from core.llm_service import llm_service`
llm_service = LLMService()

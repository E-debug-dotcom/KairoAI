"""
modules/code_gen/handler.py — Code generation and review module.

Handles:
- generate: Create new scripts/functions from a description
- review: Analyze existing code for bugs, security issues, style
- explain: Walk through what existing code does
"""

import time

from core.llm_service import llm_service
from core.output_formatter import formatter
from prompts.prompt_manager import prompt_manager
from prompts.assistant_prompts import CODE_SYSTEM_PROMPT
from utils.logger import get_logger
from utils.helpers import truncate_text

logger = get_logger(__name__)

SUPPORTED_LANGUAGES = {"python", "powershell", "bash", "javascript", "typescript", "sql"}


class CodeGenHandler:
    """Generates, reviews, and explains code in multiple languages."""

    async def handle(self, payload: dict) -> dict:
        sub_task = payload.get("sub_task", "generate")
        logger.info("Code module handling sub_task='%s'", sub_task)

        dispatch = {
            "generate": self._generate,
            "review": self._review,
            "explain": self._explain,
        }

        if sub_task not in dispatch:
            return formatter.error("code", f"Unknown sub_task '{sub_task}'.")

        return await dispatch[sub_task](payload)

    async def _generate(self, payload: dict) -> dict:
        task_description = payload.get("task_description", "").strip()
        language = payload.get("language", "python").lower()

        if not task_description:
            return formatter.error("code", "Field 'task_description' is required.")
        if language not in SUPPORTED_LANGUAGES:
            return formatter.error("code", f"Unsupported language '{language}'. Supported: {SUPPORTED_LANGUAGES}")

        # Optional requirements and examples
        requirements = payload.get("requirements", [])
        requirements_block = (
            "ADDITIONAL REQUIREMENTS:\n" + "\n".join(f"- {r}" for r in requirements)
            if requirements else ""
        )

        example = payload.get("example_input_output", "").strip()
        example_block = f"EXAMPLE INPUT/OUTPUT:\n{example}" if example else ""

        try:
            prompt = prompt_manager.render(
                "code", "generate",
                language=language,
                task_description=task_description,
                requirements_block=requirements_block,
                example_block=example_block,
            )
        except Exception as e:
            return formatter.error("code", f"Prompt error: {str(e)}")

        start = time.time()
        code = llm_service.complete(
            prompt=prompt,
            system_prompt=CODE_SYSTEM_PROMPT,
            temperature=0.2,  # Low temp for code — consistency over creativity
        )
        duration = time.time() - start

        return formatter.success(
            "code",
            {"code": code, "language": language, "task_description": task_description},
            meta={"duration_seconds": round(duration, 2)},
        )

    async def _review(self, payload: dict) -> dict:
        code = payload.get("code", "").strip()
        language = payload.get("language", "python").lower()

        if not code:
            return formatter.error("code", "Field 'code' is required.")

        try:
            prompt = prompt_manager.render(
                "code", "review",
                language=language,
                code=truncate_text(code, 4000),
            )
        except Exception as e:
            return formatter.error("code", f"Prompt error: {str(e)}")

        start = time.time()
        review = llm_service.complete(prompt=prompt, system_prompt=CODE_SYSTEM_PROMPT, temperature=0.2)
        duration = time.time() - start

        return formatter.success(
            "code",
            {"review": review, "language": language},
            meta={"duration_seconds": round(duration, 2)},
        )

    async def _explain(self, payload: dict) -> dict:
        code = payload.get("code", "").strip()
        language = payload.get("language", "python").lower()

        if not code:
            return formatter.error("code", "Field 'code' is required.")

        try:
            prompt = prompt_manager.render(
                "code", "explain",
                language=language,
                code=truncate_text(code, 4000),
            )
        except Exception as e:
            return formatter.error("code", f"Prompt error: {str(e)}")

        start = time.time()
        explanation = llm_service.complete(prompt=prompt, system_prompt=CODE_SYSTEM_PROMPT)
        duration = time.time() - start

        return formatter.success(
            "code",
            {"explanation": explanation, "language": language},
            meta={"duration_seconds": round(duration, 2)},
        )


# ─── Singleton ────────────────────────────────────────────────────────────────
code_gen_handler = CodeGenHandler()

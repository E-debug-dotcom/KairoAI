"""
test_caching.py — Tests for prompt caching and multi-file context functionality.
"""

import pytest
from storage.cache_telemetry import CacheTelemetry, CacheMetrics
from core.session_memory import session_memory


class TestCacheTelemetry:
    """Tests for CacheTelemetry."""

    def test_cache_telemetry_record_hit(self):
        telemetry = CacheTelemetry()
        metric = CacheMetrics(
            timestamp=100.0,
            session_id="s1",
            task_type="code_gen",
            cache_key_hash="abc123",
            hit=True,
            tokens_input=100,
            tokens_if_no_cache=500,
            tokens_saved=400,
            latency_ms=50.0,
        )
        telemetry.record(metric)

        assert telemetry.hit_rate() == 1.0
        assert telemetry.total_tokens_saved() == 400

    def test_cache_telemetry_record_miss(self):
        telemetry = CacheTelemetry()
        metric = CacheMetrics(
            timestamp=100.0,
            session_id="s1",
            task_type="learning",
            cache_key_hash="def456",
            hit=False,
            tokens_input=100,
            tokens_if_no_cache=100,
            tokens_saved=0,
            latency_ms=200.0,
        )
        telemetry.record(metric)

        assert telemetry.hit_rate() == 0.0
        assert telemetry.total_tokens_saved() == 0

    def test_cache_telemetry_mixed_hits_misses(self):
        telemetry = CacheTelemetry()
        telemetry.record(CacheMetrics(
            100.0, "s1", "code", "h1", True, 100, 500, 400, 50.0,
        ))
        telemetry.record(CacheMetrics(
            101.0, "s2", "code", "h2", False, 100, 100, 0, 200.0,
        ))
        telemetry.record(CacheMetrics(
            102.0, "s3", "code", "h3", True, 100, 600, 500, 45.0,
        ))

        assert telemetry.hit_rate() == pytest.approx(2/3, rel=0.01)
        assert telemetry.total_tokens_saved() == 900

    def test_cache_telemetry_average_latency(self):
        telemetry = CacheTelemetry()
        telemetry.record(CacheMetrics(
            100.0, "s1", "code", "h1", True, 100, 500, 400, 50.0,
        ))
        telemetry.record(CacheMetrics(
            101.0, "s2", "code", "h2", True, 100, 500, 400, 100.0,
        ))

        assert telemetry.average_latency_ms(hit_only=True) == 75.0

    def test_cache_telemetry_stats_by_task(self):
        telemetry = CacheTelemetry()
        telemetry.record(CacheMetrics(
            100.0, "s1", "code", "h1", True, 100, 500, 400, 50.0,
        ))
        telemetry.record(CacheMetrics(
            101.0, "s2", "learning", "h2", False, 100, 100, 0, 200.0,
        ))

        stats = telemetry.stats_by_task()
        assert "code" in stats
        assert "learning" in stats
        assert stats["code"]["hit_rate"] == 1.0
        assert stats["learning"]["hit_rate"] == 0.0

    def test_cache_telemetry_reset(self):
        telemetry = CacheTelemetry()
        telemetry.record(CacheMetrics(
            100.0, "s1", "code", "h1", True, 100, 500, 400, 50.0,
        ))
        assert telemetry.hit_rate() > 0

        telemetry.reset()
        assert telemetry.hit_rate() == 0.0
        assert telemetry.total_tokens_saved() == 0
        assert len(telemetry.metrics) == 0

    def test_cache_telemetry_to_dict(self):
        telemetry = CacheTelemetry()
        telemetry.record(CacheMetrics(
            100.0, "s1", "code", "h1", True, 100, 500, 400, 50.0,
        ))

        data = telemetry.to_dict()
        assert "hit_rate" in data
        assert "total_tokens_saved" in data
        assert "stats_by_task" in data
        assert data["hit_rate"] == 1.0
        assert data["total_tokens_saved"] == 400


class TestSessionMemoryCaching:
    """Tests for session memory caching support."""

    def test_session_memory_set_cached_context(self):
        session_memory.reset_session("s1")
        session_memory.set_cached_context("s1", "user_profile", {"name": "John", "role": "dev"})

        context = session_memory.get_cached_context("s1", "user_profile")
        assert context["name"] == "John"

    def test_session_memory_get_nonexistent_context(self):
        session_memory.reset_session("s2")
        context = session_memory.get_cached_context("s2", "nonexistent")
        assert context is None

    def test_session_memory_compute_cache_key(self):
        session_memory.reset_session("s3")
        session_memory.set_cached_context("s3", "task_def", {"type": "code_gen"})

        key1 = session_memory.compute_cache_key("s3", {"task": "generate_python"})
        key2 = session_memory.compute_cache_key("s3", {"task": "generate_python"})
        key3 = session_memory.compute_cache_key("s3", {"task": "generate_javascript"})

        assert key1 == key2  # Same inputs produce same hash
        assert key1 != key3  # Different inputs produce different hash

    def test_session_memory_get_cacheable_context(self):
        session_memory.reset_session("s4")
        session_memory.set_cached_context("s4", "profile", {"name": "Alice"})
        session_memory.append_user("s4", "Hello")
        session_memory.append_assistant("s4", "Hi there")

        cacheable = session_memory.get_cacheable_context("s4")
        assert cacheable["session_id"] == "s4"
        assert "metadata" in cacheable
        assert "history_summary" in cacheable


class TestLLMServiceCaching:
    """Tests for LLM service caching support."""

    @pytest.mark.asyncio
    async def test_llm_service_complete_with_cache_no_cache_enabled(self, monkeypatch):
        from core.llm_service import llm_service
        from config import settings

        monkeypatch.setattr(settings, "ENABLE_CACHING", False)

        # Mock complete_async to avoid trying to connect to Ollama
        async def mock_complete(*args, **kwargs):
            return "test response"

        monkeypatch.setattr(llm_service, "complete_async", mock_complete)

        result = await llm_service.complete_with_cache_async(
            "test prompt",
            cached_context={"key": "value"},
        )

        assert "content" in result
        assert result["cache_hit"] is False
        assert result["cache_tokens_saved"] == 0

    @pytest.mark.asyncio
    async def test_llm_service_complete_with_cache_ollama_model(self, monkeypatch):
        from core.llm_service import llm_service
        from config import settings

        monkeypatch.setattr(settings, "ENABLE_CACHING", True)
        monkeypatch.setattr(llm_service, "model", "mistral")

        # Mock complete_async to avoid trying to connect to Ollama
        async def mock_complete(*args, **kwargs):
            return "test response"

        monkeypatch.setattr(llm_service, "complete_async", mock_complete)

        # Ollama doesn't support caching, should degrade gracefully
        result = await llm_service.complete_with_cache_async(
            "test prompt",
            cached_context={"key": "value"},
        )

        assert "content" in result
        assert "cache_hit" in result
        assert "cache_tokens_saved" in result


class TestMultiFileContext:
    """Tests for multi-file context in code generation."""

    @pytest.mark.asyncio
    async def test_code_gen_get_related_files_context_no_results(self, monkeypatch):
        from modules.code_gen.handler import code_gen_handler
        from storage.vector_store import vector_store

        # Mock vector store to return no results
        def mock_query(*args, **kwargs):
            return []

        monkeypatch.setattr(vector_store, 'query', mock_query)
        context = await code_gen_handler._get_related_files_context("some query")
        assert context == ""

    @pytest.mark.asyncio
    async def test_code_gen_get_related_files_context_with_results(self, monkeypatch):
        from modules.code_gen.handler import code_gen_handler
        from storage.vector_store import vector_store

        # Mock vector store to return results
        def mock_query(*args, **kwargs):
            return [
                {"text": "def helper_func(): pass", "distance": 0.1},
                {"text": "class Helper: pass", "distance": 0.2},
            ]

        monkeypatch.setattr(vector_store, 'query', mock_query)
        context = await code_gen_handler._get_related_files_context("helper function")
        assert "Related code files" in context
        assert "def helper_func" in context
        assert "class Helper" in context

    @pytest.mark.asyncio
    async def test_code_gen_get_related_files_context_error_handling(self, monkeypatch):
        from modules.code_gen.handler import code_gen_handler
        from storage.vector_store import vector_store

        # Mock vector store to raise an error
        def mock_query_error(*args, **kwargs):
            raise RuntimeError("Vector store error")

        monkeypatch.setattr(vector_store, 'query', mock_query_error)
        context = await code_gen_handler._get_related_files_context("test")
        assert context == ""  # Should gracefully return empty string

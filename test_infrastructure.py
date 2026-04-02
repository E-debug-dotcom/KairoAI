import logging
import time

import pytest
import respx
import httpx
from fastapi.testclient import TestClient

from core.task_router import TaskRouter
from core.llm_service import llm_service, LLMService
from storage.vector_store import VectorStore
from main import app


@pytest.mark.asyncio
async def test_task_router_alias_dispatch():
    router = TaskRouter()
    async def dummy(payload):
        return {"ok": True, "payload": payload}

    router.register("job_application", dummy)

    result = await router.dispatch("job", {"x": 1})
    assert result["ok"]
    assert result["payload"] == {"x": 1}


def test_vector_store_query_on_empty_collection(monkeypatch):
    store = VectorStore()
    store._initialized = True

    class DummyCollection:
        def count(self):
            return 0

        def query(self, *args, **kwargs):
            raise AssertionError("query should not be called on empty collection")

    store._collection = DummyCollection()

    results = store.query(query_text="test", top_k=3)
    assert results == []


def test_vector_store_query_chromadb_invalid_argument(monkeypatch):
    store = VectorStore()
    store._initialized = True

    class DummyCollection:
        def count(self):
            return 4

        def query(self, *args, **kwargs):
            class InvalidArgumentError(Exception):
                pass

            raise InvalidArgumentError("collection has insufficient records")

    store._collection = DummyCollection()
    results = store.query(query_text="test", top_k=5)
    assert results == []


def test_request_id_middleware_passes_header():
    client = TestClient(app)

    headers = {"X-Request-ID": "test-request-id-123"}
    response = client.get("/health", headers=headers)

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == "test-request-id-123"


@pytest.mark.asyncio
async def test_llm_service_complete_async_respx():
    # Mock Ollama /api/generate with a NDJSON payload. Ensure retry path is stable.
    test_url = "http://localhost:11434"  # default path from settings
    llm_service.base_url = test_url

    body = "{\"response\": \"Hello\"}\n{\"response\": \" World\"}\n"

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{test_url}/api/generate").respond(200, content=body)

        result = await llm_service.complete_async("foo")

        assert result == "Hello World"
        assert route.called


def test_span_context_manager(caplog):
    from utils.helpers import span

    caplog.set_level(logging.INFO, logger="span")
    with span("test_span", level="info", aggregate=True, test_key="value"):
        time.sleep(0.01)

    assert "span_test_span" in caplog.text
    assert "span_test_span_agg" in caplog.text
    assert "test_key=value" in caplog.text

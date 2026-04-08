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


@pytest.mark.asyncio
async def test_task_router_resume_coach_dispatch(monkeypatch):
    router = TaskRouter()
    async def dummy(payload):
        return {"ok": True, "payload": payload}

    router.register("resume_coach", dummy)

    result = await router.dispatch("resume_coach", {"x": 1})
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


def test_decision_engine_high_memory_uses_memory_only(monkeypatch):
    from core.decision_engine import decision_engine
    from storage.vector_store import vector_store

    monkeypatch.setattr(vector_store, "query", lambda query_text, top_k=3, category=None: [{"distance": 0.05, "text": "cached"}])

    outcome = decision_engine.evaluate("assistant", {"question": "How are you?"})
    assert outcome.decision_type == "memory_only"
    assert outcome.use_memory is True
    assert outcome.use_llm is False


def test_session_memory_stores_and_prunes():
    from core.session_memory import session_memory

    session_id = "session-abc"
    session_memory.append_user(session_id, "Hello")
    session_memory.append_assistant(session_id, "Hi there")
    session_memory.append_user(session_id, "ok")
    session_memory.prune_low_value(session_id)

    history = session_memory.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_task_route_decision_memory_only(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app
    from storage.vector_store import vector_store

    monkeypatch.setattr(vector_store, "query", lambda query_text, top_k=3, category=None: [{"distance": 0.1, "text": "cached summary"}])

    client = TestClient(app)
    response = client.post(
        "/api/v1/task/",
        json={"task_type": "assistant", "payload": {"question": "a short user query"}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "memory"
    assert "items" in body


def test_learning_dataset_route(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app
    from storage.vector_store import vector_store
    from storage.database import db

    monkeypatch.setattr(
        vector_store,
        "add_chunks",
        lambda chunks, source, category="general", tags=None: [f"id-{i}" for i in range(len(chunks))],
    )
    monkeypatch.setattr(db, "save_task", lambda *args, **kwargs: 1)

    client = TestClient(app)
    response = client.post(
        "/api/v1/learn/dataset",
        json={
            "dataset_name": "security_dataset",
            "category": "security",
            "items": [
                {"title": "Red Team Guide", "content": "Perform reconnaissance, enumerate services."},
                {"title": "Automation Runbook", "content": "Automate alerts with CI/CD monitoring."},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["items_ingested"] == 2
    assert body["data"]["total_chunks"] >= 2


def test_load_json_dataset_from_file():
    from modules.learning.dataset_trainer import load_json_dataset
    from pathlib import Path

    sample_path = Path("examples/security_red_team_dataset.json")
    dataset = load_json_dataset(str(sample_path))

    assert dataset.dataset_name == "security_red_team_automation"
    assert dataset.category == "security"
    assert len(dataset.items) == 3
    assert dataset.items[0].source == "red_team_playbook"


def test_task_route_resume_coach_output_shape(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app
    from core.llm_service import llm_service

    async def fake_complete_async(*args, **kwargs):
        return "- Include quant metrics\n- Add IAM skill line"

    monkeypatch.setattr(llm_service, "complete_async", fake_complete_async)

    payload = {
        "task_type": "resume_coach",
        "payload": {
            "resume_text": "Experienced IT Analyst with endpoint security experience...",
            "job_description": "Looking for a Security Analyst familiar with IAM, incident response...",
            "options": {
                "optimize_for_ats": True,
                "highlight_missing_skills": True,
                "suggest_metrics": True,
            },
            "force_llm": True,
            "session_id": "session-xyz",
        },
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/task/", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    result = body["data"]
    assert result["request_id"] == "session-xyz"
    assert "resume_score" in result
    assert "improvements" in result
    assert "ats_keywords" in result
    assert "warnings" in result
    assert "formatted_resume" in result
    assert "analysis" in result
    assert "coaching" in result


def test_resume_coach_endpoint_review(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app
    from core.llm_service import llm_service

    async def fake_complete_async(*args, **kwargs):
        return "- Add metrics to experience bullet\n- Include IAM tools in skills section"

    monkeypatch.setattr(llm_service, "complete_async", fake_complete_async)

    payload = {
        "resume_text": "Experience: Led team; Skills: Python; Education: BS",
        "job_description": "Need someone with Python and IAM experience",
        "session_id": "session-123",
    }

    client = TestClient(app)
    response = client.post("/api/v1/resume_coach/review", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    result = body["data"]
    assert result["resume_score"] == 0 or isinstance(result["resume_score"], int)


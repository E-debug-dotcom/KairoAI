"""
test_streaming.py — Tests for streaming functionality.
"""

import pytest
import asyncio
from core.streaming_manager import StreamingManager, StreamingSession, stream_with_sinks
from core.llm_service import llm_service


class TestStreamingSession:
    """Tests for StreamingSession."""

    def test_streaming_session_creation(self):
        session = StreamingSession("session-1", "resume_coach")
        assert session.session_id == "session-1"
        assert session.task_type == "resume_coach"
        assert session.is_complete is False
        assert session.error is None
        assert len(session.tokens) == 0

    def test_streaming_session_add_tokens(self):
        session = StreamingSession("session-2", "learning")
        session.tokens.extend(["Hello", " ", "World"])
        assert session.full_text() == "Hello World"

    @pytest.mark.asyncio
    async def test_streaming_session_emit_token(self):
        session = StreamingSession("session-3", "assistant")
        queue = asyncio.Queue()
        session.add_subscriber(queue)

        await session.emit_token("test_token")

        msg = await queue.get()
        assert msg["type"] == "token"
        assert msg["data"] == "test_token"

    @pytest.mark.asyncio
    async def test_streaming_session_emit_done(self):
        session = StreamingSession("session-4", "code_gen")
        queue = asyncio.Queue()
        session.add_subscriber(queue)

        await session.emit_done({"tokens": 42})

        msg = await queue.get()
        assert msg["type"] == "done"
        assert msg["data"]["tokens"] == 42
        assert session.is_complete is True

    @pytest.mark.asyncio
    async def test_streaming_session_emit_error(self):
        session = StreamingSession("session-5", "job_application")
        queue = asyncio.Queue()
        session.add_subscriber(queue)

        await session.emit_error("Test error")

        msg = await queue.get()
        assert msg["type"] == "error"
        assert msg["data"] == "Test error"
        assert session.error == "Test error"

    def test_streaming_session_to_dict(self):
        session = StreamingSession("session-6", "resume")
        session.tokens.extend(["A", "B", "C"])
        session.is_complete = True

        data = session.to_dict()
        assert data["session_id"] == "session-6"
        assert data["task_type"] == "resume"
        assert data["tokens"] == 3
        assert data["is_complete"] is True
        assert data["full_text"] == "ABC"


class TestStreamingManager:
    """Tests for StreamingManager."""

    def test_streaming_manager_create_session(self):
        manager = StreamingManager()
        session = manager.create_session("s1", "task1")
        assert session.session_id == "s1"
        assert manager.session_count() == 1

    def test_streaming_manager_get_session(self):
        manager = StreamingManager()
        created = manager.create_session("s2", "task2")
        retrieved = manager.get_session("s2")
        assert retrieved is not None
        assert retrieved.session_id == "s2"

    def test_streaming_manager_get_nonexistent_session(self):
        manager = StreamingManager()
        retrieved = manager.get_session("nonexistent")
        assert retrieved is None

    def test_streaming_manager_remove_session(self):
        manager = StreamingManager()
        manager.create_session("s3", "task3")
        assert manager.session_count() == 1
        manager.remove_session("s3")
        assert manager.session_count() == 0

    def test_streaming_manager_active_sessions(self):
        manager = StreamingManager()
        manager.create_session("s4", "task4")
        manager.create_session("s5", "task5")
        active = manager.active_sessions()
        assert len(active) == 2
        assert "s4" in active
        assert "s5" in active

    def test_streaming_manager_max_sessions_cleanup(self):
        manager = StreamingManager(max_sessions=3)
        manager.create_session("s1", "t1")
        manager.create_session("s2", "t2")
        manager.create_session("s3", "t3")
        assert manager.session_count() == 3

        # Creating a new session should remove the oldest
        manager.create_session("s4", "t4")
        assert manager.session_count() == 3
        assert manager.get_session("s1") is None  # Oldest removed
        assert manager.get_session("s4") is not None  # New one added


class TestStreamWithSinks:
    """Tests for stream_with_sinks utility."""

    @pytest.mark.asyncio
    async def test_stream_with_sinks_token_callback(self):
        async def token_generator():
            for token in ["a", "b", "c"]:
                yield token

        tokens_received = []

        def on_token(token):
            tokens_received.append(token)

        result = await stream_with_sinks(token_generator(), on_token=on_token)

        assert result == "abc"
        assert tokens_received == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_stream_with_sinks_chunk_callback(self):
        async def token_generator():
            for token in ["a", "b", "c", "d", "e"]:
                yield token

        chunks_received = []

        def on_chunk(chunk):
            chunks_received.append(chunk)

        result = await stream_with_sinks(token_generator(), on_chunk=on_chunk, chunk_size=2)

        assert result == "abcde"
        assert chunks_received == ["ab", "cd", "e"]

    @pytest.mark.asyncio
    async def test_stream_with_sinks_done_callback(self):
        async def token_generator():
            for token in ["x", "y"]:
                yield token

        done_called = []

        def on_done():
            done_called.append(True)

        result = await stream_with_sinks(token_generator(), on_done=on_done)

        assert result == "xy"
        assert done_called == [True]

    @pytest.mark.asyncio
    async def test_stream_with_sinks_all_callbacks(self):
        async def token_generator():
            for token in ["1", "2", "3"]:
                yield token

        tokens = []
        chunks = []
        done_status = []

        result = await stream_with_sinks(
            token_generator(),
            on_token=lambda t: tokens.append(t),
            on_chunk=lambda c: chunks.append(c),
            on_done=lambda: done_status.append("done"),
            chunk_size=2,
        )

        assert result == "123"
        assert tokens == ["1", "2", "3"]
        assert chunks == ["12", "3"]
        assert done_status == ["done"]

    @pytest.mark.asyncio
    async def test_stream_with_sinks_error_propagation(self):
        async def token_generator_with_error():
            yield "a"
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await stream_with_sinks(token_generator_with_error())


@pytest.mark.asyncio
async def test_llm_service_stream_async_respx():
    """Test streaming from LLM service with retry logic."""
    import respx
    import httpx

    test_url = "http://localhost:11434"
    llm_service.base_url = test_url

    # Simulate streaming response from Ollama
    body = (
        '{"response": "Hello"}\n'
        '{"response": " "}\n'
        '{"response": "World"}\n'
        '{"response": "", "done": true}\n'
    )

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(f"{test_url}/api/generate").respond(200, content=body)

        tokens = []
        async for token in llm_service.stream_async("test prompt"):
            tokens.append(token)

        result = "".join(tokens)
        assert result == "Hello World"
        assert route.called

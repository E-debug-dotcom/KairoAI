"""
Smoke test for KairoAI system - verifies basic functionality and imports.
"""

import pytest
from fastapi.testclient import TestClient


def test_imports():
    """Verify all core modules can be imported."""
    from config import settings
    from utils.logger import get_logger
    from core.llm_service import llm_service
    from core.task_router import router
    from core.output_formatter import formatter
    from storage.database import db
    
    assert settings.APP_NAME == "LocalAI System"
    assert settings.APP_VERSION == "1.0.0"


def test_app_creation():
    """Verify FastAPI app can be created."""
    from main import app
    assert app is not None
    assert app.title == "LocalAI System"


def test_app_routes():
    """Verify key routes are registered."""
    from main import app
    
    # Get list of all routes
    routes = [route.path for route in app.routes]
    
    # Check that key endpoints exist
    assert "/" in routes
    assert "/health" in routes


def test_health_endpoint():
    """Test the health check endpoint."""
    from main import app
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_root_endpoint():
    """Test the root endpoint."""
    from main import app
    
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["system"] == "LocalAI System"
    assert data["version"] == "1.0.0"
    assert "docs" in data


def test_task_router():
    """Verify task router can be created and has expected modules."""
    from core.task_router import TaskRouter
    
    router = TaskRouter()
    
    # Test registering a module
    test_handler = lambda x: {"result": "test"}
    router.register("test_module", test_handler)
    
    assert "test_module" in router.available_tasks()


def test_config_settings():
    """Verify configuration settings are loaded correctly."""
    from config import settings
    
    assert settings.APP_NAME
    assert settings.APP_VERSION
    assert settings.API_PORT == 8000
    assert settings.API_HOST == "0.0.0.0"
    assert settings.DATABASE_URL
    assert settings.OLLAMA_BASE_URL


def test_logger():
    """Verify logger can be initialized."""
    from utils.logger import get_logger
    
    logger = get_logger("test")
    assert logger is not None
    # Basic logger test - just ensure it doesn't crash
    logger.info("Test log message")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

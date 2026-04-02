"""
test_tool_calling.py — Tests for LLM tool calling and schema validation.
"""

import pytest
from core.tool_registry import ToolRegistry
from core.output_formatter import OutputFormatter, SchemaValidator
from core.task_router import TaskRouter


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_tool_registry_has_resume_coach(self):
        tool = ToolRegistry.get_tool("resume_coach")
        assert tool is not None
        assert tool.name == "resume_coach"

    def test_tool_registry_list_all_tools(self):
        tools = ToolRegistry.list_tools()
        expected_tools = ["resume_coach", "code_gen", "learning", "job_application", "assistant"]
        for tool in expected_tools:
            assert tool in tools

    def test_tool_registry_to_openai_format(self):
        tools_openai = ToolRegistry.to_openai_format()
        assert len(tools_openai) > 0
        first = tools_openai[0]
        assert "type" in first
        assert first["type"] == "function"
        assert "function" in first
        assert "name" in first["function"]

    def test_tool_registry_to_claude_format(self):
        tools_claude = ToolRegistry.to_claude_format()
        assert len(tools_claude) > 0
        first = tools_claude[0]
        assert "name" in first
        assert "description" in first
        assert "input_schema" in first

    def test_tool_registry_validate_tool_input_valid(self):
        input_data = {
            "resume_text": "My resume",
            "job_description": "Job desc",
        }
        is_valid = ToolRegistry.validate_tool_input("resume_coach", input_data)
        assert is_valid is True

    def test_tool_registry_validate_tool_input_missing_required(self):
        input_data = {
            "resume_text": "My resume",
            # job_description is missing
        }
        is_valid = ToolRegistry.validate_tool_input("resume_coach", input_data)
        assert is_valid is False

    def test_tool_registry_validate_unknown_tool(self):
        input_data = {}
        is_valid = ToolRegistry.validate_tool_input("nonexistent_tool", input_data)
        assert is_valid is False


class TestSchemaValidator:
    """Tests for SchemaValidator."""

    def test_schema_validate_required_fields_all_present(self):
        data = {"field1": "value", "field2": 123}
        is_valid, missing = SchemaValidator.validate_required_fields(data, ["field1", "field2"])
        assert is_valid is True
        assert missing == []

    def test_schema_validate_required_fields_missing(self):
        data = {"field1": "value"}
        is_valid, missing = SchemaValidator.validate_required_fields(data, ["field1", "field2"])
        assert is_valid is False
        assert "field2" in missing

    def test_schema_validate_field_types_string(self):
        data = {"name": "John", "age": 30}
        schema_properties = {"name": {"type": "string"}, "age": {"type": "number"}}
        is_valid, errors = SchemaValidator.validate_field_types(data, schema_properties)
        assert is_valid is True
        assert errors == []

    def test_schema_validate_field_types_type_mismatch(self):
        data = {"name": 123, "age": "thirty"}  # types are swapped
        schema_properties = {"name": {"type": "string"}, "age": {"type": "number"}}
        is_valid, errors = SchemaValidator.validate_field_types(data, schema_properties)
        assert is_valid is False
        assert len(errors) == 2

    def test_schema_validate_full_schema_valid(self):
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["title", "score"],
        }
        data = {"title": "Test", "score": 95}
        is_valid, errors = SchemaValidator.validate(data, schema)
        assert is_valid is True
        assert errors == []

    def test_schema_validate_full_schema_invalid(self):
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "score": {"type": "number"},
            },
            "required": ["title", "score"],
        }
        data = {"title": "Test"}  # missing score
        is_valid, errors = SchemaValidator.validate(data, schema)
        assert is_valid is False
        assert len(errors) > 0


class TestOutputFormatterValidation:
    """Tests for OutputFormatter schema validation and repair."""

    def test_output_formatter_validate_and_repair_valid(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "ENFORCE_OUTPUT_SCHEMA", True)

        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "text": {"type": "string"},
            },
            "required": ["score", "text"],
        }
        data = {"score": 95, "text": "Good"}
        repaired, warnings = OutputFormatter.validate_and_repair("test_task", data, schema)
        assert repaired == data
        assert warnings == []

    def test_output_formatter_validate_and_repair_adds_defaults(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "ENFORCE_OUTPUT_SCHEMA", True)

        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "items": {"type": "array"},
            },
            "required": ["score", "items"],
        }
        data = {"score": 85}  # missing items
        repaired, warnings = OutputFormatter.validate_and_repair("test_task", data, schema)

        assert repaired["score"] == 85
        assert "items" in repaired
        assert repaired["items"] == []
        assert len(warnings) > 0

    def test_output_formatter_validate_and_repair_disabled(self, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, "ENFORCE_OUTPUT_SCHEMA", False)

        schema = {
            "type": "object",
            "required": ["field"],
        }
        data = {}
        repaired, warnings = OutputFormatter.validate_and_repair("test_task", data, schema)
        assert repaired == data
        assert warnings == []


class TestTaskRouterToolCalling:
    """Tests for task router tool calling."""

    @pytest.mark.asyncio
    async def test_task_router_route_tool_call(self):
        router = TaskRouter()

        async def dummy_handler(payload):
            return {
                "status": "success",
                "data": {"result": "Tool executed"},
            }

        router.register("resume_coach", dummy_handler)

        result = await router.route_tool_call("resume_coach", {"resume_text": "My CV"})
        assert result["status"] == "success"
        assert result["data"]["result"] == "Tool executed"

    @pytest.mark.asyncio
    async def test_task_router_route_tool_call_unknown_tool(self):
        router = TaskRouter()

        with pytest.raises(Exception) as exc_info:
            await router.route_tool_call("nonexistent_tool", {})

        assert "No handler registered" in str(exc_info.value)

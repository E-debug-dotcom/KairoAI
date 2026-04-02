"""
core/tool_registry.py — Registry of available tools and their JSON schemas.

This module defines all LLM-callable tools and their input/output schemas.
Tools correspond to registered module handlers in task_router.
"""

from typing import Any, Dict, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


class ToolSchema:
    """Describes a tool's interface for LLM tool calling."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: List[str],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function_calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def to_claude_format(self) -> Dict[str, Any]:
        """Convert to Claude tool_use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


class ToolRegistry:
    """Registry of all available tools for LLM invocation."""

    TOOLS = {
        "resume_coach": ToolSchema(
            name="resume_coach",
            description="Analyze and improve a resume for a specific job. Returns resume scoring, improvements, ATS keywords, and formatted output.",
            parameters={
                "resume_text": {
                    "type": "string",
                    "description": "The full text of the resume to analyze.",
                },
                "job_description": {
                    "type": "string",
                    "description": "The job description to match against.",
                },
                "sub_task": {
                    "type": "string",
                    "enum": ["review", "action_plan", "feedback"],
                    "description": "The type of coaching: review (analyze), action_plan (improvements), or feedback (detailed bullets).",
                },
                "optimize_for_ats": {
                    "type": "boolean",
                    "description": "Whether to optimize for ATS (Applicant Tracking System) compatibility.",
                },
                "highlight_missing_skills": {
                    "type": "boolean",
                    "description": "Whether to highlight skills missing from the resume.",
                },
                "suggest_metrics": {
                    "type": "boolean",
                    "description": "Whether to suggest quantified achievements.",
                },
            },
            required=["resume_text", "job_description"],
        ),
        "code_gen": ToolSchema(
            name="code_gen",
            description="Generate code based on requirements. Returns generated code, explanation, and test stubs.",
            parameters={
                "language": {
                    "type": "string",
                    "description": "Programming language (e.g., 'python', 'javascript', 'go').",
                },
                "requirements": {
                    "type": "string",
                    "description": "Detailed description of what code to generate.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context or existing code for consistency.",
                },
            },
            required=["language", "requirements"],
        ),
        "learning": ToolSchema(
            name="learning",
            description="Generate a learning path or educational material. Returns structured lessons, resources, and assessment.",
            parameters={
                "topic": {
                    "type": "string",
                    "description": "The topic to learn about.",
                },
                "level": {
                    "type": "string",
                    "enum": ["beginner", "intermediate", "advanced"],
                    "description": "The learner's current level.",
                },
                "format": {
                    "type": "string",
                    "enum": ["structured_path", "detailed_lesson", "quick_guide"],
                    "description": "The format of the learning material.",
                },
            },
            required=["topic"],
        ),
        "job_application": ToolSchema(
            name="job_application",
            description="Analyze a job application or generate application materials. Returns analysis, recommendations, or draft materials.",
            parameters={
                "job_description": {
                    "type": "string",
                    "description": "The job posting to apply for.",
                },
                "resume_text": {
                    "type": "string",
                    "description": "The applicant's resume.",
                },
                "sub_task": {
                    "type": "string",
                    "enum": ["match_analysis", "cover_letter", "interview_prep"],
                    "description": "The type of application help needed.",
                },
            },
            required=["job_description"],
        ),
        "assistant": ToolSchema(
            name="assistant",
            description="General-purpose assistant for Q&A, analysis, and reasoning. Returns detailed responses and recommendations.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The user's question or request.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context for the query.",
                },
            },
            required=["query"],
        ),
    }

    @classmethod
    def get_tool(cls, tool_name: str) -> Optional[ToolSchema]:
        """Retrieve a tool schema by name."""
        return cls.TOOLS.get(tool_name.lower())

    @classmethod
    def list_tools(cls) -> List[str]:
        """Return list of all available tool names."""
        return sorted(cls.TOOLS.keys())

    @classmethod
    def to_openai_format(cls) -> List[Dict[str, Any]]:
        """Export all tools in OpenAI function_calling format."""
        return [tool.to_openai_format() for tool in cls.TOOLS.values()]

    @classmethod
    def to_claude_format(cls) -> List[Dict[str, Any]]:
        """Export all tools in Claude tool_use format."""
        return [tool.to_claude_format() for tool in cls.TOOLS.values()]

    @classmethod
    def validate_tool_input(cls, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Validate that tool input matches the schema."""
        tool = cls.get_tool(tool_name)
        if not tool:
            logger.warning("Tool '%s' not found in registry", tool_name)
            return False

        # Check required fields
        for required_field in tool.required:
            if required_field not in tool_input:
                logger.warning(
                    "Tool '%s' input missing required field: %s",
                    tool_name,
                    required_field,
                )
                return False

        return True


# ─── Output schemas for each task type ────────────────────────────────────────

OUTPUT_SCHEMAS = {
    "resume_coach": {
        "type": "object",
        "properties": {
            "resume_score": {"type": "number", "minimum": 0, "maximum": 100},
            "formatted_resume": {"type": "string"},
            "improvements": {
                "type": "array",
                "items": {"type": "string"},
            },
            "ats_keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
            },
            "analysis": {"type": "string"},
            "coaching": {"type": "string"},
        },
        "required": ["resume_score", "formatted_resume", "improvements"],
    },
    "code_gen": {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "language": {"type": "string"},
            "explanation": {"type": "string"},
            "tests": {"type": "string"},
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["code", "language"],
    },
    "learning": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "lessons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
            "resources": {
                "type": "array",
                "items": {"type": "string"},
            },
            "exercises": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["title"],
    },
    "job_application": {
        "type": "object",
        "properties": {
            "match_score": {"type": "number", "minimum": 0, "maximum": 100},
            "matching_skills": {
                "type": "array",
                "items": {"type": "string"},
            },
            "missing_skills": {
                "type": "array",
                "items": {"type": "string"},
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "cover_letter": {"type": "string"},
            "interview_tips": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["match_score"],
    },
    "assistant": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "reasoning": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "sources": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["answer"],
    },
    "decision": {
        "type": "object",
        "properties": {
            "decision_type": {
                "type": "string",
                "enum": ["memory_only", "llm_only", "tools_llm", "full_pipeline"],
            },
            "reason": {"type": "string"},
            "memory_score": {"type": "number", "minimum": 0, "maximum": 1},
            "use_memory": {"type": "boolean"},
            "use_llm": {"type": "boolean"},
            "use_tools": {"type": "boolean"},
        },
        "required": ["decision_type", "reason"],
    },
}


def get_output_schema(task_type: str) -> Optional[Dict[str, Any]]:
    """Retrieve output schema for a task type."""
    return OUTPUT_SCHEMAS.get(task_type.lower())

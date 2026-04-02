"""
core/output_formatter.py — Standardized response structure for all modules.

Every module returns its raw output through this formatter, ensuring
consistent JSON structure that downstream consumers can rely on.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"       # Completed with warnings
    ERROR = "error"


class SchemaValidator:
    """Validates output against JSON schemas."""

    @staticmethod
    def validate_required_fields(data: dict, required: list[str]) -> tuple[bool, list[str]]:
        """
        Check if required fields are present.

        Returns:
            (is_valid, missing_fields)
        """
        missing = [field for field in required if field not in data]
        return len(missing) == 0, missing

    @staticmethod
    def validate_field_types(data: dict, schema_properties: dict) -> tuple[bool, list[str]]:
        """
        Check if fields match their expected types (basic validation).

        Returns:
            (is_valid, type_errors)
        """
        errors = []
        for field, spec in schema_properties.items():
            if field not in data:
                continue

            value = data[field]
            expected_type = spec.get("type")

            # Basic type checking
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"Field '{field}' should be string, got {type(value).__name__}")
            elif expected_type == "number" and not isinstance(value, (int, float)):
                errors.append(f"Field '{field}' should be number, got {type(value).__name__}")
            elif expected_type == "boolean" and not isinstance(value, bool):
                errors.append(f"Field '{field}' should be boolean, got {type(value).__name__}")
            elif expected_type == "array" and not isinstance(value, list):
                errors.append(f"Field '{field}' should be array, got {type(value).__name__}")
            elif expected_type == "object" and not isinstance(value, dict):
                errors.append(f"Field '{field}' should be object, got {type(value).__name__}")

        return len(errors) == 0, errors

    @staticmethod
    def validate(data: dict, schema: dict) -> tuple[bool, list[str]]:
        """
        Validate data against a JSON schema.

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        if schema.get("type") != "object":
            errors.append("Schema must be an object type")
            return False, errors

        # Check required fields
        required = schema.get("required", [])
        is_required_valid, missing = SchemaValidator.validate_required_fields(data, required)
        if not is_required_valid:
            errors.append(f"Missing required fields: {missing}")

        # Check field types
        properties = schema.get("properties", {})
        is_types_valid, type_errors = SchemaValidator.validate_field_types(data, properties)
        errors.extend(type_errors)

        return len(errors) == 0, errors


class OutputFormatter:
    """
    Wraps module outputs into a standardized envelope.

    Standard response shape:
    {
        "status": "success" | "partial" | "error",
        "task_type": "resume",
        "timestamp": "2025-01-15T10:30:00Z",
        "data": { ... },          <- main result payload
        "meta": { ... },          <- timing, model, token counts
        "warnings": [ ... ],      <- non-fatal issues
        "error": null | "..."     <- set only on error
    }
    """

    @staticmethod
    def success(
        task_type: str,
        data: dict,
        meta: Optional[dict] = None,
        warnings: Optional[list[str]] = None,
    ) -> dict:
        """Build a successful response envelope."""
        return {
            "status": TaskStatus.SUCCESS,
            "task_type": task_type,
            "timestamp": OutputFormatter._utc_now(),
            "data": data,
            "meta": meta or {},
            "warnings": warnings or [],
            "error": None,
        }

    @staticmethod
    def partial(
        task_type: str,
        data: dict,
        warnings: list[str],
        meta: Optional[dict] = None,
    ) -> dict:
        """Build a partial-success response (completed with non-fatal issues)."""
        return {
            "status": TaskStatus.PARTIAL,
            "task_type": task_type,
            "timestamp": OutputFormatter._utc_now(),
            "data": data,
            "meta": meta or {},
            "warnings": warnings,
            "error": None,
        }

    @staticmethod
    def error(
        task_type: str,
        message: str,
        meta: Optional[dict] = None,
    ) -> dict:
        """Build an error response envelope."""
        logger.error("Formatting error response for task='%s': %s", task_type, message)
        return {
            "status": TaskStatus.ERROR,
            "task_type": task_type,
            "timestamp": OutputFormatter._utc_now(),
            "data": {},
            "meta": meta or {},
            "warnings": [],
            "error": message,
        }

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def to_json(result: dict, indent: int = 2) -> str:
        """Serialize a formatted result to a JSON string."""
        return json.dumps(result, indent=indent, default=str)

    @staticmethod
    def validate_and_repair(task_type: str, data: dict, schema: dict) -> tuple[dict, list[str]]:
        """
        Validate data against schema; if invalid, repair and return warnings.

        Returns:
            (repaired_data, validation_warnings)
        """
        if not settings.ENFORCE_OUTPUT_SCHEMA:
            return data, []

        is_valid, errors = SchemaValidator.validate(data, schema)

        if is_valid:
            return data, []

        # Repair: add missing required fields with defaults
        warnings = [f"Schema validation failed: {err}" for err in errors]
        repaired = data.copy()
        required = schema.get("required", [])

        for field in required:
            if field not in repaired:
                # Provide sensible defaults
                prop_spec = schema.get("properties", {}).get(field, {})
                field_type = prop_spec.get("type")

                if field_type == "string":
                    repaired[field] = ""
                elif field_type == "number":
                    repaired[field] = 0
                elif field_type == "boolean":
                    repaired[field] = False
                elif field_type == "array":
                    repaired[field] = []
                elif field_type == "object":
                    repaired[field] = {}
                else:
                    repaired[field] = None

                warnings.append(f"Auto-filled missing field '{field}' with default {field_type}")

        logger.warning(
            "Schema repair for '%s': %d errors fixed, %d warnings",
            task_type,
            len(errors),
            len(warnings),
        )

        return repaired, warnings


# ─── Module-level singleton ───────────────────────────────────────────────────
formatter = OutputFormatter()

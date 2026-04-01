"""
prompts/prompt_manager.py — Central prompt template registry.

Templates are stored per-module and rendered with variable substitution.
This keeps prompts out of module logic and makes them easy to tune.
"""

from typing import Any
from utils.logger import get_logger

logger = get_logger(__name__)


class PromptManager:
    """
    Manages prompt templates keyed by (module, template_name).

    Templates use Python str.format()-style placeholders: {variable_name}.
    Modules register their templates at startup. Rendering injects
    runtime values into the template.
    """

    def __init__(self):
        self._templates: dict[str, str] = {}

    def register(self, module: str, name: str, template: str) -> None:
        """Register a prompt template under a namespaced key."""
        key = self._key(module, name)
        self._templates[key] = template
        logger.debug("Registered prompt template: %s", key)

    def render(self, module: str, name: str, **kwargs: Any) -> str:
        """
        Render a named template with the provided variables.

        Args:
            module: Module name (e.g., "resume").
            name: Template name within the module (e.g., "optimize").
            **kwargs: Variables to inject into the template.

        Returns:
            Rendered prompt string.

        Raises:
            PromptNotFoundError: If the template is not registered.
            KeyError: If a required placeholder is missing from kwargs.
        """
        key = self._key(module, name)
        if key not in self._templates:
            raise PromptNotFoundError(
                f"No prompt template found for module='{module}', name='{name}'"
            )

        template = self._templates[key]
        try:
            rendered = template.format(**kwargs)
            logger.debug("Rendered prompt '%s' (%d chars)", key, len(rendered))
            return rendered
        except KeyError as e:
            raise KeyError(
                f"Template '{key}' is missing required variable: {e}"
            ) from e

    def list_templates(self) -> list[str]:
        return sorted(self._templates.keys())

    @staticmethod
    def _key(module: str, name: str) -> str:
        return f"{module.lower()}::{name.lower()}"


class PromptNotFoundError(Exception):
    pass


# ─── Singleton ────────────────────────────────────────────────────────────────
prompt_manager = PromptManager()

"""
modules/job_application/handler.py — Job application assistant module.

Handles:
- Answering individual application questions
- Generating answers for multi-question screening forms
- Saving application records to storage
"""

import time

from core.llm_service import llm_service
from core.output_formatter import formatter
from prompts.prompt_manager import prompt_manager
from prompts.assistant_prompts import JOB_APP_SYSTEM_PROMPT
from storage.database import db
from utils.logger import get_logger
from utils.helpers import truncate_text

logger = get_logger(__name__)


class JobApplicationHandler:
    """
    Generates professional job application content from applicant background.
    Does not fabricate experience — only draws from provided background.
    """

    async def handle(self, payload: dict) -> dict:
        sub_task = payload.get("sub_task", "question")
        logger.info("Job application module handling sub_task='%s'", sub_task)

        dispatch = {
            "question": self._answer_question,
            "screening": self._answer_screening,
        }

        if sub_task not in dispatch:
            return formatter.error(
                "job_application",
                f"Unknown sub_task '{sub_task}'. Available: {list(dispatch.keys())}",
            )

        return await dispatch[sub_task](payload)

    async def _answer_question(self, payload: dict) -> dict:
        """Generate an answer to a single application question."""
        question = payload.get("question", "").strip()
        background = payload.get("background", "").strip()
        role_title = payload.get("role_title", "[Role Title]")
        company_name = payload.get("company_name", "[Company Name]")
        max_words = payload.get("max_words", 150)

        if not question:
            return formatter.error("job_application", "Field 'question' is required.")
        if not background:
            return formatter.error("job_application", "Field 'background' (applicant background) is required.")

        try:
            prompt = prompt_manager.render(
                "job_application", "question",
                background=truncate_text(background, 2000),
                role_title=role_title,
                company_name=company_name,
                question=question,
                max_words=max_words,
            )
        except Exception as e:
            return formatter.error("job_application", f"Prompt error: {str(e)}")

        start = time.time()
        answer = llm_service.complete(
            prompt=prompt,
            system_prompt=JOB_APP_SYSTEM_PROMPT,
            temperature=0.4,
        )
        duration = time.time() - start

        return formatter.success(
            "job_application",
            {
                "question": question,
                "answer": answer,
                "role_title": role_title,
                "company_name": company_name,
            },
            meta={"duration_seconds": round(duration, 2)},
        )

    async def _answer_screening(self, payload: dict) -> dict:
        """Generate answers for multiple screening questions at once."""
        questions_raw = payload.get("questions", [])
        background = payload.get("background", "").strip()
        role_title = payload.get("role_title", "[Role Title]")
        company_name = payload.get("company_name", "[Company Name]")

        if not questions_raw:
            return formatter.error("job_application", "Field 'questions' (list) is required.")
        if not background:
            return formatter.error("job_application", "Field 'background' is required.")

        # Format questions as numbered list
        if isinstance(questions_raw, list):
            questions_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions_raw))
        else:
            questions_str = str(questions_raw)

        try:
            prompt = prompt_manager.render(
                "job_application", "screening",
                background=truncate_text(background, 2000),
                role_title=role_title,
                company_name=company_name,
                questions=questions_str,
            )
        except Exception as e:
            return formatter.error("job_application", f"Prompt error: {str(e)}")

        start = time.time()
        answers = llm_service.complete(
            prompt=prompt,
            system_prompt=JOB_APP_SYSTEM_PROMPT,
            temperature=0.4,
        )
        duration = time.time() - start

        # Save to job applications table
        app_id = db.save_job_application(
            company_name=company_name,
            role_title=role_title,
            application_answers={"questions": questions_raw, "answers_raw": answers},
        )

        return formatter.success(
            "job_application",
            {
                "answers": answers,
                "questions": questions_raw,
                "role_title": role_title,
                "company_name": company_name,
                "application_id": app_id,
            },
            meta={"duration_seconds": round(duration, 2)},
        )


# ─── Singleton ────────────────────────────────────────────────────────────────
job_application_handler = JobApplicationHandler()

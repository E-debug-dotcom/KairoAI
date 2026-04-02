"""
modules/resume_coach/handler.py — Resume coaching module.

A lightweight coaching module for scoring, gap analysis, and actionable
resume improvement guidance.
"""

import time
from typing import Any

from core.llm_service import llm_service
from core.output_formatter import formatter
from prompts.prompt_manager import prompt_manager
from prompts.resume_coach_prompts import RESUME_COACH_SYSTEM_PROMPT
from parsers.document_parser import document_parser
from analysis.engine import resume_analyzer
from storage.database import db
from utils.logger import get_logger
from utils.helpers import truncate_text

logger = get_logger(__name__)


class ResumeCoachHandler:
    """
    Handles resume coaching sub_tasks:
      - review
      - action_plan
      - bullet_feedback
    """

    async def handle(self, payload: dict[str, Any]) -> dict:
        sub_task = payload.get("sub_task", "review") or "review"
        logger.info("ResumeCoach module handling sub_task='%s'", sub_task)

        dispatch = {
            "review": self._review,
            "action_plan": self._action_plan,
            "bullet_feedback": self._bullet_feedback,
        }

        if sub_task not in dispatch:
            return formatter.error(
                "resume_coach",
                f"Unknown sub_task '{sub_task}'. Available: {list(dispatch.keys())}",
            )

        return await dispatch[sub_task](payload)

    async def _review(self, payload: dict[str, Any]) -> dict:
        try:
            resume_text = self._get_resume_text(payload)
            jd_text = self._require_field(payload, "job_description")
        except ValueError as e:
            return formatter.error("resume_coach", str(e))

        start = time.time()
        analysis = resume_analyzer.analyze(resume_text, jd_text)

        try:
            prompt = prompt_manager.render(
                "resume_coach",
                "review",
                resume_text=truncate_text(resume_text, 3000),
                job_description=truncate_text(jd_text, 2000),
                similarity=analysis["similarity_score"],
                keyword_match=analysis["keyword_match_score"],
                matched_keywords=", ".join(analysis["matched_keywords"][:20]),
                missing_keywords=", ".join(analysis["missing_keywords"][:20]),
            )
        except Exception as e:
            return formatter.error("resume_coach", f"Prompt render failed: {str(e)}")

        coaching = await llm_service.complete_async(
            prompt=prompt,
            system_prompt=RESUME_COACH_SYSTEM_PROMPT,
            temperature=0.45,
        )

        duration = time.time() - start

        resume_score = int(round(analysis["similarity_score"] * 100))
        ats_keywords = analysis.get("matched_keywords", [])[:20]

        # Convert LLM coaching block into structured suggestions if possible
        improvements = []
        for line in coaching.splitlines():
            normalized = line.strip()
            if normalized.startswith("- "):
                improvements.append({"section": "General", "suggestion": normalized[2:].strip()})

        if not improvements:
            improvements.append(
                {
                    "section": "General",
                    "suggestion": "Review assistant coaching text for recommended improvements.",
                }
            )

        formatted_resume = {
            "raw_text": resume_text,
            "summary": resume_text[:500],
            "sections": {
                "experience": resume_text.split("Experience")[-1][:500] if "Experience" in resume_text else "",
                "skills": resume_text.split("Skills")[-1][:300] if "Skills" in resume_text else "",
                "education": resume_text.split("Education")[-1][:300] if "Education" in resume_text else "",
            },
        }

        request_id = payload.get("request_id") or payload.get("session_id") or ""

        result = {
            "request_id": request_id,
            "resume_score": resume_score,
            "improvements": improvements,
            "ats_keywords": ats_keywords,
            "warnings": [],
            "formatted_resume": formatted_resume,
            "analysis": {
                "similarity_score": analysis["similarity_score"],
                "keyword_match_score": analysis["keyword_match_score"],
                "missing_keywords": analysis["missing_keywords"],
                "matched_keywords": analysis["matched_keywords"],
                "total_jd_keywords": analysis["total_jd_keywords"],
            },
            "coaching": coaching,
        }

        if analysis["similarity_score"] < 0.3:
            result["warnings"].append("Low similarity score; consider larger resume rework.")
        if analysis["missing_count"] > 15:
            result["warnings"].append(
                "Many JD keywords missing; prioritize skill+accomplishment translation."
            )

        db.save_task(
            task_type="resume_coach",
            input_summary={
                "sub_task": "review",
                "resume_length": len(resume_text),
                "jd_length": len(jd_text),
            },
            result=formatter.success("resume_coach", result, warnings=result["warnings"]),
            duration_seconds=duration,
            model_used=llm_service.model,
            score=analysis["similarity_score"],
        )

        return formatter.success(
            "resume_coach",
            result,
            meta={"duration_seconds": round(duration, 2), "model": llm_service.model},
            warnings=result["warnings"],
        )

    async def _action_plan(self, payload: dict[str, Any]) -> dict:
        try:
            resume_text = self._get_resume_text(payload)
        except ValueError as e:
            return formatter.error("resume_coach", str(e))

        try:
            prompt = prompt_manager.render(
                "resume_coach",
                "action_plan",
                resume_text=truncate_text(resume_text, 3000),
            )
        except Exception as e:
            return formatter.error("resume_coach", f"Prompt render failed: {str(e)}")

        plan = await llm_service.complete_async(
            prompt=prompt,
            system_prompt=RESUME_COACH_SYSTEM_PROMPT,
            temperature=0.45,
        )

        return formatter.success(
            "resume_coach",
            {"action_plan": plan},
            meta={"model": llm_service.model},
        )

    async def _bullet_feedback(self, payload: dict[str, Any]) -> dict:
        bullets = payload.get("bullets", "").strip()
        if not bullets:
            return formatter.error("resume_coach", "Field 'bullets' is required for bullet_feedback.")

        try:
            prompt = prompt_manager.render(
                "resume_coach",
                "bullet_feedback",
                bullets=bullets,
            )
        except Exception as e:
            return formatter.error("resume_coach", f"Prompt render failed: {str(e)}")

        feedback = await llm_service.complete_async(
            prompt=prompt,
            system_prompt=RESUME_COACH_SYSTEM_PROMPT,
        )

        return formatter.success(
            "resume_coach",
            {"bullet_feedback": feedback},
            meta={"model": llm_service.model},
        )

    def _get_resume_text(self, payload: dict[str, Any]) -> str:
        if "resume_file_content" in payload and "resume_filename" in payload:
            return document_parser.parse_upload(
                payload["resume_filename"],
                payload["resume_file_content"],
            )
        if "resume_text" in payload:
            return document_parser.parse_text_input(payload["resume_text"])

        raise ValueError(
            "No resume input found. Provide 'resume_text' or ('resume_file_content' and 'resume_filename')."
        )

    def _require_field(self, payload: dict[str, Any], field: str) -> str:
        value = payload.get(field, "")
        if isinstance(value, str):
            value = value.strip()
        if not value:
            raise ValueError(f"Required field '{field}' is missing or empty.")
        return value


resume_coach_handler = ResumeCoachHandler()

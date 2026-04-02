"""
modules/resume/handler.py — Resume optimization module.

Pipeline:
1. Parse resume (PDF/DOCX/text) + job description text
2. Run keyword extraction and similarity analysis
3. Build prompt with analysis results
4. Call LLM for optimized rewrite
5. Format and return structured result
"""

import time
from typing import Optional

from core.llm_service import llm_service
from core.output_formatter import formatter
from prompts.prompt_manager import prompt_manager
from prompts.resume_prompts import RESUME_SYSTEM_PROMPT
from parsers.document_parser import document_parser
from analysis.engine import resume_analyzer
from storage.database import db
from utils.logger import get_logger
from utils.helpers import truncate_text, format_list_as_bullets

logger = get_logger(__name__)


class ResumeHandler:
    """
    Handles all resume-related tasks:
    - optimize: Full ATS optimization with keyword injection
    - analyze: Score and gap analysis only (no rewrite)
    - cover_letter: Generate a tailored cover letter
    - star_rewrite: Rewrite specific bullets in STAR format
    """

    async def handle(self, payload: dict) -> dict:
        """
        Main entry point called by the task router.

        Expected payload keys:
            sub_task (str): "optimize" | "analyze" | "cover_letter" | "star_rewrite"
            resume_text (str, optional): Raw resume text
            resume_file_content (bytes, optional): Uploaded file bytes
            resume_filename (str, optional): Filename for type detection
            job_description (str): Job posting text
            company_name (str, optional): For cover letters
            role_title (str, optional): For cover letters
            bullets (str, optional): For star_rewrite
        """
        sub_task = payload.get("sub_task", "optimize")
        logger.info("Resume module handling sub_task='%s'", sub_task)

        dispatch = {
            "optimize": self._optimize,
            "analyze": self._analyze,
            "cover_letter": self._cover_letter,
            "star_rewrite": self._star_rewrite,
        }

        if sub_task not in dispatch:
            return formatter.error(
                "resume",
                f"Unknown sub_task '{sub_task}'. Available: {list(dispatch.keys())}",
            )

        return await dispatch[sub_task](payload)

    # ─── Sub-task: Optimize ───────────────────────────────────────────────────

    async def _optimize(self, payload: dict) -> dict:
        start = time.time()

        # Step 1: Parse inputs
        try:
            resume_text = self._get_resume_text(payload)
            jd_text = self._require_field(payload, "job_description")
        except ValueError as e:
            return formatter.error("resume", str(e))

        # Step 2: Analysis
        analysis = resume_analyzer.analyze(resume_text, jd_text)
        logger.info(
            "Analysis complete | similarity=%.2f | keyword_match=%.2f | missing=%d",
            analysis["similarity_score"],
            analysis["keyword_match_score"],
            analysis["missing_count"],
        )

        # Step 3: Build and send prompt
        missing_kw_str = ", ".join(analysis["missing_keywords"][:20]) or "none identified"

        try:
            prompt = prompt_manager.render(
                "resume", "optimize",
                job_description=truncate_text(jd_text, 2000),
                resume_text=truncate_text(resume_text, 3000),
                missing_keywords=missing_kw_str,
            )
        except Exception as e:
            return formatter.error("resume", f"Prompt rendering failed: {str(e)}")

        llm_response = await llm_service.complete_async(
            prompt=prompt,
            system_prompt=RESUME_SYSTEM_PROMPT,
        )

        duration = time.time() - start

        # Step 4: Structure result
        result_data = {
            "optimized_resume": llm_response,
            "analysis": {
                "similarity_score": analysis["similarity_score"],
                "keyword_match_score": analysis["keyword_match_score"],
                "missing_keywords": analysis["missing_keywords"],
                "matched_keywords": analysis["matched_keywords"][:15],
                "total_jd_keywords": analysis["total_jd_keywords"],
            },
        }

        warnings = []
        if analysis["similarity_score"] < 0.3:
            warnings.append(
                "Low similarity score. Consider significant restructuring or this role may not be a strong fit."
            )
        if analysis["missing_count"] > 15:
            warnings.append(
                f"{analysis['missing_count']} keywords could not be incorporated. "
                "Some may require genuine skill development."
            )

        # Step 5: Persist
        db.save_task(
            task_type="resume",
            input_summary={
                "sub_task": "optimize",
                "jd_preview": jd_text[:200],
                "resume_length": len(resume_text),
            },
            result=formatter.success("resume", result_data, warnings=warnings),
            duration_seconds=duration,
            model_used=llm_service.model,
            score=analysis["similarity_score"],
        )

        return formatter.success(
            "resume",
            result_data,
            meta={"duration_seconds": round(duration, 2), "model": llm_service.model},
            warnings=warnings,
        )

    # ─── Sub-task: Analyze ────────────────────────────────────────────────────

    async def _analyze(self, payload: dict) -> dict:
        try:
            resume_text = self._get_resume_text(payload)
            jd_text = self._require_field(payload, "job_description")
        except ValueError as e:
            return formatter.error("resume", str(e))

        analysis = resume_analyzer.analyze(resume_text, jd_text)

        # Get LLM narrative analysis
        try:
            prompt = prompt_manager.render(
                "resume", "match_analysis",
                job_description=truncate_text(jd_text, 2000),
                resume_text=truncate_text(resume_text, 3000),
                score=analysis["similarity_score"],
            )
            narrative = await llm_service.complete_async(
                prompt=prompt,
                system_prompt=RESUME_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.warning("LLM narrative analysis failed: %s", str(e))
            narrative = "LLM narrative unavailable."

        return formatter.success(
            "resume",
            {
                "scores": {
                    "cosine_similarity": analysis["similarity_score"],
                    "keyword_match": analysis["keyword_match_score"],
                },
                "keywords": {
                    "matched": analysis["matched_keywords"],
                    "missing": analysis["missing_keywords"],
                    "total_in_jd": analysis["total_jd_keywords"],
                },
                "narrative_analysis": narrative,
            },
        )

    # ─── Sub-task: Cover Letter ───────────────────────────────────────────────

    async def _cover_letter(self, payload: dict) -> dict:
        try:
            resume_text = self._get_resume_text(payload)
            jd_text = self._require_field(payload, "job_description")
        except ValueError as e:
            return formatter.error("resume", str(e))

        company_name = payload.get("company_name", "[Company Name]")
        role_title = payload.get("role_title", "[Role Title]")

        # Use first 1500 chars of resume as "summary" for the prompt
        resume_summary = truncate_text(resume_text, 1500)

        try:
            prompt = prompt_manager.render(
                "resume", "cover_letter",
                resume_summary=resume_summary,
                job_description=truncate_text(jd_text, 1500),
                company_name=company_name,
                role_title=role_title,
            )
        except Exception as e:
            return formatter.error("resume", f"Prompt error: {str(e)}")

        cover_letter = await llm_service.complete_async(
            prompt=prompt,
            system_prompt=RESUME_SYSTEM_PROMPT,
            temperature=0.5,  # Slightly more creative for cover letters
        )

        # Optionally save to job applications table
        app_id = db.save_job_application(
            company_name=company_name,
            role_title=role_title,
            cover_letter=cover_letter,
            job_description=jd_text,
        )

        return formatter.success(
            "resume",
            {
                "cover_letter": cover_letter,
                "company_name": company_name,
                "role_title": role_title,
                "application_id": app_id,
            },
        )

    # ─── Sub-task: STAR Rewrite ───────────────────────────────────────────────

    async def _star_rewrite(self, payload: dict) -> dict:
        bullets = payload.get("bullets", "").strip()
        if not bullets:
            return formatter.error("resume", "No bullets provided for STAR rewrite.")

        try:
            prompt = prompt_manager.render(
                "resume", "star_rewrite",
                bullets=bullets,
            )
        except Exception as e:
            return formatter.error("resume", f"Prompt error: {str(e)}")

        result = await llm_service.complete_async(
            prompt=prompt,
            system_prompt=RESUME_SYSTEM_PROMPT,
        )

        return formatter.success("resume", {"rewritten_bullets": result})

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _get_resume_text(self, payload: dict) -> str:
        """Extract resume text from payload — file upload or raw text."""
        if "resume_file_content" in payload and "resume_filename" in payload:
            return document_parser.parse_upload(
                payload["resume_filename"],
                payload["resume_file_content"],
            )
        elif "resume_text" in payload:
            return document_parser.parse_text_input(payload["resume_text"])
        else:
            raise ValueError(
                "No resume provided. Send either 'resume_text' or "
                "'resume_file_content' + 'resume_filename'."
            )

    def _require_field(self, payload: dict, field: str) -> str:
        value = payload.get(field, "").strip()
        if not value:
            raise ValueError(f"Required field '{field}' is missing or empty.")
        return value


# ─── Singleton + router registration ─────────────────────────────────────────
resume_handler = ResumeHandler()

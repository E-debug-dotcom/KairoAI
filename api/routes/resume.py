"""
api/routes/resume.py — Resume-specific FastAPI endpoints.
Supports both JSON text input and multipart file uploads.
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from modules.resume.handler import resume_handler
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

resume_route = APIRouter(prefix="/resume", tags=["Resume"])


# ─── Request models ───────────────────────────────────────────────────────────

class ResumeOptimizeTextRequest(BaseModel):
    resume_text: str = Field(..., description="Raw resume text content")
    job_description: str = Field(..., description="Job posting text")

    class Config:
        json_schema_extra = {
            "example": {
                "resume_text": "John Smith\nIT Administrator...",
                "job_description": "We are looking for a Senior IT Specialist...",
            }
        }


class ResumeAnalyzeRequest(BaseModel):
    resume_text: str
    job_description: str


class CoverLetterRequest(BaseModel):
    resume_text: str
    job_description: str
    company_name: Optional[str] = "[Company Name]"
    role_title: Optional[str] = "[Role Title]"


class STARRewriteRequest(BaseModel):
    bullets: str = Field(..., description="Newline-separated bullet points to rewrite")


# ─── Endpoints ────────────────────────────────────────────────────────────────

@resume_route.post(
    "/optimize",
    summary="Optimize resume for a job description (text input)",
    description="Rewrites experience using STAR method, injects missing keywords, and formats for ATS.",
)
async def optimize_resume_text(request: ResumeOptimizeTextRequest):
    payload = {
        "sub_task": "optimize",
        "resume_text": request.resume_text,
        "job_description": request.job_description,
    }
    result = await resume_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@resume_route.post(
    "/optimize/upload",
    summary="Optimize resume for a job description (file upload)",
    description="Upload a PDF or DOCX resume. Provide job description as form field.",
)
async def optimize_resume_upload(
    resume_file: UploadFile = File(..., description="Resume PDF or DOCX"),
    job_description: str = Form(..., description="Job posting text"),
):
    # Validate file size
    content = await resume_file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB",
        )

    # Validate extension
    filename = resume_file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '.{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    payload = {
        "sub_task": "optimize",
        "resume_file_content": content,
        "resume_filename": filename,
        "job_description": job_description,
    }
    result = await resume_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@resume_route.post(
    "/analyze",
    summary="Score resume against a job description",
    description="Returns similarity score, matched/missing keywords, and LLM narrative analysis.",
)
async def analyze_resume(request: ResumeAnalyzeRequest):
    payload = {
        "sub_task": "analyze",
        "resume_text": request.resume_text,
        "job_description": request.job_description,
    }
    result = await resume_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@resume_route.post(
    "/cover-letter",
    summary="Generate a tailored cover letter",
)
async def generate_cover_letter(request: CoverLetterRequest):
    payload = {
        "sub_task": "cover_letter",
        "resume_text": request.resume_text,
        "job_description": request.job_description,
        "company_name": request.company_name,
        "role_title": request.role_title,
    }
    result = await resume_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@resume_route.post(
    "/star-rewrite",
    summary="Rewrite experience bullets using the STAR method",
)
async def star_rewrite(request: STARRewriteRequest):
    payload = {"sub_task": "star_rewrite", "bullets": request.bullets}
    result = await resume_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result

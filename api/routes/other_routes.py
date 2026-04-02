"""
api/routes/assistant.py  — General assistant endpoints
api/routes/job_app.py    — Job application endpoints
api/routes/code_gen.py   — Code generation endpoints
api/routes/history.py    — Task history endpoint
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ASSISTANT ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional

from modules.assistant.handler import assistant_handler
from modules.job_application.handler import job_application_handler
from modules.code_gen.handler import code_gen_handler
from storage.database import db
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Routers ──────────────────────────────────────────────────────────────────
assistant_route = APIRouter(prefix="/assistant", tags=["Assistant"])
job_app_route = APIRouter(prefix="/job", tags=["Job Application"])
code_route = APIRouter(prefix="/code", tags=["Code Generation"])
history_route = APIRouter(prefix="/history", tags=["History"])


# ─── Assistant models ─────────────────────────────────────────────────────────

class AssistantQueryRequest(BaseModel):
    question: str = Field(..., description="Technical question to answer")
    context: Optional[str] = Field(None, description="Additional background context")
    history: Optional[list[dict[str, str]]] = Field(
        None,
        description="Conversation history: [{role: user|assistant, content: ...}]",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I find all failed login attempts in Windows Event Viewer?",
                "context": "Windows Server 2022 environment, 300 users",
            }
        }


class AssistantExplainRequest(BaseModel):
    topic: str = Field(..., description="Concept to explain")
    level: Literal["beginner", "intermediate", "expert"] = "intermediate"
    context: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Kerberos authentication",
                "level": "intermediate",
            }
        }



class AssistantChatRequest(BaseModel):
    message: str = Field(..., description="User chat message")
    session_id: str = Field("default", description="Conversation/session identifier")
    category: Optional[str] = Field(None, description="Optional memory category filter")
    top_k: int = Field(3, ge=1, le=10, description="Number of memory chunks to retrieve")


# ─── Assistant endpoints ──────────────────────────────────────────────────────

@assistant_route.post(
    "/query",
    summary="Ask a technical question",
    description="Answers IT, cybersecurity, scripting, and infrastructure questions.",
)
async def assistant_query(request: AssistantQueryRequest):
    payload = {
        "sub_task": "query",
        "question": request.question,
        "context": request.context or "",
        "history": request.history or [],
    }
    result = await assistant_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@assistant_route.post(
    "/explain",
    summary="Explain a technical concept",
    description="Returns a structured explanation with examples, adapted to the specified skill level.",
)
async def assistant_explain(request: AssistantExplainRequest):
    payload = {
        "sub_task": "explain",
        "topic": request.topic,
        "level": request.level,
        "context": request.context or "",
    }
    result = await assistant_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@assistant_route.post(
    "/chat",
    summary="Interactive chat with memory retrieval",
    description="Conversational endpoint that grounds responses using taught memory.",
)
async def assistant_chat(request: AssistantChatRequest):
    payload = {
        "sub_task": "chat",
        "message": request.message,
        "session_id": request.session_id,
        "category": request.category,
        "top_k": request.top_k,
    }
    result = await assistant_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# JOB APPLICATION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

class JobQuestionRequest(BaseModel):
    question: str = Field(..., description="Application question to answer")
    background: str = Field(..., description="Applicant's professional background")
    role_title: Optional[str] = "[Role Title]"
    company_name: Optional[str] = "[Company Name]"
    max_words: int = Field(150, ge=50, le=500)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Why do you want to work at our company?",
                "background": "IT Analyst with 3 years experience in M365 administration...",
                "role_title": "IT Support Specialist",
                "company_name": "Acme Corp",
                "max_words": 150,
            }
        }


class JobScreeningRequest(BaseModel):
    questions: list[str] = Field(..., description="List of screening questions")
    background: str = Field(..., description="Applicant's professional background")
    role_title: Optional[str] = "[Role Title]"
    company_name: Optional[str] = "[Company Name]"


@job_app_route.post(
    "/question",
    summary="Answer a single job application question",
)
async def answer_job_question(request: JobQuestionRequest):
    payload = {
        "sub_task": "question",
        "question": request.question,
        "background": request.background,
        "role_title": request.role_title,
        "company_name": request.company_name,
        "max_words": request.max_words,
    }
    result = await job_application_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@job_app_route.post(
    "/screening",
    summary="Answer multiple screening questions at once",
)
async def answer_screening_questions(request: JobScreeningRequest):
    payload = {
        "sub_task": "screening",
        "questions": request.questions,
        "background": request.background,
        "role_title": request.role_title,
        "company_name": request.company_name,
    }
    result = await job_application_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@job_app_route.get(
    "/applications",
    summary="List saved job applications",
)
async def list_applications(status: Optional[str] = None):
    return {"applications": db.get_job_applications(status=status)}


# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

class CodeGenerateRequest(BaseModel):
    task_description: str = Field(..., description="What the script should do")
    language: str = Field("python", description="python | powershell | bash | javascript | sql")
    requirements: Optional[list[str]] = Field(None, description="Additional requirements list")
    example_input_output: Optional[str] = Field(None, description="Input/output example")

    class Config:
        json_schema_extra = {
            "example": {
                "task_description": "Script that checks Active Directory for accounts inactive for 90+ days and exports to CSV",
                "language": "powershell",
                "requirements": [
                    "Include error handling",
                    "Log results to a file",
                    "Accept output path as a parameter",
                ],
            }
        }


class CodeReviewRequest(BaseModel):
    code: str = Field(..., description="Code to review")
    language: str = Field("python", description="Language of the code")


class CodeExplainRequest(BaseModel):
    code: str = Field(..., description="Code to explain")
    language: str = Field("python")


@code_route.post(
    "/generate",
    summary="Generate a script or function",
    description="Creates production-ready code with error handling and comments.",
)
async def generate_code(request: CodeGenerateRequest):
    payload = {
        "sub_task": "generate",
        "task_description": request.task_description,
        "language": request.language,
        "requirements": request.requirements or [],
        "example_input_output": request.example_input_output or "",
    }
    result = await code_gen_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@code_route.post("/review", summary="Review code for bugs and improvements")
async def review_code(request: CodeReviewRequest):
    payload = {"sub_task": "review", "code": request.code, "language": request.language}
    result = await code_gen_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@code_route.post("/explain", summary="Explain what code does")
async def explain_code(request: CodeExplainRequest):
    payload = {"sub_task": "explain", "code": request.code, "language": request.language}
    result = await code_gen_handler.handle(payload)
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORY ROUTE
# ═══════════════════════════════════════════════════════════════════════════════

@history_route.get(
    "/",
    summary="Get task history",
    description="Returns recent task records. Filter by task_type if needed.",
)
async def get_history(task_type: Optional[str] = None, limit: int = 50):
    return {"history": db.get_task_history(task_type=task_type, limit=limit)}


@history_route.get(
    "/{record_id}",
    summary="Get full result for a past task",
)
async def get_task_result(record_id: int):
    result = db.get_task_result(record_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"No record found with id={record_id}")
    return result

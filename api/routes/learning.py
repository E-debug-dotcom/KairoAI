"""api/routes/learning.py — Teachable memory endpoints."""

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from modules.learning.handler import learning_handler

learning_route = APIRouter(prefix="/learn", tags=["Learning"])


class TeachTextRequest(BaseModel):
    text: str = Field(..., description="Knowledge text to store")
    source: str = Field("user_note", description="Source label")
    category: str = Field("general", description="Category, e.g., security, iam, automation")
    tags: list[str] = Field(default_factory=list, description="Optional tags")


class SearchKnowledgeRequest(BaseModel):
    query: str
    category: Optional[str] = None
    top_k: int = Field(5, ge=1, le=15)


@learning_route.post("/text", summary="Teach the assistant using raw text")
async def teach_text(request: TeachTextRequest):
    result = await learning_handler.handle(
        {
            "sub_task": "teach_text",
            "text": request.text,
            "source": request.source,
            "category": request.category,
            "tags": request.tags,
        }
    )
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@learning_route.post("/upload", summary="Teach the assistant using a document")
async def teach_upload(
    file: UploadFile = File(...),
    category: str = Form("general"),
    tags: str = Form(""),
):
    content = await file.read()
    result = await learning_handler.handle(
        {
            "sub_task": "teach_document",
            "filename": file.filename,
            "content": content,
            "category": category,
            "tags": [t.strip() for t in tags.split(",") if t.strip()],
        }
    )
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@learning_route.post("/search", summary="Search stored knowledge memory")
async def search_knowledge(request: SearchKnowledgeRequest):
    result = await learning_handler.handle(
        {
            "sub_task": "search",
            "query": request.query,
            "category": request.category,
            "top_k": request.top_k,
        }
    )
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result

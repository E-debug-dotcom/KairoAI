"""api/routes/learning.py — Teachable memory endpoints."""

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from modules.learning.dataset_schemas import TeachDatasetRequest
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


@learning_route.post("/dataset", summary="Teach the assistant using a structured dataset")
async def teach_dataset(request: TeachDatasetRequest):
    result = await learning_handler.handle(
        {
            "sub_task": "teach_dataset",
            "dataset_name": request.dataset_name,
            "category": request.category,
            "items": [item.model_dump() for item in request.items],
        }
    )
    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@learning_route.post("/dataset/upload", summary="Teach the assistant using a JSON dataset file")
async def teach_dataset_upload(
    file: UploadFile = File(...),
    category: str = Form("general"),
):
    content = await file.read()
    try:
        payload = json.loads(content.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Dataset JSON parse failed: {exc}")

    try:
        if isinstance(payload, dict) and "items" in payload:
            request = TeachDatasetRequest(
                dataset_name=payload.get("dataset_name") or file.filename,
                category=category,
                items=payload["items"],
            )
        elif isinstance(payload, list):
            request = TeachDatasetRequest(
                dataset_name=file.filename,
                category=category,
                items=payload,
            )
        else:
            raise ValueError(
                "Dataset payload must be a JSON list of items or an object with an 'items' array."
            )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = await learning_handler.handle(
        {
            "sub_task": "teach_dataset",
            "dataset_name": request.dataset_name,
            "category": request.category,
            "items": [item.model_dump() for item in request.items],
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

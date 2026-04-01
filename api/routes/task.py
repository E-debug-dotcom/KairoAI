"""
api/routes/task.py — Generic /task endpoint that uses the task router.
api/routes/resume.py — Resume-specific endpoints with file upload support.
api/routes/assistant.py — Assistant endpoints.
api/routes/job_application.py — Job application endpoints.
api/routes/code_gen.py — Code generation endpoints.
"""

# ─────────────────────────────────────────────────────────────────────────────
# GENERIC TASK ROUTE
# ─────────────────────────────────────────────────────────────────────────────
# api/routes/task.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional

from core.task_router import router as task_router, TaskNotFoundError
from utils.logger import get_logger

logger = get_logger(__name__)

task_route = APIRouter(prefix="/task", tags=["Generic Task"])


class GenericTaskRequest(BaseModel):
    task_type: str = Field(..., description="One of: resume, assistant, job_application, code")
    sub_task: Optional[str] = Field(None, description="Module-specific sub-task")
    payload: dict[str, Any] = Field(default_factory=dict, description="Task-specific parameters")

    class Config:
        json_schema_extra = {
            "example": {
                "task_type": "assistant",
                "sub_task": "query",
                "payload": {
                    "question": "What is the difference between symmetric and asymmetric encryption?",
                }
            }
        }


@task_route.post(
    "/",
    summary="Generic task dispatcher",
    description="Routes to any registered module using task_type + sub_task.",
)
async def dispatch_task(request: GenericTaskRequest):
    payload = {**request.payload, "sub_task": request.sub_task or ""}
    try:
        result = await task_router.dispatch(request.task_type, payload)
        return result
    except TaskNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Task dispatch error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")


@task_route.get("/available", summary="List available task types")
async def list_tasks():
    return {
        "available_tasks": task_router.available_tasks(),
        "usage": "POST /api/v1/task with {task_type, sub_task, payload}",
    }

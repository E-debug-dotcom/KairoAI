"""Dataset schemas for learning ingestion and validation."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DatasetItem(BaseModel):
    title: Optional[str] = Field(None, description="Optional title for the dataset item")
    source: Optional[str] = Field(None, description="Optional source identifier")
    content: str = Field(..., description="Document content or text to ingest")
    category: Optional[str] = Field(None, description="Optional item-level category")
    tags: list[str] = Field(default_factory=list, description="Optional tags for the item")


class TeachDatasetRequest(BaseModel):
    dataset_name: Optional[str] = Field(
        "dataset",
        description="Friendly name for the dataset",
    )
    category: str = Field(
        "general",
        description="Default category for items missing an explicit category",
    )
    items: list[DatasetItem]

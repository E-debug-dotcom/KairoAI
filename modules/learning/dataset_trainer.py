"""Local training utility for dataset ingestion into the learning vector store."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from modules.learning.dataset_schemas import TeachDatasetRequest
from modules.learning.handler import learning_handler
from utils.logger import get_logger

logger = get_logger(__name__)


def load_json_dataset(path: str) -> TeachDatasetRequest:
    """Load and validate a JSON dataset file for training."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with path_obj.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "items" in payload:
        dataset_name = payload.get("dataset_name") or path_obj.stem
        category = payload.get("category", "general")
        items = payload["items"]
    elif isinstance(payload, list):
        dataset_name = path_obj.stem
        category = "general"
        items = payload
    else:
        raise ValueError(
            "Dataset JSON must be either a list of items or an object with an 'items' array."
        )

    return TeachDatasetRequest(dataset_name=dataset_name, category=category, items=items)


async def ingest_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    return await learning_handler.handle(payload)


def ingest_dataset_file(path: str, category: str | None = None, dataset_name: str | None = None) -> dict[str, Any]:
    dataset = load_json_dataset(path)
    if category:
        dataset.category = category
    if dataset_name:
        dataset.dataset_name = dataset_name

    payload = {
        "sub_task": "teach_dataset",
        "dataset_name": dataset.dataset_name,
        "category": dataset.category,
        "items": [item.model_dump() for item in dataset.items],
    }
    return asyncio.run(ingest_dataset(payload))


def ingest_dataset_directory(path: str, category: str | None = None) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    results: dict[str, Any] = {
        "files": [],
        "total_items": 0,
        "total_chunks": 0,
    }

    for file_path in sorted(path_obj.glob("*.json")):
        try:
            response = ingest_dataset_file(str(file_path), category=category)
            results["files"].append({
                "file": file_path.name,
                "status": response.get("status"),
                "items_ingested": response.get("data", {}).get("items_ingested", 0),
                "total_chunks": response.get("data", {}).get("total_chunks", 0),
            })
            results["total_items"] += response.get("data", {}).get("items_ingested", 0) or 0
            results["total_chunks"] += response.get("data", {}).get("total_chunks", 0) or 0
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", file_path, exc)
            results["files"].append({
                "file": file_path.name,
                "status": "error",
                "error": str(exc),
            })

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a JSON dataset or directory of dataset JSON files into KairoAI learning memory."
    )
    parser.add_argument(
        "path",
        help="Path to a JSON dataset file or a directory containing JSON dataset files.",
    )
    parser.add_argument(
        "--category",
        help="Default category for the dataset items.",
        default=None,
    )
    parser.add_argument(
        "--dataset-name",
        help="Override the dataset name used for ingestion.",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_obj = Path(args.path)

    if path_obj.is_dir():
        result = ingest_dataset_directory(str(path_obj), category=args.category)
    else:
        result = ingest_dataset_file(str(path_obj), category=args.category, dataset_name=args.dataset_name)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

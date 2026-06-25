"""Upload the Synataric agent golden dataset to LangSmith."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from langsmith import Client

from src.agent_eval_dataset import load_golden_dataset, validate_golden_dataset


DEFAULT_DATASET_NAME = "Synataric-Agent-Golden-Dataset-V1"
DEFAULT_PROJECT_NAME = "Synataric-Agent-Evals"


def _dataset_id(dataset: Any) -> Any:
    return getattr(dataset, "id", None) or getattr(dataset, "dataset_id", None)


def _read_dataset(client: Client, dataset_name: str) -> Any | None:
    try:
        return client.read_dataset(dataset_name=dataset_name)
    except Exception:
        return None


def _delete_dataset(client: Client, dataset: Any) -> None:
    dataset_id = _dataset_id(dataset)
    if dataset_id is None:
        raise RuntimeError("Could not determine LangSmith dataset id for deletion.")
    client.delete_dataset(dataset_id=dataset_id)


def _existing_example_ids(client: Client, dataset: Any) -> set[str]:
    dataset_id = _dataset_id(dataset)
    if dataset_id is None:
        return set()

    existing_ids: set[str] = set()
    try:
        examples = client.list_examples(dataset_id=dataset_id)
    except TypeError:
        examples = client.list_examples(dataset_name=getattr(dataset, "name", None))

    for example in examples:
        metadata = getattr(example, "metadata", None) or {}
        example_id = metadata.get("id")
        if example_id:
            existing_ids.add(example_id)
    return existing_ids


def _example_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "inputs": {
            "query": row["query"],
            "agent_mode": row["agent_mode"],
        },
        "outputs": {
            "expected_intent": row["expected_intent"],
            "expected_status": row["expected_status"],
            "expected_tools": row["expected_tools"],
            "expected_tool_sequence": row["expected_tool_sequence"],
            "expected_sources": row["expected_sources"],
            "expected_answer_criteria": row["expected_answer_criteria"],
            "forbidden_behavior": row["forbidden_behavior"],
            "requires_human": row["requires_human"],
            "expected_human_question": row["expected_human_question"],
        },
        "metadata": {
            "id": row["id"],
            "scenario_type": row["scenario_type"],
            "difficulty": row["difficulty"],
            "notes": row["notes"],
            "dataset_version": "v1",
            "project": DEFAULT_PROJECT_NAME,
        },
    }


def upload_dataset(csv_path: Path, dataset_name: str, recreate: bool = False) -> int:
    summary = validate_golden_dataset(csv_path)
    print(f"Validation summary: {summary}")
    if summary["errors"]:
        print("Dataset validation failed; upload aborted.")
        return 1

    if not os.getenv("LANGCHAIN_API_KEY"):
        print("LANGCHAIN_API_KEY is required to upload to LangSmith.")
        return 0

    client = Client()
    dataset = _read_dataset(client, dataset_name)

    if dataset and recreate:
        print(f"Recreating existing dataset: {dataset_name}")
        _delete_dataset(client, dataset)
        dataset = None
    elif dataset:
        print(f"Dataset already exists: {dataset_name}")
        print("Existing examples with matching metadata ids will be skipped. Use --recreate to rebuild.")

    if dataset is None:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Golden evaluation dataset for Synataric Router Agent and ReAct Care Planner.",
        )

    existing_ids = _existing_example_ids(client, dataset)
    dataset_id = _dataset_id(dataset)
    uploaded = 0
    skipped = 0

    for row in load_golden_dataset(csv_path):
        if row["id"] in existing_ids:
            skipped += 1
            continue

        payload = _example_payload(row)
        client.create_example(
            inputs=payload["inputs"],
            outputs=payload["outputs"],
            metadata=payload["metadata"],
            dataset_id=dataset_id,
        )
        uploaded += 1

    print(f"Dataset name: {dataset_name}")
    print(f"Uploaded examples: {uploaded}")
    print(f"Skipped existing examples: {skipped}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload Synataric golden eval dataset to LangSmith.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--csv-path", type=Path, default=Path("data/evals/synataric_agent_golden_dataset.csv"))
    parser.add_argument("--recreate", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return upload_dataset(args.csv_path, args.dataset_name, args.recreate)


if __name__ == "__main__":
    sys.exit(main())

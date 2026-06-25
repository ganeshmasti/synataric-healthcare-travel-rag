"""Utilities for loading and validating the Synataric agent golden dataset."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_PATH = Path("data/evals/synataric_agent_golden_dataset.csv")

REQUIRED_COLUMNS = [
    "id",
    "agent_mode",
    "scenario_type",
    "difficulty",
    "query",
    "expected_intent",
    "expected_status",
    "expected_tools",
    "expected_tool_sequence",
    "expected_sources",
    "expected_answer_criteria",
    "forbidden_behavior",
    "requires_human",
    "expected_human_question",
    "notes",
]

VALID_AGENT_MODES = {"router_agent", "react_care_planner"}
VALID_SCENARIO_TYPES = {"happy_path", "edge_case", "known_failure", "adversarial"}
VALID_EXPECTED_STATUSES = {
    "complete",
    "needs_human",
    "unsafe",
    "out_of_scope",
    "no_evidence",
    "error",
}


def _to_bool(value: str) -> bool | str:
    normalized = value.strip().upper()
    if normalized == "TRUE":
        return True
    if normalized == "FALSE":
        return False
    return value


def load_golden_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Load the golden dataset CSV as a list of row dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows: list[dict[str, Any]] = []
        for row in reader:
            normalized = dict(row)
            normalized["requires_human"] = _to_bool(normalized.get("requires_human", ""))
            rows.append(normalized)
        return rows


def validate_golden_dataset(path: Path = DATASET_PATH) -> dict[str, Any]:
    """Validate required structure and values for the golden dataset."""
    errors: list[str] = []

    if not path.exists():
        return {
            "rows": 0,
            "agent_mode_counts": {},
            "scenario_type_counts": {},
            "errors": [f"Dataset file not found: {path}"],
        }

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        rows = list(reader)

    if len(rows) != 40:
        errors.append(f"Expected exactly 40 rows, found {len(rows)}")

    ids = [row.get("id", "") for row in rows]
    duplicate_ids = sorted({row_id for row_id, count in Counter(ids).items() if count > 1})
    if duplicate_ids:
        errors.append(f"Duplicate ids: {', '.join(duplicate_ids)}")

    agent_mode_counts = Counter(row.get("agent_mode", "") for row in rows)
    scenario_type_counts = Counter(row.get("scenario_type", "") for row in rows)

    invalid_agent_modes = sorted(set(agent_mode_counts) - VALID_AGENT_MODES)
    if invalid_agent_modes:
        errors.append(f"Invalid agent_mode values: {', '.join(invalid_agent_modes)}")

    invalid_scenario_types = sorted(set(scenario_type_counts) - VALID_SCENARIO_TYPES)
    if invalid_scenario_types:
        errors.append(f"Invalid scenario_type values: {', '.join(invalid_scenario_types)}")

    invalid_statuses = sorted(
        {row.get("expected_status", "") for row in rows} - VALID_EXPECTED_STATUSES
    )
    if invalid_statuses:
        errors.append(f"Invalid expected_status values: {', '.join(invalid_statuses)}")

    return {
        "rows": len(rows),
        "agent_mode_counts": dict(agent_mode_counts),
        "scenario_type_counts": dict(scenario_type_counts),
        "errors": errors,
    }


def print_dataset_summary(path: Path = DATASET_PATH) -> None:
    """Print validation counts and errors for the golden dataset."""
    summary = validate_golden_dataset(path)
    print(f"Rows: {summary['rows']}")
    print(f"Agent modes: {summary['agent_mode_counts']}")
    print(f"Scenario types: {summary['scenario_type_counts']}")
    if summary["errors"]:
        print("Errors:")
        for error in summary["errors"]:
            print(f"- {error}")
    else:
        print("Errors: none")


if __name__ == "__main__":
    print_dataset_summary()

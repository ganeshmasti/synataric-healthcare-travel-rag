"""Validate the Synataric care intent router fine-tuning CSV."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "data" / "finetune" / "synataric_care_router_tickets.csv"

REQUIRED_COLUMNS = ["id", "ticket", "label", "scenario_type", "difficulty", "notes"]
ALLOWED_LABELS = {
    "provider_search",
    "cost_estimate",
    "travel_planning",
    "recovery_guidance",
    "risk_checklist",
    "find_evidence",
    "care_plan_multistep",
    "needs_clarification",
    "unsafe_medical",
    "out_of_scope",
    "general_navigation",
}
ALLOWED_SCENARIO_TYPES = {"happy_path", "edge_case", "known_failure", "adversarial"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
EXPECTED_TOTAL_ROWS = 550
EXPECTED_ROWS_PER_LABEL = 50


def _load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], [f"Dataset not found: {path}"]

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames or []

    errors = []
    if columns != REQUIRED_COLUMNS:
        errors.append(f"Columns must be exactly {REQUIRED_COLUMNS}; found {columns}")
    return rows, errors


def validate_dataset(path: Path = DATASET_PATH) -> tuple[list[dict[str, str]], list[str]]:
    rows, errors = _load_rows(path)
    if not rows:
        return rows, errors

    tickets = [row.get("ticket", "").strip() for row in rows]
    labels = [row.get("label", "").strip() for row in rows]
    scenario_types = [row.get("scenario_type", "").strip() for row in rows]
    difficulties = [row.get("difficulty", "").strip() for row in rows]
    label_counts = Counter(labels)

    if len(rows) != EXPECTED_TOTAL_ROWS:
        errors.append(f"Expected {EXPECTED_TOTAL_ROWS} rows; found {len(rows)}")

    unknown_labels = sorted(set(labels) - ALLOWED_LABELS)
    if unknown_labels:
        errors.append(f"Unknown labels: {unknown_labels}")

    missing_labels = sorted(ALLOWED_LABELS - set(labels))
    if missing_labels:
        errors.append(f"Missing labels: {missing_labels}")

    for label in sorted(ALLOWED_LABELS):
        count = label_counts.get(label, 0)
        if count != EXPECTED_ROWS_PER_LABEL:
            errors.append(f"Label {label} expected {EXPECTED_ROWS_PER_LABEL} rows; found {count}")

    blank_ticket_count = sum(1 for ticket in tickets if not ticket)
    if blank_ticket_count:
        errors.append(f"Blank ticket count: {blank_ticket_count}")

    duplicate_count = len(tickets) - len(set(tickets))
    if duplicate_count:
        errors.append(f"Duplicate ticket count: {duplicate_count}")

    unknown_scenarios = sorted(set(scenario_types) - ALLOWED_SCENARIO_TYPES)
    if unknown_scenarios:
        errors.append(f"Unknown scenario_type values: {unknown_scenarios}")

    unknown_difficulties = sorted(set(difficulties) - ALLOWED_DIFFICULTIES)
    if unknown_difficulties:
        errors.append(f"Unknown difficulty values: {unknown_difficulties}")

    for index, row in enumerate(rows, start=2):
        for column in REQUIRED_COLUMNS:
            if column != "notes" and not row.get(column, "").strip():
                errors.append(f"Row {index} has blank required field: {column}")

    return rows, errors


def _print_counter(title: str, values: list[str]) -> None:
    print(title)
    for key, count in sorted(Counter(values).items()):
        print(f"  {key}: {count}")


def main() -> int:
    rows, errors = validate_dataset()
    tickets = [row.get("ticket", "").strip() for row in rows]

    print(f"total rows: {len(rows)}")
    _print_counter("label counts:", [row.get("label", "").strip() for row in rows])
    _print_counter("scenario_type counts:", [row.get("scenario_type", "").strip() for row in rows])
    _print_counter("difficulty counts:", [row.get("difficulty", "").strip() for row in rows])
    print(f"duplicate count: {len(tickets) - len(set(tickets))}")
    print("validation errors:")
    if errors:
        for error in errors:
            print(f"  - {error}")
        return 1
    print("  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export the Synataric router dataset for the official Week 5 SFT notebook."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_CSV = PROJECT_ROOT / "data" / "finetune" / "synataric_care_router_tickets.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "finetune" / "synataric_finetuning_preference_dataset.csv"
INPUT_COLUMNS = ["id", "ticket", "label", "scenario_type", "difficulty", "notes"]
OUTPUT_COLUMNS = ["TEST_ID", "USER_INPUT", "TARGET_INTENT"]
EXPECTED_ROW_COUNT = 550
EXPECTED_PER_LABEL = 50
ALLOWED_LABELS = [
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
]


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        if columns != INPUT_COLUMNS:
            raise ValueError(f"Expected columns {INPUT_COLUMNS}; found {columns}")
        return list(reader)


def _validate_export_rows(rows: list[dict[str, str]]) -> None:
    errors: list[str] = []
    label_counts = Counter(row.get("TARGET_INTENT", "").strip() for row in rows)

    if len(rows) != EXPECTED_ROW_COUNT:
        errors.append(f"Expected {EXPECTED_ROW_COUNT} rows; found {len(rows)}")

    missing_labels = sorted(set(ALLOWED_LABELS) - set(label_counts))
    if missing_labels:
        errors.append(f"Missing labels: {missing_labels}")

    unknown_labels = sorted(set(label_counts) - set(ALLOWED_LABELS))
    if unknown_labels:
        errors.append(f"Unknown labels: {unknown_labels}")

    for label in ALLOWED_LABELS:
        count = label_counts.get(label, 0)
        if count != EXPECTED_PER_LABEL:
            errors.append(f"Label {label} expected {EXPECTED_PER_LABEL}; found {count}")

    blank_user_inputs = sum(1 for row in rows if not row.get("USER_INPUT", "").strip())
    blank_target_intents = sum(1 for row in rows if not row.get("TARGET_INTENT", "").strip())
    if blank_user_inputs:
        errors.append(f"Blank USER_INPUT rows: {blank_user_inputs}")
    if blank_target_intents:
        errors.append(f"Blank TARGET_INTENT rows: {blank_target_intents}")

    if errors:
        raise ValueError("; ".join(errors))


def export_dataset(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV) -> Path:
    source_rows = _load_rows(input_csv)
    export_rows = [
        {
            "TEST_ID": row["id"].strip(),
            "USER_INPUT": row["ticket"].strip(),
            "TARGET_INTENT": row["label"].strip(),
        }
        for row in source_rows
    ]
    _validate_export_rows(export_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(export_rows)

    label_counts = Counter(row["TARGET_INTENT"] for row in export_rows)
    print(f"Output path: {output_csv.relative_to(PROJECT_ROOT)}")
    print("Label counts:")
    for label in ALLOWED_LABELS:
        print(f"  {label}: {label_counts[label]}")
    return output_csv


def main() -> int:
    try:
        export_dataset()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Create baseline predictions from the existing Synataric intent router."""

from __future__ import annotations

import csv
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT_CSV = PROJECT_ROOT / "data" / "finetune" / "synataric_care_router_tickets.csv"
OUTPUT_CSV = PROJECT_ROOT / "reports" / "finetune" / "baseline_existing_router_predictions.csv"
MODEL_NAME = "existing_synataric_router"
TRAIN_RATIO = 0.8
SEED = 42

LABELS = [
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

INTENT_TO_LABEL = {
    "provider_search": "provider_search",
    "cost_estimate": "cost_estimate",
    "travel_planning": "travel_planning",
    "recovery_guidance": "recovery_guidance",
    "risk_checklist": "risk_checklist",
    "find_evidence": "find_evidence",
    "needs_clarification": "needs_clarification",
    "unsafe_medical": "unsafe_medical",
    "out_of_scope": "out_of_scope",
    "general_navigation": "general_navigation",
}


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"id", "ticket", "label"}
        missing_columns = sorted(required_columns - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(f"Missing required dataset columns: {missing_columns}")
        return list(reader)


def _stratified_validation_split(
    rows: list[dict[str, str]], train_ratio: float = TRAIN_RATIO, seed: int = SEED
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    rng = random.Random(seed)
    validation_rows: list[dict[str, str]] = []
    for label in LABELS:
        label_rows = list(grouped[label])
        if not label_rows:
            raise ValueError(f"No rows found for label: {label}")
        rng.shuffle(label_rows)
        train_count = int(len(label_rows) * train_ratio)
        validation_rows.extend(label_rows[train_count:])

    rng.shuffle(validation_rows)
    return validation_rows


def _map_intent_to_label(intent: str) -> str:
    return INTENT_TO_LABEL.get(intent, intent)


def _predict_ticket(ticket: str) -> tuple[str, float]:
    from src.agent_intents import classify_intent

    start = time.perf_counter()
    try:
        classification = classify_intent(ticket, {})
        predicted_label = _map_intent_to_label(str(classification.intent).strip())
    except Exception:
        predicted_label = "error"
    latency_seconds = time.perf_counter() - start
    return predicted_label, latency_seconds


def write_baseline_predictions(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV) -> Path:
    _load_dotenv_if_available()

    rows = _read_rows(input_csv)
    validation_rows = _stratified_validation_split(rows)
    label_counts = Counter(row["label"] for row in validation_rows)
    if len(validation_rows) != 110:
        raise ValueError(f"Expected 110 validation rows; found {len(validation_rows)}")
    unexpected_counts = {
        label: count for label, count in label_counts.items() if count != 10
    }
    missing_labels = sorted(set(LABELS) - set(label_counts))
    if unexpected_counts or missing_labels:
        raise ValueError(
            f"Expected 10 validation rows per label; counts={dict(sorted(label_counts.items()))}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "ticket",
                "true_label",
                "predicted_label",
                "latency_seconds",
                "model_name",
            ],
        )
        writer.writeheader()

        total = len(validation_rows)
        for index, row in enumerate(validation_rows, start=1):
            ticket = row["ticket"].strip()
            predicted_label, latency_seconds = _predict_ticket(ticket)
            writer.writerow(
                {
                    "id": row["id"],
                    "ticket": ticket,
                    "true_label": row["label"].strip(),
                    "predicted_label": predicted_label,
                    "latency_seconds": f"{latency_seconds:.6f}",
                    "model_name": MODEL_NAME,
                }
            )
            if index % 10 == 0:
                print(f"[{index}/{total}] predicted ...")

    return output_csv


def main() -> int:
    output_csv = write_baseline_predictions()
    relative_output = output_csv.relative_to(PROJECT_ROOT)
    print(f"Baseline predictions written to {relative_output}")
    print(
        "python scripts/evaluate_synataric_router_predictions.py "
        "--predictions-csv reports/finetune/baseline_existing_router_predictions.csv "
        "--output-dir reports/finetune/baseline_existing_router"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

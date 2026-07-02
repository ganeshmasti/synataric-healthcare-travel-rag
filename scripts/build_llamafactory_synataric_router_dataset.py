"""Convert the Synataric care router CSV to LLaMA Factory ShareGPT JSON."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "finetune" / "synataric_care_router_tickets.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "finetune" / "llamafactory"
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
INSTRUCTION = (
    "You are Synataric Care Router. Classify the user message into exactly one route label. "
    "Return only one label from this list: "
    f"{', '.join(LABELS)}.\n\n"
    "User message: {ticket}"
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _stratified_split(
    rows: list[dict[str, str]], train_ratio: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    rng = random.Random(seed)

    for label in LABELS:
        label_rows = list(grouped[label])
        if not label_rows:
            raise ValueError(f"No rows found for label: {label}")
        rng.shuffle(label_rows)
        train_count = int(len(label_rows) * train_ratio)
        train_rows.extend(label_rows[:train_count])
        val_rows.extend(label_rows[train_count:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _to_sharegpt(row: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    return {
        "conversations": [
            {
                "from": "human",
                "value": INSTRUCTION.format(ticket=row["ticket"].strip()),
            },
            {
                "from": "gpt",
                "value": row["label"].strip(),
            },
        ]
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _dataset_info_snippet() -> dict[str, dict[str, object]]:
    return {
        "synataric_care_router": {
            "file_name": "synataric_care_router_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }


def build_dataset(input_csv: Path, output_dir: Path, train_ratio: float, seed: int) -> tuple[Path, Path, Path]:
    rows = _read_rows(input_csv)
    train_rows, val_rows = _stratified_split(rows, train_ratio=train_ratio, seed=seed)

    train_path = output_dir / "synataric_care_router_train.json"
    val_path = output_dir / "synataric_care_router_val.json"
    info_path = output_dir / "dataset_info_snippet.json"

    _write_json(train_path, [_to_sharegpt(row) for row in train_rows])
    _write_json(val_path, [_to_sharegpt(row) for row in val_rows])
    _write_json(info_path, _dataset_info_snippet())

    print(f"train rows: {len(train_rows)}")
    print(f"validation rows: {len(val_rows)}")
    print(f"labels in train: {dict(sorted(Counter(row['label'] for row in train_rows).items()))}")
    print(f"labels in validation: {dict(sorted(Counter(row['label'] for row in val_rows).items()))}")
    print("output paths:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {info_path}")
    print(
        "Next: copy the JSON files into LLaMA-Factory/data and merge "
        "dataset_info_snippet.json into LLaMA-Factory/data/dataset_info.json."
    )
    return train_path, val_path, info_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dataset(args.input_csv, args.output_dir, args.train_ratio, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate Synataric local care intent router predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
REQUIRED_COLUMNS = ["id", "ticket", "true_label", "predicted_label"]
CRITICAL_RECALL_LABELS = [
    "unsafe_medical",
    "out_of_scope",
    "needs_clarification",
    "care_plan_multistep",
]
PARTIAL_CREDIT_PAIRS = {
    frozenset(("travel_planning", "recovery_guidance")),
    frozenset(("cost_estimate", "general_navigation")),
}


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return list(reader)


def _try_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _supports_provider_multistep_partial(ticket: str) -> bool:
    text = ticket.lower()
    provider_terms = ("provider", "hospital", "clinic", "doctor", "specialist")
    planning_terms = ("cost", "budget", "travel", "recovery", "risk", "red flag", "care plan")
    return any(term in text for term in provider_terms) and any(term in text for term in planning_terms)


def route_execution_score(row: dict[str, str]) -> float:
    true_label = row["true_label"].strip()
    predicted_label = row["predicted_label"].strip()
    if predicted_label == true_label:
        return 1.0

    no_partial = {"unsafe_medical", "out_of_scope", "needs_clarification"}
    if true_label in no_partial or predicted_label in no_partial:
        return 0.0

    pair = frozenset((true_label, predicted_label))
    if pair in PARTIAL_CREDIT_PAIRS:
        return 0.5

    if pair == frozenset(("provider_search", "care_plan_multistep")):
        return 0.5 if _supports_provider_multistep_partial(row.get("ticket", "")) else 0.0

    return 0.0


def _manual_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    true_labels = [row["true_label"].strip() for row in rows]
    predicted_labels = [row["predicted_label"].strip() for row in rows]
    valid_prediction = [label in ALLOWED_LABELS for label in predicted_labels]
    invalid_count = valid_prediction.count(False)
    correct_count = sum(
        1 for true, predicted in zip(true_labels, predicted_labels) if true == predicted
    )

    per_label: dict[str, dict[str, float | int]] = {}
    for label in ALLOWED_LABELS:
        tp = sum(1 for true, predicted in zip(true_labels, predicted_labels) if true == label and predicted == label)
        fp = sum(1 for true, predicted in zip(true_labels, predicted_labels) if true != label and predicted == label)
        fn = sum(1 for true, predicted in zip(true_labels, predicted_labels) if true == label and predicted != label)
        support = sum(1 for true in true_labels if true == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_precision = sum(float(metrics["precision"]) for metrics in per_label.values()) / len(ALLOWED_LABELS)
    macro_recall = sum(float(metrics["recall"]) for metrics in per_label.values()) / len(ALLOWED_LABELS)
    macro_f1 = sum(float(metrics["f1"]) for metrics in per_label.values()) / len(ALLOWED_LABELS)

    matrix_labels = ALLOWED_LABELS + ["INVALID"]
    confusion_matrix = {true: {predicted: 0 for predicted in matrix_labels} for true in ALLOWED_LABELS}
    for true, predicted in zip(true_labels, predicted_labels):
        if true not in ALLOWED_LABELS:
            continue
        predicted_key = predicted if predicted in ALLOWED_LABELS else "INVALID"
        confusion_matrix[true][predicted_key] += 1

    scores = [route_execution_score(row) for row in rows]

    return {
        "total_examples": total,
        "accuracy": correct_count / total if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion_matrix": confusion_matrix,
        "confusion_matrix_labels": matrix_labels,
        "invalid_output_count": invalid_count,
        "invalid_output_rate": invalid_count / total if total else 0.0,
        "route_execution_score": sum(scores) / len(scores) if scores else 0.0,
        "critical_recalls": {
            label: float(per_label[label]["recall"]) for label in CRITICAL_RECALL_LABELS
        },
    }


def _compute_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    metrics = _manual_metrics(rows)
    try:
        from sklearn.metrics import classification_report
    except Exception:
        metrics["metrics_backend"] = "standard_library"
        return metrics

    true_labels = [row["true_label"].strip() for row in rows]
    predicted_labels = [
        row["predicted_label"].strip()
        if row["predicted_label"].strip() in ALLOWED_LABELS
        else "INVALID"
        for row in rows
    ]
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=ALLOWED_LABELS,
        output_dict=True,
        zero_division=0,
    )
    metrics["macro_precision"] = float(report["macro avg"]["precision"])
    metrics["macro_recall"] = float(report["macro avg"]["recall"])
    metrics["macro_f1"] = float(report["macro avg"]["f1-score"])
    for label in ALLOWED_LABELS:
        label_metrics = report.get(label, {})
        metrics["per_label"][label] = {
            "precision": float(label_metrics.get("precision", 0.0)),
            "recall": float(label_metrics.get("recall", 0.0)),
            "f1": float(label_metrics.get("f1-score", 0.0)),
            "support": int(label_metrics.get("support", 0)),
        }
    metrics["critical_recalls"] = {
        label: float(metrics["per_label"][label]["recall"]) for label in CRITICAL_RECALL_LABELS
    }
    metrics["metrics_backend"] = "sklearn"
    return metrics


def _latency_summary(rows: list[dict[str, str]]) -> float | None:
    values = [_try_float(row.get("latency_seconds")) for row in rows]
    latencies = [value for value in values if value is not None]
    if not latencies:
        return None
    return sum(latencies) / len(latencies)


def _model_name(rows: list[dict[str, str]]) -> str | None:
    names = sorted({row.get("model_name", "").strip() for row in rows if row.get("model_name", "").strip()})
    if not names:
        return None
    if len(names) == 1:
        return names[0]
    return ", ".join(names)


def _write_confusion_matrix_csv(path: Path, metrics: dict[str, Any]) -> None:
    labels = metrics["confusion_matrix_labels"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label"] + labels)
        for true_label in ALLOWED_LABELS:
            row = [true_label]
            row.extend(metrics["confusion_matrix"][true_label][predicted] for predicted in labels)
            writer.writerow(row)


def _write_confusion_matrix_png(path: Path, metrics: dict[str, Any]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    try:
        labels = metrics["confusion_matrix_labels"]
        values = [
            [metrics["confusion_matrix"][true_label][predicted] for predicted in labels]
            for true_label in ALLOWED_LABELS
        ]
        fig_width = max(10, len(labels) * 0.8)
        fig_height = max(7, len(ALLOWED_LABELS) * 0.55)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(values, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(range(len(ALLOWED_LABELS)), labels=ALLOWED_LABELS)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Synataric Router Confusion Matrix")
        for i, row in enumerate(values):
            for j, count in enumerate(row):
                if count:
                    ax.text(j, i, str(count), ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception:
        return False
    return True


def _format_pct(value: float) -> str:
    return f"{value:.3f}"


def _write_markdown_report(
    path: Path,
    metrics: dict[str, Any],
    model_name: str | None,
    average_latency: float | None,
    matrix_csv_path: Path,
    matrix_png_path: Path | None,
) -> None:
    lines = [
        "# Synataric Local Care Intent Router Evaluation",
        "",
        "## Summary",
    ]
    if model_name:
        lines.append(f"- model name: {model_name}")
    lines.extend(
        [
            f"- total examples: {metrics['total_examples']}",
            f"- accuracy: {_format_pct(metrics['accuracy'])}",
            f"- macro precision: {_format_pct(metrics['macro_precision'])}",
            f"- macro recall: {_format_pct(metrics['macro_recall'])}",
            f"- macro F1: {_format_pct(metrics['macro_f1'])}",
            f"- invalid output rate: {_format_pct(metrics['invalid_output_rate'])}",
            f"- average route_execution_score: {_format_pct(metrics['route_execution_score'])}",
        ]
    )
    if average_latency is not None:
        lines.append(f"- average latency seconds: {average_latency:.3f}")

    lines.extend(["", "## Critical Safety/Workflow Recalls"])
    for label in CRITICAL_RECALL_LABELS:
        lines.append(f"- {label} recall: {_format_pct(metrics['critical_recalls'][label])}")

    lines.extend(
        [
            "",
            "## Per-label Report",
            "",
            "| Label | Precision | Recall | F1 | Support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in ALLOWED_LABELS:
        label_metrics = metrics["per_label"][label]
        lines.append(
            "| {label} | {precision} | {recall} | {f1} | {support} |".format(
                label=label,
                precision=_format_pct(float(label_metrics["precision"])),
                recall=_format_pct(float(label_metrics["recall"])),
                f1=_format_pct(float(label_metrics["f1"])),
                support=label_metrics["support"],
            )
        )

    png_text = str(matrix_png_path) if matrix_png_path else "not created because matplotlib is unavailable"
    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            f"- CSV: {matrix_csv_path}",
            f"- PNG: {png_text}",
            "",
            "## Route Execution Score",
            "",
            "The route_execution_score gives 1.0 for an exact route match, 0.5 for selected related workflow routes, and 0.0 otherwise. Unsafe medical, out-of-scope, and clarification misses receive no partial credit.",
            "",
            f"Average route_execution_score: {_format_pct(metrics['route_execution_score'])}",
            "",
            "## Interpretation",
            "",
            "Unsafe_medical and out_of_scope recall matter for safety. Needs_clarification recall matters because the agent should ask instead of guessing. Care_plan_multistep recall matters because those requests should route to the ReAct Care Planner. A wrong label is not just a classification error; it can trigger the wrong downstream workflow.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json_report(
    path: Path,
    metrics: dict[str, Any],
    model_name: str | None,
    average_latency: float | None,
    output_paths: dict[str, str | None],
) -> None:
    payload = dict(metrics)
    payload["model_name"] = model_name
    payload["average_latency_seconds"] = average_latency
    payload["output_paths"] = output_paths
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def evaluate_predictions(predictions_csv: Path, output_dir: Path) -> dict[str, Path | None]:
    predictions_csv = _resolve_path(predictions_csv)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_predictions(predictions_csv)
    if not rows:
        raise ValueError("Predictions CSV contains no rows")

    unknown_true_labels = sorted(
        {row["true_label"].strip() for row in rows if row["true_label"].strip() not in ALLOWED_LABELS}
    )
    if unknown_true_labels:
        raise ValueError(f"Unknown true_label values: {unknown_true_labels}")

    metrics = _compute_metrics(rows)
    average_latency = _latency_summary(rows)
    model_name = _model_name(rows)

    markdown_path = output_dir / "router_classification_report.md"
    json_path = output_dir / "router_classification_report.json"
    matrix_csv_path = output_dir / "router_confusion_matrix.csv"
    matrix_png_path = output_dir / "router_confusion_matrix.png"

    _write_confusion_matrix_csv(matrix_csv_path, metrics)
    png_created = _write_confusion_matrix_png(matrix_png_path, metrics)
    final_png_path = matrix_png_path if png_created else None

    output_paths = {
        "markdown_report": str(markdown_path),
        "json_report": str(json_path),
        "confusion_matrix_csv": str(matrix_csv_path),
        "confusion_matrix_png": str(final_png_path) if final_png_path else None,
    }
    _write_markdown_report(
        markdown_path,
        metrics,
        model_name,
        average_latency,
        matrix_csv_path,
        final_png_path,
    )
    _write_json_report(json_path, metrics, model_name, average_latency, output_paths)

    print(f"total examples: {metrics['total_examples']}")
    print(f"accuracy: {_format_pct(metrics['accuracy'])}")
    print(f"macro precision: {_format_pct(metrics['macro_precision'])}")
    print(f"macro recall: {_format_pct(metrics['macro_recall'])}")
    print(f"macro F1: {_format_pct(metrics['macro_f1'])}")
    print(f"invalid output rate: {_format_pct(metrics['invalid_output_rate'])}")
    print(f"average route_execution_score: {_format_pct(metrics['route_execution_score'])}")
    if average_latency is not None:
        print(f"average latency seconds: {average_latency:.3f}")
    print("output paths:")
    for key, value in output_paths.items():
        print(f"  {key}: {value}")

    return {
        "markdown_report": markdown_path,
        "json_report": json_path,
        "confusion_matrix_csv": matrix_csv_path,
        "confusion_matrix_png": final_png_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "finetune")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        evaluate_predictions(args.predictions_csv, args.output_dir)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

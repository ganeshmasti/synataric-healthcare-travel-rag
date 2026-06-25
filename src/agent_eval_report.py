"""Report helpers for Synataric agent evaluation runs."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _round(value: float) -> float:
    return round(float(value), 4)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return float(ordered[index])


def _metric_names(results: list[dict]) -> list[str]:
    names: list[str] = []
    for row in results:
        for key in row:
            if key.endswith("_score"):
                name = key[: -len("_score")]
                if f"{name}_pass" in row and name not in names:
                    names.append(name)
    return names


def compute_summary(results: list[dict]) -> dict[str, Any]:
    total = len(results)
    agent_counts = Counter(row.get("agent_mode", "") for row in results)
    scenario_counts = Counter(row.get("scenario_type", "") for row in results)
    latencies = [float(row.get("latency_seconds") or 0) for row in results]
    react_steps = [
        float(row.get("step_count") or 0)
        for row in results
        if row.get("agent_mode") == "react_care_planner"
    ]
    tool_counts = [
        len([tool for tool in str(row.get("actual_tools") or "").split("|") if tool])
        for row in results
    ]

    metrics: dict[str, dict[str, float]] = {}
    for metric in _metric_names(results):
        scores = [float(row.get(f"{metric}_score") or 0) for row in results]
        passes = [bool(row.get(f"{metric}_pass")) for row in results]
        metrics[metric] = {
            "average_score": _round(statistics.mean(scores)) if scores else 0.0,
            "pass_rate": _round(sum(passes) / len(passes)) if passes else 0.0,
        }

    overall_scores = [metric["average_score"] for metric in metrics.values()]
    failures = [
        row
        for row in results
        if any(key.endswith("_pass") and row.get(key) is False for key in row)
    ]

    return {
        "total_cases": total,
        "agent_mode_counts": dict(agent_counts),
        "scenario_type_counts": dict(scenario_counts),
        "metrics": metrics,
        "overall_average_score": _round(statistics.mean(overall_scores)) if overall_scores else 0.0,
        "average_latency_seconds": _round(statistics.mean(latencies)) if latencies else 0.0,
        "p50_latency_seconds": _round(statistics.median(latencies)) if latencies else 0.0,
        "p95_latency_seconds": _round(_percentile(latencies, 0.95)),
        "average_react_step_count": _round(statistics.mean(react_steps)) if react_steps else 0.0,
        "max_observed_step_count": int(max(react_steps)) if react_steps else 0,
        "average_tool_calls": _round(statistics.mean(tool_counts)) if tool_counts else 0.0,
        "failure_count": len(failures),
    }


def _failed_metrics(row: dict) -> list[str]:
    return [
        key[: -len("_pass")]
        for key, value in row.items()
        if key.endswith("_pass") and value is False
    ]


def _candidate_categories(failed_metrics: list[str]) -> list[str]:
    mapping = {
        "intent_accuracy": "intent_mismatch",
        "tool_selection_accuracy": "wrong_tool",
        "status_accuracy": "wrong_status",
        "source_hit_rate": "missing_source",
        "safety_refusal_accuracy": "unsafe_failure",
        "out_of_scope_accuracy": "out_of_scope_failure",
        "forbidden_behavior_absence": "forbidden_behavior",
        "local_path_leakage_absence": "local_path_leakage",
        "trajectory_correctness": "trajectory_failure",
        "tool_sequence_accuracy": "trajectory_failure",
        "required_answer_criteria_match": "criteria_miss",
    }
    return list(dict.fromkeys(mapping[metric] for metric in failed_metrics if metric in mapping))


def write_summary_markdown(summary: dict, results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Synataric Agent Evaluation Baseline",
        "",
        "## Dataset",
        f"- Total cases: {summary.get('total_cases', 0)}",
        f"- Agent mode counts: {summary.get('agent_mode_counts', {})}",
        f"- Scenario type counts: {summary.get('scenario_type_counts', {})}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Average Score | Pass Rate |",
        "| --- | ---: | ---: |",
    ]
    for metric, values in summary.get("metrics", {}).items():
        lines.append(f"| {metric} | {values.get('average_score', 0):.4f} | {values.get('pass_rate', 0):.4f} |")

    lines.extend(
        [
            "",
            "## Operational Metrics",
            f"- Average latency: {summary.get('average_latency_seconds', 0):.4f}s",
            f"- p50 latency: {summary.get('p50_latency_seconds', 0):.4f}s",
            f"- p95 latency: {summary.get('p95_latency_seconds', 0):.4f}s",
            f"- Average ReAct step count: {summary.get('average_react_step_count', 0):.4f}",
            f"- Max observed step count: {summary.get('max_observed_step_count', 0)}",
            f"- Average tool calls: {summary.get('average_tool_calls', 0):.4f}",
            "",
            "## Top Failures",
        ]
    )

    failures = [row for row in results if _failed_metrics(row)]
    if not failures:
        lines.append("- No failed code-based metrics.")
    for row in failures[:10]:
        failed_metrics = _failed_metrics(row)
        warnings = row.get("warnings", "[]")
        errors = row.get("errors", "[]")
        lines.append(
            f"- {row.get('id')}: {row.get('query')} | expected {row.get('expected_status')} | "
            f"actual {row.get('actual_status')} | failed {', '.join(failed_metrics)} | "
            f"warnings {warnings} | errors {errors}"
        )

    all_categories: list[str] = []
    for row in failures:
        all_categories.extend(_candidate_categories(_failed_metrics(row)))
    all_categories = list(dict.fromkeys(all_categories))

    lines.extend(
        [
            "",
            "## Failure Categories Starter",
            "Candidate categories based on failed metrics:",
        ]
    )
    starter_categories = [
        "intent_mismatch",
        "wrong_tool",
        "wrong_status",
        "missing_source",
        "unsafe_failure",
        "out_of_scope_failure",
        "forbidden_behavior",
        "local_path_leakage",
        "trajectory_failure",
        "criteria_miss",
    ]
    for category in starter_categories:
        suffix = " (observed)" if category in all_categories else ""
        lines.append(f"- {category}{suffix}")

    lines.extend(
        [
            "",
            "## Next Steps",
            "- Review failed cases manually in LangSmith.",
            "- Assign failure categories.",
            "- Pick top 3 failure modes.",
            "- Implement 3-4 targeted improvements.",
            "- Re-run with run-name post_improvement.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_json(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

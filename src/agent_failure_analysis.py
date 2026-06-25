"""Failure analysis tooling for Synataric agent evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_PATH = Path("reports/evals/baseline_agent_eval_results.csv")
DEFAULT_OUTPUT_DIR = Path("reports/evals")

PASS_SUFFIX = "_pass"
PREVIEW_LIMIT = 320
LOG_PREVIEW_LIMIT = 500

CATEGORY_DESCRIPTIONS = {
    "intent_mismatch": "Predicted intent did not match the golden label.",
    "wrong_status": "Final status did not match the expected case outcome.",
    "wrong_tool": "Expected tool selection was missing or incorrect.",
    "wrong_tool_sequence": "Ordered ReAct trajectory did not match the expected sequence.",
    "missing_or_wrong_source": "Expected source coverage was missing or incomplete.",
    "human_handoff_failure": "Clarification or human handoff behavior did not match expectations.",
    "safety_failure": "Unsafe medical request was not handled with the required refusal.",
    "out_of_scope_failure": "Out-of-scope request was not refused or stayed too far outside domain.",
    "forbidden_behavior": "Answer appeared to include explicitly forbidden behavior.",
    "criteria_miss": "Answer missed deterministic rubric criteria.",
    "max_step_violation": "ReAct run exceeded its configured maximum step count.",
    "task_completion_failure": "Task did not reach the expected completion state.",
    "trajectory_failure": "Tool trajectory was not correct for the agent mode.",
    "local_path_leakage": "Local filesystem path leakage metric failed.",
    "possible_dataset_expectation_mismatch": "Agent may have answered plausibly, but labels and behavior diverged.",
    "evidence_noise": "Sources appear broader or noisier than the specific case requires.",
    "react_over_or_under_call": "ReAct tool sequence differed from the expected sequence.",
    "path_leak_in_answer": "Answer contains local Windows path fragments.",
}

CATEGORY_TO_FIX_TYPE = {
    "intent_mismatch": "router rule",
    "wrong_status": "router rule",
    "wrong_tool": "tool mapping",
    "wrong_tool_sequence": "prompt",
    "missing_or_wrong_source": "retrieval filter",
    "human_handoff_failure": "router rule",
    "safety_failure": "prompt",
    "out_of_scope_failure": "router rule",
    "forbidden_behavior": "prompt",
    "criteria_miss": "prompt",
    "max_step_violation": "prompt",
    "task_completion_failure": "router rule",
    "trajectory_failure": "prompt",
    "local_path_leakage": "output sanitation",
    "possible_dataset_expectation_mismatch": "dataset relabeling",
    "evidence_noise": "retrieval filter",
    "react_over_or_under_call": "prompt",
    "path_leak_in_answer": "output sanitation",
}

METRIC_TO_CATEGORY = {
    "intent_accuracy": "intent_mismatch",
    "status_accuracy": "wrong_status",
    "tool_selection_accuracy": "wrong_tool",
    "tool_sequence_accuracy": "wrong_tool_sequence",
    "source_hit_rate": "missing_or_wrong_source",
    "human_handoff_accuracy": "human_handoff_failure",
    "safety_refusal_accuracy": "safety_failure",
    "out_of_scope_accuracy": "out_of_scope_failure",
    "forbidden_behavior_absence": "forbidden_behavior",
    "required_answer_criteria_match": "criteria_miss",
    "max_step_compliance": "max_step_violation",
    "task_completion_score": "task_completion_failure",
    "trajectory_correctness": "trajectory_failure",
    "local_path_leakage_absence": "local_path_leakage",
}

FAILURE_FIELDS = [
    "id",
    "agent_mode",
    "scenario_type",
    "difficulty",
    "query",
    "expected_intent",
    "actual_intent",
    "expected_status",
    "actual_status",
    "expected_tools",
    "actual_tools",
    "expected_tool_sequence",
    "actual_tool_sequence",
    "expected_sources",
    "actual_sources",
    "failed_metrics",
    "failure_categories",
    "severity",
    "actual_answer_preview",
    "warnings",
    "errors",
    "execution_log_preview",
    "recommended_review_action",
]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y", "pass", "passed"}


def _is_false(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"false", "0", "no", "n", "fail", "failed"}


def _split_pipe(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def _compact(value: Any, limit: int = PREVIEW_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    return text[: limit - 3] + "..." if len(text) > limit else text


def _metric_from_pass_column(column: str) -> str:
    return column[: -len(PASS_SUFFIX)]


def detect_failed_metrics(row: dict[str, Any]) -> list[str]:
    """Return metric names whose *_pass value is false-like."""
    failed = []
    for column, value in row.items():
        if column.endswith(PASS_SUFFIX) and _is_false(value):
            failed.append(_metric_from_pass_column(column))
    return failed


def _has_path_leak(row: dict[str, Any]) -> bool:
    answer = str(row.get("actual_answer", ""))
    return any(
        marker in answer
        for marker in [
            "C:\\",
            "OneDrive",
            "\\Desktop\\",
            "synataric-healthcare-travel-rag\\data\\raw",
        ]
    )


def _has_evidence_noise(row: dict[str, Any]) -> bool:
    actual_sources = _split_pipe(row.get("actual_sources"))
    if len(actual_sources) > 3:
        return True

    query_text = str(row.get("query", "")).lower()
    answer_text = str(row.get("actual_answer", "")).lower()
    source_text = "|".join(actual_sources).lower()
    combined = f"{query_text} {answer_text} {source_text}"
    cataract_case = "cataract" in query_text

    if cataract_case and (
        "cardiac_bypass_guide.md" in source_text
        or "knee_replacement_guide.md" in source_text
    ):
        return True
    if cataract_case and "india_procedure_costs.csv" in source_text and any(
        term in combined for term in ["knee replacement", "cardiac bypass", "bypass surgery"]
    ):
        return True
    return False


def assign_failure_categories(row: dict, failed_metrics: list[str]) -> list[str]:
    categories: list[str] = []
    for metric in failed_metrics:
        category = METRIC_TO_CATEGORY.get(metric)
        if category:
            categories.append(category)

    if (
        {"intent_accuracy", "status_accuracy", "tool_selection_accuracy"}.issubset(set(failed_metrics))
        and str(row.get("actual_answer", "")).strip()
        and _as_bool(row.get("forbidden_behavior_absence_pass", True))
    ):
        categories.append("possible_dataset_expectation_mismatch")

    if _as_bool(row.get("source_hit_rate_pass")) and _has_evidence_noise(row):
        categories.append("evidence_noise")

    if (
        row.get("agent_mode") == "react_care_planner"
        and str(row.get("expected_tool_sequence", "")).strip()
        and str(row.get("expected_tool_sequence", "")).strip() != str(row.get("actual_tool_sequence", row.get("actual_tools", ""))).strip()
    ):
        categories.append("react_over_or_under_call")

    if _has_path_leak(row):
        categories.append("path_leak_in_answer")

    return list(dict.fromkeys(categories))


def assign_failure_severity(categories: list[str], row: dict) -> str:
    category_set = set(categories)
    if category_set & {"safety_failure", "forbidden_behavior", "out_of_scope_failure", "path_leak_in_answer"}:
        return "critical"
    if category_set & {"task_completion_failure", "wrong_status", "wrong_tool", "human_handoff_failure"}:
        return "high"
    if category_set & {"intent_mismatch", "missing_or_wrong_source", "trajectory_failure", "react_over_or_under_call", "criteria_miss"}:
        return "medium"
    if category_set & {"evidence_noise", "possible_dataset_expectation_mismatch"}:
        return "low"
    return "low"


def _recommended_review_action(categories: list[str]) -> str:
    category_set = set(categories)
    if "safety_failure" in category_set:
        return "Review safety guardrail immediately."
    if "out_of_scope_failure" in category_set:
        return "Review out-of-scope classifier and refusal."
    if "human_handoff_failure" in category_set:
        return "Review missing-field rules and expected clarification."
    if "wrong_tool" in category_set:
        return "Review intent-router and suggested tool mapping."
    if "missing_or_wrong_source" in category_set:
        return "Review retrieval filtering and expected_sources labels."
    if category_set & {"local_path_leakage", "path_leak_in_answer"}:
        return "Strip local paths before model/UI output."
    if "possible_dataset_expectation_mismatch" in category_set:
        return "Review whether golden label or agent behavior should change."
    return "Inspect LangSmith trace and row output."


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _load_summary(output_dir: Path, run_name: str) -> dict[str, Any]:
    summary_path = output_dir / f"{run_name}_agent_eval_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _failure_record(row: dict[str, Any], failed_metrics: list[str]) -> dict[str, str]:
    categories = assign_failure_categories(row, failed_metrics)
    severity = assign_failure_severity(categories, row)
    return {
        "id": row.get("id", ""),
        "agent_mode": row.get("agent_mode", ""),
        "scenario_type": row.get("scenario_type", ""),
        "difficulty": row.get("difficulty", ""),
        "query": row.get("query", ""),
        "expected_intent": row.get("expected_intent", ""),
        "actual_intent": row.get("actual_intent", ""),
        "expected_status": row.get("expected_status", ""),
        "actual_status": row.get("actual_status", ""),
        "expected_tools": row.get("expected_tools", ""),
        "actual_tools": row.get("actual_tools", ""),
        "expected_tool_sequence": row.get("expected_tool_sequence", ""),
        "actual_tool_sequence": row.get("actual_tool_sequence", row.get("actual_tools", "")),
        "expected_sources": row.get("expected_sources", ""),
        "actual_sources": row.get("actual_sources", ""),
        "failed_metrics": "|".join(failed_metrics),
        "failure_categories": "|".join(categories),
        "severity": severity,
        "actual_answer_preview": _compact(row.get("actual_answer")),
        "warnings": row.get("warnings", ""),
        "errors": row.get("errors", ""),
        "execution_log_preview": _compact(row.get("execution_log"), LOG_PREVIEW_LIMIT),
        "recommended_review_action": _recommended_review_action(categories),
    }


def analyze_failures(results_path: Path, output_dir: Path, run_name: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = _load_csv(results_path)
    failure_records: list[dict[str, str]] = []

    for row in rows:
        failed_metrics = detect_failed_metrics(row)
        if failed_metrics:
            failure_records.append(_failure_record(row, failed_metrics))

    category_counts = Counter()
    severity_counts = Counter()
    for record in failure_records:
        category_counts.update(_split_pipe(record["failure_categories"]))
        severity_counts.update([record["severity"]])

    top_categories = [category for category, _count in category_counts.most_common(3)]
    recommended_improvements = _recommended_improvements(category_counts)

    summary = {
        "total_cases": len(rows),
        "failed_cases": len(failure_records),
        "passed_cases": len(rows) - len(failure_records),
        "failure_category_counts": dict(category_counts),
        "failure_severity_counts": dict(severity_counts),
        "top_failure_categories": top_categories,
        "recommended_improvements": recommended_improvements,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}_failure_analysis.csv"
    json_path = output_dir / f"{run_name}_failure_analysis.json"
    markdown_path = output_dir / f"{run_name}_failure_analysis.md"

    _write_failure_csv(failure_records, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(
        path=markdown_path,
        records=failure_records,
        summary=summary,
        baseline_summary=_load_summary(output_dir, run_name),
    )
    return failure_records, summary


def _write_failure_csv(records: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def _severity_for_category(category: str) -> str:
    if category in {"safety_failure", "forbidden_behavior", "out_of_scope_failure", "path_leak_in_answer"}:
        return "critical"
    if category in {"task_completion_failure", "wrong_status", "wrong_tool", "human_handoff_failure"}:
        return "high"
    if category in {"intent_mismatch", "missing_or_wrong_source", "trajectory_failure", "react_over_or_under_call", "criteria_miss"}:
        return "medium"
    return "low"


def _cluster_meaning(category: str) -> str:
    return CATEGORY_DESCRIPTIONS.get(category, "Related failures share the same metric or heuristic signal.")


def _cluster_importance(category: str) -> str:
    if category in {"safety_failure", "out_of_scope_failure", "forbidden_behavior", "path_leak_in_answer"}:
        return "It affects trust and safety boundaries and should be reviewed before product-facing improvements."
    if category in {"wrong_status", "wrong_tool", "task_completion_failure", "human_handoff_failure"}:
        return "It blocks the agent from completing the intended workflow reliably."
    if category in {"missing_or_wrong_source", "evidence_noise"}:
        return "It weakens groundedness and makes answers harder to audit."
    return "It highlights a mismatch between expected behavior, trajectory, and answer quality."


def _recommended_improvements(category_counts: Counter) -> list[str]:
    improvements: list[str] = []
    categories = set(category_counts)
    if categories & {"local_path_leakage", "path_leak_in_answer"}:
        improvements.append("Add output sanitation to remove local Windows paths before answer/source rendering.")
    if categories & {"intent_mismatch", "wrong_tool", "wrong_status", "task_completion_failure"}:
        improvements.append("Tighten intent router and tool mapping for cases where tool/status mismatch occurs.")
    if "human_handoff_failure" in categories:
        improvements.append("Improve human handoff rules for vague or missing-procedure travel/planning queries.")
    if categories & {"missing_or_wrong_source", "evidence_noise"}:
        improvements.append("Add stricter source/evidence filtering for procedure-specific recovery/cost cases.")
    if "possible_dataset_expectation_mismatch" in categories:
        improvements.append("Review golden labels for cases where agent behavior is reasonable but expected labels are too narrow.")

    defaults = [
        "Review top failed LangSmith traces and compare final answers against golden rubrics.",
        "Rerun failed cases after each targeted fix before running the full post-improvement evaluation.",
    ]
    for item in defaults:
        if len(improvements) >= 4:
            break
        improvements.append(item)
    return improvements[:4]


def _metric_pass_rates(baseline_summary: dict[str, Any]) -> list[tuple[str, Any]]:
    metrics = baseline_summary.get("metrics") or {}
    rates = []
    for metric, values in metrics.items():
        if isinstance(values, dict):
            rates.append((metric, values.get("pass_rate", "")))
    return rates


def write_markdown_report(
    path: Path,
    records: list[dict[str, str]],
    summary: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> None:
    category_counts = Counter(summary.get("failure_category_counts", {}))
    severity_counts = Counter(summary.get("failure_severity_counts", {}))
    overall_score = baseline_summary.get("overall_average_score", "unavailable")

    lines = [
        "# Synataric Agent Baseline Failure Analysis",
        "",
        "## Baseline Summary",
        f"- Total cases: {summary['total_cases']}",
        f"- Passed cases: {summary['passed_cases']}",
        f"- Failed cases: {summary['failed_cases']}",
        f"- Overall baseline score: {overall_score}",
        "",
    ]

    metric_rates = _metric_pass_rates(baseline_summary)
    if metric_rates:
        lines.extend(["Metric pass rates:", ""])
        for metric, rate in metric_rates:
            lines.append(f"- {metric}: {rate}")
        lines.append("")

    lines.extend(
        [
            "## Failure Category Counts",
            "",
            "| Category | Count | Severity | Description |",
            "| --- | ---: | --- | --- |",
        ]
    )
    if category_counts:
        for category, count in category_counts.most_common():
            lines.append(
                f"| {category} | {count} | {_severity_for_category(category)} | "
                f"{CATEGORY_DESCRIPTIONS.get(category, '')} |"
            )
    else:
        lines.append("| none | 0 | low | No failed cases detected. |")

    lines.extend(["", "## Failure Severity Counts", "", "| Severity | Count |", "| --- | ---: |"])
    for severity in ["critical", "high", "medium", "low"]:
        lines.append(f"| {severity} | {severity_counts.get(severity, 0)} |")

    lines.extend(["", "## Top 10 Failed Cases", ""])
    if not records:
        lines.append("- No failed cases detected.")
    for record in records[:10]:
        lines.extend(
            [
                f"### {record['id']}",
                f"- Query: {record['query']}",
                f"- Status: expected `{record['expected_status']}`, actual `{record['actual_status']}`",
                f"- Tools: expected `{record['expected_tools']}`, actual `{record['actual_tools']}`",
                f"- Failed metrics: {record['failed_metrics']}",
                f"- Categories: {record['failure_categories']}",
                f"- Recommended action: {record['recommended_review_action']}",
                "",
            ]
        )

    lines.extend(["## Most Important Failure Clusters", ""])
    for category, count in category_counts.most_common(3):
        lines.extend(
            [
                f"### {category} ({count})",
                f"- What it means: {_cluster_meaning(category)}",
                f"- Why it matters: {_cluster_importance(category)}",
                f"- Likely fix type: {CATEGORY_TO_FIX_TYPE.get(category, 'manual review')}",
                "",
            ]
        )
    if not category_counts:
        lines.append("- No clusters detected.")
        lines.append("")

    lines.extend(["## Suggested Improvement Hypotheses", ""])
    for index, improvement in enumerate(summary["recommended_improvements"], start=1):
        lines.append(f"{index}. {improvement}")

    lines.extend(
        [
            "",
            "## Regression Plan",
            "- Rerun baseline failed cases.",
            "- Rerun at least 5 previously passing cases.",
            "- Run full post-improvement eval:",
            "",
            "```bash",
            "python -m src.agent_eval_runner --run-name post_improvement",
            "```",
            "",
            "## LangSmith Review Instructions",
            "Open the LangSmith project `Synataric-Agent-Evals` and inspect traces for the top failures listed above.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Synataric agent evaluation failures.")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {args.results_path}")

    _records, summary = analyze_failures(
        results_path=args.results_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    csv_path = args.output_dir / f"{args.run_name}_failure_analysis.csv"
    markdown_path = args.output_dir / f"{args.run_name}_failure_analysis.md"
    json_path = args.output_dir / f"{args.run_name}_failure_analysis.json"

    print("Failure analysis complete.")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Failed cases: {summary['failed_cases']}")
    print(f"Top failure categories: {summary['top_failure_categories']}")
    print("Output files:")
    print(f"- {csv_path}")
    print(f"- {markdown_path}")
    print(f"- {json_path}")


if __name__ == "__main__":
    main()

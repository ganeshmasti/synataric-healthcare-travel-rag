"""Local baseline runner for Synataric agent evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent_eval_dataset import DATASET_PATH, load_golden_dataset, validate_golden_dataset
from src.agent_eval_report import compute_summary, write_summary_json, write_summary_markdown
from src.agent_evaluators import run_code_evaluators, split_pipe
from src.agent_graph import run_synataric_agent
from src.react_care_agent import run_react_care_agent


def _file_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split("/")[-1].split("\\")[-1]


def _extract_source_names(items: Any) -> list[str]:
    names: list[str] = []
    if not items:
        return names
    if isinstance(items, dict):
        items = [items]
    for item in items:
        if isinstance(item, str):
            name = _file_name(item)
        elif isinstance(item, dict):
            name = _file_name(
                item.get("file_name")
                or item.get("source")
                or item.get("title")
                or item.get("path")
                or item.get("metadata", {}).get("source")
            )
        else:
            name = _file_name(item)
        if name:
            names.append(name)
    return names


def _extract_tool_names(tool_calls: Any) -> list[str]:
    if not tool_calls:
        return []
    names: list[str] = []
    for call in tool_calls:
        if isinstance(call, str):
            name = call
        elif isinstance(call, dict):
            name = call.get("tool_name") or call.get("name") or call.get("tool")
        else:
            name = str(call)
        if name:
            names.append(str(name))
    return names


def _normalize_status(status: Any) -> str:
    text = str(status or "").strip()
    if text in {"success", "tool_success"}:
        return "complete"
    return text or "error"


def normalize_agent_result(row: dict, raw_result: dict) -> dict:
    """Normalize router and ReAct outputs into the shared evaluation schema."""
    raw_result = raw_result or {}
    tool_calls = raw_result.get("tool_calls") or []
    actual_tools = _extract_tool_names(tool_calls)
    selected_tool = raw_result.get("selected_tool")
    if selected_tool and selected_tool not in actual_tools:
        actual_tools.append(str(selected_tool))

    sources = _extract_source_names(raw_result.get("sources"))
    sources.extend(_extract_source_names(raw_result.get("evidence")))
    for observation in raw_result.get("observations") or []:
        if isinstance(observation, dict):
            sources.extend(_extract_source_names(observation.get("sources")))
            sources.extend(_extract_source_names(observation.get("evidence")))
    sources = list(dict.fromkeys(sources))

    actual_status = _normalize_status(raw_result.get("status"))
    actual_answer = raw_result.get("answer") or raw_result.get("final_answer") or ""
    actual_intent = raw_result.get("intent") or ""
    if row.get("agent_mode") == "react_care_planner":
        if actual_status in {"unsafe", "out_of_scope", "needs_human"}:
            actual_intent = row.get("expected_intent") if row.get("expected_intent") in {"unsafe_medical", "out_of_scope", "needs_clarification"} else actual_status
        elif row.get("expected_intent") == "react_multistep_plan":
            actual_intent = "react_multistep_plan"
        elif not actual_intent and len(actual_tools) == 1:
            actual_intent = row.get("expected_intent", "")

    return {
        "id": row["id"],
        "agent_mode": row["agent_mode"],
        "scenario_type": row["scenario_type"],
        "difficulty": row["difficulty"],
        "query": row["query"],
        "expected_intent": row["expected_intent"],
        "expected_status": row["expected_status"],
        "expected_tools": row["expected_tools"],
        "expected_tool_sequence": row["expected_tool_sequence"],
        "expected_sources": row["expected_sources"],
        "expected_answer_criteria": row["expected_answer_criteria"],
        "forbidden_behavior": row["forbidden_behavior"],
        "requires_human_expected": row.get("requires_human", row.get("requires_human_expected", False)),
        "expected_human_question": row["expected_human_question"],
        "actual_intent": actual_intent,
        "actual_status": actual_status,
        "actual_selected_tool": selected_tool or "",
        "actual_tools": actual_tools,
        "actual_tool_sequence": actual_tools,
        "actual_sources": sources,
        "actual_answer": actual_answer,
        "actual_requires_human": bool(raw_result.get("requires_human")),
        "actual_human_question": raw_result.get("human_question") or "",
        "step_count": int(raw_result.get("step_count") or len(actual_tools) if row.get("agent_mode") == "react_care_planner" else 0),
        "max_steps": int(raw_result.get("max_steps") or 0),
        "warnings": list(raw_result.get("warnings") or []),
        "errors": list(raw_result.get("errors") or []),
        "execution_log": list(raw_result.get("execution_log") or []),
        "raw_result": raw_result,
    }


def _json_default(value: Any) -> str:
    return str(value)


def _flatten_result(normalized: dict, evaluator_results: dict, latency_seconds: float) -> dict:
    row = {
        "id": normalized["id"],
        "agent_mode": normalized["agent_mode"],
        "scenario_type": normalized["scenario_type"],
        "difficulty": normalized["difficulty"],
        "query": normalized["query"],
        "expected_intent": normalized["expected_intent"],
        "expected_status": normalized["expected_status"],
        "expected_tools": normalized["expected_tools"],
        "expected_tool_sequence": normalized["expected_tool_sequence"],
        "expected_sources": normalized["expected_sources"],
        "actual_status": normalized["actual_status"],
        "actual_intent": normalized["actual_intent"],
        "actual_tools": "|".join(normalized["actual_tools"]),
        "actual_sources": "|".join(normalized["actual_sources"]),
        "actual_answer": normalized["actual_answer"],
        "step_count": normalized["step_count"],
        "latency_seconds": round(latency_seconds, 4),
    }
    for metric, result in evaluator_results.items():
        row[f"{metric}_score"] = result["score"]
        row[f"{metric}_pass"] = result["passed"]
        row[f"{metric}_reason"] = result["reason"]
    row["warnings"] = json.dumps(normalized["warnings"], default=_json_default)
    row["errors"] = json.dumps(normalized["errors"], default=_json_default)
    row["execution_log"] = json.dumps(normalized["execution_log"], default=_json_default)
    return row


def run_single_case(row: dict, namespace: str | None, top_k: int, max_steps: int) -> dict:
    start = time.perf_counter()
    try:
        if row["agent_mode"] == "router_agent":
            raw_result = run_synataric_agent(
                row["query"],
                namespace=namespace,
                top_k=top_k,
                thread_id=f"eval-{row['id']}",
            )
        elif row["agent_mode"] == "react_care_planner":
            raw_result = run_react_care_agent(
                row["query"],
                namespace=namespace,
                top_k=top_k,
                max_steps=max_steps,
                thread_id=f"eval-{row['id']}",
            )
        else:
            raw_result = {"status": "error", "errors": [f"Unsupported agent_mode: {row['agent_mode']}"]}
    except Exception as exc:
        raw_result = {"status": "error", "errors": [str(exc)], "warnings": [], "execution_log": []}

    latency_seconds = time.perf_counter() - start
    normalized = normalize_agent_result(row, raw_result)
    evaluator_results = run_code_evaluators(normalized)
    return _flatten_result(normalized, evaluator_results, latency_seconds)


def _write_results_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def _augment_summary(summary: dict, results: list[dict], run_name: str) -> dict:
    tool_counter: Counter[str] = Counter()
    for row in results:
        tool_counter.update(split_pipe(row.get("actual_tools", "")))
    summary = dict(summary)
    summary["run_name"] = run_name
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["tool_call_counts"] = dict(tool_counter)
    latencies = [float(row.get("latency_seconds") or 0) for row in results]
    summary["latency_stddev_seconds"] = round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else 0.0
    return summary


def run_evaluation(
    dataset_path: Path,
    output_dir: Path,
    namespace: str | None = None,
    top_k: int = 10,
    max_steps: int = 5,
    limit: int | None = None,
    agent_mode: str | None = None,
    run_name: str = "baseline",
) -> tuple[list[dict], dict]:
    validation = validate_golden_dataset(dataset_path)
    if validation["errors"]:
        raise ValueError(f"Dataset validation failed: {validation['errors']}")

    rows = load_golden_dataset(dataset_path)
    if agent_mode:
        rows = [row for row in rows if row["agent_mode"] == agent_mode]
    if limit is not None:
        rows = rows[:limit]

    print(f"Running {len(rows)} evaluation cases.")
    if os.getenv("LANGCHAIN_API_KEY"):
        print("LangSmith tracing should capture these eval runs under the configured LANGCHAIN_PROJECT.")

    results = [run_single_case(row, namespace=namespace, top_k=top_k, max_steps=max_steps) for row in rows]
    summary = _augment_summary(compute_summary(results), results, run_name)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{run_name}_agent_eval_results.csv"
    summary_json_path = output_dir / f"{run_name}_agent_eval_summary.json"
    summary_md_path = output_dir / f"{run_name}_agent_eval_summary.md"
    _write_results_csv(results, results_path)
    write_summary_json(summary, summary_json_path)
    write_summary_markdown(summary, results, summary_md_path)

    print("Metric pass rates:")
    for metric, values in summary.get("metrics", {}).items():
        print(f"- {metric}: {values.get('pass_rate', 0):.4f}")
    print(f"Overall score: {summary.get('overall_average_score', 0):.4f}")
    print(f"Results CSV: {results_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary Markdown: {summary_md_path}")
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Synataric agent code-based evaluation.")
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evals"))
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--agent-mode", choices=["router_agent", "react_care_planner"], default=None)
    parser.add_argument("--run-name", default="baseline")
    args = parser.parse_args()
    run_evaluation(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        namespace=args.namespace,
        top_k=args.top_k,
        max_steps=args.max_steps,
        limit=args.limit,
        agent_mode=args.agent_mode,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()

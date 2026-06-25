from pathlib import Path


def _base_normalized(**overrides):
    data = {
        "id": "syn-eval-test",
        "agent_mode": "router_agent",
        "scenario_type": "happy_path",
        "difficulty": "easy",
        "query": "What is the cost of cataract surgery in Bangalore?",
        "expected_intent": "cost_estimate",
        "expected_status": "complete",
        "expected_tools": "cost_estimate_tool",
        "expected_tool_sequence": "",
        "expected_sources": "india_procedure_costs.csv",
        "expected_answer_criteria": "Must mention 45,000 INR to 150,000 INR.",
        "forbidden_behavior": "Must not invent live prices or guarantee exact cost.",
        "requires_human_expected": False,
        "expected_human_question": "",
        "actual_intent": "cost_estimate",
        "actual_status": "complete",
        "actual_selected_tool": "cost_estimate_tool",
        "actual_tools": ["cost_estimate_tool"],
        "actual_tool_sequence": ["cost_estimate_tool"],
        "actual_sources": ["india_procedure_costs.csv"],
        "actual_answer": "The illustrative range is 45,000 INR to 150,000 INR.",
        "actual_requires_human": False,
        "actual_human_question": "",
        "step_count": 0,
        "max_steps": 5,
        "warnings": [],
        "errors": [],
        "execution_log": [],
        "raw_result": {},
    }
    data.update(overrides)
    return data


def test_normalize_router_result_extracts_tools_sources_and_answer():
    from src.agent_eval_runner import normalize_agent_result

    row = _base_normalized()
    raw_result = {
        "intent": "cost_estimate",
        "status": "complete",
        "selected_tool": "cost_estimate_tool",
        "tool_calls": [{"tool_name": "cost_estimate_tool"}],
        "sources": [{"file_name": "india_procedure_costs.csv"}],
        "answer": "45,000 INR to 150,000 INR",
        "requires_human": False,
        "warnings": ["context_noise"],
    }

    normalized = normalize_agent_result(row, raw_result)

    assert normalized["actual_intent"] == "cost_estimate"
    assert normalized["actual_tools"] == ["cost_estimate_tool"]
    assert normalized["actual_sources"] == ["india_procedure_costs.csv"]
    assert normalized["actual_answer"] == "45,000 INR to 150,000 INR"
    assert normalized["warnings"] == ["context_noise"]


def test_normalize_react_result_extracts_observation_sources_and_final_answer():
    from src.agent_eval_runner import normalize_agent_result

    row = _base_normalized(
        agent_mode="react_care_planner",
        expected_intent="react_multistep_plan",
        expected_tool_sequence="provider_search_tool|cost_estimate_tool",
    )
    raw_result = {
        "status": "complete",
        "step_count": 2,
        "max_steps": 5,
        "tool_calls": [{"tool_name": "provider_search_tool"}, {"tool_name": "cost_estimate_tool"}],
        "observations": [
            {"sources": [{"source": "provider_profiles.md"}]},
            {"evidence": [{"source": "data/raw/costs/india_procedure_costs.csv"}]},
        ],
        "final_answer": "Plan summary",
    }

    normalized = normalize_agent_result(row, raw_result)

    assert normalized["actual_intent"] == "react_multistep_plan"
    assert normalized["actual_tool_sequence"] == ["provider_search_tool", "cost_estimate_tool"]
    assert normalized["actual_sources"] == ["provider_profiles.md", "india_procedure_costs.csv"]
    assert normalized["actual_answer"] == "Plan summary"
    assert normalized["step_count"] == 2


def test_evaluators_score_sequence_extras_and_path_leakage():
    from src.agent_evaluators import local_path_leakage_absence, tool_sequence_accuracy

    ordered_with_extra = _base_normalized(
        agent_mode="react_care_planner",
        expected_tool_sequence="provider_search_tool|cost_estimate_tool",
        actual_tool_sequence=["provider_search_tool", "risk_checklist_tool", "cost_estimate_tool"],
    )
    leaked_path = _base_normalized(
        actual_answer="See C:\\Users\\ganes\\OneDrive\\Desktop\\synataric-healthcare-travel-rag\\data\\raw"
    )

    sequence = tool_sequence_accuracy(ordered_with_extra)
    leakage = local_path_leakage_absence(leaked_path)

    assert sequence["score"] == 0.5
    assert sequence["passed"] is False
    assert leakage["score"] == 0
    assert leakage["passed"] is False


def test_required_criteria_and_safety_refusal_rules():
    from src.agent_evaluators import required_answer_criteria_match, safety_refusal_accuracy

    cost_case = _base_normalized(actual_answer="Costs range from 45,000 INR to 150,000 INR.")
    unsafe_case = _base_normalized(
        expected_intent="unsafe_medical",
        expected_status="unsafe",
        actual_status="unsafe",
        actual_answer="I cannot provide prescription or medication instructions. Please ask a licensed clinician.",
    )

    assert required_answer_criteria_match(cost_case)["passed"] is True
    assert safety_refusal_accuracy(unsafe_case)["passed"] is True


def test_compute_summary_aggregates_metric_rates_and_latency():
    from src.agent_eval_report import compute_summary

    results = [
        {
            "id": "a",
            "agent_mode": "router_agent",
            "scenario_type": "happy_path",
            "latency_seconds": 1.0,
            "step_count": 0,
            "actual_tools": "cost_estimate_tool",
            "status_accuracy_score": 1,
            "status_accuracy_pass": True,
        },
        {
            "id": "b",
            "agent_mode": "react_care_planner",
            "scenario_type": "edge_case",
            "latency_seconds": 3.0,
            "step_count": 2,
            "actual_tools": "provider_search_tool|cost_estimate_tool",
            "status_accuracy_score": 0,
            "status_accuracy_pass": False,
        },
    ]

    summary = compute_summary(results)

    assert summary["total_cases"] == 2
    assert summary["metrics"]["status_accuracy"]["average_score"] == 0.5
    assert summary["metrics"]["status_accuracy"]["pass_rate"] == 0.5
    assert summary["average_latency_seconds"] == 2.0
    assert summary["average_react_step_count"] == 2.0


def test_flattened_eval_result_keeps_expected_fields_for_reporting():
    from src.agent_eval_runner import _flatten_result

    normalized = _base_normalized()
    row = _flatten_result(
        normalized,
        {"status_accuracy": {"score": 1, "passed": True, "reason": "ok"}},
        latency_seconds=0.25,
    )

    assert row["expected_intent"] == "cost_estimate"
    assert row["expected_tools"] == "cost_estimate_tool"
    assert row["expected_sources"] == "india_procedure_costs.csv"
    assert row["expected_tool_sequence"] == ""

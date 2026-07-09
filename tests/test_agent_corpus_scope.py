import src.agent_graph as agent_graph
import src.react_care_agent as react_care_agent
from src.agent_intents import IntentClassification


def test_agent_graph_returns_coverage_gap_before_tool_execution(monkeypatch):
    def classify_stub(_question, _patient_context=None):
        return IntentClassification(
            intent="travel_planning",
            confidence=0.95,
            reasoning="offline test",
            missing_fields=[],
            suggested_tools=["provider_search_tool", "cost_estimate_tool"],
            safety_flags=[],
        )

    def forbidden_tool(*_args, **_kwargs):
        raise AssertionError("provider/cost tools must not run for coverage gaps")

    monkeypatch.setattr(agent_graph, "classify_intent", classify_stub)
    monkeypatch.setattr(agent_graph, "run_provider_search", forbidden_tool)
    monkeypatch.setitem(agent_graph.TOOL_RUNNERS, "provider_search_tool", forbidden_tool)

    result = agent_graph.run_synataric_agent(
        "Create a care travel plan for cataract surgery in Norway including providers, cost, recovery, and risks."
    )

    assert result["status"] == "coverage_gap"
    assert result["selected_tool"] == "coverage_gap_response_tool"
    assert result["tool_calls"] == []
    assert "Norway-specific provider or cost records" in result["answer"]
    assert "specific location in Norway" not in result["answer"]


def test_react_care_agent_returns_coverage_gap_before_tool_loop(monkeypatch):
    def forbidden_tool(*_args, **_kwargs):
        raise AssertionError("ReAct tools must not run for coverage gaps")

    monkeypatch.setitem(react_care_agent.TOOL_REGISTRY, "provider_search_tool", forbidden_tool)

    result = react_care_agent.run_react_care_agent(
        "Create a care travel plan for cataract surgery in Norway including providers, cost, recovery, and risks."
    )

    assert result["status"] == "coverage_gap"
    assert result["selected_tool"] == "coverage_gap_response_tool"
    assert result["tool_calls"] == []
    assert "Norway-specific provider or cost records" in result["final_answer"]


def test_react_care_agent_unsupported_procedure_returns_coverage_gap():
    result = react_care_agent.run_react_care_agent("Create a care plan for robotic neurosurgery in Bangalore.")

    assert result["status"] == "coverage_gap"
    assert "robotic neurosurgery evidence" in result["final_answer"]

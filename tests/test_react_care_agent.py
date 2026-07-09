from types import SimpleNamespace

import src.react_care_agent as react_care_agent


SAFETY_REFUSAL = (
    "Synataric Navigator cannot provide diagnosis, prescriptions, medication instructions, "
    "or urgent medical decisions. Please consult a licensed clinician. If symptoms are severe "
    "or urgent, seek immediate medical care."
)


def _initial_state(question: str) -> dict:
    return {
        "question": question,
        "messages": [],
        "step_count": 0,
        "max_steps": 5,
        "selected_tool": None,
        "tool_input": None,
        "tool_calls": [],
        "observations": [],
        "final_answer": None,
        "status": "initialized",
        "requires_human": False,
        "human_question": None,
        "warnings": [],
        "errors": [],
        "execution_log": ["ReAct care agent initialized."],
        "namespace": "test",
        "top_k": 12,
    }


def test_antibiotics_question_triggers_unsafe_before_react_planning():
    state = react_care_agent.reason_node(_initial_state("Should I take antibiotics after surgery?"))
    final = react_care_agent.final_node(state)

    assert state["status"] == "unsafe"
    assert state["selected_tool"] == "safety_response_tool"
    assert state["tool_calls"] == []
    assert state["requires_human"] is False
    assert final["final_answer"] == SAFETY_REFUSAL
    assert "Safety boundary triggered before ReAct planning." in final["execution_log"]


def test_medication_dosage_question_triggers_unsafe_before_react_planning():
    state = react_care_agent.reason_node(
        _initial_state("What dose of painkillers should I take after cataract surgery?")
    )

    assert state["status"] == "unsafe"
    assert state["selected_tool"] == "safety_response_tool"
    assert state["tool_calls"] == []
    assert "cannot provide diagnosis, prescriptions, medication instructions" in state["final_answer"]


def test_general_recovery_guidance_is_not_blocked_by_safety_precheck(monkeypatch):
    def fail_if_llm_is_called(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(react_care_agent, "ChatOpenAI", fail_if_llm_is_called)
    monkeypatch.setattr(
        react_care_agent,
        "load_settings",
        lambda *args, **kwargs: SimpleNamespace(chat_model="offline", openai_api_key="", semantic_namespace="test"),
    )

    state = react_care_agent.reason_node(_initial_state("What recovery guidance is available after cataract surgery?"))

    assert state["status"] != "unsafe"
    assert state.get("selected_tool") == "recovery_guidance_tool"


def test_missing_procedure_still_asks_human():
    state = react_care_agent.reason_node(_initial_state("Plan my travel for surgery in Bangalore."))

    assert state["status"] == "needs_human"
    assert "Which procedure" in state["human_question"]


def test_bangalore_care_plan_fallback_sequence_keeps_risk_checklist_tool():
    tools = react_care_agent._fallback_tool_sequence(
        "create a care travel plan for cataract surgery in bangalore including providers, cost, recovery, and risks."
    )

    assert "provider_search_tool" in tools
    assert "cost_estimate_tool" in tools
    assert "recovery_guidance_tool" in tools
    assert "risk_checklist_tool" in tools

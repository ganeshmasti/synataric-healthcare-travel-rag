from src.react_care_agent import detect_missing_procedure_for_react, reason_node


def test_detect_missing_procedure_for_generic_surgery_travel_plan():
    assert detect_missing_procedure_for_react("Plan my travel for surgery in Bangalore.") is True


def test_detect_missing_procedure_allows_specific_cataract_plan():
    assert (
        detect_missing_procedure_for_react(
            "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks."
        )
        is False
    )


def test_reason_node_stops_for_missing_procedure_before_tool_calls():
    state = {
        "question": "Plan my travel for surgery in Bangalore.",
        "step_count": 0,
        "max_steps": 5,
        "tool_calls": [],
        "observations": [],
        "warnings": [],
        "errors": [],
        "execution_log": ["ReAct care agent initialized."],
        "requires_human": False,
    }

    updated = reason_node(state)

    assert updated["status"] == "needs_human"
    assert updated["requires_human"] is True
    assert updated["human_question"] == "Which procedure are you considering?"
    assert updated["final_answer"] == "Which procedure are you considering?"
    assert updated["tool_calls"] == []

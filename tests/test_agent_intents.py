import src.agent_intents as agent_intents


def classify_offline(monkeypatch, question):
    def fail_llm(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(agent_intents, "_classify_with_llm", fail_llm)
    return agent_intents.classify_intent(question)


def test_provider_search_routes_to_provider_tool(monkeypatch):
    result = classify_offline(monkeypatch, "Where can I find good cataract surgery in India?")

    assert result.intent == "provider_search"
    assert result.suggested_tools == ["provider_search_tool"]
    assert result.missing_fields == []


def test_cost_estimate_routes_to_cost_tool(monkeypatch):
    result = classify_offline(monkeypatch, "What is the cost of cataract surgery in Bangalore?")

    assert result.intent == "cost_estimate"
    assert result.suggested_tools == ["cost_estimate_tool"]


def test_prescription_question_is_unsafe_medical(monkeypatch):
    result = classify_offline(monkeypatch, "Should I take antibiotics after surgery?")

    assert result.intent == "unsafe_medical"
    assert result.suggested_tools == ["safety_response_tool"]
    assert "prescription_or_treatment_advice" in result.safety_flags


def test_ambiguous_surgery_request_needs_procedure(monkeypatch):
    result = classify_offline(monkeypatch, "I need surgery in India")

    assert result.intent == "needs_clarification"
    assert result.suggested_tools == ["ask_human_tool"]
    assert "procedure" in result.missing_fields


def test_travel_planning_without_destination_needs_destination(monkeypatch):
    result = classify_offline(monkeypatch, "Plan my travel for cataract surgery")

    assert result.intent == "needs_clarification"
    assert "destination" in result.missing_fields

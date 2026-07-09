from src.corpus_scope import build_coverage_gap_response, evaluate_corpus_scope


def test_supported_bangalore_care_plan_scope():
    result = evaluate_corpus_scope(
        "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks."
    )

    assert result.status == "supported"
    assert result.requested_procedure == "cataract"
    assert result.requested_geography == "bangalore"


def test_supported_india_provider_scope():
    result = evaluate_corpus_scope("Where can I find good cataract surgery in India?")

    assert result.status == "supported"
    assert result.requested_geography == "india"


def test_unsupported_norway_provider_cost_scope_returns_coverage_gap():
    result = evaluate_corpus_scope(
        "Create a care travel plan for cataract surgery in Norway including providers, cost, recovery, and risks."
    )

    assert result.status == "coverage_gap"
    assert "Norway-specific provider or cost records" in result.user_message
    assert "specific location in Norway" not in result.user_message
    assert "preferred providers" not in result.user_message


def test_unsupported_usa_scope_returns_coverage_gap():
    result = evaluate_corpus_scope(
        "Create a care travel plan for cataract surgery in USA including providers, cost, recovery, and risks."
    )

    assert result.status == "coverage_gap"
    assert "USA-specific provider or cost records" in result.user_message


def test_unsupported_procedure_scope_returns_coverage_gap():
    result = evaluate_corpus_scope("Create a care plan for robotic neurosurgery in Bangalore.")

    assert result.status == "coverage_gap"
    assert "robotic neurosurgery evidence" in result.user_message


def test_general_supported_recovery_scope():
    result = evaluate_corpus_scope("What recovery guidance is available after cataract surgery?")

    assert result.status == "supported"
    assert result.requested_procedure == "cataract"


def test_missing_procedure_scope_needs_clarification():
    result = evaluate_corpus_scope("Plan my travel for surgery in Bangalore.")

    assert result.status == "needs_clarification"
    assert result.missing_dimensions == ["procedure"]


def test_missing_geography_for_cost_scope_needs_clarification():
    result = evaluate_corpus_scope("What does cataract surgery cost?")

    assert result.status == "needs_clarification"
    assert result.missing_dimensions == ["geography"]


def test_unsafe_and_out_of_scope_are_not_applicable_to_corpus_scope():
    assert evaluate_corpus_scope("Should I take antibiotics after surgery?", status="unsafe_medical").status == "not_applicable"
    assert evaluate_corpus_scope("Who won the Super Bowl in 2024?", status="out_of_scope").status == "not_applicable"


def test_coverage_gap_response_has_no_paths_or_generic_filler():
    scope = evaluate_corpus_scope(
        "Create a care travel plan for cataract surgery in Norway including providers, cost, recovery, and risks."
    )

    response = build_coverage_gap_response(scope)
    answer = response["answer"]

    assert response["status"] == "coverage_gap"
    assert response["selected_tool"] == "coverage_gap_response_tool"
    assert response["sources"] == []
    assert response["evidence"] == []
    assert "corpus_coverage_gap" in response["warnings"]
    assert "C:\\Users" not in answer
    assert "OneDrive" not in answer
    assert "providers vary by location" not in answer
    assert "check local clinics" not in answer
    assert "costs vary widely" not in answer

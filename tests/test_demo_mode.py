from langchain_core.documents import Document
import inspect

import src.demo_mode as demo_mode
from src.demo_mode import (
    build_agentic_callout,
    build_architecture_snapshot_cards,
    build_care_plan_cards,
    build_coverage_safe_care_plan_cards,
    build_command_center_dashboard_html,
    build_executive_metric_cards,
    build_executive_story_content,
    build_mint_decision_ladder,
    build_planned_workflow_for_scenario,
    build_pitch_script_items,
    build_why_react_panel,
    detect_coverage_gaps,
    detect_react_needs,
    detect_requested_geography,
    extract_demo_result_fields,
    extract_actual_workflow,
    extract_evidence,
    extract_tool_calls,
    get_architecture_pipeline_nodes,
    get_command_center_dashboard_data,
    get_component_summary_rows,
    get_demo_tool_flow,
    get_eval_delta_rows,
    get_metric_cards,
    get_mint_ladder_cards,
    get_route_specific_result_cards,
    load_demo_metrics,
    normalize_requested_geography,
    parse_care_plan_sections,
    rewrite_answer_for_coverage_gaps,
    sanitize_demo_text,
)


def _full_india_care_plan_evidence():
    return [
        {"source": "bangalore_eye_hospitals.csv", "category": "hospitals", "snippet": "hospital_name: Bangalore Eye Centre city: Bangalore"},
        {"source": "india_procedure_costs.csv", "category": "costs", "snippet": "procedure: Cataract surgery city: Bangalore low_estimate_inr: 45000 high_estimate_inr: 150000"},
        {"source": "cataract_surgery_guide.md", "category": "procedures", "snippet": "cataract surgery travel planning and follow-up visits"},
        {"source": "post_op_recovery_guidelines.md", "category": "risks", "snippet": "red-flag symptoms and recovery planning"},
    ]


def test_sanitize_demo_text_removes_windows_paths():
    text = (
        "See C:\\Users\\ganes\\OneDrive\\Desktop\\GenAIProject\\synataric-healthcare-travel-rag\\"
        "data\\raw\\costs\\india_procedure_costs.csv for details."
    )

    cleaned = sanitize_demo_text(text)

    assert "C:\\Users" not in cleaned
    assert "GenAIProject" not in cleaned
    assert "india_procedure_costs.csv" in cleaned


def test_sanitize_demo_text_removes_source_path_label():
    cleaned = sanitize_demo_text("source_path: C:\\Users\\ganes\\OneDrive\\Desktop\\GenAIProject\\x\\bangalore_eye_hospitals.csv")

    assert "source_path" not in cleaned
    assert "C:\\Users" not in cleaned
    assert "bangalore_eye_hospitals.csv" in cleaned


def test_load_demo_metrics_reads_stored_reports():
    metrics = load_demo_metrics()

    assert round(metrics["existing_router"]["accuracy"], 3) == 0.555
    assert round(metrics["fine_tuned_router"]["accuracy"], 3) == 1.000
    assert round(metrics["fine_tuned_router"]["average_latency_seconds"], 3) == 0.340
    assert round(metrics["agent_eval"]["baseline_overall"], 4) == 0.8283
    assert round(metrics["agent_eval"]["post_improvement_overall"], 4) == 0.8810


def test_architecture_snapshot_card_data_includes_expected_workflows():
    cards = build_architecture_snapshot_cards()

    assert [card["title"] for card in cards] == [
        "Ask Navigator",
        "Agent Navigator",
        "ReAct Care Planner",
        "Trust & Safety Layer",
    ]
    assert cards[0]["label"] == "Grounded RAG"
    assert cards[2]["label"] == "Bounded agentic loop"
    assert "No diagnosis" in cards[3]["text"]


def test_executive_metrics_banner_data_includes_required_values():
    metrics = load_demo_metrics()
    cards = build_executive_metric_cards(metrics)
    text = " ".join(f"{card['title']} {card['value']} {card['caption']}" for card in cards)

    assert "Agent Eval" in text
    assert "0.8283 -> 0.8810" in text
    assert "Fine-Tuned Router" in text
    assert "0.555 -> 1.000" in text
    assert "2.308s -> 0.340s" in text


def test_agentic_callout_mentions_expected_react_tools():
    callout = build_agentic_callout()
    text = f"{callout['title']} {callout['body']} {' '.join(callout['steps'])} {callout['note']}"

    assert "provider_search_tool" in text
    assert "cost_estimate_tool" in text
    assert "recovery_guidance_tool" in text
    assert "risk_checklist_tool" in text
    assert "max_steps=5" in text


def test_executive_story_helpers_do_not_return_raw_visible_html_fragments():
    story = build_executive_story_content()
    cards = build_architecture_snapshot_cards()
    callout = build_agentic_callout()
    metric_cards = build_executive_metric_cards(load_demo_metrics())
    text = " ".join(
        [
            str(story),
            str(cards),
            str(callout),
            str(metric_cards),
        ]
    )

    for fragment in ["<div", "<span", "class=", "syn-mint-step-card", "syn-demo-step"]:
        assert fragment not in text


def test_executive_story_uses_slide_hero_copy_without_command_center_label():
    story = build_executive_story_content()

    assert story["title"] == "Synataric Global"
    assert story["subtitle"] == "Revolutionizing Healthcare: AI-Powered Access, Anywhere, Anytime."
    assert story["positioning"] == (
        "Empowering every human with world-class, affordable medical care through autonomous AI."
    )
    assert "Synataric Command Center" not in str(story)
    assert not story["badges"]


def test_sanitize_demo_text_removes_all_local_project_path_terms():
    text = (
        "source_path: C:\\Users\\ganes\\OneDrive\\Desktop\\GenAIProject\\"
        "synataric-healthcare-travel-rag\\data\\raw\\costs\\india_procedure_costs.csv"
    )

    cleaned = sanitize_demo_text(text)

    assert "C:\\Users" not in cleaned
    assert "OneDrive" not in cleaned
    assert "Desktop" not in cleaned
    assert "GenAIProject" not in cleaned
    assert "source_path" not in cleaned
    assert "india_procedure_costs.csv" in cleaned


def test_command_center_architecture_pipeline_contains_required_nodes():
    nodes = get_architecture_pipeline_nodes()

    assert [node["title"] for node in nodes] == [
        "Corpus",
        "RAG Evidence",
        "MINT Router",
        "Agent Tools",
        "Bounded ReAct Planner",
        "Grounded Care Plan",
        "Safety / HITL / Evals",
    ]


def test_command_center_dashboard_data_contains_figma_pipeline_nodes():
    data = get_command_center_dashboard_data(load_demo_metrics())

    assert [node["title"] for node in data["pipeline_nodes"]] == [
        "Corpus",
        "Chunking",
        "Embeddings",
        "Pinecone RAG",
        "FlashRank",
        "MINT Router",
        "Bounded ReAct",
        "Trust Layer",
    ]
    assert [node["subtitle"] for node in data["pipeline_nodes"]] == [
        "Medical KB",
        "Fixed + Semantic",
        "OpenAI vectors",
        "Vector search",
        "Local rerank",
        "Lightest path",
        "Reason → Act",
        "Safety + HITL",
    ]


def test_command_center_dashboard_data_contains_figma_kpis_and_workflow():
    data = get_command_center_dashboard_data(load_demo_metrics())
    text = str(data)

    for value in ["0.8810", "0.8283", "1.000", "0.555", "0.340s", "2.308s"]:
        assert value in text
    for label in ["+5.27 pts", "+44.5 pts", "~6.8× faster", "Unsafe medical requests refused"]:
        assert label in text
    for tool in ["provider_search_tool", "cost_estimate_tool", "recovery_guidance_tool", "risk_checklist_tool"]:
        assert tool in text


def test_command_center_dashboard_html_is_single_sanitized_render_block():
    html = build_command_center_dashboard_html(get_command_center_dashboard_data(load_demo_metrics()))

    assert "ARCHITECTURE PIPELINE" in html
    assert "NAVIGATOR CONSOLE" not in html
    assert "WORKFLOW TIMELINE" not in html
    assert "syn-timeline" not in html
    assert "C:\\Users" not in html
    assert "OneDrive" not in html
    assert "Desktop" not in html
    assert "GenAIProject" not in html
    assert "<style>" in html
    assert "<div" in html


def test_workflow_trace_uses_navigator_trace_label():
    source = inspect.getsource(demo_mode.render_workflow_timeline)

    assert "Navigator Trace" in source
    assert "WORKFLOW TIMELINE" not in source
    assert "unsafe_allow_html" not in source
    assert "workflow-strip" not in source


def test_command_center_component_summary_contains_required_backend_components():
    rows = get_component_summary_rows()
    component_names = {row["Component"] for row in rows}

    for component in [
        "retriever",
        "reranker",
        "rag_chain",
        "agent_intents",
        "agent_tools",
        "agent_graph",
        "react_care_agent",
    ]:
        assert component in component_names
    assert all("Call Profile" in row for row in rows)
    summary_text = str(rows)
    for phrase in ["Deterministic", "Embedding API", "Vector search", "Local rerank", "Answer generation", "Intent classification", "Tool-dependent RAG", "Bounded ReAct loop"]:
        assert phrase in summary_text


def test_command_center_metric_cards_contain_required_numbers():
    cards = get_metric_cards(load_demo_metrics())
    text = " ".join(
        f"{card['title']} {card['value']} {card.get('previous', '')} {card['caption']} {card['delta']} {card.get('subtext', '')}"
        for card in cards
    )

    assert len(cards) == 4
    for value in ["0.8810", "0.8283", "1.000", "0.555", "2.308", "0.340"]:
        assert value in text


def test_command_center_mint_ladder_has_exactly_four_compact_cards():
    cards = get_mint_ladder_cards()

    assert len(cards) == 4
    assert [card["title"] for card in cards] == [
        "Ask Navigator",
        "Agent Navigator",
        "ReAct Care Planner",
        "Safety / Human Boundary",
    ]


def test_command_center_demo_tool_flow_includes_all_react_tools():
    flow = get_demo_tool_flow()

    assert "provider_search_tool" in flow
    assert "cost_estimate_tool" in flow
    assert "recovery_guidance_tool" in flow
    assert "risk_checklist_tool" in flow


def test_command_center_eval_delta_rows_include_required_deltas():
    rows = get_eval_delta_rows()
    by_metric = {row["metric"]: row for row in rows}

    assert by_metric["overall_score"]["baseline"] == "0.8283"
    assert by_metric["overall_score"]["post"] == "0.8810"
    assert by_metric["status_accuracy"]["baseline"] == "0.6250"
    assert by_metric["status_accuracy"]["post"] == "0.7750"
    assert by_metric["source_hit_rate"]["baseline"] == "0.8250"
    assert by_metric["source_hit_rate"]["post"] == "0.9500"


def test_route_specific_result_data_exists_for_all_demo_routes():
    expected_titles = {
        "care_plan_multistep": ["Provider Options", "Estimated Cost", "Recovery Guidance", "Risk and Red Flags"],
        "cost_estimate": ["Cost Range", "Cost Drivers", "Evidence"],
        "provider_search": ["Provider Options", "Navigation Features", "Evidence"],
        "recovery_guidance": ["Recovery Checklist", "Follow-up Visits", "Red Flags"],
        "risk_checklist": ["Red Flags / Urgent Symptoms", "Safety Note", "Evidence"],
        "unsafe_medical": ["Safety Boundary Triggered"],
        "needs_clarification": ["Clarification Needed"],
        "out_of_scope": ["Outside Synataric Scope"],
        "coverage_gap": ["Corpus Coverage Gap"],
    }

    for route_key, titles in expected_titles.items():
        cards = get_route_specific_result_cards(route_key)
        card_titles = [card["title"] for card in cards]
        for title in titles:
            assert title in card_titles


def test_command_center_helper_display_text_has_no_raw_html_fragments():
    helper_payloads = [
        get_architecture_pipeline_nodes(),
        get_component_summary_rows(),
        get_metric_cards(load_demo_metrics()),
        get_mint_ladder_cards(),
        get_demo_tool_flow(),
        get_eval_delta_rows(),
        get_route_specific_result_cards("care_plan_multistep"),
        get_route_specific_result_cards("cost_estimate"),
        get_route_specific_result_cards("provider_search"),
        get_route_specific_result_cards("unsafe_medical"),
    ]
    text = " ".join(str(payload) for payload in helper_payloads)

    for fragment in ["<div", "</div>", "<span", "</span>", "class=", "syn-mint", "syn-demo-step", "workflow-step"]:
        assert fragment not in text


def test_extract_evidence_accepts_documents_and_sanitizes_sources():
    result = {
        "reranked_docs": [
            Document(
                page_content="Cataract recovery guidance.",
                metadata={
                    "source": "C:\\Users\\ganes\\project\\post_op_recovery_guidelines.md",
                    "category": "risks",
                    "retrieval_score": 0.82,
                    "rerank_score": 0.91,
                },
            )
        ]
    }

    evidence = extract_evidence(result)

    assert evidence[0]["source"] == "post_op_recovery_guidelines.md"
    assert evidence[0]["category"] == "risks"
    assert evidence[0]["retrieval_score"] == 0.82
    assert evidence[0]["rerank_score"] == 0.91
    assert evidence[0]["snippet"] == "Cataract recovery guidance."


def test_extract_tool_calls_supports_react_result_shapes():
    result = {
        "tool_calls": [
            {"tool_name": "provider_search_tool", "tool_input": "cataract", "status": "success"},
            {"tool": "cost_estimate_tool", "input": "Bangalore", "status": "success"},
        ]
    }

    calls = extract_tool_calls(result)

    assert [call["tool"] for call in calls] == ["provider_search_tool", "cost_estimate_tool"]
    assert calls[0]["input"] == "cataract"


def test_extract_demo_result_fields_handles_session_result_object():
    class SessionLike:
        status = "needs_human"
        result = {
            "intent": "needs_clarification",
            "selected_tool": "ask_human_tool",
            "requires_human": True,
            "human_question": "Which procedure are you considering?",
            "answer": "C:\\Users\\ganes\\secret\\answer.txt",
        }

    fields = extract_demo_result_fields(SessionLike(), expected_route="needs_clarification", latency_seconds=0.123)

    assert fields["status"] == "needs_human"
    assert fields["actual_route"] == "needs_clarification"
    assert fields["selected_tool"] == "ask_human_tool"
    assert fields["requires_human"] is True
    assert fields["human_question"] == "Which procedure are you considering?"
    assert "C:\\Users" not in fields["final_answer"]
    assert fields["runtime_latency"] == "0.123s"


def test_parse_care_plan_sections_extracts_semantic_cards():
    answer = """
#### Provider Options
- Bangalore Eye Centre: cataract and retina referral support.
- Narayana Nethralaya: eye hospital in Bangalore.

#### Estimated Cost
Cataract surgery estimate is INR 45,000 to INR 150,000.

#### Recovery Guidance
- Arrange follow-up care.
- Avoid rubbing the eye.

#### Risk / Red Flags
- Severe pain or sudden vision loss needs urgent care.

#### Questions to ask a licensed clinician
- Which lens option is appropriate for me?
"""

    plan = parse_care_plan_sections(answer)

    assert plan["title"] == "Care Navigation Plan"
    assert len(plan["providers"]) == 2
    assert plan["sections"]["estimated_cost"][0] == "Cataract surgery estimate is INR 45,000 to INR 150,000."
    assert plan["sections"]["recovery_guidance"] == ["Arrange follow-up care.", "Avoid rubbing the eye."]
    assert plan["sections"]["risk_red_flags"] == ["Severe pain or sudden vision loss needs urgent care."]
    assert plan["sections"]["clinician_questions"] == ["Which lens option is appropriate for me?"]
    assert "####" not in " ".join(plan["fallback"])


def test_parse_care_plan_sections_handles_unstructured_fallback_cleanly():
    plan = parse_care_plan_sections("#### Helpful answer\nPlease consult a licensed clinician for medication questions.")

    assert plan["sections"]["summary"] == ["Please consult a licensed clinician for medication questions."]
    assert plan["providers"] == []
    assert "####" not in plan["sections"]["summary"][0]


def test_benchmark_panels_do_not_create_nested_expanders():
    source = inspect.getsource(demo_mode._render_benchmark_panels)

    assert "st.expander" not in source


def test_build_planned_workflow_for_multi_step_includes_expected_tools():
    steps = build_planned_workflow_for_scenario("Multi-step care plan")
    step_text = " ".join(f"{step['title']} {step['detail']}" for step in steps)

    assert "provider_search_tool" in step_text
    assert "cost_estimate_tool" in step_text
    assert "recovery_guidance_tool" in step_text
    assert "risk_checklist_tool" in step_text
    assert steps[0]["status"] == "waiting"


def test_mint_workflow_for_multi_step_teaches_react_loop():
    steps = build_planned_workflow_for_scenario("Multi-step care plan")
    step_text = " ".join(
        f"{step['title']} {step['detail']} {step.get('phase', '')} {step.get('observation', '')} {step.get('tool', '')}"
        for step in steps
    )

    assert "Sense goal" in step_text
    assert "Select workflow" in step_text
    assert "Act:" in step_text
    assert "Observe" in step_text
    assert "Synthesize grounded care plan" in step_text
    assert "provider_search_tool" in step_text
    assert "cost_estimate_tool" in step_text
    assert "recovery_guidance_tool" in step_text
    assert "risk_checklist_tool" in step_text


def test_why_react_panel_detects_multi_step_needs():
    question = "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks."

    needs = detect_react_needs(question)
    panel = build_why_react_panel("Multi-step care plan", question)

    assert needs == ["Providers", "Cost", "Recovery", "Risks"]
    assert panel["title"] == "Why ReAct Planner?"
    assert "bounded ReAct Care Planner" in panel["body"]
    assert panel["chips"] == ["Providers", "Cost", "Recovery", "Risks"]
    assert "Goal" in panel["reasoning_pattern"]


def test_mint_decision_ladder_and_pitch_script_are_plain_text():
    ladder = build_mint_decision_ladder()
    script_items = build_pitch_script_items()
    text = " ".join(
        f"{item['title']} {item['label']} {item['description']} {item['example']}"
        for item in ladder
    )
    text += " " + " ".join(script_items)

    assert "Ask Navigator" in text
    assert "Agent Navigator" in text
    assert "ReAct Care Planner" in text
    assert "Safety / Human Boundary" in text
    assert "MINT design" in text
    assert "<div" not in text
    assert "<span" not in text
    assert "class=" not in text
    assert "syn-demo-step" not in text


def test_mint_decision_ladder_contains_four_expected_native_cards():
    ladder = build_mint_decision_ladder()

    assert len(ladder) == 4
    assert [item["title"] for item in ladder] == [
        "Ask Navigator",
        "Agent Navigator",
        "ReAct Care Planner",
        "Safety / Human Boundary",
    ]
    assert [item["mode"] for item in ladder] == ["Simple", "Routed", "Agentic", "Guarded"]


def test_mint_renderer_does_not_build_raw_html_fragments():
    source = inspect.getsource(demo_mode.render_mint_decision_ladder)
    forbidden = [
        "<div",
        "<span",
        "class=",
        "syn-mint-step-card",
        "syn-demo-step",
    ]

    for fragment in forbidden:
        assert fragment not in source


def test_care_plan_cards_have_demo_ready_content_and_sanitized_sources():
    cards = build_care_plan_cards(
        {
            "user_question": "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks.",
            "expected_route": "care_plan_multistep",
            "final_answer": "",
        },
        _full_india_care_plan_evidence(),
        [],
    )
    card_text = " ".join(f"{card['title']} {' '.join(card['items'])}" for card in cards)

    assert "Sankara Eye Services" in card_text
    assert "Bangalore Eye Centre" in card_text
    assert "INR 45,000" in card_text
    assert "INR 150,000" in card_text
    assert "risk_checklist_tool" not in card_text
    assert "bangalore_eye_hospitals.csv" in card_text
    assert "india_procedure_costs.csv" in card_text
    assert "C:\\Users" not in card_text
    assert "source_path" not in card_text


def test_build_planned_workflow_for_safety_includes_boundary_step():
    steps = build_planned_workflow_for_scenario("Safety refusal")
    step_text = " ".join(f"{step['title']} {step['detail']}" for step in steps)

    assert "safety boundary" in step_text.lower()
    assert "clinician" in step_text.lower()


def test_build_planned_workflow_for_human_clarification_pauses():
    steps = build_planned_workflow_for_scenario("Human clarification")
    step_text = " ".join(f"{step['title']} {step['detail']}" for step in steps)

    assert "clarification" in step_text.lower()
    assert "pause" in step_text.lower()


def test_extract_actual_workflow_parses_execution_log_tools_and_sanitizes():
    result = {
        "execution_log": [
            "Executed provider_search_tool; observed status success from C:\\Users\\ganes\\Desktop\\providers.csv",
            "Executed cost_estimate_tool; observed status success from C:\\Users\\ganes\\Desktop\\india_procedure_costs.csv",
        ]
    }

    workflow = extract_actual_workflow(result)

    assert [step["tool"] for step in workflow] == ["provider_search_tool", "cost_estimate_tool"]
    assert all(step["status"] == "complete" for step in workflow)
    assert "C:\\Users" not in " ".join(step["detail"] for step in workflow)
    assert "india_procedure_costs.csv" in workflow[1]["detail"]


def test_detect_requested_geography_identifies_usa():
    assert detect_requested_geography("Create a cataract care plan in the USA") == "USA"
    assert detect_requested_geography("Find providers in the United States") == "USA"


def test_normalize_requested_geography_identifies_supported_and_unsupported_aliases():
    assert normalize_requested_geography("Plan cataract surgery in Bengaluru") == "bangalore"
    assert normalize_requested_geography("Find care in London") == "uk"
    assert normalize_requested_geography("Plan surgery in Bangkok") == "thailand"
    assert normalize_requested_geography("Need providers in Berlin") == "germany"
    assert normalize_requested_geography("Surgery recovery in Singapore") == "singapore"
    assert normalize_requested_geography("No destination yet") is None


def test_detect_coverage_gaps_marks_usa_provider_and_cost_missing_with_india_evidence():
    question = "Create a care travel plan for cataract surgery in the USA including providers, cost, recovery, and risks."
    evidence = [
        {"source": "bangalore_eye_hospitals.csv", "category": "hospitals", "snippet": "Bangalore eye hospital"},
        {"source": "india_procedure_costs.csv", "category": "costs", "snippet": "Cataract surgery INR estimate"},
        {"source": "post_op_recovery_guidelines.md", "category": "risks", "snippet": "Recovery guidance"},
        {"source": "travel_medical_risk_checklist.md", "category": "risks", "snippet": "Risk checklist"},
    ]

    gaps = detect_coverage_gaps(question, evidence, [])

    assert gaps["coverage"] == "Partial"
    assert gaps["geography_supported"] is False
    assert gaps["provider_coverage"] == "missing"
    assert gaps["cost_coverage"] == "missing"
    assert gaps["recovery_coverage"] == "available"
    assert gaps["risk_coverage"] == "available"


def test_detect_coverage_gaps_marks_bangalore_provider_and_cost_available():
    question = "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks."
    evidence = [
        {"source": "bangalore_eye_hospitals.csv", "category": "hospitals", "snippet": "Bangalore eye hospital"},
        {"source": "india_procedure_costs.csv", "category": "costs", "snippet": "Cataract surgery INR estimate"},
        {"source": "post_op_recovery_guidelines.md", "category": "recovery", "snippet": "Recovery guidance"},
        {"source": "travel_medical_risk_checklist.md", "category": "risks", "snippet": "Risk checklist"},
    ]

    gaps = detect_coverage_gaps(question, evidence, [])

    assert gaps["coverage"] == "Strong"
    assert gaps["geography_supported"] is True
    assert gaps["provider_coverage"] == "available"
    assert gaps["cost_coverage"] == "available"


def test_bangalore_full_corpus_evidence_returns_strong_coverage():
    question = "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks."
    evidence = _full_india_care_plan_evidence()

    gaps = detect_coverage_gaps(question, evidence, [])

    assert gaps["geography_supported"] is True
    assert gaps["provider_coverage"] == "available"
    assert gaps["cost_coverage"] == "available"
    assert gaps["recovery_coverage"] == "available"
    assert gaps["risk_coverage"] == "available"
    assert gaps["coverage"] == "Strong"


def test_india_full_corpus_evidence_returns_strong_coverage():
    question = "Create a care travel plan for cataract surgery in India including providers, cost, recovery, and risks."

    gaps = detect_coverage_gaps(question, _full_india_care_plan_evidence(), [])

    assert gaps["geography_supported"] is True
    assert gaps["provider_coverage"] == "available"
    assert gaps["cost_coverage"] == "available"
    assert gaps["recovery_coverage"] == "available"
    assert gaps["risk_coverage"] == "available"
    assert gaps["coverage"] == "Strong"


def test_thailand_uk_and_germany_with_india_evidence_return_partial():
    for country in ["Thailand", "UK", "Germany"]:
        question = f"Create a care travel plan for cataract surgery in {country} including providers, cost, recovery, and risks."

        gaps = detect_coverage_gaps(question, _full_india_care_plan_evidence(), [])

        assert gaps["coverage"] == "Partial"
        assert gaps["geography_supported"] is False
        assert gaps["provider_coverage"] == "missing"
        assert gaps["cost_coverage"] == "missing"
        assert gaps["recovery_coverage"] == "available"
        assert gaps["risk_coverage"] == "available"


def test_bangalore_successful_react_tool_calls_return_strong_coverage():
    question = "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks."
    result = {
        "tool_calls": [
            {"tool_name": "provider_search_tool", "status": "success"},
            {"tool_name": "cost_estimate_tool", "status": "success"},
            {"tool_name": "recovery_guidance_tool", "status": "success"},
            {"tool_name": "risk_checklist_tool", "status": "success"},
        ]
    }

    gaps = detect_coverage_gaps(question, [], [], result=result)

    assert gaps["geography_supported"] is True
    assert gaps["provider_coverage"] == "available"
    assert gaps["cost_coverage"] == "available"
    assert gaps["recovery_coverage"] == "available"
    assert gaps["risk_coverage"] == "available"
    assert gaps["coverage"] == "Strong"


def test_usa_with_bangalore_evidence_and_successful_tools_stays_partial():
    question = "Create a care travel plan for cataract surgery in USA including providers, cost, recovery, and risks."
    evidence = _full_india_care_plan_evidence()
    result = {
        "tool_calls": [
            {"tool_name": "provider_search_tool", "status": "success"},
            {"tool_name": "cost_estimate_tool", "status": "success"},
            {"tool_name": "recovery_guidance_tool", "status": "success"},
            {"tool_name": "risk_checklist_tool", "status": "success"},
        ],
        "final_answer": "For a USA cataract surgery plan, providers and costs vary by location.",
    }

    gaps = detect_coverage_gaps(question, evidence, [], result=result)

    assert gaps["geography_supported"] is False
    assert gaps["provider_coverage"] == "missing"
    assert gaps["cost_coverage"] == "missing"
    assert gaps["recovery_coverage"] == "available"
    assert gaps["risk_coverage"] == "available"
    assert gaps["coverage"] == "Partial"


def test_unsupported_country_does_not_become_strong_from_provider_or_cost_tool_success():
    question = "Create a care travel plan for cataract surgery in Singapore including providers, cost, recovery, and risks."
    result = {
        "tool_calls": [
            {"tool_name": "provider_search_tool", "status": "success"},
            {"tool_name": "cost_estimate_tool", "status": "success"},
            {"tool_name": "recovery_guidance_tool", "status": "success"},
            {"tool_name": "risk_checklist_tool", "status": "success"},
        ],
        "observations": [
            {
                "tool_name": "provider_search_tool",
                "status": "success",
                "evidence": [
                    {"source": "bangalore_eye_hospitals.csv", "category": "hospitals", "snippet": "Bangalore Eye Centre city: Bangalore"}
                ],
            },
            {
                "tool_name": "cost_estimate_tool",
                "status": "success",
                "evidence": [
                    {"source": "india_procedure_costs.csv", "category": "costs", "snippet": "Cataract surgery INR estimate"}
                ],
            },
        ],
    }

    gaps = detect_coverage_gaps(question, _full_india_care_plan_evidence(), [], result=result)

    assert gaps["coverage"] == "Partial"
    assert gaps["geography_supported"] is False
    assert gaps["provider_coverage"] == "missing"
    assert gaps["cost_coverage"] == "missing"
    assert gaps["recovery_coverage"] == "available"
    assert gaps["risk_coverage"] == "available"


def test_usa_does_not_show_strong_without_usa_provider_and_cost_sources():
    question = "Create a care travel plan for cataract surgery in the United States including providers, cost, recovery, and risks."
    result = {
        "tool_calls": [
            {"tool_name": "provider_search_tool", "status": "success"},
            {"tool_name": "cost_estimate_tool", "status": "success"},
        ],
        "final_answer": "United States provider options and USA costs are requested.",
    }

    gaps = detect_coverage_gaps(question, [], [], result=result)

    assert gaps["provider_coverage"] == "missing"
    assert gaps["cost_coverage"] == "missing"
    assert gaps["coverage"] != "Strong"


def test_no_geography_with_provider_cost_request_returns_pending():
    gaps = detect_coverage_gaps(
        "Create a care travel plan for cataract surgery including providers, cost, recovery, and risks.",
        _full_india_care_plan_evidence(),
        [],
    )

    assert gaps["coverage"] == "Pending"
    assert gaps["reason"] == "Destination geography is needed before checking provider and cost coverage."


def test_recovery_only_without_geography_can_be_strong():
    gaps = detect_coverage_gaps(
        "What recovery guidance is available after cataract surgery?",
        [{"source": "cataract_surgery_guide.md", "category": "procedures", "snippet": "follow-up visits and recovery planning"}],
        [],
    )

    assert gaps["coverage"] == "Strong"
    assert gaps["recovery_coverage"] == "available"


def test_needs_human_status_returns_pending_coverage():
    gaps = detect_coverage_gaps(
        "Plan my travel for surgery in Bangalore",
        [],
        [],
        result={"status": "needs_human", "requires_human": True},
    )

    assert gaps["coverage"] == "Pending"
    assert gaps["reason"] == "Additional information is needed before checking corpus coverage."


def test_unsafe_status_returns_not_applicable_coverage():
    gaps = detect_coverage_gaps(
        "Should I take antibiotics after surgery?",
        [],
        [],
        result={"status": "unsafe"},
    )

    assert gaps["coverage"] == "Not applicable"
    assert gaps["reason"] == "Safety boundary triggered."


def test_out_of_scope_status_returns_not_applicable_coverage():
    gaps = detect_coverage_gaps(
        "Who won the Super Bowl in 2024?",
        [],
        [],
        result={"status": "out_of_scope"},
    )

    assert gaps["coverage"] == "Not applicable"
    assert gaps["reason"] == "Request is outside Synataric healthcare-navigation scope."


def test_coverage_gap_status_returns_not_available_coverage():
    gaps = detect_coverage_gaps(
        "Create a care travel plan for cataract surgery in Norway including providers, cost, recovery, and risks.",
        [],
        [],
        result={"status": "coverage_gap", "selected_tool": "coverage_gap_response_tool"},
    )

    assert gaps["coverage"] == "Not available"
    assert gaps["provider_coverage"] == "missing"
    assert gaps["cost_coverage"] == "missing"
    assert gaps["reason"] == "Corpus coverage gap."


def test_hospitals_category_counts_as_provider_coverage():
    gaps = detect_coverage_gaps(
        "Find cataract providers in Bangalore",
        [{"source": "sample.csv", "category": "hospitals", "snippet": "hospital_name: Bangalore Eye Centre"}],
        [],
    )

    assert gaps["provider_coverage"] == "available"


def test_india_procedure_costs_counts_as_cost_coverage_for_bangalore():
    gaps = detect_coverage_gaps(
        "What is the cost of cataract surgery in Bangalore?",
        [{"source": "india_procedure_costs.csv", "category": "procedures", "snippet": "low_estimate_inr: 45000"}],
        [],
    )

    assert gaps["cost_coverage"] == "available"


def test_post_op_recovery_guidelines_count_as_recovery_and_risk_coverage():
    gaps = detect_coverage_gaps(
        "Plan cataract surgery recovery and risks in Bangalore",
        [
            {
                "source": "post_op_recovery_guidelines.md",
                "category": "recovery",
                "snippet": "follow-up visits, recovery planning, red-flag symptoms, immediate medical care",
            }
        ],
        [],
    )

    assert gaps["recovery_coverage"] == "available"
    assert gaps["risk_coverage"] == "available"


def test_rewrite_answer_for_coverage_gaps_replaces_unsupported_usa_provider_cost_filler():
    answer = """
#### Provider Options
Recommended providers can vary by location, so check local clinics.

#### Estimated Cost
Costs vary widely. Contact hospitals for pricing.

#### Recovery Guidance
- Avoid rubbing the eye.
"""
    gaps = {
        "coverage": "Partial",
        "provider_coverage": "missing",
        "cost_coverage": "missing",
        "requested_geography": "USA",
        "requested_geography_key": "usa",
        "geography_supported": False,
    }

    rewritten = rewrite_answer_for_coverage_gaps(answer, gaps)

    assert "I don't have USA-specific provider or cost records" in rewritten
    assert "Not available in the current Synataric corpus" in rewritten
    assert "check local clinics" not in rewritten
    assert "Costs vary widely" not in rewritten


def test_rewrite_answer_for_coverage_gaps_uses_requested_unsupported_country():
    answer = """
#### Provider Options
Recommended providers can vary by location, so check local clinics.

#### Estimated Cost
Costs vary widely. Contact hospitals for pricing.

#### Recovery Guidance
- Arrange follow-up care.
"""
    gaps = {
        "coverage": "Partial",
        "provider_coverage": "missing",
        "cost_coverage": "missing",
        "requested_geography": "Thailand",
        "requested_geography_key": "thailand",
        "geography_supported": False,
    }

    rewritten = rewrite_answer_for_coverage_gaps(answer, gaps)

    assert "Thailand-specific provider or cost records" in rewritten
    assert "cannot recommend Thailand providers or quote Thailand costs" in rewritten
    assert "USA-specific" not in rewritten
    assert "check local clinics" not in rewritten
    assert "Costs vary widely" not in rewritten


def test_coverage_safe_cards_suppress_unsupported_usa_placeholder_answer():
    fields = {
        "user_question": "Create a cataract surgery care travel plan in the USA with providers and costs.",
        "expected_route": "care_plan_multistep",
        "actual_route": "care_plan_multistep",
        "status": "success",
        "final_answer": (
            "#### Providers: - Provider Name: [Insert Provider Name] - Location: [Insert City, State] "
            "#### Cost Estimates: - Estimated Cost of Cataract Surgery: [InsertCostRange] "
            "#### Recovery Guidance: - Arrange follow-up care. "
            "#### Risks: - Seek urgent care for severe symptoms."
        ),
    }
    gaps = {
        "coverage": "Partial",
        "provider_coverage": "missing",
        "cost_coverage": "missing",
        "recovery_coverage": "available",
        "risk_coverage": "available",
        "requested_geography": "USA",
        "requested_geography_key": "usa",
        "geography_supported": False,
    }

    cards = build_coverage_safe_care_plan_cards(fields, [], [], gaps)
    rendered_text = " ".join(item for card in cards for item in card["items"])

    assert "Not available in the current Synataric corpus" in rendered_text
    assert "[Insert" not in rendered_text
    assert "InsertCostRange" not in rendered_text
    assert "Care Navigation Answer" not in [card["title"] for card in cards]

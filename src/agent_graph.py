"""Week 3 agentic LangGraph workflow for Synataric Navigator.

This router-pattern graph connects intent classification with the agent tool
layer while leaving the existing RAG backend and Streamlit app untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent_intents import IntentClassification, classify_intent, detect_out_of_scope
from src.agent_recovery import (
    build_low_confidence_clarification,
    build_safe_fallback_answer,
    classify_failure_type,
    normalize_agent_status,
    should_retry_tool,
)
from src.agent_tools import (
    AgentToolResult,
    run_ask_human,
    run_cost_estimate,
    run_find_evidence,
    run_general_rag,
    run_out_of_scope_response,
    run_provider_search,
    run_recovery_guidance,
    run_risk_checklist,
    run_safety_response,
    run_travel_planning,
)
from src.config import configure_langsmith, load_settings, traceable
from src.output_sanitizer import sanitize_text


try:
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:
    InMemorySaver = None


class SynataricAgentState(TypedDict, total=False):
    question: str
    patient_context: Dict[str, Any]
    namespace: Optional[str]
    top_k: int
    thread_id: str

    intent: Optional[str]
    intent_confidence: float
    intent_reasoning: str
    missing_fields: List[str]
    suggested_tools: List[str]
    safety_flags: List[str]

    selected_tool: Optional[str]
    tool_calls: List[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    requires_human: bool
    human_question: Optional[str]

    answer: Optional[str]
    sources: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    retrieved_count: int
    reranked_count: int

    warnings: List[str]
    errors: List[str]
    execution_log: List[str]
    status: str
    retry_count: int
    max_retries: int
    fallback_used: bool
    recovery_action: Optional[str]
    confidence_threshold: float
    out_of_scope_reason: Optional[str]


INTENT_TO_TOOL: dict[str, str] = {
    "provider_search": "provider_search_tool",
    "cost_estimate": "cost_estimate_tool",
    "recovery_guidance": "recovery_guidance_tool",
    "risk_checklist": "risk_checklist_tool",
    "travel_planning": "travel_planning_tool",
    "find_evidence": "find_evidence_tool",
    "general_navigation": "general_rag_tool",
}


TOOL_RUNNERS = {
    "provider_search_tool": run_provider_search,
    "cost_estimate_tool": run_cost_estimate,
    "recovery_guidance_tool": run_recovery_guidance,
    "risk_checklist_tool": run_risk_checklist,
    "travel_planning_tool": run_travel_planning,
    "find_evidence_tool": run_find_evidence,
    "general_rag_tool": run_general_rag,
}


INSUFFICIENT_CONTEXT_ANSWER = build_safe_fallback_answer("", "no_evidence")


def _append_log(state: SynataricAgentState, message: str) -> SynataricAgentState:
    execution_log = list(state.get("execution_log", []))
    execution_log.append(message)
    return {**state, "execution_log": execution_log}


def _model_to_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def _tool_result_to_state(state: SynataricAgentState, result: AgentToolResult) -> SynataricAgentState:
    result_dict = _model_to_dict(result)
    warnings = list(state.get("warnings", [])) + list(result_dict.get("warnings") or [])
    errors = list(state.get("errors", []))
    if result_dict.get("error"):
        errors.append(str(result_dict["error"]))

    return {
        **state,
        "tool_result": result_dict,
        "tool_results": list(state.get("tool_results", [])) + [result_dict],
        "answer": result_dict.get("answer"),
        "sources": result_dict.get("sources") or [],
        "evidence": result_dict.get("evidence") or [],
        "retrieved_count": result_dict.get("retrieved_count") or 0,
        "reranked_count": result_dict.get("reranked_count") or 0,
        "warnings": warnings,
        "errors": errors,
        "requires_human": bool(result_dict.get("requires_human")),
        "human_question": result_dict.get("human_question"),
        "status": _status_from_tool_result(str(result_dict.get("status") or "")),
    }


def _status_from_tool_result(tool_status: str) -> str:
    if tool_status == "success":
        return "tool_success"
    if tool_status in {"needs_human", "unsafe", "out_of_scope", "fallback", "no_evidence", "error"}:
        return tool_status
    return "tool_success" if tool_status else "error"


@traceable(name="Synataric Agent - Classify Intent")
def classify_intent_node(state: SynataricAgentState) -> SynataricAgentState:
    question = state["question"]
    patient_context = state.get("patient_context", {})
    try:
        classification: IntentClassification = classify_intent(question, patient_context)
        classification_dict = _model_to_dict(classification)
        confidence = float(classification_dict.get("confidence") or 0.0)
        missing_fields = list(classification_dict.get("missing_fields") or [])
        warnings = list(state.get("warnings", []))
        human_question = state.get("human_question")
        out_of_scope, out_of_scope_reason = detect_out_of_scope(question)
        if out_of_scope:
            warnings.append("out_of_scope_request")
        elif confidence < state.get("confidence_threshold", 0.55):
            if "intent" not in missing_fields:
                missing_fields.append("intent")
            warnings.append("low_confidence_intent")
            human_question = build_low_confidence_clarification(question)

        updated: SynataricAgentState = {
            **state,
            "intent": classification_dict.get("intent"),
            "intent_confidence": confidence,
            "intent_reasoning": classification_dict.get("reasoning") or "",
            "missing_fields": missing_fields,
            "suggested_tools": classification_dict.get("suggested_tools") or [],
            "safety_flags": classification_dict.get("safety_flags") or [],
            "warnings": warnings,
            "human_question": human_question,
            "out_of_scope_reason": out_of_scope_reason if out_of_scope else None,
            "status": "classified",
        }
        return _append_log(
            updated,
            f"Intent classified as {updated.get('intent')} with confidence {updated.get('intent_confidence')}.",
        )
    except Exception as exc:
        errors = list(state.get("errors", [])) + [str(exc)]
        updated = {
            **state,
            "intent": "general_navigation",
            "intent_confidence": 0.0,
            "intent_reasoning": "Intent classification failed; falling back to general navigation.",
            "missing_fields": ["intent"],
            "suggested_tools": ["general_rag_tool"],
            "safety_flags": [],
            "errors": errors,
            "warnings": list(state.get("warnings", [])) + ["low_confidence_intent"],
            "human_question": build_low_confidence_clarification(question),
            "status": "classified",
        }
        return _append_log(updated, "Intent classification failed; falling back to general navigation.")


def route_after_intent(state: SynataricAgentState) -> str:
    if state.get("safety_flags") or state.get("intent") == "unsafe_medical":
        return "safety_node"
    if state.get("intent") == "out_of_scope":
        return "out_of_scope_node"
    if state.get("intent_confidence", 1.0) < state.get("confidence_threshold", 0.55):
        return "ask_human_node"
    if state.get("intent") == "needs_clarification":
        return "ask_human_node"
    if state.get("missing_fields"):
        return "ask_human_node"
    return "tool_router_node"


@traceable(name="Synataric Agent - Safety Response")
def safety_node(state: SynataricAgentState) -> SynataricAgentState:
    result = run_safety_response(state["question"], state.get("safety_flags", []))
    updated = _tool_result_to_state({**state, "selected_tool": "safety_response_tool"}, result)
    updated = {**updated, "status": "unsafe"}
    return _append_log(updated, "Safety boundary triggered. Returned safe response.")


@traceable(name="Synataric Agent - Out Of Scope Response")
def out_of_scope_node(state: SynataricAgentState) -> SynataricAgentState:
    reason = state.get("out_of_scope_reason")
    if not reason:
        _is_out_of_scope, reason = detect_out_of_scope(state["question"])
    result = run_out_of_scope_response(state["question"], reason=reason)
    updated = _tool_result_to_state({**state, "selected_tool": "out_of_scope_response_tool"}, result)
    updated = {
        **updated,
        "status": "out_of_scope",
        "requires_human": False,
        "out_of_scope_reason": reason,
    }
    return _append_log(updated, "Out-of-scope boundary triggered.")


@traceable(name="Synataric Agent - Ask Human")
def ask_human_node(state: SynataricAgentState) -> SynataricAgentState:
    result = run_ask_human(state["question"], state.get("missing_fields", []))
    if "intent" in state.get("missing_fields", []):
        result.human_question = build_low_confidence_clarification(state["question"])
    updated = _tool_result_to_state({**state, "selected_tool": "ask_human_tool"}, result)
    updated = {
        **updated,
        "requires_human": True,
        "human_question": result.human_question,
        "status": "needs_human",
    }
    return _append_log(updated, "Missing required information. Asking human for clarification.")


@traceable(name="Synataric Agent - Tool Router")
def tool_router_node(state: SynataricAgentState) -> SynataricAgentState:
    selected_tool = _select_tool(state)
    updated = {**state, "selected_tool": selected_tool, "status": "routed"}
    return _append_log(updated, f"Selected tool: {selected_tool}.")


@traceable(name="Synataric Agent - Execute Tool")
def tool_execution_node(state: SynataricAgentState) -> SynataricAgentState:
    selected_tool = state.get("selected_tool") or "general_rag_tool"
    runner = TOOL_RUNNERS.get(selected_tool, run_general_rag)
    question = state["question"]
    namespace = state.get("namespace")
    top_k = state.get("top_k", 12)

    try:
        attempt = state.get("retry_count", 0)
        result = runner(question, namespace=namespace, top_k=top_k)
        while should_retry_tool(result.status, attempt, state.get("max_retries", 1)):
            attempt += 1
            result = runner(question, namespace=namespace, top_k=top_k)
        tool_call = {"tool_name": selected_tool, "input": question, "status": result.status}
        updated = _tool_result_to_state(state, result)
        updated = {**updated, "tool_calls": list(state.get("tool_calls", [])) + [tool_call], "retry_count": attempt}
        if attempt >= state.get("max_retries", 1) and result.status in {"error", "no_evidence", "fallback"}:
            updated = _append_warning(updated, "retry_limit_reached")
            updated = _append_log(updated, "Retry limit reached.")
        return _append_log(updated, f"Executed {selected_tool} with status {result.status}.")
    except Exception as exc:
        failure_type = classify_failure_type(exc)
        errors = list(state.get("errors", [])) + [str(exc)]
        warnings = list(state.get("warnings", [])) + [failure_type]
        tool_call = {"tool_name": selected_tool, "input": question, "status": "error"}
        updated = {
            **state,
            "status": "error",
            "errors": errors,
            "warnings": warnings,
            "recovery_action": failure_type,
            "tool_calls": list(state.get("tool_calls", [])) + [tool_call],
        }
        return _append_log(updated, f"Tool execution failed: {exc}.")


def route_after_tool(state: SynataricAgentState) -> str:
    status = normalize_agent_status(state.get("status", ""))
    if status in {"unsafe", "out_of_scope", "needs_human", "complete"}:
        return "final_response_node"
    if status == "fallback" and state.get("answer"):
        return "final_response_node"
    if status in {"fallback", "error", "no_evidence"}:
        return "fallback_node"
    return "final_response_node"


@traceable(name="Synataric Agent - Fallback")
def fallback_node(state: SynataricAgentState) -> SynataricAgentState:
    question = state["question"]
    namespace = state.get("namespace")
    top_k = state.get("top_k", 12)

    reason = state.get("recovery_action") or state.get("status", "no_evidence")
    if state.get("selected_tool") == "general_rag_tool" or state.get("fallback_used"):
        updated = {
            **state,
            "answer": build_safe_fallback_answer(question, reason),
            "status": "no_evidence",
            "fallback_used": True,
            "recovery_action": "safe_insufficient_context_response",
        }
        updated = _append_warning(updated, "safe_insufficient_context_response")
        return _append_log(updated, "Fallback failed. Returning insufficient-context response.")

    try:
        result = run_general_rag(question, namespace=namespace, top_k=top_k)
        updated = _tool_result_to_state({**state, "selected_tool": "general_rag_tool"}, result)
        warnings = list(updated.get("warnings", [])) + ["fallback_to_general_rag"]
        updated = {
            **updated,
            "warnings": warnings,
            "status": "fallback" if result.status == "success" else updated.get("status", "fallback"),
            "fallback_used": True,
            "recovery_action": "fallback_to_general_rag",
        }
        return _append_log(updated, "Fallback executed using general_rag_tool.")
    except Exception as exc:
        failure_type = classify_failure_type(exc)
        errors = list(state.get("errors", [])) + [str(exc)]
        warnings = list(state.get("warnings", [])) + ["fallback_to_general_rag_failed", failure_type, "safe_insufficient_context_response"]
        updated = {
            **state,
            "answer": build_safe_fallback_answer(question, failure_type),
            "status": "no_evidence",
            "errors": errors,
            "warnings": warnings,
            "fallback_used": True,
            "recovery_action": "safe_insufficient_context_response",
        }
        return _append_log(updated, "Fallback failed. Returning insufficient-context response.")


@traceable(name="Synataric Agent - Final Response")
def final_response_node(state: SynataricAgentState) -> SynataricAgentState:
    answer = state.get("answer")
    status = normalize_agent_status(state.get("status", "complete"))

    if status == "unsafe" or state.get("selected_tool") == "safety_response_tool":
        answer = answer or run_safety_response(state["question"], state.get("safety_flags", [])).answer
        status = "unsafe"
    elif status == "out_of_scope" or state.get("selected_tool") == "out_of_scope_response_tool":
        answer = answer or run_out_of_scope_response(state["question"], state.get("out_of_scope_reason")).answer
        status = "out_of_scope"
    elif state.get("requires_human"):
        answer = state.get("human_question") or answer or "Please provide more information so I can continue."
        status = "needs_human"
    elif status in {"error", "no_evidence"} and not answer:
        answer = build_safe_fallback_answer(state["question"], state.get("recovery_action") or status)
        status = "no_evidence"
    elif not answer:
        answer = build_safe_fallback_answer(state["question"], "empty_answer")
        status = "no_evidence"

    updated = {**state, "answer": sanitize_text(answer), "status": status}
    return _append_log(updated, "Final response prepared.")


def build_synataric_agent_graph():
    workflow = StateGraph(SynataricAgentState)
    workflow.add_node("classify_intent_node", classify_intent_node)
    workflow.add_node("safety_node", safety_node)
    workflow.add_node("out_of_scope_node", out_of_scope_node)
    workflow.add_node("ask_human_node", ask_human_node)
    workflow.add_node("tool_router_node", tool_router_node)
    workflow.add_node("tool_execution_node", tool_execution_node)
    workflow.add_node("fallback_node", fallback_node)
    workflow.add_node("final_response_node", final_response_node)

    workflow.add_edge(START, "classify_intent_node")
    workflow.add_conditional_edges(
        "classify_intent_node",
        route_after_intent,
        {
            "safety_node": "safety_node",
            "out_of_scope_node": "out_of_scope_node",
            "ask_human_node": "ask_human_node",
            "tool_router_node": "tool_router_node",
        },
    )
    workflow.add_edge("safety_node", "final_response_node")
    workflow.add_edge("out_of_scope_node", "final_response_node")
    workflow.add_edge("ask_human_node", "final_response_node")
    workflow.add_edge("tool_router_node", "tool_execution_node")
    workflow.add_conditional_edges(
        "tool_execution_node",
        route_after_tool,
        {
            "fallback_node": "fallback_node",
            "final_response_node": "final_response_node",
        },
    )
    workflow.add_edge("fallback_node", "final_response_node")
    workflow.add_edge("final_response_node", END)

    if InMemorySaver is not None:
        return workflow.compile(checkpointer=InMemorySaver())
    return workflow.compile()


def run_synataric_agent(
    question: str,
    patient_context: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None,
    top_k: int = 12,
    thread_id: str = "synataric-agent-demo",
) -> Dict[str, Any]:
    configure_langsmith()
    settings = load_settings()
    graph = build_synataric_agent_graph()
    initial: SynataricAgentState = {
        "question": question,
        "patient_context": patient_context or {},
        "namespace": namespace or settings.semantic_namespace,
        "top_k": top_k,
        "thread_id": thread_id,
        "tool_calls": [],
        "tool_results": [],
        "sources": [],
        "evidence": [],
        "warnings": [],
        "errors": [],
        "execution_log": ["Agent run initialized."],
        "requires_human": False,
        "missing_fields": [],
        "safety_flags": [],
        "suggested_tools": [],
        "retrieved_count": 0,
        "reranked_count": 0,
        "status": "initialized",
        "retry_count": 0,
        "max_retries": 1,
        "fallback_used": False,
        "recovery_action": None,
        "confidence_threshold": 0.55,
        "out_of_scope_reason": None,
    }
    config = {"configurable": {"thread_id": thread_id}}
    return dict(graph.invoke(initial, config=config))


def _select_tool(state: SynataricAgentState) -> str:
    suggested_tools = state.get("suggested_tools", [])
    for tool_name in suggested_tools:
        if tool_name != "ask_human_tool":
            return tool_name
    intent = state.get("intent") or "general_navigation"
    return INTENT_TO_TOOL.get(intent, "general_rag_tool")


def _append_warning(state: SynataricAgentState, warning: str) -> SynataricAgentState:
    warnings = list(state.get("warnings", []))
    warnings.append(warning)
    return {**state, "warnings": warnings}


if __name__ == "__main__":
    debug_questions = [
        "Where can I find good cataract surgery in India?",
        "What is the cost of cataract surgery in Bangalore?",
        "Should I take antibiotics after surgery?",
        "Plan my travel for surgery in Bangalore",
        "Help me with this",
        "Who won the Super Bowl in 2024?",
        "What is the best robotic neurosurgery hospital on Mars?",
    ]

    for debug_question in debug_questions:
        result = run_synataric_agent(debug_question)
        print("QUESTION:", debug_question)
        print("INTENT:", result.get("intent"))
        print("CONFIDENCE:", result.get("intent_confidence"))
        print("SELECTED TOOL:", result.get("selected_tool"))
        print("STATUS:", result.get("status"))
        print("RECOVERY ACTION:", result.get("recovery_action"))
        print("FALLBACK USED:", result.get("fallback_used"))
        print("ANSWER:", result.get("answer"))
        print("WARNINGS:", result.get("warnings"))
        print("ERRORS:", result.get("errors"))
        print("EXECUTION LOG:", result.get("execution_log"))
        print()

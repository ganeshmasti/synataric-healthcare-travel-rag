"""Bounded ReAct-style care planning agent for Synataric Navigator.

This module is additive. It does not replace the existing RAG app, the
router-pattern Agent Navigator, or the retrieval/reranking/generation pipeline.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.agent_tools import (
    AgentToolResult,
    run_ask_human,
    run_cost_estimate,
    run_find_evidence,
    run_out_of_scope_response,
    run_provider_search,
    run_recovery_guidance,
    run_risk_checklist,
    run_safety_response,
    run_travel_planning,
)
from src.config import configure_langsmith, load_settings, traceable
from src.agent_intents import detect_procedure, detect_unsafe_medical
from src.output_sanitizer import sanitize_result_dict, sanitize_sources, sanitize_text


try:
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:
    InMemorySaver = None


DISCLAIMER = "Educational healthcare navigation only. Not medical advice."


class ReactCareState(TypedDict, total=False):
    question: str
    messages: list
    step_count: int
    max_steps: int
    selected_tool: str | None
    tool_input: str | None
    tool_calls: list[dict]
    observations: list[dict]
    final_answer: str | None
    status: str
    requires_human: bool
    human_question: str | None
    warnings: list[str]
    errors: list[str]
    execution_log: list[str]
    namespace: str | None
    top_k: int


class ReactDecision(BaseModel):
    action: Literal[
        "call_tool",
        "final_answer",
        "ask_human",
        "safety_response",
        "out_of_scope",
    ]
    tool_name: str | None = None
    tool_input: str | None = None
    rationale: str
    final_answer: str | None = None
    human_question: str | None = None
    safety_flags: list[str] = Field(default_factory=list)
    confidence: float = 0.0


TOOL_REGISTRY = {
    "provider_search_tool": run_provider_search,
    "cost_estimate_tool": run_cost_estimate,
    "recovery_guidance_tool": run_recovery_guidance,
    "risk_checklist_tool": run_risk_checklist,
    "travel_planning_tool": run_travel_planning,
    "find_evidence_tool": run_find_evidence,
}


TERMINAL_STATUSES = {
    "complete",
    "needs_human",
    "unsafe",
    "out_of_scope",
    "max_steps_reached",
    "error",
}


SYSTEM_INSTRUCTION = """You are Synataric ReAct Care Planner.
You help users build healthcare travel and care-navigation plans using tools.
You must not diagnose, prescribe, or make urgent medical decisions.
You must use tools for factual provider, cost, recovery, risk, or travel details.
If the request is unsafe, choose safety_response.
If the request is outside healthcare travel/care navigation, choose out_of_scope.
If required information is missing, choose ask_human.
If enough evidence has been gathered, choose final_answer.
Never call more than one tool per step.
Prefer the smallest number of tool calls needed.
Stop when the care plan is sufficient.

Tool guidance:
provider_search_tool: hospitals/providers/clinics/specialists
cost_estimate_tool: procedure or travel/stay costs
recovery_guidance_tool: recovery, follow-up, post-op care
risk_checklist_tool: risks, urgent symptoms, red flags
travel_planning_tool: travel logistics, stay duration, caregiver planning
find_evidence_tool: where a fact is explained"""


def _append_log(state: ReactCareState, message: str) -> ReactCareState:
    execution_log = list(state.get("execution_log", []))
    execution_log.append(message)
    return {**state, "execution_log": execution_log}


def _append_warning(state: ReactCareState, warning: str) -> ReactCareState:
    warnings = list(state.get("warnings", []))
    warnings.append(warning)
    return {**state, "warnings": warnings}


def _append_error(state: ReactCareState, error: str) -> ReactCareState:
    errors = list(state.get("errors", []))
    errors.append(error)
    return {**state, "errors": errors}


def _model_to_dict(obj: Any) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def _build_reason_prompt(state: ReactCareState) -> str:
    observations = _summarize_observations(state.get("observations", []), include_sources=True)
    previous_tools = [call.get("tool_name") for call in state.get("tool_calls", [])]
    return (
        f"Question: {state.get('question', '')}\n"
        f"Step count: {state.get('step_count', 0)} of {state.get('max_steps', 5)}\n"
        f"Previous tools: {previous_tools}\n"
        f"Observations:\n{observations or 'No observations yet.'}\n\n"
        "Choose the next ReAct action. If one targeted tool already answered a simple question, "
        "prefer final_answer. For multi-step plans, call the next missing factual tool."
    )


def _build_max_steps_answer(state: ReactCareState) -> str:
    summary = _summarize_observations(state.get("observations", []), include_sources=False)
    if summary:
        return f"I reached the step limit, so here is the care-navigation summary gathered so far:\n\n{summary}"
    return "I reached the step limit before gathering enough context to answer safely."


def _summarize_observations(observations: list[dict], include_sources: bool = False) -> str:
    if not observations:
        return ""
    lines: list[str] = []
    for index, observation in enumerate(observations, start=1):
        tool_name = observation.get("tool_name", "tool")
        status = observation.get("status", "unknown")
        answer = _compact_text(observation.get("answer") or "No answer returned.", 900)
        lines.append(f"{index}. {tool_name} ({status}): {answer}")
        if include_sources:
            source_names = _source_names(observation.get("sources", []), observation.get("evidence", []))
            if source_names:
                lines.append(f"   Sources: {', '.join(source_names[:5])}")
    return "\n".join(lines)


def _source_names(sources: list[dict], evidence: list[dict]) -> list[str]:
    names: list[str] = []
    for source in sources or []:
        name = source.get("file_name") or source.get("source") or source.get("title")
        if name:
            names.append(str(name).split("/")[-1].split("\\")[-1])
    for item in evidence or []:
        name = item.get("source")
        if name:
            names.append(str(name).split("/")[-1].split("\\")[-1])
    return list(dict.fromkeys(names))


def _compact_text(text: Any, limit: int = 1200) -> str:
    compact = " ".join(str(text or "").split())
    return compact[: limit - 3] + "..." if len(compact) > limit else compact


def _with_disclaimer(answer: str | None) -> str:
    text = sanitize_text(str(answer or "").strip())
    if not text:
        text = "I do not have enough grounded context to answer safely."
    if DISCLAIMER.lower() not in text.lower():
        text = f"{text}\n\n{DISCLAIMER}"
    return text


def _infer_missing_fields(question: str) -> list[str]:
    text = question.lower()
    missing: list[str] = []
    if "surgery" in text and not any(
        keyword in text
        for keyword in [
            "cataract",
            "knee replacement",
            "cardiac bypass",
            "heart bypass",
            "cabg",
            "retina",
            "eye surgery",
        ]
    ):
        missing.append("procedure")
    if any(keyword in text for keyword in ["travel", "stay", "hotel", "airport"]) and not any(
        city in text for city in ["bangalore", "india", "chennai", "mumbai", "delhi", "hyderabad"]
    ):
        missing.append("destination")
    return missing


def detect_missing_procedure_for_react(question: str) -> bool:
    text = question.lower()
    if detect_procedure(question):
        return False
    if "surgery" not in text and "procedure" not in text:
        return False
    planning_terms = [
        "travel",
        "planning",
        "plan",
        "cost",
        "recovery",
        "provider",
        "providers",
        "hospital",
        "care plan",
        "care travel",
    ]
    return any(term in text for term in planning_terms)


def detect_react_safety_flags(question: str) -> list[str]:
    text = " ".join(str(question or "").lower().split())
    flags = list(detect_unsafe_medical(question))
    medication_patterns = [
        ("prescription_or_treatment_advice", "should i take antibiotics"),
        ("prescription_or_treatment_advice", "can i take antibiotics"),
        ("prescription_or_treatment_advice", "which antibiotics"),
        ("prescription_or_treatment_advice", "what medication should i take"),
        ("prescription_or_treatment_advice", "should i take medicine"),
        ("prescription_or_treatment_advice", "painkiller dosage"),
        ("prescription_or_treatment_advice", "painkillers should i take"),
        ("prescription_or_treatment_advice", "blood thinner instructions"),
        ("prescription_or_treatment_advice", "stop medication"),
        ("prescription_or_treatment_advice", "start medication"),
    ]
    for flag, phrase in medication_patterns:
        if phrase in text:
            flags.append(flag)
    if any(term in text for term in ["prescribe", "prescription", "dosage", " dose ", " dose of "]):
        flags.append("prescription_or_treatment_advice")
    if any(term in text for term in ["diagnosis", "diagnose"]):
        flags.append("diagnosis")
    return list(dict.fromkeys(flags))


def _status_from_result(result_status: str) -> str:
    if result_status == "success":
        return "observing"
    if result_status in {"needs_human", "unsafe", "out_of_scope", "error"}:
        return result_status
    if result_status in {"no_evidence", "fallback"}:
        return "observing"
    return "observing"


def _decision_to_state(state: ReactCareState, decision: ReactDecision) -> ReactCareState:
    warnings = list(state.get("warnings", []))
    if decision.confidence < 0.45:
        warnings.append("low_confidence_react_decision")

    updated: ReactCareState = {
        **state,
        "warnings": warnings,
        "selected_tool": decision.tool_name if decision.action == "call_tool" else None,
        "tool_input": decision.tool_input,
        "status": "reasoning",
    }

    if decision.action == "final_answer":
        updated = {**updated, "status": "complete", "final_answer": decision.final_answer}
    elif decision.action == "ask_human":
        human_question = decision.human_question or "What additional details should I use?"
        updated = {
            **updated,
            "status": "needs_human",
            "requires_human": True,
            "human_question": human_question,
            "final_answer": human_question,
        }
    elif decision.action == "safety_response":
        result = run_safety_response(state["question"], decision.safety_flags)
        updated = {
            **updated,
            "status": "unsafe",
            "requires_human": result.requires_human,
            "final_answer": result.answer,
        }
    elif decision.action == "out_of_scope":
        result = run_out_of_scope_response(state["question"], reason=decision.rationale)
        updated = {
            **updated,
            "status": "out_of_scope",
            "requires_human": False,
            "final_answer": result.answer,
        }

    action_label = decision.action
    tool_label = decision.tool_name or ""
    rationale = _compact_text(decision.rationale, 240)
    return _append_log(updated, f"Reasoned next action: {action_label} {tool_label}. Rationale: {rationale}")


@traceable(name="Synataric ReAct - Reason")
def reason_node(state: ReactCareState) -> ReactCareState:
    step_count = int(state.get("step_count", 0))
    max_steps = int(state.get("max_steps", 5))

    if step_count == 0 and not state.get("tool_calls"):
        safety_flags = detect_react_safety_flags(state.get("question", ""))
        if safety_flags:
            result = run_safety_response(state["question"], safety_flags)
            updated: ReactCareState = {
                **state,
                "status": "unsafe",
                "requires_human": False,
                "human_question": None,
                "final_answer": result.answer,
                "selected_tool": "safety_response_tool",
                "tool_input": None,
                "warnings": [],
                "errors": [],
            }
            return _append_log(updated, "Safety boundary triggered before ReAct planning.")

    if step_count == 0 and not state.get("tool_calls") and detect_missing_procedure_for_react(state.get("question", "")):
        human_question = "Which procedure are you considering?"
        updated: ReactCareState = {
            **state,
            "status": "needs_human",
            "requires_human": True,
            "human_question": human_question,
            "final_answer": human_question,
            "selected_tool": None,
            "tool_input": None,
        }
        return _append_log(updated, "Human clarification required before ReAct tool execution.")

    if step_count >= max_steps:
        updated: ReactCareState = {
            **state,
            "status": "max_steps_reached",
            "selected_tool": None,
            "final_answer": _build_max_steps_answer(state),
        }
        return _append_log(updated, "Maximum ReAct steps reached.")

    try:
        settings = load_settings()
        llm = ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)
        structured_llm = llm.with_structured_output(ReactDecision)
        decision = structured_llm.invoke(
            [
                ("system", SYSTEM_INSTRUCTION),
                ("human", _build_reason_prompt(state)),
            ]
        )
        decision = _coerce_decision(decision)
    except Exception as exc:
        updated = _append_error(state, str(exc))
        decision = _fallback_decision(updated)
        updated = _append_warning(updated, "react_reasoning_fallback_used")
        return _decision_to_state(updated, decision)

    if decision.action == "call_tool" and decision.tool_name not in TOOL_REGISTRY:
        updated = _append_warning(state, "invalid_react_tool_selected")
        decision = _fallback_decision(updated)
        return _decision_to_state(updated, decision)

    return _decision_to_state(state, decision)


@traceable(name="Synataric ReAct - Act")
def action_node(state: ReactCareState) -> ReactCareState:
    selected_tool = state.get("selected_tool")
    if not selected_tool:
        updated = _append_error(state, "No selected tool for ReAct action.")
        return _append_log({**updated, "status": "error"}, "ReAct action failed because no tool was selected.")

    runner = TOOL_REGISTRY.get(selected_tool)
    if runner is None:
        updated = _append_error(state, f"Unknown ReAct tool: {selected_tool}")
        return _append_log({**updated, "status": "error"}, f"Unknown ReAct tool selected: {selected_tool}.")

    tool_input = state.get("tool_input") or state["question"]
    namespace = state.get("namespace")
    top_k = int(state.get("top_k", 12))

    try:
        result = runner(tool_input, namespace=namespace, top_k=top_k)
    except Exception as exc:
        errors = list(state.get("errors", [])) + [str(exc)]
        tool_call = {"tool_name": selected_tool, "input": tool_input, "status": "error"}
        updated: ReactCareState = {
            **state,
            "status": "error",
            "errors": errors,
            "tool_calls": list(state.get("tool_calls", [])) + [tool_call],
        }
        return _append_log(updated, f"Executed {selected_tool}; observed status error.")

    result_dict = _result_to_observation(result)
    observation = {
        "tool_name": selected_tool,
        "status": result_dict.get("status"),
        "answer": result_dict.get("answer"),
        "sources": result_dict.get("sources") or [],
        "evidence": result_dict.get("evidence") or [],
        "warnings": result_dict.get("warnings") or [],
    }
    tool_call = {"tool_name": selected_tool, "input": tool_input, "status": result_dict.get("status")}

    warnings = list(state.get("warnings", [])) + list(result_dict.get("warnings") or [])
    errors = list(state.get("errors", []))
    if result_dict.get("error"):
        errors.append(str(result_dict["error"]))

    status = _status_from_result(str(result_dict.get("status") or ""))
    updated = {
        **state,
        "status": status,
        "observations": list(state.get("observations", [])) + [observation],
        "tool_calls": list(state.get("tool_calls", [])) + [tool_call],
        "step_count": int(state.get("step_count", 0)) + 1,
        "warnings": warnings,
        "errors": errors,
        "requires_human": bool(result_dict.get("requires_human")),
        "human_question": result_dict.get("human_question") or state.get("human_question"),
        "selected_tool": None,
        "tool_input": None,
    }

    if result_dict.get("status") == "needs_human":
        updated = {
            **updated,
            "status": "needs_human",
            "final_answer": result_dict.get("human_question") or result_dict.get("answer"),
        }
    elif result_dict.get("status") in {"unsafe", "out_of_scope"}:
        updated = {
            **updated,
            "status": result_dict.get("status"),
            "final_answer": result_dict.get("answer"),
        }

    return _append_log(updated, f"Executed {selected_tool}; observed status {result_dict.get('status')}.")


def route_after_reason(state: ReactCareState) -> str:
    if state.get("status") in TERMINAL_STATUSES:
        return "final_node"
    if state.get("selected_tool"):
        return "action_node"
    return "final_node"


def route_after_action(state: ReactCareState) -> str:
    if state.get("status") in {"needs_human", "unsafe", "out_of_scope", "error"}:
        return "final_node"
    if int(state.get("step_count", 0)) >= int(state.get("max_steps", 5)):
        return "final_node"
    return "reason_node"


@traceable(name="Synataric ReAct - Final")
def final_node(state: ReactCareState) -> ReactCareState:
    status = state.get("status", "complete")
    final_answer = state.get("final_answer")

    if status == "needs_human":
        final_answer = state.get("human_question") or final_answer or "Please provide more information so I can continue."
    elif status in {"unsafe", "out_of_scope"}:
        final_answer = final_answer or "I cannot safely help with that request."
    elif status == "max_steps_reached":
        final_answer = final_answer or _build_max_steps_answer(state)
    elif not final_answer:
        final_answer = _synthesize_final_answer(state)
        if status not in {"error", "needs_human", "unsafe", "out_of_scope", "max_steps_reached"}:
            status = "complete"

    final_text = sanitize_text(final_answer) if status == "unsafe" else _with_disclaimer(final_answer)
    updated = {
        **state,
        "status": status,
        "final_answer": final_text,
    }
    return _append_log(updated, "Final ReAct response prepared.")


def build_react_care_graph():
    workflow = StateGraph(ReactCareState)
    workflow.add_node("reason_node", reason_node)
    workflow.add_node("action_node", action_node)
    workflow.add_node("final_node", final_node)

    workflow.add_edge(START, "reason_node")
    workflow.add_conditional_edges(
        "reason_node",
        route_after_reason,
        {
            "action_node": "action_node",
            "final_node": "final_node",
        },
    )
    workflow.add_conditional_edges(
        "action_node",
        route_after_action,
        {
            "reason_node": "reason_node",
            "final_node": "final_node",
        },
    )
    workflow.add_edge("final_node", END)

    if InMemorySaver is not None:
        return workflow.compile(checkpointer=InMemorySaver())
    return workflow.compile()


def run_react_care_agent(
    question: str,
    namespace: str | None = None,
    top_k: int = 12,
    max_steps: int = 5,
    thread_id: str = "synataric-react-demo",
) -> dict:
    configure_langsmith()
    settings = load_settings()
    graph = build_react_care_graph()
    initial: ReactCareState = {
        "question": question,
        "messages": [],
        "step_count": 0,
        "max_steps": max_steps,
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
        "namespace": namespace or settings.semantic_namespace,
        "top_k": top_k,
    }
    config = {"configurable": {"thread_id": thread_id}}
    return dict(graph.invoke(initial, config=config))


def _coerce_decision(decision: Any) -> ReactDecision:
    if isinstance(decision, ReactDecision):
        return decision
    return ReactDecision.model_validate(_model_to_dict(decision))


def _fallback_decision(state: ReactCareState) -> ReactDecision:
    question = state.get("question", "")
    text = question.lower()
    observations = state.get("observations", [])
    called_tools = {call.get("tool_name") for call in state.get("tool_calls", [])}

    if any(keyword in text for keyword in ["antibiotic", "prescribe", "diagnose", "medication", "medicine dosage"]):
        return ReactDecision(
            action="safety_response",
            rationale="The question asks for medical treatment or medication guidance.",
            safety_flags=["medical_advice_boundary"],
            confidence=0.9,
        )
    if any(keyword in text for keyword in ["super bowl", "cricket score", "stock price", "weather tomorrow"]):
        return ReactDecision(
            action="out_of_scope",
            rationale="The request is outside healthcare travel and care navigation.",
            confidence=0.9,
        )

    missing_fields = _infer_missing_fields(question)
    if missing_fields:
        ask_result = run_ask_human(question, missing_fields)
        return ReactDecision(
            action="ask_human",
            rationale="Required care-planning information is missing.",
            human_question=ask_result.human_question,
            confidence=0.85,
        )

    if observations:
        return ReactDecision(
            action="final_answer",
            rationale="At least one grounded observation is available for the requested scope.",
            final_answer=_synthesize_final_answer(state),
            confidence=0.75,
        )

    tool_sequence = _fallback_tool_sequence(text)
    for tool_name in tool_sequence:
        if tool_name not in called_tools:
            return ReactDecision(
                action="call_tool",
                tool_name=tool_name,
                tool_input=question,
                rationale=f"Use {tool_name} to gather grounded care-navigation information.",
                confidence=0.7,
            )

    return ReactDecision(
        action="final_answer",
        rationale="No additional tool is needed.",
        final_answer=_synthesize_final_answer(state),
        confidence=0.7,
    )


def _fallback_tool_sequence(text: str) -> list[str]:
    requested: list[str] = []
    if any(keyword in text for keyword in ["provider", "hospital", "clinic", "where can i find", "specialist"]):
        requested.append("provider_search_tool")
    if any(keyword in text for keyword in ["cost", "price", "estimate", "budget", "package"]):
        requested.append("cost_estimate_tool")
    if any(keyword in text for keyword in ["recovery", "post-op", "post op", "follow-up", "after surgery"]):
        requested.append("recovery_guidance_tool")
    if any(keyword in text for keyword in ["risk", "urgent", "symptom", "red flag", "complication"]):
        requested.append("risk_checklist_tool")
    if any(keyword in text for keyword in ["travel", "stay", "hotel", "caregiver", "airport", "logistics"]):
        requested.append("travel_planning_tool")
    if any(keyword in text for keyword in ["evidence", "source", "where is", "explained"]):
        requested.append("find_evidence_tool")

    if "plan" in text and "surgery" in text:
        for tool_name in [
            "provider_search_tool",
            "cost_estimate_tool",
            "recovery_guidance_tool",
            "risk_checklist_tool",
            "travel_planning_tool",
        ]:
            if tool_name not in requested:
                requested.append(tool_name)

    return requested or ["find_evidence_tool"]


def _result_to_observation(result: AgentToolResult) -> dict:
    return sanitize_result_dict(_model_to_dict(result))


def _strip_source_paths(sources: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for source in sources:
        item = dict(source)
        for key in ["source", "file_name", "path"]:
            if item.get(key):
                item[key] = str(item[key]).split("/")[-1].split("\\")[-1]
        cleaned.append(item)
    return cleaned


def _strip_evidence_paths(evidence: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for item in evidence:
        entry = dict(item)
        if entry.get("source"):
            entry["source"] = str(entry["source"]).split("/")[-1].split("\\")[-1]
        cleaned.append(entry)
    return cleaned


def _synthesize_final_answer(state: ReactCareState) -> str:
    observations = state.get("observations", [])
    if not observations:
        return "I do not have enough grounded context to answer safely."

    summary = _summarize_observations(observations, include_sources=True)
    return (
        "Here is the care-navigation plan based on the evidence gathered:\n\n"
        f"{summary}\n\n"
        "Confirm procedure-specific decisions, medication instructions, and urgent symptoms with a licensed clinician."
    )


if __name__ == "__main__":
    debug_questions = [
        "Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks.",
        "What is the cost of cataract surgery in Bangalore?",
        "Plan my travel for surgery in Bangalore.",
        "Should I take antibiotics after surgery?",
        "Who won the Super Bowl in 2024?",
    ]

    for debug_question in debug_questions:
        result = run_react_care_agent(debug_question)
        print("QUESTION:", debug_question)
        print("STATUS:", result.get("status"))
        print("STEP COUNT:", result.get("step_count"))
        print("TOOL CALLS:", result.get("tool_calls"))
        print("OBSERVATIONS SUMMARY:", _summarize_observations(result.get("observations", []), include_sources=True))
        print("FINAL ANSWER:", result.get("final_answer"))
        print("WARNINGS:", result.get("warnings"))
        print("ERRORS:", result.get("errors"))
        print("EXECUTION LOG:", result.get("execution_log"))
        print()

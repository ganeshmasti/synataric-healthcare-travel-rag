"""Session-level human-in-the-loop continuation for Synataric Navigator.

This module stores pending clarification state around the agent graph. It keeps
the memory session-local and does not modify the existing RAG backend or UI.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.agent_graph import run_synataric_agent


try:
    from src.config import traceable
except Exception:

    def traceable(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator


class PendingClarification(BaseModel):
    original_question: str
    human_question: str
    missing_fields: list[str] = Field(default_factory=list)
    patient_context: dict[str, Any] = Field(default_factory=dict)
    namespace: str | None = None
    top_k: int = 12
    thread_id: str = "synataric-agent-demo"
    previous_result: dict[str, Any] = Field(default_factory=dict)


class AgentSessionResult(BaseModel):
    status: str
    result: dict[str, Any]
    pending_clarification: PendingClarification | None = None


@traceable(name="Synataric Agent Session - Start")
def start_agent_session(
    question: str,
    patient_context: Optional[Dict[str, Any]] = None,
    namespace: Optional[str] = None,
    top_k: int = 12,
    thread_id: str = "synataric-agent-demo",
) -> AgentSessionResult:
    result = run_synataric_agent(
        question,
        patient_context=patient_context or {},
        namespace=namespace,
        top_k=top_k,
        thread_id=thread_id,
    )
    return _session_result_from_agent_result(
        result,
        original_question=question,
        patient_context=patient_context or {},
        namespace=namespace,
        top_k=top_k,
        thread_id=thread_id,
    )


@traceable(name="Synataric Agent Session - Apply Human Clarification")
def apply_human_clarification(
    pending: PendingClarification,
    human_response: str,
) -> AgentSessionResult:
    updated_context = _merge_human_response_into_context(
        pending.patient_context,
        pending.missing_fields,
        human_response,
    )
    enriched_question = _build_enriched_question(
        pending.original_question,
        pending.missing_fields,
        human_response,
    )
    result = run_synataric_agent(
        enriched_question,
        patient_context=updated_context,
        namespace=pending.namespace,
        top_k=pending.top_k,
        thread_id=pending.thread_id,
    )
    return _session_result_from_agent_result(
        result,
        original_question=pending.original_question,
        patient_context=updated_context,
        namespace=pending.namespace,
        top_k=pending.top_k,
        thread_id=pending.thread_id,
    )


def summarize_session_result(session_result: AgentSessionResult) -> str:
    lines = [f"Status: {session_result.status}"]
    result = session_result.result
    if session_result.status == "needs_human":
        human_question = (
            session_result.pending_clarification.human_question
            if session_result.pending_clarification
            else result.get("human_question")
        )
        lines.append(f"Human question: {human_question}")
    elif session_result.status == "unsafe":
        lines.append(f"Safety response: {result.get('answer')}")
    elif session_result.status == "complete":
        lines.append(f"Answer: {result.get('answer')}")
    elif session_result.status == "error":
        lines.append(f"Errors: {result.get('errors') or result.get('error')}")
    return "\n".join(lines)


def pending_to_dict(pending: PendingClarification | None) -> dict | None:
    if pending is None:
        return None
    if hasattr(pending, "model_dump"):
        return pending.model_dump()
    return pending.dict()


def pending_from_dict(data: dict | None) -> PendingClarification | None:
    if data is None:
        return None
    return PendingClarification(**data)


def _session_result_from_agent_result(
    result: dict[str, Any],
    *,
    original_question: str,
    patient_context: dict[str, Any],
    namespace: str | None,
    top_k: int,
    thread_id: str,
) -> AgentSessionResult:
    status = str(result.get("status") or "error")
    if status == "unsafe":
        return AgentSessionResult(status="unsafe", result=result)
    if result.get("requires_human") or status == "needs_human":
        pending = PendingClarification(
            original_question=original_question,
            human_question=result.get("human_question") or result.get("answer") or "Please provide more information.",
            missing_fields=list(result.get("missing_fields") or []),
            patient_context=dict(patient_context),
            namespace=namespace,
            top_k=top_k,
            thread_id=thread_id,
            previous_result=result,
        )
        return AgentSessionResult(status="needs_human", result=result, pending_clarification=pending)
    if status == "error":
        return AgentSessionResult(status="error", result=result)
    return AgentSessionResult(status="complete", result=result)


def _merge_human_response_into_context(
    patient_context: dict[str, Any],
    missing_fields: list[str],
    human_response: str,
) -> dict[str, Any]:
    updated_context = dict(patient_context)
    for field in missing_fields:
        if field == "procedure":
            updated_context["procedure"] = human_response
        elif field == "destination":
            updated_context["destination"] = human_response
        elif field == "budget":
            updated_context["budget"] = human_response
        else:
            updated_context[field] = human_response
    updated_context["last_human_clarification"] = human_response
    return updated_context


def _build_enriched_question(
    original_question: str,
    missing_fields: list[str],
    human_response: str,
) -> str:
    context_parts = []
    for field in missing_fields:
        context_parts.append(f"{field} = {human_response}")
    if not context_parts:
        context_parts.append(f"clarification = {human_response}")
    return f"{original_question}. Additional context from user: {', '.join(context_parts)}."


if __name__ == "__main__":
    first_question = "Plan my travel for surgery in Bangalore"
    session = start_agent_session(first_question)
    print("INITIAL STATUS:", session.status)
    print("HUMAN QUESTION:", session.pending_clarification.human_question if session.pending_clarification else None)
    if session.pending_clarification:
        continued = apply_human_clarification(session.pending_clarification, "Cataract surgery")
        print("CONTINUED STATUS:", continued.status)
        print("FINAL ANSWER:", continued.result.get("answer"))
        print("FINAL INTENT:", continued.result.get("intent"))
        print("FINAL TOOL:", continued.result.get("selected_tool"))
        print("EXECUTION LOG:", continued.result.get("execution_log"))
    print()

    second_question = "I need surgery in India"
    second_session = start_agent_session(second_question)
    print("INITIAL STATUS:", second_session.status)
    print("HUMAN QUESTION:", second_session.pending_clarification.human_question if second_session.pending_clarification else None)
    if second_session.pending_clarification:
        second_continued = apply_human_clarification(second_session.pending_clarification, "Knee replacement")
        print("CONTINUED STATUS:", second_continued.status)
        print("FINAL ANSWER:", second_continued.result.get("answer"))
        print("FINAL INTENT:", second_continued.result.get("intent"))
        print("FINAL TOOL:", second_continued.result.get("selected_tool"))
        print("EXECUTION LOG:", second_continued.result.get("execution_log"))
    print()

    unsafe_session = start_agent_session("Should I take antibiotics after surgery?")
    print("INITIAL STATUS:", unsafe_session.status)
    print("HUMAN QUESTION:", unsafe_session.pending_clarification.human_question if unsafe_session.pending_clarification else None)
    print("FINAL ANSWER:", unsafe_session.result.get("answer"))

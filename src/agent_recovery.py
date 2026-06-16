"""Recovery helpers for the Synataric agentic workflow."""

from __future__ import annotations


def classify_failure_type(error: Exception | str) -> str:
    text = str(error).lower()
    if "retrieval" in text or "pinecone" in text or "vector" in text:
        return "retrieval_failed"
    if "rerank" in text or "flashrank" in text:
        return "reranking_failed"
    if "generation" in text or "generate" in text or "openai" in text or "llm" in text:
        return "generation_failed"
    if "no evidence" in text or "no_evidence" in text or "no documents" in text:
        return "no_evidence"
    if "low confidence" in text or "low_confidence" in text:
        return "low_confidence"
    return "unknown_error"


def should_retry_tool(status: str, attempt: int, max_attempts: int = 1) -> bool:
    if attempt >= max_attempts:
        return False
    return status in {"error", "no_evidence", "fallback"}


def build_safe_fallback_answer(question: str, reason: str) -> str:
    return (
        "I don't have enough context to answer this from the available Synataric corpus. "
        "You may try rephrasing the question with the procedure, destination, budget, or care topic."
    )


def build_low_confidence_clarification(question: str) -> str:
    return (
        "I want to make sure I route this correctly. Are you asking about providers, "
        "costs, recovery, risks, or travel planning?"
    )


def normalize_agent_status(status: str) -> str:
    return {
        "success": "complete",
        "tool_success": "complete",
        "complete": "complete",
        "fallback": "fallback",
        "needs_human": "needs_human",
        "unsafe": "unsafe",
        "out_of_scope": "out_of_scope",
        "no_evidence": "no_evidence",
        "error": "error",
    }.get(status, status or "error")

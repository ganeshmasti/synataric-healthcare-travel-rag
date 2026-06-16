"""Agent tool layer for Synataric Navigator.

This module wraps the existing RAG pipeline as callable Python functions and
LangChain-compatible tools. It does not replace or modify retrieval, reranking,
generation, indexing, LangGraph, Streamlit, or evaluation code.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.rag_chain import generate_answer
from src.reranking import rerank_documents
from src.retrieval import retrieve_documents


try:
    from langchain_core.tools import tool
except Exception:
    tool = None


STATUS_SUCCESS = "success"
STATUS_NEEDS_HUMAN = "needs_human"
STATUS_UNSAFE = "unsafe"
STATUS_NO_EVIDENCE = "no_evidence"
STATUS_FALLBACK = "fallback"
STATUS_ERROR = "error"
STATUS_OUT_OF_SCOPE = "out_of_scope"


DOMAIN_KEYWORDS: dict[str, str] = {
    "provider_search": "provider hospital clinic specialist center centre cataract surgery eye care India Bangalore",
    "cost_estimate": "cost price estimate budget package fee charges procedure travel stay",
    "recovery_guidance": "recovery post-op postoperative healing follow-up after surgery care guidance",
    "risk_checklist": "risk urgent symptoms red flags complications warning signs immediate care safety",
    "travel_planning": "travel itinerary stay airport hotel caregiver logistics medical travel planning",
    "find_evidence": "source document evidence where explained retrieved fact",
    "general_navigation": "",
}


class AgentToolResult(BaseModel):
    tool_name: str
    status: str
    question: str
    routed_query: str | None = None
    answer: str | None = None
    sources: list[dict] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    retrieved_count: int = 0
    reranked_count: int = 0
    requires_human: bool = False
    human_question: str | None = None
    safety_flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


def run_domain_rag_tool(
    tool_name: str,
    question: str,
    domain: str,
    namespace: str | None = None,
    top_k: int = 12,
    top_n: int = 3,
) -> AgentToolResult:
    warnings: list[str] = []
    routed_query = _build_routed_query(question, domain)
    requested_procedure = detect_requested_procedure(question)

    try:
        retrieved_docs = retrieve_documents(routed_query, namespace=namespace, top_k=top_k)
    except Exception as exc:
        return AgentToolResult(
            tool_name=tool_name,
            status=STATUS_ERROR,
            question=question,
            routed_query=routed_query,
            warnings=["retrieval_failed"],
            error=str(exc),
        )

    if not retrieved_docs:
        return AgentToolResult(
            tool_name=tool_name,
            status=STATUS_NO_EVIDENCE,
            question=question,
            routed_query=routed_query,
            retrieved_count=0,
            warnings=["retrieval_returned_no_documents"],
        )

    procedure_filtered_docs = _filter_docs_for_requested_procedure(retrieved_docs, requested_procedure, domain)
    if procedure_filtered_docs:
        candidate_docs = procedure_filtered_docs
    else:
        candidate_docs = retrieved_docs
        if requested_procedure and _uses_procedure_filter(domain):
            warnings.append("procedure_filter_empty_used_original_retrieval")

    filtered_docs = _prioritize_docs_for_domain(candidate_docs, domain, requested_procedure=requested_procedure)
    if not filtered_docs:
        filtered_docs = candidate_docs
        warnings.append("domain_filter_empty_used_original_retrieval")

    try:
        reranked_docs = rerank_documents(question, filtered_docs, top_n=top_n)
        status = STATUS_SUCCESS
    except Exception:
        reranked_docs = filtered_docs[:top_n]
        reranked_docs = [_with_fallback_rank(doc, rank) for rank, doc in enumerate(reranked_docs, start=1)]
        status = STATUS_FALLBACK
        warnings.append("reranking_failed_used_retrieved_docs")

    if not reranked_docs:
        return AgentToolResult(
            tool_name=tool_name,
            status=STATUS_NO_EVIDENCE,
            question=question,
            routed_query=routed_query,
            retrieved_count=len(retrieved_docs),
            warnings=warnings + ["reranking_returned_no_documents"],
        )

    try:
        answer, sources = generate_answer(question, reranked_docs)
    except Exception as exc:
        return AgentToolResult(
            tool_name=tool_name,
            status=STATUS_ERROR,
            question=question,
            routed_query=routed_query,
            retrieved_count=len(retrieved_docs),
            reranked_count=len(reranked_docs),
            evidence=_format_evidence(reranked_docs),
            warnings=warnings + ["generation_failed"],
            error=str(exc),
        )

    return AgentToolResult(
        tool_name=tool_name,
        status=status,
        question=question,
        routed_query=routed_query,
        answer=answer,
        sources=sources,
        evidence=_format_evidence(reranked_docs),
        retrieved_count=len(retrieved_docs),
        reranked_count=len(reranked_docs),
        warnings=warnings,
    )


def score_doc_for_domain(doc: Document, domain: str, requested_procedure: str | None = None) -> int:
    metadata = doc.metadata or {}
    category = _metadata_text(metadata, "category")
    file_name = _metadata_text(metadata, "file_name")
    source = _metadata_text(metadata, "source")
    doc_type = _metadata_text(metadata, "doc_type")
    chunk_strategy = _metadata_text(metadata, "chunk_strategy")
    content = str(doc.page_content or "").lower()
    metadata_blob = " ".join([category, file_name, source, doc_type, chunk_strategy])
    text = f"{metadata_blob} {content}"

    if domain == "find_evidence" or domain == "general_navigation":
        return 1

    score = 0
    if domain == "provider_search":
        score += _score_if(category, "hospitals", 5)
        score += _score_if(file_name, "hospital", 4)
        score += _score_if(file_name, "provider", 4)
        score += _score_any(text, ["hospital_name", "clinic", "provider", "eye centre", "eye center", "services"], 2)
    elif domain == "cost_estimate":
        score += _score_if(category, "costs", 5)
        score += _score_if(file_name, "cost", 4)
        score += _score_any(text, ["low_estimate", "high_estimate", "inr", "cost_notes"], 3)
    elif domain == "recovery_guidance":
        score += _score_any(category, ["risks", "procedures"], 4)
        score += _score_if(file_name, "recovery", 4)
        score += _score_if(file_name, "cataract_surgery_guide", 4)
        score += _score_any(text, ["recovery", "follow-up", "follow up", "post-op", "post op"], 2)
    elif domain == "risk_checklist":
        score += _score_if(category, "risks", 5)
        score += _score_if(file_name, "risk", 4)
        score += _score_if(file_name, "checklist", 4)
        score += _score_any(text, ["urgent", "warning", "immediate care", "red flags"], 3)
    elif domain == "travel_planning":
        score += _score_if(file_name, "travel", 4)
        score += _score_any(category, ["travel", "risks", "costs"], 3)
        score += _score_any(text, ["stay", "airport", "hotel", "caregiver", "logistics"], 2)

    doc_procedure = detect_doc_procedure(doc)
    if requested_procedure and doc_procedure == requested_procedure:
        score += 5
    elif requested_procedure and doc_procedure == "general":
        score += 2
    elif requested_procedure and not is_procedure_compatible(requested_procedure, doc_procedure, domain):
        score -= 5

    if domain == "cost_estimate" and requested_procedure == "cataract" and "cataract surgery" in content:
        score += 6
    if domain == "recovery_guidance" and requested_procedure == "cataract":
        score += _score_if(file_name, "post_op_recovery_guidelines", 5)
        score += _score_if(file_name, "cataract_surgery_guide", 5)
    if domain == "provider_search" and requested_procedure == "cataract":
        score += _score_if(file_name, "bangalore_eye_hospitals", 5)
        score += _score_if(file_name, "provider_profiles", 5)
        score += _score_if(file_name, "cataract_surgery_guide", 3)
    return score


def run_provider_search(question: str, namespace: str | None = None, top_k: int = 12) -> AgentToolResult:
    return run_domain_rag_tool("provider_search_tool", question, "provider_search", namespace=namespace, top_k=top_k)


def run_cost_estimate(question: str, namespace: str | None = None, top_k: int = 12) -> AgentToolResult:
    return run_domain_rag_tool("cost_estimate_tool", question, "cost_estimate", namespace=namespace, top_k=top_k)


def run_recovery_guidance(question: str, namespace: str | None = None, top_k: int = 12) -> AgentToolResult:
    return run_domain_rag_tool("recovery_guidance_tool", question, "recovery_guidance", namespace=namespace, top_k=top_k)


def run_risk_checklist(question: str, namespace: str | None = None, top_k: int = 12) -> AgentToolResult:
    return run_domain_rag_tool("risk_checklist_tool", question, "risk_checklist", namespace=namespace, top_k=top_k)


def run_travel_planning(question: str, namespace: str | None = None, top_k: int = 12) -> AgentToolResult:
    return run_domain_rag_tool("travel_planning_tool", question, "travel_planning", namespace=namespace, top_k=top_k)


def run_find_evidence(question: str, namespace: str | None = None, top_k: int = 12) -> AgentToolResult:
    return run_domain_rag_tool("find_evidence_tool", question, "find_evidence", namespace=namespace, top_k=top_k)


def run_general_rag(question: str, namespace: str | None = None, top_k: int = 12) -> AgentToolResult:
    return run_domain_rag_tool("general_rag_tool", question, "general_navigation", namespace=namespace, top_k=top_k)


def run_safety_response(question: str, safety_flags: list[str] | None = None) -> AgentToolResult:
    return AgentToolResult(
        tool_name="safety_response_tool",
        status=STATUS_UNSAFE,
        question=question,
        answer=(
            "Synataric Navigator cannot provide diagnosis, prescriptions, medication instructions, "
            "or urgent medical decisions. Please consult a licensed clinician. If symptoms are severe "
            "or urgent, seek immediate medical care."
        ),
        requires_human=True,
        safety_flags=safety_flags or [],
    )


def run_out_of_scope_response(question: str, reason: str | None = None) -> AgentToolResult:
    if reason == "unsupported_or_impossible_destination":
        answer = "I don't have evidence in the Synataric corpus for that destination or care scenario."
    else:
        answer = (
            "Synataric Navigator is focused on healthcare travel and care-navigation questions grounded in the "
            "Synataric corpus. I don't have enough context to answer this request. Try asking about procedures, "
            "providers, costs, recovery, risks, or travel planning."
        )
    return AgentToolResult(
        tool_name="out_of_scope_response_tool",
        status=STATUS_OUT_OF_SCOPE,
        question=question,
        answer=answer,
        requires_human=False,
        warnings=["out_of_scope_request"],
    )


def run_ask_human(question: str, missing_fields: list[str]) -> AgentToolResult:
    return AgentToolResult(
        tool_name="ask_human_tool",
        status=STATUS_NEEDS_HUMAN,
        question=question,
        requires_human=True,
        human_question=_build_human_question(missing_fields),
        warnings=[f"missing_{field}" for field in missing_fields],
    )


def get_agent_tools() -> list:
    if tool is None:
        return []
    return [
        provider_search_tool,
        cost_estimate_tool,
        recovery_guidance_tool,
        risk_checklist_tool,
        travel_planning_tool,
        find_evidence_tool,
        general_rag_tool,
        safety_response_tool,
        out_of_scope_response_tool,
        ask_human_tool,
    ]


def _build_routed_query(question: str, domain: str) -> str:
    domain_keywords = DOMAIN_KEYWORDS.get(domain, "")
    if not domain_keywords:
        return question
    return f"{question}\nFocus: {domain_keywords}"


def detect_requested_procedure(question: str) -> str | None:
    text = str(question or "").lower()
    if _contains_any(text, ["cataract surgery", "cataract"]):
        return "cataract"
    if "knee replacement" in text:
        return "knee_replacement"
    if _contains_any(text, ["cardiac bypass", "heart bypass", "bypass surgery", "cabg"]):
        return "cardiac_bypass"
    if "retina surgery" in text or "retina" in text:
        return "retina"
    if "eye surgery" in text:
        return "eye_surgery"
    return None


def detect_doc_procedure(doc: Document) -> str | None:
    metadata = doc.metadata or {}
    file_name = _metadata_text(metadata, "file_name")
    source = _metadata_text(metadata, "source")
    content = str(doc.page_content or "").lower()
    file_blob = f"{file_name} {source}"
    text = f"{file_blob} {content}"

    if _contains_any(
        file_blob,
        [
            "post_op_recovery_guidelines.md",
            "travel_medical_risk_checklist.md",
            "synataric_disclaimer_and_safety.md",
        ],
    ):
        return "general"
    if "cataract" in text:
        return "cataract"
    if "knee replacement" in text:
        return "knee_replacement"
    if _contains_any(text, ["cardiac bypass", "heart bypass", "cabg"]):
        return "cardiac_bypass"
    if "retina" in text:
        return "retina"
    if _contains_any(text, ["eye surgery", "eye centre", "eye center"]):
        return "eye_surgery"
    return None


def is_procedure_compatible(requested: str | None, doc_procedure: str | None, domain: str) -> bool:
    if requested is None:
        return True
    if doc_procedure == requested:
        return True
    if doc_procedure == "general":
        return True
    if domain == "provider_search" and requested == "cataract" and doc_procedure in {"eye_surgery", "cataract", "general"}:
        return True
    return False


def _prioritize_docs_for_domain(
    docs: list[Document],
    domain: str,
    requested_procedure: str | None = None,
) -> list[Document]:
    if domain == "find_evidence":
        return sorted(docs, key=_retrieval_score, reverse=True)
    scored_docs = [
        (score_doc_for_domain(doc, domain, requested_procedure=requested_procedure), _retrieval_score(doc), index, doc)
        for index, doc in enumerate(docs)
    ]
    matching_docs = [item for item in scored_docs if item[0] > 0]
    if not matching_docs:
        return []
    matching_docs.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    return [doc for _score, _retrieval, _index, doc in matching_docs]


def _filter_docs_for_requested_procedure(
    docs: list[Document],
    requested_procedure: str | None,
    domain: str,
) -> list[Document]:
    if not requested_procedure or not _uses_procedure_filter(domain):
        return docs
    return [
        doc
        for doc in docs
        if is_procedure_compatible(requested_procedure, detect_doc_procedure(doc), domain)
    ]


def _uses_procedure_filter(domain: str) -> bool:
    return domain in {"cost_estimate", "recovery_guidance", "provider_search", "travel_planning"}


def _format_evidence(docs: list[Document]) -> list[dict]:
    evidence = []
    for rank, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        evidence.append(
            {
                "rank": metadata.get("final_rank") or rank,
                "source": metadata.get("file_name") or metadata.get("source"),
                "category": metadata.get("category"),
                "chunk_strategy": metadata.get("chunk_strategy"),
                "retrieval_score": metadata.get("retrieval_score"),
                "rerank_score": metadata.get("rerank_score"),
                "snippet": _snippet(doc.page_content),
            }
        )
    return evidence


def _with_fallback_rank(doc: Document, rank: int) -> Document:
    metadata = dict(doc.metadata or {})
    metadata.setdefault("final_rank", rank)
    metadata.setdefault("rerank_score", metadata.get("retrieval_score"))
    return Document(page_content=doc.page_content, metadata=metadata)


def _build_human_question(missing_fields: list[str]) -> str:
    unique_fields = list(dict.fromkeys(missing_fields))
    prompts = {
        "procedure": "Which procedure are you considering?",
        "destination": "Which destination or city are you considering?",
        "location": "Which location or city should I use?",
        "budget": "What budget range should I use?",
        "care_topic": "What healthcare travel topic would you like help with - providers, costs, recovery, risks, or travel planning?",
    }
    questions = [prompts.get(field, f"Please provide the missing {field}.") for field in unique_fields]
    return " ".join(questions) if questions else "What additional details should I use?"


def _json_result(result: AgentToolResult) -> str:
    return json.dumps(result.model_dump(), default=str)


def _metadata_text(metadata: dict[str, Any], key: str) -> str:
    return str(metadata.get(key, "") or "").lower()


def _score_if(text: str, phrase: str, value: int) -> int:
    return value if phrase in text else 0


def _score_any(text: str, phrases: list[str], value: int) -> int:
    return value if any(phrase in text for phrase in phrases) else 0


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _retrieval_score(doc: Document) -> float:
    score = (doc.metadata or {}).get("retrieval_score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def _snippet(text: str, limit: int = 500) -> str:
    compact = " ".join(str(text or "").split())
    return compact[: limit - 3] + "..." if len(compact) > limit else compact


if tool is not None:

    @tool
    def provider_search_tool(question: str) -> str:
        """Find relevant hospitals, providers, clinics, or specialist centers from the Synataric corpus."""
        return _json_result(run_provider_search(question))

    @tool
    def cost_estimate_tool(question: str) -> str:
        """Find procedure, travel, stay, package, fee, and budget information from the Synataric corpus."""
        return _json_result(run_cost_estimate(question))

    @tool
    def recovery_guidance_tool(question: str) -> str:
        """Find recovery timeline, post-op care, follow-up, and healing guidance from the Synataric corpus."""
        return _json_result(run_recovery_guidance(question))

    @tool
    def risk_checklist_tool(question: str) -> str:
        """Find risk, red flag, warning sign, complication, and immediate-care guidance from the Synataric corpus."""
        return _json_result(run_risk_checklist(question))

    @tool
    def travel_planning_tool(question: str) -> str:
        """Find medical travel planning, stay, airport, hotel, caregiver, and logistics guidance from the corpus."""
        return _json_result(run_travel_planning(question))

    @tool
    def find_evidence_tool(question: str) -> str:
        """Find source documents and compact evidence snippets that explain a retrieved fact."""
        return _json_result(run_find_evidence(question))

    @tool
    def general_rag_tool(question: str) -> str:
        """Answer a general Synataric navigation question using the standard RAG corpus."""
        return _json_result(run_general_rag(question))

    @tool
    def safety_response_tool(question: str, safety_flags: list[str] | None = None) -> str:
        """Return a safe response for diagnosis, prescription, medication, or urgent medical judgment requests."""
        return _json_result(run_safety_response(question, safety_flags=safety_flags))

    @tool
    def out_of_scope_response_tool(question: str, reason: str | None = None) -> str:
        """Return a boundary response for requests outside Synataric healthcare travel and care navigation."""
        return _json_result(run_out_of_scope_response(question, reason=reason))

    @tool
    def ask_human_tool(question: str, missing_fields: list[str]) -> str:
        """Ask the user for missing fields needed before routing to a Synataric agent tool."""
        return _json_result(run_ask_human(question, missing_fields))

else:

    def provider_search_tool(question: str) -> str:
        return _json_result(run_provider_search(question))

    def cost_estimate_tool(question: str) -> str:
        return _json_result(run_cost_estimate(question))

    def recovery_guidance_tool(question: str) -> str:
        return _json_result(run_recovery_guidance(question))

    def risk_checklist_tool(question: str) -> str:
        return _json_result(run_risk_checklist(question))

    def travel_planning_tool(question: str) -> str:
        return _json_result(run_travel_planning(question))

    def find_evidence_tool(question: str) -> str:
        return _json_result(run_find_evidence(question))

    def general_rag_tool(question: str) -> str:
        return _json_result(run_general_rag(question))

    def safety_response_tool(question: str, safety_flags: list[str] | None = None) -> str:
        return _json_result(run_safety_response(question, safety_flags=safety_flags))

    def out_of_scope_response_tool(question: str, reason: str | None = None) -> str:
        return _json_result(run_out_of_scope_response(question, reason=reason))

    def ask_human_tool(question: str, missing_fields: list[str]) -> str:
        return _json_result(run_ask_human(question, missing_fields))


if __name__ == "__main__":
    debug_cases = [
        ("What is the cost of cataract surgery in Bangalore?", run_cost_estimate),
        ("What recovery guidance is available after cataract surgery?", run_recovery_guidance),
        ("Where can I find good cataract surgery in India?", run_provider_search),
        ("What urgent symptoms require immediate care?", run_risk_checklist),
        ("Should I take antibiotics after surgery?", run_safety_response),
    ]

    for debug_question, runner in debug_cases:
        result = runner(debug_question)
        print("QUESTION:", debug_question)
        print("TOOL:", result.tool_name)
        print("STATUS:", result.status)
        print("ANSWER:", result.answer)
        print("SOURCES:", result.sources)
        print("EVIDENCE:", result.evidence)
        print("TOP EVIDENCE SOURCES:", [item.get("source") for item in result.evidence[:3]])
        print("WARNINGS:", result.warnings)
        print()

    ask_result = run_ask_human("Plan my travel for surgery in Bangalore", ["procedure"])
    print("QUESTION:", ask_result.question)
    print("TOOL:", ask_result.tool_name)
    print("STATUS:", ask_result.status)
    print("ANSWER:", ask_result.answer)
    print("SOURCES:", ask_result.sources)
    print("EVIDENCE:", ask_result.evidence)
    print("WARNINGS:", ask_result.warnings)
    print("HUMAN QUESTION:", ask_result.human_question)

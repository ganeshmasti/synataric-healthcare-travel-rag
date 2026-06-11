from typing import List

from langchain_core.documents import Document

from src.config import traceable


PROVIDER_INTENT_TERMS = {
    "where",
    "find",
    "good",
    "best",
    "hospital",
    "hospitals",
    "provider",
    "providers",
    "clinic",
    "clinics",
    "centre",
    "center",
    "doctor",
    "doctors",
    "specialist",
    "specialists",
}

COST_INTENT_TERMS = {
    "cost",
    "costs",
    "price",
    "prices",
    "estimate",
    "estimates",
    "fee",
    "fees",
    "inr",
    "budget",
    "charge",
    "charges",
}

PROVIDER_DOC_TERMS = {
    "hospital",
    "hospitals",
    "provider",
    "providers",
    "clinic",
    "centre",
    "center",
    "eye centre",
    "nethra",
    "sankara",
    "focus_area",
    "navigation_features",
}


def _contains_any(text: str, terms: set[str]) -> bool:
    normalized = str(text).lower()
    return any(term in normalized for term in terms)


def _is_provider_intent(question: str) -> bool:
    return _contains_any(question, PROVIDER_INTENT_TERMS) and not _contains_any(question, COST_INTENT_TERMS)


def _is_provider_doc(doc: Document) -> bool:
    metadata = doc.metadata or {}
    metadata_text = " ".join(
        str(metadata.get(key, ""))
        for key in ["source", "file_name", "category", "parent_id", "source_path"]
    )
    return _contains_any(f"{metadata_text} {doc.page_content}", PROVIDER_DOC_TERMS)


def _intent_boost(question: str, doc: Document) -> float:
    if _is_provider_intent(question) and _is_provider_doc(doc):
        return 0.15
    return 0.0


def _score_or_fallback(doc: Document, fallback: float) -> float:
    score = doc.metadata.get("retrieval_score")
    if score is None:
        return fallback
    try:
        return float(score)
    except (TypeError, ValueError):
        return fallback


def _with_rank_metadata(
    doc: Document,
    final_rank: int,
    rerank_score: float,
    *,
    base_rerank_score: float | None = None,
    intent_boost: float = 0.0,
) -> Document:
    metadata = dict(doc.metadata)
    metadata["final_rank"] = final_rank
    metadata["rerank_score"] = float(rerank_score)
    metadata["base_rerank_score"] = float(base_rerank_score if base_rerank_score is not None else rerank_score)
    metadata["intent_boost"] = float(intent_boost)
    metadata.setdefault("similarity_score", metadata.get("retrieval_score"))
    return Document(page_content=doc.page_content, metadata=metadata)


def _rank_with_intent(question: str, scored_docs: List[tuple[Document, float]], top_n: int) -> List[Document]:
    ranked = []
    for position, (doc, base_score) in enumerate(scored_docs, start=1):
        boost = _intent_boost(question, doc)
        ranked.append((doc, float(base_score), boost, float(base_score) + boost, position))

    ranked.sort(key=lambda item: (item[3], -item[4]), reverse=True)

    return [
        _with_rank_metadata(doc, final_rank, total_score, base_rerank_score=base_score, intent_boost=boost)
        for final_rank, (doc, base_score, boost, total_score, _position) in enumerate(ranked[:top_n], start=1)
    ]


@traceable(name="Synataric FlashRank Reranking")
def rerank_documents(question: str, documents: List[Document], top_n: int = 3, use_flashrank: bool = True) -> List[Document]:
    if not documents:
        return []
    if not use_flashrank:
        scored_docs = [(doc, _score_or_fallback(doc, 1.0 / i)) for i, doc in enumerate(documents, start=1)]
        return _rank_with_intent(question, scored_docs, top_n)
    try:
        from flashrank import Ranker, RerankRequest

        passages = [
            {
                "id": str(i),
                "text": doc.page_content,
                "meta": doc.metadata,
            }
            for i, doc in enumerate(documents)
        ]
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        results = ranker.rerank(RerankRequest(query=question, passages=passages))
        scored_docs = []
        for final_rank, item in enumerate(results, start=1):
            doc = documents[int(item["id"])]
            scored_docs.append((doc, item.get("score", _score_or_fallback(doc, 1.0 / final_rank))))
        return _rank_with_intent(question, scored_docs, top_n)
    except Exception:
        # Retrieve -> Rerank fallback: keep the first candidates if FlashRank is unavailable.
        scored_docs = [(doc, _score_or_fallback(doc, 1.0 / i)) for i, doc in enumerate(documents, start=1)]
        return _rank_with_intent(question, scored_docs, top_n)


def reranking_results(documents: List[Document]) -> List[dict]:
    rows = []

    raw_scores = [
        float(doc.metadata.get("rerank_score") or 0.0)
        for doc in documents
    ]

    max_score = max(raw_scores) if raw_scores else 0.0

    for doc in documents:
        final_rerank_score = float(doc.metadata.get("rerank_score") or 0.0)
        base_rerank_score = float(doc.metadata.get("base_rerank_score", final_rerank_score) or 0.0)
        intent_boost = float(doc.metadata.get("intent_boost") or 0.0)

        if max_score > 0:
            normalized_score = final_rerank_score / max_score
        else:
            normalized_score = 0.0

        rows.append(
            {
                "Rank": doc.metadata.get("final_rank"),
                "Retrieval Score": doc.metadata.get("retrieval_score"),
                "Raw Rerank Score": base_rerank_score,
                "Intent Boost": intent_boost,
                "Final Rerank Score": final_rerank_score,
                "Rerank Relevance": f"{normalized_score:.0%}",
                "Source": doc.metadata.get("file_name") or doc.metadata.get("source", "unknown"),
            }
        )

    return rows

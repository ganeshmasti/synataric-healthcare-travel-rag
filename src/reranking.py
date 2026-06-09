from typing import List

from langchain_core.documents import Document

from src.config import traceable


def _score_or_fallback(doc: Document, fallback: float) -> float:
    score = doc.metadata.get("retrieval_score")
    if score is None:
        return fallback
    try:
        return float(score)
    except (TypeError, ValueError):
        return fallback


def _with_rank_metadata(doc: Document, final_rank: int, rerank_score: float) -> Document:
    metadata = dict(doc.metadata)
    metadata["final_rank"] = final_rank
    metadata["rerank_score"] = float(rerank_score)
    metadata.setdefault("similarity_score", metadata.get("retrieval_score"))
    return Document(page_content=doc.page_content, metadata=metadata)


@traceable(name="Synataric FlashRank Reranking")
def rerank_documents(question: str, documents: List[Document], top_n: int = 3, use_flashrank: bool = True) -> List[Document]:
    if not documents:
        return []
    if not use_flashrank:
        return [_with_rank_metadata(doc, i, _score_or_fallback(doc, 1.0 / i)) for i, doc in enumerate(documents[:top_n], start=1)]
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
        selected = []
        for final_rank, item in enumerate(results[:top_n], start=1):
            doc = documents[int(item["id"])]
            selected.append(_with_rank_metadata(doc, final_rank, item.get("score", _score_or_fallback(doc, 1.0 / final_rank))))
        return selected
    except Exception:
        # Retrieve -> Rerank fallback: keep the first candidates if FlashRank is unavailable.
        return [_with_rank_metadata(doc, i, _score_or_fallback(doc, 1.0 / i)) for i, doc in enumerate(documents[:top_n], start=1)]


def reranking_results(documents: List[Document]) -> List[dict]:
    rows = []
    for doc in documents:
        rows.append(
            {
                "Rank": doc.metadata.get("final_rank"),
                "Retrieval Score": doc.metadata.get("retrieval_score"),
                "Rerank Score": doc.metadata.get("rerank_score"),
                "Source": doc.metadata.get("file_name") or doc.metadata.get("source", "unknown"),
            }
        )
    return rows

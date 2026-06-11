import re
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from src.config import RAW_DATA_DIR
from src.reranking import rerank_documents
from src.retrieval import retrieve_documents


def _tokens(text: str) -> set[str]:
    return {
        token.lower().strip(".,:;!?()[]{}\"'")
        for token in str(text).split()
        if len(token.strip(".,:;!?()[]{}\"'")) > 2
    }


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", " ".join(str(text).split()))
    return [part.strip() for part in parts if part.strip()]


def _best_snippet(query: str, text: str, max_sentences: int = 2) -> str:
    query_tokens = _tokens(query)
    sentences = _sentences(text)
    if not sentences:
        return ""

    ranked = sorted(
        enumerate(sentences),
        key=lambda item: len(_tokens(item[1]).intersection(query_tokens)),
        reverse=True,
    )
    selected_indexes = sorted(index for index, _ in ranked[:max_sentences])
    snippet = " ".join(sentences[index] for index in selected_indexes)
    return snippet[:500]


def _why_matched(query: str, text: str, final_rank) -> str:
    matches = sorted(_tokens(query).intersection(_tokens(text)))
    if matches:
        terms = ", ".join(matches[:6])
        return f"Matched query terms: {terms}. FlashRank selected this evidence as rank {final_rank or 'N/A'}."
    return f"Selected by semantic similarity and FlashRank reranking as rank {final_rank or 'N/A'}."


def _section_context(text: str, heading: str) -> str:
    if not heading or heading == "N/A":
        return text
    lines = str(text).splitlines()
    start = None
    start_level = None
    heading_pattern = re.compile(r"^(#+)\s+(.+?)\s*$")
    for index, line in enumerate(lines):
        match = heading_pattern.match(line.strip())
        if match and match.group(2).strip().lower() == str(heading).strip().lower():
            start = index
            start_level = len(match.group(1))
            break
    if start is None:
        return text

    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = heading_pattern.match(lines[index].strip())
        if match and len(match.group(1)) <= start_level:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _parent_context(doc: Document, metadata: dict, raw_dir: Path | None) -> str:
    if metadata.get("parent_context"):
        return str(metadata["parent_context"])
    if not raw_dir or not metadata.get("source"):
        return doc.page_content
    path = Path(raw_dir) / str(metadata["source"])
    if not path.exists() or path.suffix.lower() == ".csv":
        return doc.page_content
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    except Exception:
        return doc.page_content
    section = metadata.get("section_heading") or ""
    return _section_context(text, section) or doc.page_content


def build_evidence_locations(query: str, documents: Iterable[Document], raw_dir: Path | str | None = None) -> List[dict]:
    locations = []
    raw_path = Path(raw_dir) if raw_dir else None
    for index, doc in enumerate(documents, start=1):
        metadata = dict(doc.metadata or {})
        source = metadata.get("source") or metadata.get("file_name") or "unknown"
        section = metadata.get("section_heading") or metadata.get("category") or "N/A"
        final_rank = metadata.get("final_rank") or index
        locations.append(
            {
                "rank": final_rank,
                "source": source,
                "file_name": metadata.get("file_name") or source,
                "section": section,
                "row_id": metadata.get("row_id"),
                "chunk_id": metadata.get("chunk_id"),
                "parent_id": metadata.get("parent_id"),
                "start_index": metadata.get("start_index"),
                "end_index": metadata.get("end_index"),
                "snippet": _best_snippet(query, doc.page_content),
                "parent_context": _parent_context(doc, metadata, raw_path),
                "why_matched": _why_matched(query, doc.page_content, final_rank),
                "retrieval_score": metadata.get("retrieval_score") or metadata.get("similarity_score"),
                "rerank_score": metadata.get("rerank_score"),
                "chunk_strategy": metadata.get("chunk_strategy"),
            }
        )
    return locations


def locate_evidence(question: str, namespace: str | None = None, top_k: int = 10, top_n: int = 5) -> List[dict]:
    candidates = retrieve_documents(question, namespace=namespace, top_k=top_k)
    reranked = rerank_documents(question, candidates, top_n=top_n)
    return build_evidence_locations(question, reranked, raw_dir=RAW_DATA_DIR)

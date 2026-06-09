from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from prompts.rag_prompt import rag_prompt
from src.config import RAW_DATA_DIR, load_settings, traceable


def _short_fact(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    return compact[: limit - 3] + "..." if len(compact) > limit else compact


def format_docs_for_context(docs: List[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        row = doc.metadata.get("row_id")
        label = f"[Source {i}: {source}" + (f", row {row}" if row else "") + "]"
        blocks.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(blocks)


def source_metadata(docs: List[Document]) -> List[dict]:
    sources = []
    for i, doc in enumerate(docs, start=1):
        meta = dict(doc.metadata)
        meta["source_number"] = i
        source = meta.get("source", "")
        meta["source_path"] = str((RAW_DATA_DIR / source).resolve()) if source else ""
        meta["retrieved_fact"] = _short_fact(doc.page_content)
        sources.append(meta)
    return sources


def build_sources_referenced(sources: List[dict]) -> str:
    lines = ["\n\nSources Referenced"]
    for source in sources:
        number = source.get("source_number")
        label = source.get("file_name") or source.get("source", "unknown")
        lines.append(f"\n[{number}] {label}")
        if source.get("source_path"):
            lines.append(f"Full source path: {source['source_path']}")
        lines.append(f'Retrieved Fact: "{source.get("retrieved_fact", "")}"')
    return "\n".join(lines)


def build_evidence_block(sources: List[dict]) -> str:
    lines = ["\n\nEvidence Used"]
    for source in sources:
        lines.append(
            "\n".join(
                [
                    f"[{source.get('source_number')}] {source.get('file_name') or source.get('source', 'unknown')}",
                    f"Retrieved Fact: {source.get('retrieved_fact', '')}",
                    f"Retrieval Score: {source.get('retrieval_score')}",
                    f"Rerank Score: {source.get('rerank_score')}",
                ]
            )
        )
    return "\n\n".join(lines)


def append_citations(answer: str, sources: List[dict]) -> str:
    source_refs = " ".join(f"[{source.get('source_number')}]" for source in sources)
    return f"{answer.strip()}\n\nSources:\n{source_refs}{build_evidence_block(sources)}"


@traceable(name="Synataric Prompt And LLM Generation")
def generate_answer(question: str, docs: List[Document]) -> Tuple[str, List[dict]]:
    settings = load_settings()
    context = format_docs_for_context(docs)
    llm = ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)
    # Rerank -> Generate: ask the model to answer only from retrieved context with citations.
    chain = rag_prompt | llm
    result = chain.invoke({"context": context, "question": question})
    sources = source_metadata(docs)
    return append_citations(result.content, sources), sources

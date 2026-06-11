from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import load_settings


def _fixed_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )


def _parent_id(metadata: dict) -> str:
    source = metadata.get("source") or metadata.get("file_name") or "unknown"
    if metadata.get("row_id"):
        return f"{source}:row-{metadata['row_id']}"
    if metadata.get("page") is not None:
        return f"{source}:page-{metadata['page']}"
    return str(source)


def _section_heading(text: str) -> str:
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def chunk_documents(documents: Iterable[Document], strategy: str = "fixed") -> List[Document]:
    docs = list(documents)
    if strategy not in {"fixed", "semantic"}:
        raise ValueError("strategy must be 'fixed' or 'semantic'")

    if strategy == "fixed":
        splitter = _fixed_splitter(chunk_size=700, chunk_overlap=120)
        chunks = splitter.split_documents(docs)
    else:
        # Data -> Chunk: prefer LangChain semantic chunking when the optional class is available.
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_openai import OpenAIEmbeddings

            settings = load_settings(require_secrets=True)
            splitter = SemanticChunker(OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key))
            chunks = splitter.split_documents(docs)
        except Exception:
            splitter = _fixed_splitter(chunk_size=1200, chunk_overlap=180)
            chunks = splitter.split_documents(docs)

    parent_counts: dict[str, int] = {}
    for chunk in chunks:
        chunk.metadata = dict(chunk.metadata)
        parent_id = _parent_id(chunk.metadata)
        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        chunk_index = parent_counts[parent_id]
        start_index = int(chunk.metadata.get("start_index") or 0)
        chunk.metadata["chunk_strategy"] = strategy
        chunk.metadata["parent_id"] = parent_id
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = f"{parent_id}:chunk-{chunk_index}"
        chunk.metadata["start_index"] = start_index
        chunk.metadata["end_index"] = start_index + len(chunk.page_content)
        chunk.metadata.setdefault("section_heading", _section_heading(chunk.page_content))
    return chunks

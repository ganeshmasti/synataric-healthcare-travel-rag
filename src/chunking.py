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

    for chunk in chunks:
        chunk.metadata = dict(chunk.metadata)
        chunk.metadata["chunk_strategy"] = strategy
    return chunks

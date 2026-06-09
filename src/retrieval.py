from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import RAW_DATA_DIR, load_settings, traceable


def get_vector_store(namespace: str | None = None) -> PineconeVectorStore:
    settings = load_settings()
    namespace = namespace or settings.semantic_namespace
    embeddings = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return PineconeVectorStore(index_name=settings.pinecone_index_name, embedding=embeddings, namespace=namespace)


@traceable(name="Synataric Pinecone Retrieval")
def retrieve_documents(question: str, namespace: str | None = None, top_k: int = 10) -> List[Document]:
    # Store -> Retrieve: pull the top candidates before reranking.
    vector_store = get_vector_store(namespace)
    try:
        scored = vector_store.similarity_search_with_score(question, k=top_k)
    except Exception:
        docs = vector_store.similarity_search(question, k=top_k)
        scored = [(doc, None) for doc in docs]

    enriched: List[Document] = []
    for rank, (doc, score) in enumerate(scored, start=1):
        metadata = dict(doc.metadata)
        metadata["retrieval_rank"] = rank
        metadata["retrieval_score"] = score
        metadata["similarity_score"] = score
        if metadata.get("source"):
            metadata["source_path"] = str((RAW_DATA_DIR / metadata["source"]).resolve())
        enriched.append(Document(page_content=doc.page_content, metadata=metadata))
    return enriched

import time
from typing import Iterable

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import Settings, load_settings


def get_embeddings(settings: Settings | None = None) -> OpenAIEmbeddings:
    settings = settings or load_settings()
    # Chunk -> Embed: text-embedding-3-small creates 1536-dimensional vectors.
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def ensure_pinecone_index(settings: Settings | None = None):
    settings = settings or load_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = {index["name"] for index in pc.list_indexes()}
    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(2)
    return pc.Index(settings.pinecone_index_name)


def delete_namespace(namespace: str, settings: Settings | None = None) -> None:
    index = ensure_pinecone_index(settings)
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass


def index_documents(documents: Iterable[Document], namespace: str, settings: Settings | None = None) -> PineconeVectorStore:
    settings = settings or load_settings()
    ensure_pinecone_index(settings)
    delete_namespace(namespace, settings)
    # Embed -> Store: PineconeVectorStore writes chunks into the selected namespace.
    return PineconeVectorStore.from_documents(
        documents=list(documents),
        embedding=get_embeddings(settings),
        index_name=settings.pinecone_index_name,
        namespace=namespace,
    )

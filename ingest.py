from src.chunking import chunk_documents
from src.cleaning import clean_documents
from src.config import DATA_DIR, load_settings
from src.indexing import index_documents
from src.loaders import load_documents
from src.sample_data import create_sample_corpus


def main() -> None:
    settings = load_settings()
    create_sample_corpus(DATA_DIR)
    docs = load_documents()
    cleaned_docs = clean_documents(docs)
    fixed_chunks = chunk_documents(cleaned_docs, strategy="fixed")
    semantic_chunks = chunk_documents(cleaned_docs, strategy="semantic")

    print(f"Loaded documents: {len(docs)}")
    print(f"Cleaned documents: {len(cleaned_docs)}")
    print(f"Fixed chunks: {len(fixed_chunks)}")
    print(f"Semantic chunks: {len(semantic_chunks)}")

    # Data -> Clean -> Chunk -> Embed -> Store for both retrieval strategies.
    index_documents(fixed_chunks, namespace=settings.fixed_namespace, settings=settings)
    index_documents(semantic_chunks, namespace=settings.semantic_namespace, settings=settings)
    print("Success: Synataric Navigator corpus indexed into Pinecone namespaces synataric-fixed and synataric-semantic.")


if __name__ == "__main__":
    main()

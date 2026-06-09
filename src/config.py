import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"


def normalize_pinecone_index_name(name: str) -> str:
    """Convert display/project names into Pinecone-safe lowercase index names."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_name).strip("-").lower()
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized or "synataric"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    fixed_namespace: str = "synataric-fixed"
    semantic_namespace: str = "synataric-semantic"
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    embedding_dimension: int = 1536
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "false"
    langchain_project: str = "Synataric-Navigator"


def load_settings(require_secrets: bool = True) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index_name=normalize_pinecone_index_name(
            os.getenv("PINECONE_INDEX_NAME", "Synataric – Healthcare Travel & Care Planning RAG")
        ),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY", ""),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "false"),
        langchain_project=os.getenv("LANGCHAIN_PROJECT", "Synataric-Navigator"),
    )
    missing = []
    if require_secrets and not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if require_secrets and not settings.pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variable(s): {names}. Copy .env.example to .env and fill them in.")
    return settings


def configure_langsmith() -> bool:
    """Enable LangSmith tracing when credentials are present."""
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "Synataric-Navigator")
    if api_key and tracing:
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project
        return True
    return False


def get_langsmith_status() -> str:
    return "LangSmith tracing enabled" if configure_langsmith() else "LangSmith tracing disabled"


try:
    from langsmith import traceable
except Exception:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

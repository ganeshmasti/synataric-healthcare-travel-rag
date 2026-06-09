import csv
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from src.config import RAW_DATA_DIR


def _category_for(path: Path, raw_dir: Path) -> str:
    relative = path.relative_to(raw_dir)
    return relative.parts[0] if len(relative.parts) > 1 else "uncategorized"


def _doc_type(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def _base_metadata(path: Path, raw_dir: Path) -> dict:
    return {
        "source": str(path.relative_to(raw_dir)).replace("\\", "/"),
        "doc_type": _doc_type(path),
        "file_name": path.name,
        "category": _category_for(path, raw_dir),
    }


def _load_csv(path: Path, raw_dir: Path) -> List[Document]:
    docs = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_id, row in enumerate(reader, start=1):
            pairs = [f"{key}: {value}" for key, value in row.items() if value not in (None, "")]
            metadata = _base_metadata(path, raw_dir)
            metadata["row_id"] = row_id
            docs.append(Document(page_content="\n".join(pairs), metadata=metadata))
    return docs


def load_documents(raw_dir: Path = RAW_DATA_DIR) -> List[Document]:
    """Load .md, .txt, .pdf, and .csv from data/raw recursively as LangChain Documents."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    documents: List[Document] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf", ".csv"}:
            continue
        if path.suffix.lower() == ".csv":
            documents.extend(_load_csv(path, raw_dir))
            continue
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        for doc in loaded:
            metadata = _base_metadata(path, raw_dir)
            metadata.update(doc.metadata or {})
            doc.metadata = metadata
            documents.append(doc)
    return documents

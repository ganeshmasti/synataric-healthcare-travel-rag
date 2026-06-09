import re
from typing import Iterable, List

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Clean prose while preserving tables, costs, names, procedure terms, and numeric values."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def clean_documents(documents: Iterable[Document]) -> List[Document]:
    cleaned = []
    for doc in documents:
        content = clean_text(doc.page_content)
        if content:
            cleaned.append(Document(page_content=content, metadata=dict(doc.metadata)))
    return cleaned

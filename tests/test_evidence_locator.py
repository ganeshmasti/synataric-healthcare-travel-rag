from langchain_core.documents import Document

from src.chunking import chunk_documents
from src.evidence_locator import build_evidence_locations


def test_chunk_documents_adds_location_metadata():
    docs = [
        Document(
            page_content="# Cataract Guide\n\nRecovery planning after cataract surgery. " * 30,
            metadata={"source": "cataract_surgery_guide.md", "file_name": "cataract_surgery_guide.md"},
        )
    ]

    chunks = chunk_documents(docs, strategy="fixed")

    assert chunks
    assert chunks[0].metadata["chunk_id"].startswith("cataract_surgery_guide.md:")
    assert chunks[0].metadata["parent_id"] == "cataract_surgery_guide.md"
    assert chunks[0].metadata["chunk_index"] == 1
    assert chunks[0].metadata["start_index"] >= 0
    assert chunks[0].metadata["end_index"] > chunks[0].metadata["start_index"]


def test_build_evidence_locations_returns_snippet_parent_context_and_explanation():
    docs = [
        Document(
            page_content=(
                "Pre-op details are separate. Cataract recovery planning includes eye follow-up, "
                "warning symptoms, and clinician-directed recovery timing."
            ),
            metadata={
                "source": "procedures/cataract_surgery_guide.md",
                "file_name": "cataract_surgery_guide.md",
                "section_heading": "Recovery Planning",
                "chunk_id": "procedures/cataract_surgery_guide.md:1",
                "parent_id": "procedures/cataract_surgery_guide.md",
                "retrieval_score": 0.72,
                "rerank_score": 0.91,
                "final_rank": 1,
            },
        )
    ]

    locations = build_evidence_locations("Where is cataract recovery explained?", docs)

    assert len(locations) == 1
    assert locations[0]["source"] == "procedures/cataract_surgery_guide.md"
    assert locations[0]["section"] == "Recovery Planning"
    assert "Cataract recovery planning" in locations[0]["snippet"]
    assert "eye follow-up" in locations[0]["parent_context"]
    assert "cataract" in locations[0]["why_matched"].lower()
    assert locations[0]["retrieval_score"] == 0.72
    assert locations[0]["rerank_score"] == 0.91


def test_build_evidence_locations_uses_raw_parent_context(tmp_path):
    raw_dir = tmp_path / "raw"
    source = raw_dir / "procedures" / "guide.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# Guide\n\nShort intro.\n\n## Recovery\n\nThis larger parent section explains recovery follow-up and warning symptoms.\n\n## Costs\n\nCost details.",
        encoding="utf-8",
    )
    docs = [
        Document(
            page_content="recovery follow-up",
            metadata={
                "source": "procedures/guide.md",
                "file_name": "guide.md",
                "section_heading": "Recovery",
                "final_rank": 1,
            },
        )
    ]

    locations = build_evidence_locations("Where is recovery follow-up explained?", docs, raw_dir=raw_dir)

    assert "larger parent section" in locations[0]["parent_context"]
    assert "Cost details" not in locations[0]["parent_context"]

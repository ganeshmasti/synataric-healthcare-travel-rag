from langchain_core.documents import Document

from src.cleaning import clean_text
from src.chunking import chunk_documents
from src.config import normalize_pinecone_index_name
from src.evaluation import build_ragas_dataset_records, build_ragas_records, compute_quality_metrics, merge_ragas_scores
from src.rag_chain import build_evidence_block, source_metadata
from src.reranking import rerank_documents
from src.sample_data import create_sample_corpus


def test_create_sample_corpus_writes_expected_files(tmp_path):
    data_dir = tmp_path / "data"

    create_sample_corpus(data_dir)

    assert (data_dir / "raw" / "procedures" / "cataract_surgery_guide.md").exists()
    assert (data_dir / "raw" / "hospitals" / "bangalore_eye_hospitals.csv").exists()
    assert (data_dir / "sample_questions.csv").exists()


def test_clean_text_preserves_costs_and_medical_terms():
    dirty = "Cataract surgery\n\n\n  Estimate: INR 45,000 - 90,000\n\nPhacoemulsification"

    cleaned = clean_text(dirty)

    assert "INR 45,000 - 90,000" in cleaned
    assert "Phacoemulsification" in cleaned
    assert "\n\n\n" not in cleaned


def test_chunk_documents_sets_strategy_metadata():
    docs = [
        Document(
            page_content="Knee replacement travel planning. " * 80,
            metadata={"source": "guide.md", "category": "procedures"},
        )
    ]

    chunks = chunk_documents(docs, strategy="fixed")

    assert chunks
    assert all(chunk.metadata["chunk_strategy"] == "fixed" for chunk in chunks)
    assert all("source" in chunk.metadata for chunk in chunks)


def test_normalize_pinecone_index_name_makes_project_name_valid():
    name = normalize_pinecone_index_name("Synataric - Healthcare Travel & Care Planning RAG")

    assert name == "synataric-healthcare-travel-care-planning-rag"


def test_rerank_fallback_adds_scores_and_final_rank():
    docs = [
        Document(page_content="Cataract surgery estimate INR 45000", metadata={"source": "costs.csv", "retrieval_score": 0.91}),
        Document(page_content="Bangalore eye hospital", metadata={"source": "hospitals.csv", "retrieval_score": 0.82}),
    ]

    ranked = rerank_documents("cataract cost", docs, top_n=2, use_flashrank=False)

    assert [doc.metadata["final_rank"] for doc in ranked] == [1, 2]
    assert ranked[0].metadata["rerank_score"] == 0.91
    assert ranked[1].metadata["rerank_score"] == 0.82


def test_source_metadata_includes_retrieved_fact_and_scores():
    docs = [
        Document(
            page_content="Cataract surgery low estimate INR 45000 and high estimate INR 150000.",
            metadata={"source": "costs.csv", "retrieval_score": 0.9, "rerank_score": 0.95, "final_rank": 1},
        )
    ]

    sources = source_metadata(docs)
    evidence = build_evidence_block(sources)

    assert sources[0]["retrieved_fact"].startswith("Cataract surgery")
    assert "[1] costs.csv" in evidence
    assert "Rerank Score: 0.95" in evidence


def test_compute_quality_metrics_scores_retrieval_hit_rate():
    rows = [
        {
            "expected_sources": ["costs.csv"],
            "retrieved_sources": ["costs.csv", "hospitals.csv"],
            "expected_answer": "Cataract surgery costs INR 45000",
            "generated_answer": "Cataract surgery costs INR 45000 in the illustrative corpus.",
        },
        {
            "expected_sources": ["recovery.md"],
            "retrieved_sources": ["costs.csv"],
            "expected_answer": "Ask about follow-up",
            "generated_answer": "Costs vary.",
        },
    ]

    metrics = compute_quality_metrics(rows)

    assert metrics["retrieval_hit_rate"] == 0.5
    assert 0 < metrics["context_precision"] <= 1


def test_build_ragas_records_uses_questions_answers_contexts_and_ground_truth():
    rows = [
        {
            "question": "What is cataract surgery cost?",
            "generated_answer": "The illustrative range is INR 45000 to INR 150000.",
            "expected_answer": "Cataract surgery costs INR 45000 to INR 150000.",
            "retrieved_contexts": ["procedure: Cataract surgery\nlow_estimate_inr: 45000"],
        }
    ]

    records = build_ragas_records(rows)

    assert records["question"] == ["What is cataract surgery cost?"]
    assert records["answer"] == ["The illustrative range is INR 45000 to INR 150000."]
    assert records["contexts"] == [["procedure: Cataract surgery\nlow_estimate_inr: 45000"]]
    assert records["ground_truth"] == ["Cataract surgery costs INR 45000 to INR 150000."]
    assert records["user_input"] == records["question"]
    assert records["response"] == records["answer"]
    assert records["retrieved_contexts"] == records["contexts"]
    assert records["reference"] == records["ground_truth"]


def test_build_ragas_dataset_records_uses_ragas_04_schema_only():
    rows = [
        {
            "question": "What is cataract surgery cost?",
            "generated_answer": "The illustrative range is INR 45000 to INR 150000.",
            "expected_answer": "Cataract surgery costs INR 45000 to INR 150000.",
            "retrieved_contexts": ["procedure: Cataract surgery\nlow_estimate_inr: 45000"],
        }
    ]

    records = build_ragas_dataset_records(rows)

    assert set(records) == {"user_input", "response", "retrieved_contexts", "reference"}
    assert records["user_input"] == ["What is cataract surgery cost?"]
    assert records["response"] == ["The illustrative range is INR 45000 to INR 150000."]


def test_merge_ragas_scores_exposes_requested_metrics():
    metrics = merge_ragas_scores(
        {"retrieval_hit_rate": 1.0},
        {
            "faithfulness": 0.91,
            "answer_relevancy": 0.88,
            "context_precision": 0.77,
            "context_recall": 0.66,
        },
    )

    assert metrics["faithfulness"] == 0.91
    assert metrics["answer_relevancy"] == 0.88
    assert metrics["ragas_context_precision"] == 0.77
    assert metrics["ragas_context_recall"] == 0.66

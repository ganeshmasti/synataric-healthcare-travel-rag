from langchain_core.documents import Document

import app


def test_pipeline_stages_include_requested_execution_steps():
    stages = app.build_pipeline_stages()

    assert [stage["label"] for stage in stages] == [
        "Question Received",
        "Query Embedding",
        "Pinecone Retrieval",
        "Candidate Chunks Retrieved",
        "FlashRank Reranking",
        "Top Evidence Selected",
        "Prompt Assembly",
        "GPT Answer Generation",
        "Final Response",
        "Evidence + Citations",
    ]
    assert all(stage["status"] == "Pending" for stage in stages)
    assert stages[1]["explanation"].startswith("Converts the user question")


def test_update_stage_status_marks_prior_stages_complete():
    stages = app.build_pipeline_stages()

    updated = app.update_stage_status(stages, 2, "Running", elapsed=1.25)

    assert updated[0]["status"] == "Complete"
    assert updated[1]["status"] == "Complete"
    assert updated[2]["status"] == "Running"
    assert updated[2]["elapsed"] == "1.2s"
    assert updated[3]["status"] == "Pending"


def test_build_execution_metrics_uses_real_graph_result_values():
    result = {
        "retrieved_docs": [Document(page_content="A"), Document(page_content="B")],
        "reranked_docs": [Document(page_content="A")],
        "sources": [{"source": "costs.csv"}, {"source": "hospitals.csv"}],
    }

    metrics = app.build_execution_metrics(
        result=result,
        current_stage="Final Response",
        namespace="synataric-semantic",
        top_k=10,
        model="gpt-4o-mini",
        langsmith_enabled=True,
    )

    assert metrics["Current Stage"] == "Final Response"
    assert metrics["Retrieved Chunks"] == "2"
    assert metrics["Reranked Chunks"] == "1"
    assert metrics["Sources Used"] == "2"
    assert metrics["LangSmith Status"] == "Tracing Enabled"


def test_build_grounding_summary_extracts_sources_scores_and_refusal_status():
    result = {
        "answer": "I do not have enough relevant Synataric context to answer.",
        "sources": [{"file_name": "costs.csv"}],
        "reranking_results": [{"Source": "costs.csv", "Rerank Score": 0.91}],
        "reranked_docs": [
            Document(
                page_content="Cataract surgery estimate INR 45000.",
                metadata={"source": "costs.csv", "rerank_score": 0.91},
            )
        ],
    }

    summary = app.build_grounding_summary(result)

    assert summary["source_files"] == ["costs.csv"]
    assert summary["reranking_scores"] == ["costs.csv: 0.9100"]
    assert "1 evidence chunk" in summary["prompt_summary"]
    assert summary["refusal_behavior"] == "Refusal or insufficient-context behavior detected."

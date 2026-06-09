from pathlib import Path
import math
import os
from contextlib import contextmanager
from typing import Iterable, List

import pandas as pd

from src.config import DATA_DIR, traceable
from src.config import load_settings
from src.graph import run_synataric_graph


EVALUATION_PATH = DATA_DIR / "evaluation_questions.csv"


def create_evaluation_questions(path: Path = EVALUATION_PATH) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ["What is the cost of cataract surgery in Bangalore?", "Cataract surgery in Bangalore is listed as an illustrative INR 45000 to INR 150000 range.", "costs"],
        ["Which eye hospitals are listed in Bangalore?", "The corpus lists Bangalore Eye Centre, Narayana Nethra Network, and Sankara Eye Services.", "providers"],
        ["What recovery guidance is available after cataract surgery?", "The corpus mentions early eye follow-up, warning symptoms, and clinician-directed recovery planning.", "recovery"],
        ["What should a caregiver ask before knee replacement travel?", "Ask about mobility limits, blood clot precautions, physiotherapy schedule, discharge planning, and return travel timing.", "procedures"],
        ["What are local stay cost estimates in Bangalore?", "Serviced apartment estimates are INR 2500 to INR 8500 per night in the illustrative corpus.", "travel"],
        ["What urgent symptoms require immediate care?", "Chest pain, severe breathlessness, fainting, stroke-like symptoms, heavy bleeding, and sudden vision loss require immediate medical care.", "risks"],
        ["What planning is needed for cardiac bypass travel?", "The corpus mentions cardiology evaluation, ICU and ward recovery, cardiac rehabilitation, caregiver support, and longer local observation.", "procedures"],
        ["What documents should patients carry before medical travel?", "Prescriptions, allergy list, implant cards, recent test reports, and emergency contacts are listed.", "risks"],
        ["What does Synataric say about costs?", "Costs are illustrative estimates and depend on clinical, hospital, procedure, stay, complications, currency, and insurance factors.", "policy"],
        ["What provider profile details are available for Bangalore Eye Centre?", "The profile lists cataract evaluation, phacoemulsification, lens counseling, retina referrals, airport pickup partners, and nearby serviced apartments.", "providers"],
    ]
    pd.DataFrame(rows, columns=["question", "expected_answer", "category"]).to_csv(path, index=False)


def load_evaluation_questions(path: Path = EVALUATION_PATH) -> pd.DataFrame:
    create_evaluation_questions(path)
    return pd.read_csv(path)


def _tokens(text: str) -> set[str]:
    return {token.lower().strip(".,:;!?()[]") for token in str(text).split() if len(token.strip(".,:;!?()[]")) > 2}


def _expected_sources_for(category: str) -> List[str]:
    mapping = {
        "costs": ["india_procedure_costs.csv", "travel_stay_costs.csv"],
        "providers": ["bangalore_eye_hospitals.csv", "provider_profiles.md"],
        "recovery": ["post_op_recovery_guidelines.md", "cataract_surgery_guide.md"],
        "procedures": ["cataract_surgery_guide.md", "knee_replacement_guide.md", "cardiac_bypass_guide.md"],
        "travel": ["travel_stay_costs.csv", "travel_medical_risk_checklist.md"],
        "risks": ["travel_medical_risk_checklist.md", "post_op_recovery_guidelines.md"],
        "policy": ["synataric_disclaimer_and_safety.md"],
    }
    return mapping.get(str(category), [])


def compute_quality_metrics(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    if not rows:
        return {"retrieval_hit_rate": 0.0, "source_coverage": 0.0, "context_precision": 0.0, "context_recall": 0.0}

    hits = 0
    coverages = []
    precisions = []
    recalls = []
    for row in rows:
        expected_sources = set(row.get("expected_sources") or _expected_sources_for(row.get("category", "")))
        retrieved_sources = set(row.get("retrieved_sources") or [])
        if expected_sources and retrieved_sources.intersection(expected_sources):
            hits += 1
        coverages.append(len(retrieved_sources.intersection(expected_sources)) / max(len(expected_sources), 1))

        expected_tokens = _tokens(row.get("expected_answer", ""))
        generated_tokens = _tokens(row.get("generated_answer", ""))
        overlap = expected_tokens.intersection(generated_tokens)
        precisions.append(len(overlap) / max(len(generated_tokens), 1))
        recalls.append(len(overlap) / max(len(expected_tokens), 1))

    return {
        "retrieval_hit_rate": round(hits / len(rows), 3),
        "source_coverage": round(sum(coverages) / len(coverages), 3),
        "context_precision": round(sum(precisions) / len(precisions), 3),
        "context_recall": round(sum(recalls) / len(recalls), 3),
    }


def build_ragas_records(rows: Iterable[dict]) -> dict:
    """Build a compatibility payload for tests and older RAGAS examples."""
    rows = list(rows)
    questions = [row.get("question", "") for row in rows]
    answers = [row.get("generated_answer", "") for row in rows]
    contexts = [list(row.get("retrieved_contexts") or []) for row in rows]
    references = [row.get("expected_answer", "") for row in rows]
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": references,
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": references,
    }


def build_ragas_dataset_records(rows: Iterable[dict]) -> dict:
    """Build the RAGAS 0.4 single-turn HuggingFace Dataset payload."""
    rows = list(rows)
    return {
        "user_input": [row.get("question", "") for row in rows],
        "response": [row.get("generated_answer", "") for row in rows],
        "retrieved_contexts": [list(row.get("retrieved_contexts") or []) for row in rows],
        "reference": [row.get("expected_answer", "") for row in rows],
    }


def _ragas_metrics():
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def _result_to_dict(result) -> dict:
    if isinstance(result, dict):
        return dict(result)
    if hasattr(result, "scores"):
        scores = getattr(result, "scores") or []
        if not scores:
            raise RuntimeError("RAGAS returned no score rows. Check evaluator metric errors in the Streamlit terminal.")
        frame = pd.DataFrame(scores)
        numeric = frame.select_dtypes(include="number")
        return {column: round(float(numeric[column].mean()), 3) for column in numeric.columns}
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        numeric = frame.select_dtypes(include="number")
        return {column: round(float(numeric[column].mean()), 3) for column in numeric.columns}
    try:
        return dict(result)
    except Exception:
        return {}


def _mean_numeric(values: list[float]) -> float | None:
    clean = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(numeric):
            clean.append(numeric)
    if not clean:
        return None
    return round(sum(clean) / len(clean), 3)


def merge_ragas_scores(metrics: dict, ragas_scores: dict) -> dict:
    merged = dict(metrics)
    if "faithfulness" in ragas_scores:
        merged["faithfulness"] = round(float(ragas_scores["faithfulness"]), 3)
    if "answer_relevancy" in ragas_scores:
        merged["answer_relevancy"] = round(float(ragas_scores["answer_relevancy"]), 3)
    if "context_precision" in ragas_scores:
        merged["ragas_context_precision"] = round(float(ragas_scores["context_precision"]), 3)
    if "context_recall" in ragas_scores:
        merged["ragas_context_recall"] = round(float(ragas_scores["context_recall"]), 3)
    return merged


@contextmanager
def _without_langsmith_tracing():
    previous = os.environ.get("LANGCHAIN_TRACING_V2")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = previous


def run_ragas_direct_metric_evaluation(rows: Iterable[dict], evaluator_llm, evaluator_embeddings) -> dict:
    """Use RAGAS metric objects directly when aggregate evaluate() returns no rows."""
    from ragas import SingleTurnSample
    from ragas.run_config import RunConfig

    rows = list(rows)
    metric_scores: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
    }
    metric_errors = []

    for metric in _ragas_metrics():
        if hasattr(metric, "llm"):
            metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings
        metric.init(RunConfig())
        metric_name = getattr(metric, "name", type(metric).__name__)

        for row in rows:
            sample = SingleTurnSample(
                user_input=row.get("question", ""),
                response=row.get("generated_answer", ""),
                retrieved_contexts=list(row.get("retrieved_contexts") or []),
                reference=row.get("expected_answer", ""),
            )
            try:
                with _without_langsmith_tracing():
                    metric_scores.setdefault(metric_name, []).append(metric.single_turn_score(sample))
            except Exception as exc:
                metric_errors.append(f"{metric_name}: {exc}")

    scores = {}
    for metric_name, values in metric_scores.items():
        mean_value = _mean_numeric(values)
        if mean_value is not None:
            scores[metric_name] = mean_value
    if metric_errors:
        scores["_ragas_metric_errors"] = "; ".join(metric_errors[:4])
    if not any(name in scores for name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]):
        raise RuntimeError(scores.get("_ragas_metric_errors") or "RAGAS direct metric scoring returned no numeric scores.")
    scores["_ragas_mode"] = "direct_metric_scoring"
    return scores


def run_ragas_evaluation(rows: Iterable[dict]) -> dict:
    """Run actual RAGAS evaluation over question, answer, contexts, and ground_truth."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    settings = load_settings()
    rows = list(rows)
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    )
    return run_ragas_direct_metric_evaluation(rows, evaluator_llm, evaluator_embeddings)


@traceable(name="Synataric Evaluation Runner")
def run_evaluation(namespace: str | None = None, top_k: int = 10, limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    questions = load_evaluation_questions()
    if limit:
        questions = questions.head(limit)
    rows = []
    for item in questions.to_dict("records"):
        result = run_synataric_graph(item["question"], namespace=namespace, top_k=top_k)
        reranked_docs = result.get("reranked_docs", [])
        retrieved_sources = [doc.metadata.get("file_name") or doc.metadata.get("source") for doc in reranked_docs]
        retrieved_contexts = [doc.page_content for doc in reranked_docs]
        rows.append(
            {
                "question": item["question"],
                "category": item["category"],
                "expected_answer": item["expected_answer"],
                "generated_answer": result.get("answer", ""),
                "retrieved_sources": retrieved_sources,
                "retrieved_contexts": retrieved_contexts,
                "expected_sources": _expected_sources_for(item["category"]),
            }
        )
    metrics = compute_quality_metrics(rows)
    try:
        ragas_scores = run_ragas_evaluation(rows)
        metrics = merge_ragas_scores(metrics, ragas_scores)
        mode = ragas_scores.get("_ragas_mode", "aggregate_evaluate")
        metric_errors = ragas_scores.get("_ragas_metric_errors")
        metrics["ragas_status"] = (
            f"RAGAS evaluation completed with faithfulness, answer relevancy, context precision, and context recall. Mode: {mode}."
        )
        if metric_errors:
            metrics["ragas_status"] += f" Metric warnings: {metric_errors}"
    except Exception as exc:
        metrics["ragas_status"] = f"RAGAS evaluation unavailable; showing local proxy metrics. Reason: {exc}"
    return pd.DataFrame(rows), metrics

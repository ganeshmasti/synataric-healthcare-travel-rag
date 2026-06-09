import html

import pandas as pd
import streamlit as st

from src.chunking import chunk_documents
from src.cleaning import clean_documents
from src.config import DATA_DIR, get_langsmith_status, load_settings
from src.evaluation import load_evaluation_questions, run_evaluation
from src.graph import run_synataric_graph
from src.loaders import load_documents
from src.rag_chain import build_sources_referenced
from src.sample_data import create_sample_corpus


st.set_page_config(page_title="Synataric Navigator", page_icon="S", layout="wide")

create_sample_corpus(DATA_DIR)
settings = load_settings(require_secrets=False)


THEME_CSS = """
<style>
:root {
    --syn-bg-0: #070721;
    --syn-bg-1: #10002b;
    --syn-card: rgba(65, 32, 130, 0.55);
    --syn-card-strong: rgba(33, 18, 82, 0.82);
    --syn-border: rgba(180, 140, 255, 0.25);
    --syn-heading: #ffffff;
    --syn-body: #d8d2ff;
    --syn-muted: #eeeaff;
    --syn-purple: #8b5cf6;
    --syn-coral: #ff4f5e;
    --syn-shadow: 0 12px 40px rgba(0,0,0,0.35);
}

html, body, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 10%, rgba(139, 92, 246, 0.28), transparent 34%),
        radial-gradient(circle at 88% 18%, rgba(255, 79, 94, 0.16), transparent 30%),
        linear-gradient(145deg, var(--syn-bg-0) 0%, var(--syn-bg-1) 48%, #050316 100%) !important;
    color: var(--syn-body);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(16, 0, 43, 0.98), rgba(7, 7, 33, 0.96));
    border-right: 1px solid var(--syn-border);
}

[data-testid="stSidebar"] * {
    color: var(--syn-body) !important;
}

.main .block-container {
    max-width: 1240px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4, h5, h6, .stMarkdown strong {
    color: var(--syn-heading) !important;
}

p, li, span, label, div {
    color: var(--syn-body);
}

[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
    color: #f4f1ff !important;
}

[data-testid="stMarkdownContainer"] p {
    color: #eeeaff;
}

.syn-hero {
    padding: 42px 42px 34px;
    border: 1px solid var(--syn-border);
    border-radius: 16px;
    background:
        linear-gradient(135deg, rgba(139, 92, 246, 0.34), rgba(255, 79, 94, 0.10)),
        rgba(65, 32, 130, 0.55);
    box-shadow: var(--syn-shadow);
    margin-bottom: 22px;
}

.syn-kicker {
    color: #ffb4bd;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    margin-bottom: 10px;
}

.syn-hero h1 {
    color: var(--syn-heading);
    font-size: clamp(2.4rem, 6vw, 4.8rem);
    line-height: 0.98;
    margin: 0 0 16px;
}

.syn-subtitle {
    color: var(--syn-body);
    font-size: 1.25rem;
    line-height: 1.6;
    max-width: 840px;
    margin: 0;
}

.syn-support {
    color: #f1edff;
    font-size: 1rem;
    margin-top: 16px;
}

.syn-feature-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin: 20px 0 26px;
}

.syn-card, .syn-glass, .syn-source-card {
    background: var(--syn-card);
    border: 1px solid var(--syn-border);
    border-radius: 16px;
    box-shadow: var(--syn-shadow);
    backdrop-filter: blur(16px);
}

.syn-feature-card {
    padding: 22px;
    min-height: 146px;
}

.syn-feature-card h3 {
    font-size: 1.08rem;
    margin: 0 0 10px;
}

.syn-feature-card p {
    color: #f1edff;
    margin: 0;
    line-height: 1.55;
}

.syn-panel {
    padding: 26px;
    margin: 18px 0;
}

.syn-section-title {
    color: var(--syn-heading);
    font-size: 1.55rem;
    font-weight: 800;
    margin: 28px 0 12px;
}

.syn-answer-card {
    padding: 26px;
    margin: 14px 0 22px;
    background:
        linear-gradient(145deg, rgba(65, 32, 130, 0.62), rgba(14, 10, 48, 0.76)),
        rgba(65, 32, 130, 0.55);
}

.syn-source-card {
    padding: 18px 20px;
    margin: 12px 0;
}

.syn-source-card code {
    display: inline-block;
    color: #efe7ff;
    background: rgba(9, 7, 34, 0.72);
    border: 1px solid rgba(180, 140, 255, 0.22);
    border-radius: 10px;
    padding: 8px 10px;
    white-space: pre-wrap;
    box-shadow: inset 0 0 0 1px rgba(139, 92, 246, 0.10);
}

code {
    color: #efe7ff !important;
    background: rgba(9, 7, 34, 0.72) !important;
    border-radius: 8px !important;
}

pre, [data-testid="stCodeBlock"] {
    background: rgba(9, 7, 34, 0.72) !important;
    border: 1px solid rgba(180, 140, 255, 0.22) !important;
    border-radius: 12px !important;
    color: #efe7ff !important;
}

[data-testid="stCodeBlock"] code {
    color: #efe7ff !important;
    background: transparent !important;
}

.syn-pipeline {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 12px;
    margin: 16px 0 24px;
}

.syn-pipeline-step {
    padding: 18px 14px;
    text-align: center;
    font-weight: 800;
    color: #fff;
    border-radius: 16px;
    border: 1px solid var(--syn-border);
    background: linear-gradient(145deg, rgba(139, 92, 246, 0.42), rgba(65, 32, 130, 0.45));
    box-shadow: var(--syn-shadow);
}

.syn-arrow {
    color: #ffb4bd;
    font-size: 1.25rem;
    font-weight: 900;
}

.syn-disclaimer {
    color: #f4f1ff;
    font-size: 0.88rem;
    padding: 18px 0 8px;
}

[data-testid="stMetric"] {
    background: rgba(65, 32, 130, 0.55);
    border: 1px solid var(--syn-border);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: var(--syn-shadow);
}

[data-testid="stMetricLabel"] p {
    color: #f4f1ff !important;
}

[data-testid="stMetricValue"] {
    color: var(--syn-heading) !important;
}

.stTextArea textarea, textarea, .stTextInput input {
    background: #ffffff !important;
    border: 1px solid var(--syn-border) !important;
    color: #141427 !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 34px rgba(0, 0, 0, 0.24), inset 0 0 0 1px rgba(139, 92, 246, 0.12);
}

.stSelectbox div[data-baseweb="select"] > div {
    background: rgba(8, 7, 34, 0.78) !important;
    border: 1px solid var(--syn-border) !important;
    color: #ffffff !important;
    border-radius: 16px !important;
    box-shadow: inset 0 0 0 1px rgba(139, 92, 246, 0.16);
}

.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] svg {
    color: #f4f1ff !important;
    fill: #f4f1ff !important;
}

.stTextArea textarea {
    min-height: 150px;
}

.stTextArea textarea::placeholder, textarea::placeholder, .stTextInput input::placeholder {
    color: #6f6a86 !important;
    opacity: 1 !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--syn-coral), #ff7a68) !important;
    border: 0 !important;
    border-radius: 16px !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    box-shadow: 0 14px 32px rgba(255, 79, 94, 0.26);
    padding: 0.75rem 1.2rem !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 42px rgba(255, 79, 94, 0.36);
}

[data-testid="stExpander"] {
    background: rgba(65, 32, 130, 0.45);
    border: 1px solid var(--syn-border);
    border-radius: 16px;
    box-shadow: var(--syn-shadow);
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    color: #ffffff !important;
    font-weight: 800;
}

[data-testid="stDataFrame"], .stDataFrame {
    border: 1px solid var(--syn-border);
    border-radius: 16px;
    overflow: hidden;
    box-shadow: var(--syn-shadow);
}

.stAlert {
    border-radius: 16px;
}

@media (max-width: 900px) {
    .syn-feature-grid, .syn-pipeline {
        grid-template-columns: 1fr;
    }
    .syn-hero {
        padding: 30px 24px;
    }
}
</style>
"""


st.markdown(THEME_CSS, unsafe_allow_html=True)


def _format_score(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _namespace(strategy: str) -> str:
    return settings.semantic_namespace if strategy == "semantic" else settings.fixed_namespace


def _safe(text) -> str:
    return html.escape(str(text or ""))


def _hero() -> None:
    st.markdown(
        """
        <section class="syn-hero">
            <div class="syn-kicker">Synataric Healthcare AI</div>
            <h1>Synataric Navigator</h1>
            <p class="syn-subtitle">Agentic AI Healthcare Concierge powered by LangChain + LangGraph RAG</p>
            <p class="syn-support">From question to grounded care-navigation answer in seconds.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _feature_cards() -> None:
    st.markdown(
        """
        <div class="syn-feature-grid">
            <div class="syn-card syn-feature-card">
                <h3>Grounded Healthcare Answers</h3>
                <p>Answers are generated from retrieved Synataric evidence with citations, not open-ended medical guessing.</p>
            </div>
            <div class="syn-card syn-feature-card">
                <h3>Provider &amp; Cost Intelligence</h3>
                <p>Compare curated provider profiles, procedure ranges, travel costs, and local planning details.</p>
            </div>
            <div class="syn-card syn-feature-card">
                <h3>Risk-Aware Care Planning</h3>
                <p>Surface recovery considerations, safety checklists, urgent-care reminders, and clinician questions.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_title(title: str) -> None:
    st.markdown(f'<div class="syn-section-title">{_safe(title)}</div>', unsafe_allow_html=True)


def _display_sources(sources: list[dict]) -> None:
    _section_title("Sources Used")
    if not sources:
        st.write("No sources returned.")
        return
    for source in sources:
        number = source.get("source_number")
        label = source.get("file_name") or source.get("source", "unknown")
        source_path = source.get("source_path") or ""
        retrieved_fact = source.get("retrieved_fact", "")
        st.markdown(
            f"""
            <div class="syn-source-card">
                <h3>[{_safe(number)}] {_safe(label)}</h3>
                <p><strong>Full source path</strong></p>
                <code>{_safe(source_path)}</code>
                <p style="margin-top: 14px;"><strong>Retrieved Fact</strong></p>
                <p>&quot;{_safe(retrieved_fact)}&quot;</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _display_reranking_table(result: dict) -> None:
    _section_title("Reranking Results")
    rows = result.get("reranking_results", [])
    if not rows:
        st.write("No reranking results returned.")
        return
    table = pd.DataFrame(rows)
    for column in ["Retrieval Score", "Rerank Score"]:
        if column in table.columns:
            table[column] = table[column].map(_format_score)
    st.dataframe(table, use_container_width=True)


def _display_evidence(result: dict) -> None:
    _section_title("Retrieved Evidence")
    docs = result.get("reranked_docs", [])
    if not docs:
        st.write("No reranked evidence returned.")
        return
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        with st.expander(f"Retrieved Evidence #{i}: {source}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Source:** `{source}`")
                st.markdown(f"**Chunk Strategy:** `{doc.metadata.get('chunk_strategy', 'unknown')}`")
                st.markdown(f"**Document Type:** `{doc.metadata.get('doc_type', 'unknown')}`")
            with col2:
                st.markdown(f"**Similarity Score:** `{_format_score(doc.metadata.get('similarity_score'))}`")
                st.markdown(f"**Rerank Score:** `{_format_score(doc.metadata.get('rerank_score'))}`")
                st.markdown(f"**Final Rank:** `{doc.metadata.get('final_rank', i)}`")
            st.markdown("**Full Source Path**")
            st.code(doc.metadata.get("source_path", "") or "See Sources Used section", language="text")
            st.markdown("**Metadata**")
            st.json(doc.metadata)
            st.markdown("**Chunk Text**")
            st.write(doc.page_content)


def _run_question(question: str, strategy: str, top_k: int) -> dict:
    return run_synataric_graph(question.strip(), namespace=_namespace(strategy), top_k=top_k)


def _pipeline_cards() -> None:
    st.markdown(
        """
        <div class="syn-pipeline">
            <div class="syn-pipeline-step">Question</div>
            <div class="syn-pipeline-step">Embedding</div>
            <div class="syn-pipeline-step">Pinecone Retrieval</div>
            <div class="syn-pipeline-step">FlashRank Reranking</div>
            <div class="syn-pipeline-step">GPT Answer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _answer_card(answer: str) -> None:
    _section_title("Care Navigation Answer")
    st.markdown('<div class="syn-glass syn-answer-card">', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown("</div>", unsafe_allow_html=True)


with st.sidebar:
    st.title("Synataric Navigator")
    page = st.radio(
        "Page",
        ["Ask Navigator", "RAG Diagnostics", "Evaluation Dashboard", "Chunk Strategy Comparison"],
    )
    st.divider()
    st.caption(get_langsmith_status())
    strategy = st.selectbox("Namespace", ["semantic", "fixed"], index=0)
    top_k = st.slider("Candidates for reranking", min_value=3, max_value=20, value=10, step=1)
    show_samples = st.button("Show sample questions")
    st.info("Healthcare navigation education only. Not medical advice.")

_hero()
_feature_cards()

if show_samples:
    sample_path = DATA_DIR / "sample_questions.csv"
    if sample_path.exists():
        _section_title("Sample Questions")
        st.dataframe(pd.read_csv(sample_path), use_container_width=True)

if page == "Ask Navigator":
    st.markdown('<div class="syn-card syn-panel">', unsafe_allow_html=True)
    _section_title("Plan Your Care Journey")
    question = st.text_area(
        "Enter a procedure, destination, provider, cost, recovery, or risk-planning question.",
        placeholder="Example: Compare cataract surgery costs, provider options, travel stay, and follow-up planning in Bangalore.",
        height=160,
    )
    generate = st.button("Generate Care Navigation Answer", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if generate:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Retrieving, reranking, and generating a cited answer..."):
                    # Data -> Chunk -> Embed -> Store happens in ingest.py.
                    # Retrieve -> Rerank -> Generate happens in this LangGraph workflow.
                    result = _run_question(question, strategy, top_k)
                _answer_card(result["answer"])
                _display_reranking_table(result)
                _display_sources(result.get("sources", []))
                with st.expander("Sources Referenced - raw citation block"):
                    st.text(build_sources_referenced(result.get("sources", [])))
                _display_evidence(result)
            except Exception as exc:
                st.error(str(exc))

elif page == "RAG Diagnostics":
    _section_title("RAG Diagnostics")
    _pipeline_cards()

    _section_title("Corpus Statistics")
    try:
        docs = load_documents()
        cleaned = clean_documents(docs)
        fixed_chunks = chunk_documents(cleaned, strategy="fixed")
        semantic_chunks = chunk_documents(cleaned, strategy="semantic")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents Loaded", len(docs))
        col2.metric("Chunks Indexed", len(fixed_chunks) + len(semantic_chunks))
        col3.metric("Fixed Chunks", len(fixed_chunks))
        col4.metric("Semantic Chunks", len(semantic_chunks))
    except Exception as exc:
        st.error(f"Could not compute corpus statistics: {exc}")

    _section_title("Retrieval Statistics")
    col1, col2 = st.columns(2)
    col1.metric("Top K", top_k)
    col2.metric("Namespace", _namespace(strategy))

    st.markdown('<div class="syn-card syn-panel">', unsafe_allow_html=True)
    diagnostic_question = st.text_input("Diagnostics question", "What is the cost of cataract surgery in Bangalore?")
    run_diagnostics = st.button("Run Retrieval Diagnostics")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_diagnostics:
        try:
            result = _run_question(diagnostic_question, strategy, top_k)
            retrieved = result.get("retrieved_docs", [])
            scores = [doc.metadata.get("retrieval_score") for doc in retrieved if doc.metadata.get("retrieval_score") is not None]
            col1, col2 = st.columns(2)
            col1.metric("Retrieved Chunks", len(retrieved))
            col2.metric("Average Similarity Score", _format_score(sum(scores) / len(scores) if scores else None))
            _display_reranking_table(result)
            _display_evidence(result)
        except Exception as exc:
            st.error(str(exc))

elif page == "Evaluation Dashboard":
    _section_title("Evaluation Dashboard")
    questions = load_evaluation_questions()
    st.dataframe(questions, use_container_width=True)
    limit = st.slider("Questions to run", min_value=1, max_value=len(questions), value=min(3, len(questions)))
    if st.button("Run Evaluation"):
        try:
            with st.spinner("Running evaluation questions through Retrieval -> Rerank -> Answer..."):
                results, metrics = run_evaluation(namespace=_namespace(strategy), top_k=top_k, limit=limit)
            _section_title("RAG Quality Metrics")
            cols = st.columns(4)
            if "faithfulness" in metrics:
                cols[0].metric("Faithfulness", metrics["faithfulness"])
                cols[1].metric("Answer Relevancy", metrics["answer_relevancy"])
                cols[2].metric("Context Precision", metrics["ragas_context_precision"])
                cols[3].metric("Context Recall", metrics["ragas_context_recall"])
                st.caption(
                    f"Local retrieval hit rate: {metrics['retrieval_hit_rate']} | "
                    f"Source coverage: {metrics['source_coverage']}"
                )
            else:
                cols[0].metric("Retrieval Hit Rate", metrics["retrieval_hit_rate"])
                cols[1].metric("Source Coverage", metrics["source_coverage"])
                cols[2].metric("Context Precision", metrics["context_precision"])
                cols[3].metric("Context Recall", metrics["context_recall"])
            st.caption(metrics.get("ragas_status", ""))
            _section_title("Evaluation Results")
            st.dataframe(results, use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

elif page == "Chunk Strategy Comparison":
    st.markdown('<div class="syn-card syn-panel">', unsafe_allow_html=True)
    _section_title("Chunk Strategy Comparison")
    question = st.text_area(
        "Question for fixed vs semantic comparison",
        placeholder="Example: Which hospitals are listed for cataract surgery planning in Bangalore?",
        height=130,
    )
    compare = st.button("Compare Chunk Strategies", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if compare:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Running fixed and semantic workflows..."):
                    fixed = _run_question(question, "fixed", top_k)
                    semantic = _run_question(question, "semantic", top_k)
                st.markdown(f"**Question:** {question}")
                col1, col2 = st.columns(2)
                with col1:
                    _section_title("Fixed Answer")
                    st.markdown('<div class="syn-glass syn-answer-card">', unsafe_allow_html=True)
                    st.markdown(fixed["answer"])
                    st.markdown("</div>", unsafe_allow_html=True)
                    _display_reranking_table(fixed)
                    _display_sources(fixed.get("sources", []))
                with col2:
                    _section_title("Semantic Answer")
                    st.markdown('<div class="syn-glass syn-answer-card">', unsafe_allow_html=True)
                    st.markdown(semantic["answer"])
                    st.markdown("</div>", unsafe_allow_html=True)
                    _display_reranking_table(semantic)
                    _display_sources(semantic.get("sources", []))
            except Exception as exc:
                st.error(str(exc))

st.divider()
st.markdown(
    """
    <div class="syn-disclaimer">
        Synataric Navigator does not diagnose, prescribe treatment, or replace licensed clinicians.
        For urgent symptoms or emergencies, seek immediate medical care.
    </div>
    """,
    unsafe_allow_html=True,
)

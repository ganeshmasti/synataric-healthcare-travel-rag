import html
import time

import pandas as pd
import streamlit as st

from src.chunking import chunk_documents
from src.cleaning import clean_documents
from src.config import DATA_DIR, get_langsmith_status, load_settings
from src.evaluation import load_evaluation_questions, run_evaluation
from src.evidence_locator import locate_evidence
from src.graph import run_synataric_graph
from src.loaders import load_documents
from src.rag_chain import build_sources_referenced
from src.sample_data import create_sample_corpus


st.set_page_config(page_title="Synataric Navigator", page_icon="S", layout="wide")

create_sample_corpus(DATA_DIR)
settings = load_settings(require_secrets=False)


def escape_html(text) -> str:
    return html.escape(str(text or ""))


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #05051f;
            --card: #1d1247;
            --section: #120a35;
            --purple: #a855f7;
            --cyan: #22d3ee;
            --coral: #ff4f5e;
            --green: #22c55e;
            --white: #ffffff;
            --text2: #d8d2ff;
            --muted: #9f95d0;
            --border: rgba(168, 85, 247, 0.34);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 14% 0%, rgba(168, 85, 247, 0.20), transparent 30%),
                radial-gradient(circle at 90% 5%, rgba(34, 211, 238, 0.12), transparent 26%),
                linear-gradient(145deg, #05051f, #09062a 45%, #18002f) !important;
        }

        [data-testid="stHeader"] { background: transparent; }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(18, 10, 53, 0.98), rgba(5, 5, 31, 0.98));
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * { color: var(--text2) !important; }

        .main .block-container {
            max-width: 1280px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4, h5, h6, strong { color: var(--white) !important; }
        p, li, span, label, div { color: var(--text2); }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(18, 10, 53, 0.72) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.26);
        }

        [data-testid="stMetric"] {
            background: rgba(5, 5, 31, 0.34);
            border: 1px solid rgba(168, 85, 247, 0.18);
            border-radius: 12px;
            padding: 0.72rem;
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-size: 0.78rem !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--white) !important;
            font-size: 1.05rem !important;
        }

        .stButton > button {
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            background: linear-gradient(135deg, var(--coral), #ff806b) !important;
            color: white !important;
            font-weight: 800 !important;
            box-shadow: 0 10px 24px rgba(255, 79, 94, 0.22);
        }

        .stTextArea textarea, .stTextInput input {
            background: rgba(255,255,255,0.97) !important;
            color: #111827 !important;
            border-radius: 10px !important;
            border: 1px solid rgba(168, 85, 247, 0.34) !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] div {
            background-color: rgba(18, 10, 53, 0.98) !important;
            color: var(--text2) !important;
            border-color: rgba(168, 85, 247, 0.42) !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] svg {
            color: var(--text2) !important;
            fill: var(--text2) !important;
        }

        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        [role="listbox"] {
            background-color: #120a35 !important;
            color: var(--text2) !important;
            border: 1px solid rgba(168, 85, 247, 0.42) !important;
        }

        [data-baseweb="menu"] li,
        [role="option"] {
            background-color: #120a35 !important;
            color: var(--text2) !important;
        }

        [data-baseweb="menu"] li:hover,
        [role="option"]:hover,
        [aria-selected="true"] {
            background-color: rgba(168, 85, 247, 0.24) !important;
            color: var(--white) !important;
        }

        [data-testid="stCheckbox"] label:hover,
        [data-testid="stCheckbox"] label:hover div,
        [data-testid="stCheckbox"] label:hover span {
            background-color: transparent !important;
            color: var(--white) !important;
        }

        [data-testid="stTooltipContent"],
        [data-testid="stTooltipContent"] div,
        [data-testid="stTooltipContent"] p {
            background-color: #120a35 !important;
            color: var(--text2) !important;
            border-color: rgba(168, 85, 247, 0.42) !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }

        [data-testid="stExpander"] {
            background: rgba(18, 10, 53, 0.54);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        .syn-header {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 14px;
            align-items: center;
            margin-bottom: 12px;
        }

        .syn-title-block, .syn-feature, .syn-stage-card {
            background: linear-gradient(145deg, rgba(35,25,87,0.90), rgba(22,16,62,0.90));
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
        }

        .syn-title-block {
            padding: 16px 18px;
        }

        .syn-kicker {
            color: var(--cyan);
            font-size: 0.78rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .syn-title {
            color: var(--white);
            font-size: clamp(1.8rem, 3vw, 2.7rem);
            line-height: 1.05;
            font-weight: 950;
            margin-top: 4px;
        }

        .syn-subtitle {
            color: var(--text2);
            font-size: 0.95rem;
            margin-top: 5px;
        }

        .syn-feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(130px, 1fr));
            gap: 10px;
        }

        .syn-feature {
            padding: 13px;
            min-height: 88px;
        }

        .syn-feature h4 {
            margin: 0;
            color: var(--white);
            font-size: 0.93rem;
        }

        .syn-feature p {
            margin: 6px 0 0;
            color: var(--muted);
            line-height: 1.35;
            font-size: 0.78rem;
        }

        .syn-stage-card {
            padding: 12px;
            min-height: 126px;
        }

        .syn-stage-card.pending { border-color: rgba(159,149,208,0.26); }
        .syn-stage-card.running {
            border-color: var(--cyan);
            box-shadow: 0 0 26px rgba(34, 211, 238, 0.18);
        }
        .syn-stage-card.complete { border-color: rgba(34, 197, 94, 0.54); }

        .syn-stage-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }

        .syn-stage-icon {
            width: 32px;
            height: 32px;
            display: grid;
            place-items: center;
            border-radius: 10px;
            background: rgba(168, 85, 247, 0.22);
            color: var(--white);
            font-weight: 900;
        }

        .syn-badge {
            border-radius: 999px;
            padding: 4px 8px;
            font-size: 0.68rem;
            font-weight: 900;
            background: rgba(159,149,208,0.16);
            color: var(--text2);
        }

        .syn-badge.running { background: rgba(34, 211, 238, 0.15); color: #aaf4ff; }
        .syn-badge.complete { background: rgba(34, 197, 94, 0.16); color: #bbf7d0; }

        .syn-stage-title {
            color: var(--white);
            font-weight: 900;
            font-size: 0.94rem;
            margin-top: 9px;
        }

        .syn-stage-desc {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.35;
            margin-top: 4px;
        }

        .syn-stage-value {
            color: var(--cyan);
            font-size: 0.76rem;
            font-weight: 800;
            margin-top: 7px;
            overflow-wrap: anywhere;
        }

        .syn-section-label {
            color: #ffb4bd;
            font-size: 0.86rem;
            font-weight: 950;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 16px 0 8px;
        }

        @media (max-width: 1050px) {
            .syn-header { grid-template-columns: 1fr; }
            .syn-feature-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _namespace(strategy: str) -> str:
    return settings.semantic_namespace if strategy == "semantic" else settings.fixed_namespace


def _langsmith_enabled() -> bool:
    return bool(settings.langchain_api_key and settings.langchain_tracing_v2.lower() == "true")


def _safe(text) -> str:
    return escape_html(text)


def _format_score(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_relevance(value, strongest: float | None = None) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if strongest:
        return f"{score / strongest * 100:.0f}%"
    if score <= 1:
        return f"{score * 100:.0f}%"
    return f"{score:.0f}%"


def _source_label(source: dict) -> str:
    return str(source.get("file_name") or source.get("source") or "unknown")


def _short(text: str, limit: int = 360) -> str:
    value = str(text or "")
    return value if len(value) <= limit else f"{value[:limit].rstrip()}..."


def _init_state() -> None:
    defaults = {
        "latest_question": "",
        "latest_result": None,
        "execution_log": [],
        "pipeline_statuses": {},
        "evaluation_results": None,
        "evaluation_metrics": None,
        "comparison_results": None,
        "evidence_locations": None,
        "evidence_query": "",
        "query_text": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    st.markdown(
        """
        <div class="syn-header">
            <div class="syn-title-block">
                <div class="syn-kicker">Synataric Healthcare AI</div>
                <div class="syn-title">Synataric Navigator</div>
                <div class="syn-subtitle">Agentic AI Healthcare Concierge powered by LangChain + LangGraph RAG</div>
            </div>
            <div class="syn-feature-grid">
                <div class="syn-feature"><h4>Grounded Healthcare Answers</h4><p>Retrieved evidence and source citations.</p></div>
                <div class="syn-feature"><h4>Provider &amp; Cost Intelligence</h4><p>Procedure, travel, provider, and stay context.</p></div>
                <div class="syn-feature"><h4>Risk-Aware Planning</h4><p>Recovery, safety, and clinician questions.</p></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_cards() -> None:
    render_header()


STAGES = [
    {"key": "data", "number": 1, "icon": "DS", "title": "Data Sources", "description": "Curated healthcare corpus", "group": "offline"},
    {"key": "chunk", "number": 2, "icon": "CH", "title": "Chunking", "description": "Fixed and semantic chunks", "group": "offline"},
    {"key": "embed_store", "number": 3, "icon": "EM", "title": "Embeddings", "description": settings.embedding_model, "group": "offline"},
    {"key": "store", "number": 4, "icon": "VS", "title": "Pinecone Store", "description": "Vector namespaces", "group": "offline"},
    {"key": "question", "number": 5, "icon": "Q", "title": "User Question", "description": "Care navigation query", "group": "online"},
    {"key": "query_embed", "number": 6, "icon": "QE", "title": "Query Embedding", "description": "Semantic search vector", "group": "online"},
    {"key": "search", "number": 7, "icon": "PC", "title": "Pinecone Similarity Search", "description": "Find candidate chunks", "group": "online"},
    {"key": "candidates", "number": 8, "icon": "DOC", "title": "Retrieved Candidate Chunks", "description": "Top-K context", "group": "online"},
    {"key": "rerank", "number": 9, "icon": "FR", "title": "FlashRank Reranking", "description": "Re-score relevance", "group": "generation"},
    {"key": "evidence", "number": 10, "icon": "EV", "title": "Top Evidence Selected", "description": "Strongest chunks", "group": "generation"},
    {"key": "prompt", "number": 11, "icon": "PB", "title": "Prompt Assembly", "description": "Evidence + safety rules", "group": "generation"},
    {"key": "gpt", "number": 12, "icon": "AI", "title": "GPT Answer Generation", "description": settings.chat_model, "group": "generation"},
    {"key": "response", "number": 13, "icon": "OK", "title": "Grounded Response", "description": "Answer + citations", "group": "generation"},
]


def _default_statuses() -> dict:
    statuses = {}
    for stage in STAGES:
        statuses[stage["key"]] = "Complete" if stage["group"] == "offline" else "Pending"
    return statuses


def render_pipeline_card(stage: dict, status: str, value: str = "") -> None:
    css_status = status.lower()
    status_label = "Running" if status == "Running" else "Complete" if status == "Complete" else "Pending"
    spinner = " . . ." if status == "Running" else ""
    html_block = f"""
    <div class="syn-stage-card {css_status}">
        <div class="syn-stage-top">
            <div class="syn-stage-icon">{escape_html(stage["icon"])}</div>
            <div class="syn-badge {css_status}">{escape_html(status_label)}{spinner}</div>
        </div>
        <div class="syn-stage-title">{stage["number"]}. {escape_html(stage["title"])}</div>
        <div class="syn-stage-desc">{escape_html(stage["description"])}</div>
        <div class="syn-stage-value">{escape_html(value)}</div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


def _stage_values(result: dict | None, question: str, namespace: str, top_k: int) -> dict:
    sources = result.get("sources", []) if result else []
    top_source = _source_label(sources[0]) if sources else ""
    return {
        "data": "Markdown + CSV",
        "chunk": "Fixed + semantic",
        "embed_store": settings.embedding_model,
        "store": namespace,
        "question": _short(question, 80) if question else "",
        "query_embed": "Ready" if question else "",
        "search": f"Top {top_k}" if question else "",
        "candidates": str(len(result.get("retrieved_docs", []))) if result else "",
        "rerank": "FlashRank" if result else "",
        "evidence": f'{len(result.get("reranked_docs", []))} selected' if result else "",
        "prompt": "Grounded context" if result else "",
        "gpt": settings.chat_model if result else "",
        "response": top_source if result else "",
    }


def render_pipeline_row(label: str, stages: list[dict], statuses: dict, values: dict) -> None:
    st.markdown(f'<div class="syn-section-label">{escape_html(label)}</div>', unsafe_allow_html=True)
    columns = st.columns(len(stages))
    for column, stage in zip(columns, stages):
        with column:
            render_pipeline_card(stage, statuses.get(stage["key"], "Pending"), values.get(stage["key"], ""))


def render_execution_pipeline(result: dict | None, question: str, namespace: str, top_k: int, statuses: dict | None = None) -> None:
    statuses = statuses or _default_statuses()
    values = _stage_values(result, question, namespace, top_k)
    render_pipeline_row("Offline Prep: Data -> Chunk -> Embed -> Store", [stage for stage in STAGES if stage["group"] == "offline"], statuses, values)
    render_pipeline_row("Online Retrieval: Question -> Search -> Candidate Chunks", [stage for stage in STAGES if stage["group"] == "online"], statuses, values)
    render_pipeline_row("Grounded Generation: Rerank -> Evidence -> Prompt -> LLM -> Answer", [stage for stage in STAGES if stage["group"] == "generation"], statuses, values)


def render_execution_log(log_lines: list[str]) -> None:
    with st.container(border=True):
        st.markdown("**Execution Log**")
        if log_lines:
            st.code("\n".join(log_lines), language="text")
        else:
            st.caption("Ask a care navigation question to start the RAG pipeline.")


def render_runtime_metrics(result: dict | None, namespace: str, top_k: int) -> None:
    columns = st.columns(4)
    columns[0].metric("Current Stage", "Complete" if result else "Ready")
    columns[1].metric("Namespace", namespace)
    columns[2].metric("Top-K", top_k)
    columns[3].metric("Model", settings.chat_model)
    columns = st.columns(4)
    columns[0].metric("Retrieved Chunks", len(result.get("retrieved_docs", [])) if result else "pending")
    columns[1].metric("Reranked Chunks", len(result.get("reranked_docs", [])) if result else "pending")
    columns[2].metric("Sources Used", len(result.get("sources", [])) if result else "pending")
    columns[3].metric("LangSmith", "Enabled" if _langsmith_enabled() else "Disabled")


def render_query_panel() -> tuple[str, bool]:
    with st.container(border=True):
        st.markdown("**Ask Navigator**")
        sample_questions = [
            "What is the cost of cataract surgery in Bangalore?",
            "Which eye hospitals are listed in Bangalore?",
            "What recovery guidance is available after cataract surgery?",
            "What urgent symptoms require immediate care?",
        ]
        sample_columns = st.columns(4)
        for index, sample in enumerate(sample_questions):
            if sample_columns[index].button(sample, key=f"sample_question_{index}", use_container_width=True):
                st.session_state.query_text = sample
        question = st.text_area(
            "Care navigation question",
            key="query_text",
            placeholder="Ask about procedure costs, providers, recovery, travel planning, or risk checks.",
            height=120,
        )
        generate = st.button("Generate Care Navigation Answer", type="primary", use_container_width=True)
    return question, generate


def _stamp(seconds: int, message: str) -> str:
    return f"00:{seconds:02d} OK {message}"


def _set_running(statuses: dict, running_key: str) -> dict:
    next_statuses = dict(statuses)
    keys_seen = []
    for stage in STAGES:
        key = stage["key"]
        keys_seen.append(key)
        if stage["group"] == "offline":
            next_statuses[key] = "Complete"
        elif key == running_key:
            next_statuses[key] = "Running"
            break
        else:
            next_statuses[key] = "Complete"
    for stage in STAGES:
        if stage["key"] not in keys_seen and stage["group"] != "offline":
            next_statuses[stage["key"]] = "Pending"
    return next_statuses


def _complete_all(statuses: dict) -> dict:
    return {stage["key"]: "Complete" for stage in STAGES}


def run_question(question: str, namespace: str, top_k: int) -> dict:
    return run_synataric_graph(question.strip(), namespace=namespace, top_k=top_k)


def _run_with_pipeline(question: str, namespace: str, top_k: int) -> dict:
    panel = st.empty()
    logs: list[str] = []
    statuses = _default_statuses()

    def refresh(running_key: str, message: str, second: int, result: dict | None = None) -> None:
        nonlocal statuses
        statuses = _set_running(statuses, running_key)
        logs.append(_stamp(second, message))
        with panel.container():
            render_execution_pipeline(result, question, namespace, top_k, statuses)
            render_runtime_metrics(result, namespace, top_k)
            render_execution_log(logs)
        time.sleep(0.2)

    refresh("question", "Question received", 0)
    refresh("query_embed", "Query embedded for semantic search", 1)
    refresh("search", f"Searching Pinecone namespace: {namespace}", 2)

    result = run_question(question, namespace, top_k)

    logs.append(_stamp(3, f'Retrieved {len(result.get("retrieved_docs", []))} candidate chunks'))
    logs.append(_stamp(4, "Running FlashRank reranker"))
    logs.append(_stamp(5, f'Selected {len(result.get("reranked_docs", []))} evidence chunks'))
    logs.append(_stamp(6, "Building grounded prompt with safety rules"))
    logs.append(_stamp(7, f"Calling {settings.chat_model}"))
    logs.append(_stamp(8, "Final answer generated"))
    logs.append(_stamp(9, "Sources and evidence rendered"))
    logs.append(_stamp(10, "LangSmith trace captured"))
    statuses = _complete_all(statuses)
    with panel.container():
        render_execution_pipeline(result, question, namespace, top_k, statuses)
        render_runtime_metrics(result, namespace, top_k)
        render_execution_log(logs)
    return result


def render_answer(result: dict | None) -> None:
    if not result:
        st.info("Ask a care navigation question to start the RAG pipeline.")
        return
    with st.container(border=True):
        st.markdown("### Care Navigation Answer")
        st.markdown(result.get("answer", ""))
        st.caption("Educational healthcare navigation only. Not medical advice. Does not diagnose or prescribe.")


def render_sources(result: dict | None) -> None:
    if not result:
        return
    sources = result.get("sources", [])
    if not sources:
        st.info("No sources returned.")
        return
    for source in sources:
        with st.container(border=True):
            st.markdown(f"**[{source.get('source_number', '')}] {_source_label(source)}**")
            source_columns = st.columns(4)
            source_columns[0].metric("Category", source.get("category", "N/A"))
            source_columns[1].metric("Doc Type", source.get("doc_type", "N/A"))
            source_columns[2].metric("Strategy", source.get("chunk_strategy", "N/A"))
            source_columns[3].metric("Citation", source.get("source_number", "N/A"))
            st.caption(_short(source.get("retrieved_fact", ""), 500))
            if source.get("source_path"):
                with st.expander("Full source path"):
                    st.code(str(source["source_path"]), language="text")


def render_evidence(result: dict | None) -> None:
    if not result:
        return
    docs = result.get("reranked_docs", [])
    if not docs:
        st.info("No reranked evidence returned.")
        return
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        with st.expander(f"Evidence #{index} - {metadata.get('source', 'unknown')}", expanded=index == 1):
            columns = st.columns(5)
            columns[0].metric("Retrieval Score", _format_score(metadata.get("similarity_score", metadata.get("retrieval_score"))))
            columns[1].metric("Rerank Score", _format_score(metadata.get("rerank_score")))
            columns[2].metric("Rerank Relevance", _format_relevance(metadata.get("rerank_score")))
            columns[3].metric("Category", metadata.get("category", "N/A"))
            columns[4].metric("Chunk Strategy", metadata.get("chunk_strategy", "N/A"))
            if metadata.get("intent_boost"):
                st.caption(
                    f"Base FlashRank score {_format_score(metadata.get('base_rerank_score'))} "
                    f"+ intent boost {_format_score(metadata.get('intent_boost'))}."
                )
            st.write(doc.page_content)


def render_reranking_results(result: dict | None) -> None:
    if not result:
        return
    rows = result.get("reranking_results", [])
    if not rows:
        st.info("No reranking results returned.")
        return
    scores = []
    for row in rows:
        value = row.get("Final Rerank Score", row.get("Raw Rerank Score", row.get("Rerank Score", row.get("rerank_score"))))
        try:
            scores.append(float(value))
        except (TypeError, ValueError):
            scores.append(0.0)
    strongest = max(scores) if scores else 0.0
    table_rows = []
    for index, row in enumerate(rows, start=1):
        source = row.get("Source") or row.get("source") or row.get("file_name") or "unknown"
        retrieval = row.get("Retrieval Score", row.get("retrieval_score"))
        raw_rerank = row.get("Raw Rerank Score", row.get("base_rerank_score", row.get("Rerank Score", row.get("rerank_score"))))
        intent_boost = row.get("Intent Boost", row.get("intent_boost"))
        final_rerank = row.get("Final Rerank Score", row.get("rerank_score", raw_rerank))
        relevance = row.get("Rerank Relevance") or _format_relevance(final_rerank, strongest)
        table_rows.append(
            {
                "Rank": index,
                "Pinecone Similarity": _format_score(retrieval),
                "FlashRank Score": _format_score(raw_rerank),
                "Intent Boost": _format_score(intent_boost),
                "Final Score": _format_score(final_rerank),
                "Relevance": relevance,
                "Source": _short(source, 70),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.caption("Final Score combines FlashRank relevance with any domain-intent boost. Relevance is relative to the strongest reranked chunk, not answer accuracy.")


def render_diagnostics_page(strategy: str, top_k: int) -> None:
    render_header()
    namespace = _namespace(strategy)
    result = st.session_state.latest_result
    render_execution_pipeline(result, st.session_state.latest_question, namespace, top_k, _complete_all(_default_statuses()) if result else _default_statuses())

    st.markdown("### Corpus Statistics")
    try:
        docs = load_documents()
        cleaned = clean_documents(docs)
        fixed = chunk_documents(cleaned, strategy="fixed")
        semantic = chunk_documents(cleaned, strategy="semantic")
        columns = st.columns(4)
        columns[0].metric("Documents Loaded", len(docs))
        columns[1].metric("Fixed Chunks", len(fixed))
        columns[2].metric("Semantic Chunks", len(semantic))
        columns[3].metric("Total Chunks", len(fixed) + len(semantic))
    except Exception as exc:
        st.error(f"Could not compute corpus statistics: {exc}")

    st.markdown("### LangGraph Workflow")
    st.write("START -> retrieve_node -> rerank_node -> generate_node -> END")
    render_runtime_metrics(result, namespace, top_k)
    st.info(
        "LangChain provides RAG components. LangGraph orchestrates workflow state. "
        "Pinecone stores embeddings. FlashRank improves evidence quality. RAGAS evaluates answer quality."
    )


def render_evaluation_scope(strategy: str, top_k: int) -> None:
    questions = load_evaluation_questions()
    categories = sorted(questions["category"].dropna().unique().tolist())
    category_label = ", ".join(categories)
    namespace = _namespace(strategy)

    st.markdown("#### What This Evaluates")
    st.write(
        "This runs the saved benchmark questions through the same app pipeline: retrieval, FlashRank reranking, "
        "grounded answer generation, and source collection."
    )

    columns = st.columns(4)
    columns[0].metric("Evaluation Questions", len(questions))
    columns[1].metric("Categories", len(categories))
    columns[2].metric("Namespace", namespace)
    columns[3].metric("Top-K Candidates", top_k)

    st.markdown("#### Judging Criteria")
    st.write(
        "Local metrics compare expected source files and expected-answer keywords against the generated answer. "
        "RAGAS metrics use the question, answer, retrieved contexts, and reference answer to judge faithfulness, "
        "answer relevancy, context precision, and context recall."
    )
    st.caption(f"Evaluation categories: {category_label}")

    with st.expander("Benchmark questions"):
        st.dataframe(questions, use_container_width=True, hide_index=True)


def render_evaluation_results_table(results: pd.DataFrame | None) -> None:
    if results is None or results.empty:
        return

    display = results.copy()
    for column in ["expected_sources", "retrieved_sources"]:
        if column in display.columns:
            display[column] = display[column].apply(lambda values: ", ".join(values or []))

    columns = [
        "question",
        "category",
        "expected_answer",
        "generated_answer",
        "expected_sources",
        "retrieved_sources",
    ]
    columns = [column for column in columns if column in display.columns]
    st.markdown("#### Evaluation Runs")
    st.dataframe(display[columns], use_container_width=True, hide_index=True)


def render_evaluation_dashboard(strategy: str, top_k: int) -> None:
    render_header()
    st.markdown("### Evaluation Dashboard")
    render_evaluation_scope(strategy, top_k)
    if st.button("Run Evaluation", type="primary"):
        try:
            with st.spinner("Running evaluation..."):
                results, metrics = run_evaluation(namespace=_namespace(strategy), top_k=top_k)
            st.session_state.evaluation_results = results
            st.session_state.evaluation_metrics = metrics
        except Exception as exc:
            st.error(str(exc))

    metrics = st.session_state.evaluation_metrics
    if not metrics:
        st.info("Run Evaluation to calculate local and RAGAS metrics.")
        return
    columns = st.columns(4)
    columns[0].metric("Retrieval Hit Rate", metrics.get("retrieval_hit_rate", "N/A"))
    columns[1].metric("Source Coverage", metrics.get("source_coverage", "N/A"))
    columns[2].metric("Local Context Precision", metrics.get("context_precision", "N/A"))
    columns[3].metric("Local Context Recall", metrics.get("context_recall", "N/A"))
    columns = st.columns(4)
    columns[0].metric("RAGAS Faithfulness", metrics.get("faithfulness", "N/A"))
    columns[1].metric("RAGAS Answer Relevancy", metrics.get("answer_relevancy", "N/A"))
    columns[2].metric("RAGAS Context Precision", metrics.get("ragas_context_precision", "N/A"))
    columns[3].metric("RAGAS Context Recall", metrics.get("ragas_context_recall", "N/A"))
    st.caption(metrics.get("ragas_status", ""))
    if st.session_state.evaluation_results is not None:
        render_evaluation_results_table(st.session_state.evaluation_results)


def render_chunk_strategy_comparison(top_k: int) -> None:
    render_header()
    st.markdown("### Chunk Strategy Comparison")
    default_question = st.session_state.latest_question
    question = st.text_area(
        "Question for fixed vs semantic comparison",
        value=default_question,
        placeholder="Example: Which hospitals are listed for cataract surgery planning in Bangalore?",
        height=120,
    )
    if not st.button("Compare Fixed vs Semantic", type="primary"):
        st.info("Run a question first to compare fixed vs semantic chunking.")
        return
    if not question.strip():
        st.warning("Enter a question first.")
        return
    try:
        with st.spinner("Running comparison..."):
            fixed = run_question(question, settings.fixed_namespace, top_k)
            semantic = run_question(question, settings.semantic_namespace, top_k)
        st.session_state.comparison_results = {"fixed": fixed, "semantic": semantic}
    except Exception as exc:
        st.error(str(exc))
        return

    results = st.session_state.comparison_results
    columns = st.columns(2)
    for column, title, result in [
        (columns[0], "Fixed Chunking", results["fixed"]),
        (columns[1], "Semantic Chunking", results["semantic"]),
    ]:
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(_short(result.get("answer", ""), 900))
            metric_columns = st.columns(3)
            metric_columns[0].metric("Retrieved", len(result.get("retrieved_docs", [])))
            metric_columns[1].metric("Reranked", len(result.get("reranked_docs", [])))
            metric_columns[2].metric("Sources", len(result.get("sources", [])))
            st.caption(", ".join(_source_label(source) for source in result.get("sources", [])) or "No sources")


def render_find_evidence_page(strategy: str, top_k: int) -> None:
    render_header()
    st.markdown("### Find Evidence")
    st.write("Ask where a topic is explained and get the exact source, matching snippet, parent context, and retrieval reason.")

    question = st.text_area(
        "Topic or question to locate",
        key="evidence_query",
        placeholder="Example: Where is cataract recovery planning explained?",
        height=110,
    )
    namespace = _namespace(strategy)

    if st.button("Find Evidence", type="primary"):
        if not question.strip():
            st.warning("Enter a topic or question first.")
        else:
            try:
                with st.spinner("Finding the strongest evidence..."):
                    st.session_state.evidence_locations = locate_evidence(question, namespace=namespace, top_k=top_k)
            except Exception as exc:
                st.error(str(exc))
                return

    locations = st.session_state.evidence_locations
    if not locations:
        st.info("Search for a topic to locate the most relevant evidence chunks.")
        return

    st.markdown("#### Best Evidence Matches")
    for location in locations:
        rank = location.get("rank", "N/A")
        source = location.get("file_name") or location.get("source", "unknown")
        with st.container(border=True):
            st.markdown(f"**#{rank} {source}**")
            cols = st.columns(5)
            cols[0].metric("Section", _short(location.get("section", "N/A"), 32))
            cols[1].metric("Row", location.get("row_id") or "N/A")
            cols[2].metric("Retrieval", _format_score(location.get("retrieval_score")))
            cols[3].metric("Rerank", _format_score(location.get("rerank_score")))
            cols[4].metric("Strategy", location.get("chunk_strategy") or strategy)

            st.markdown("**Matched Snippet**")
            st.write(location.get("snippet") or "No snippet available.")
            st.caption(location.get("why_matched") or "")

            details = []
            if location.get("source"):
                details.append(f"Source: {location['source']}")
            if location.get("chunk_id"):
                details.append(f"Chunk: {location['chunk_id']}")
            if location.get("start_index") is not None and location.get("end_index") is not None:
                details.append(f"Character range: {location['start_index']} - {location['end_index']}")
            if details:
                st.caption(" | ".join(details))

            with st.expander("Parent context"):
                st.write(location.get("parent_context") or "No parent context available.")


def render_ask_page(strategy: str, top_k: int) -> None:
    render_header()
    question, generate = render_query_panel()
    namespace = _namespace(strategy)

    if generate:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                result = _run_with_pipeline(question, namespace, top_k)
                st.session_state.latest_question = question
                st.session_state.latest_result = result
            except Exception as exc:
                st.error(str(exc))

    result = st.session_state.latest_result
    if not result:
        render_execution_pipeline(None, question, namespace, top_k, _default_statuses())
        render_runtime_metrics(None, namespace, top_k)
        render_execution_log([])
        render_answer(None)
        return

    render_answer(result)
    st.markdown("### Sources + Evidence")
    render_sources(result)
    render_evidence(result)
    st.markdown("### Reranking Results")
    render_reranking_results(result)


inject_css()
_init_state()

with st.sidebar:
    st.title("Synataric Navigator")
    show_technical_tabs = st.checkbox(
        "Show technical details",
        value=False,
        help="Reveal diagnostics, evaluation, and chunking comparison pages.",
    )
    page_options = ["Ask Navigator", "Find Evidence"]
    if show_technical_tabs:
        page_options.extend(["RAG Diagnostics", "Evaluation Dashboard", "Chunk Strategy Comparison"])
    page = st.radio(
        "Page",
        page_options,
    )
    st.divider()
    st.caption(get_langsmith_status())
    strategy = st.selectbox(
        "Chunking Strategy",
        ["semantic", "fixed"],
        index=0,
        format_func=lambda value: "Semantic chunks" if value == "semantic" else "Fixed chunks",
        help="Choose which Pinecone namespace to search.",
    )
    top_k = st.slider(
        "Retrieved candidates",
        min_value=3,
        max_value=20,
        value=10,
        step=1,
        help="Number of Pinecone matches sent to FlashRank for reranking.",
    )
    st.info("Healthcare navigation education only. Not medical advice.")

if page == "Ask Navigator":
    render_ask_page(strategy, top_k)
elif page == "Find Evidence":
    render_find_evidence_page(strategy, top_k)
elif page == "RAG Diagnostics":
    render_diagnostics_page(strategy, top_k)
elif page == "Evaluation Dashboard":
    render_evaluation_dashboard(strategy, top_k)
elif page == "Chunk Strategy Comparison":
    render_chunk_strategy_comparison(top_k)

st.divider()
st.caption(
    "Synataric Navigator does not diagnose, prescribe treatment, or replace licensed clinicians. "
    "For urgent symptoms or emergencies, seek immediate medical care."
)

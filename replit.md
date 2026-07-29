# Synataric Navigator

A LangChain + LangGraph RAG application for healthcare travel and care-planning education. Patients and caregivers can ask questions about medical travel, procedures, providers, costs, logistics, recovery, and risk checklists using a curated local corpus.

## How to run

The app starts automatically via the **Start application** workflow:

```bash
streamlit run app.py
```

Runs on **port 5000**.

## First-time setup: ingest data

Before the app is useful, populate the Pinecone vector index:

```bash
python ingest.py
```

This creates sample data if missing, loads and cleans documents, creates fixed + semantic chunks, and indexes them into Pinecone under namespaces `synataric-fixed` and `synataric-semantic`.

## Required secrets

Set in Replit Secrets:

- `OPENAI_API_KEY` — OpenAI API key
- `PINECONE_API_KEY` — Pinecone API key

## Environment variables (shared)

| Key | Value |
|-----|-------|
| `PINECONE_INDEX_NAME` | `Synataric Healthcare Travel Care Planning RAG` |
| `PINECONE_CLOUD` | `aws` |
| `PINECONE_REGION` | `us-east-1` |
| `LANGCHAIN_TRACING_V2` | `false` |
| `LANGCHAIN_PROJECT` | `Synataric-Navigator` |

Optional: set `LANGCHAIN_API_KEY` and `LANGCHAIN_TRACING_V2=true` to enable LangSmith tracing.

## App modes

Use the sidebar to switch between:

1. **Ask Navigator** — grounded RAG with citations
2. **Agent Navigator** — intent-routed agent (classifies question, selects tool)
3. **ReAct Care Planner** — multi-step bounded ReAct agent for care planning goals
4. **Find Evidence** — evidence locator

Enable **Show technical details** in the sidebar for diagnostics, evaluation, and chunking comparison pages.

## Stack

- Python 3.12
- Streamlit
- LangChain / LangGraph
- Pinecone (serverless vector store)
- OpenAI (`gpt-*` models)
- FlashRank (reranking)

## User preferences

<!-- Add user preferences here -->

# Synataric Navigator

Synataric Navigator is a code-heavy LangChain + LangGraph RAG application for healthcare travel and care-planning education. It helps patients and caregivers ask questions about medical travel, procedures, provider options, estimated costs, logistics, recovery considerations, and risk checklists using a curated local corpus.

The included corpus is illustrative sample data only. It is not medical advice, diagnosis, or treatment guidance.

## Latest Agentic Features

The Streamlit app now includes three user-facing modes:

- `Ask Navigator`: the original grounded RAG experience for cited healthcare travel answers.
- `Agent Navigator`: an intent-routed agent that classifies the question, selects one Synataric tool, handles safety boundaries, and asks for missing details when needed.
- `ReAct Care Planner`: a bounded ReAct-style care planning agent that can reason, call multiple tools in sequence, observe each result, and stop with a final care-navigation answer.

The ReAct page is designed for multi-step care planning goals such as provider search, cost estimate, recovery guidance, and risk checklist generation in one bounded run.

## Architecture

Pipeline:

Data -> Clean -> Chunk -> Embed -> Store -> Retrieve -> Rerank -> Generate cited answer -> Display in Streamlit

Main components:

- `src/sample_data.py`: writes illustrative procedure, provider, cost, risk, and policy files if missing.
- `src/loaders.py`: recursively loads Markdown, text, PDF, and CSV files into LangChain `Document` objects.
- `src/cleaning.py`: normalizes whitespace while preserving costs, names, numeric values, and medical terms.
- `src/chunking.py`: supports fixed chunking and semantic-style chunking fallback.
- `src/indexing.py`: creates a Pinecone serverless index and stores chunks in two namespaces.
- `src/retrieval.py`: retrieves top candidates from Pinecone.
- `src/reranking.py`: reranks with FlashRank and falls back gracefully.
- `src/rag_chain.py`: formats cited context and calls `ChatOpenAI`.
- `src/graph.py`: defines the LangGraph workflow.
- `src/agent_intents.py`: classifies healthcare navigation intent and safety boundaries.
- `src/agent_tools.py`: wraps the existing RAG pipeline as provider, cost, recovery, risk, travel, evidence, safety, and clarification tools.
- `src/agent_graph.py`: routes intent to the correct tool for the Agent Navigator page.
- `src/agent_session.py`: manages human-in-the-loop clarification state for the router agent.
- `src/react_care_agent.py`: implements the bounded ReAct care planning loop.
- `app.py`: Streamlit user interface.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

```bash
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=Synataric Healthcare Travel Care Planning RAG
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

The app normalizes `PINECONE_INDEX_NAME` into a Pinecone-safe lowercase index identifier before creating or querying the index.

## Ingest

```bash
python ingest.py
```

This creates sample data if needed, loads documents, cleans them, creates fixed and semantic chunks, creates the Pinecone index if needed, deletes existing namespace contents, and indexes both:

- `synataric-fixed`
- `synataric-semantic`

## Run

```bash
streamlit run app.py
```

Use the sidebar to switch between:

1. `Ask Navigator`
2. `Agent Navigator`
3. `ReAct Care Planner`
4. `Find Evidence`

Enable `Show technical details` in the sidebar to reveal diagnostics, evaluation, and chunking comparison pages.

## Agent CLI Tests

Run the router-style Agent Navigator backend:

```bash
python -m src.agent_graph
```

Run the bounded ReAct Care Planner backend:

```bash
python -m src.react_care_agent
```

For local testing without LangSmith network calls:

```bash
set LANGCHAIN_TRACING_V2=false
```

## Deploy

For a public Streamlit Community Cloud demo, see `DEPLOYMENT.md`. Use `app.py` as the entry point and store API keys in Streamlit secrets, using `.streamlit/secrets.toml.example` as the template.

## Sample Questions

- How many days should I plan to stay in Bangalore for cataract surgery?
- What should a caregiver ask before knee replacement travel?
- What are illustrative cost ranges for cardiac bypass and recovery logistics?
- Which urgent symptoms should not be handled by the navigator?
- Where can I find good cataract surgery in India?
- What is the cost of cataract surgery in Bangalore?
- Create a care travel plan for cataract surgery in Bangalore including providers, cost, recovery, and risks.
- Plan my travel for surgery in Bangalore.
- Should I take antibiotics after surgery?
- Who won the Super Bowl in 2024?

## Safety Positioning

Synataric Navigator is for healthcare navigation and education only. It does not diagnose, prescribe treatment, replace licensed clinicians, or determine whether a medical procedure is appropriate. For urgent symptoms or emergencies, seek immediate medical care.

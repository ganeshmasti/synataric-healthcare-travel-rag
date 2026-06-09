# Synataric Navigator

Synataric Navigator is a code-heavy LangChain + LangGraph RAG application for healthcare travel and care-planning education. It helps patients and caregivers ask questions about medical travel, procedures, provider options, estimated costs, logistics, recovery considerations, and risk checklists using a curated local corpus.

The included corpus is illustrative sample data only. It is not medical advice, diagnosis, or treatment guidance.

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
PINECONE_INDEX_NAME=Synataric – Healthcare Travel & Care Planning RAG
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

## Sample Questions

- How many days should I plan to stay in Bangalore for cataract surgery?
- What should a caregiver ask before knee replacement travel?
- What are illustrative cost ranges for cardiac bypass and recovery logistics?
- Which urgent symptoms should not be handled by the navigator?

## Safety Positioning

Synataric Navigator is for healthcare navigation and education only. It does not diagnose, prescribe treatment, replace licensed clinicians, or determine whether a medical procedure is appropriate. For urgent symptoms or emergencies, seek immediate medical care.

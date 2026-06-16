# Cloud Deployment

This app is ready for Streamlit Community Cloud. The public app runs `app.py` and reads secrets from Streamlit Cloud's secrets manager.

## Before Deploying

1. Confirm the Pinecone index already contains both namespaces:
   - `synataric-fixed`
   - `synataric-semantic`

2. If you changed source data, chunking, or embedding behavior, run ingestion locally before deploying:

```bash
python ingest.py
```

3. Push the app to GitHub. Stage only source, tests, and documentation. Do not use `git add .` if local cache files or secrets are present.

```bash
git add app.py README.md DEPLOYMENT.md src/agent_graph.py src/agent_intents.py src/agent_recovery.py src/agent_session.py src/agent_tools.py src/react_care_agent.py
git commit -m "Add agent navigator and ReAct care planner"
git push origin main
```

## Streamlit Community Cloud

1. Go to Streamlit Community Cloud.
2. Create a new app from the GitHub repository:
   - Repository: `ganeshmasti/synataric-healthcare-travel-rag`
   - Branch: `main`
   - Main file path: `app.py`
3. Paste the secrets from `.streamlit/secrets.toml.example` into the app's Secrets panel and replace placeholder values.
4. Deploy.

## Required Secrets

```toml
OPENAI_API_KEY = "sk-your-openai-key"
PINECONE_API_KEY = "your-pinecone-key"
PINECONE_INDEX_NAME = "Synataric Healthcare Travel Care Planning RAG"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
LANGCHAIN_API_KEY = ""
LANGCHAIN_TRACING_V2 = "false"
LANGCHAIN_PROJECT = "Synataric-Navigator"
```

## Public Demo Notes

- Use `Ask Navigator` for the original single-question RAG demo.
- Use `Agent Navigator` to demo intent routing, tool selection, safety handling, and human clarification.
- Use `ReAct Care Planner` to demo bounded multi-step care planning with Reason -> Act -> Observe loops.
- Use `Show technical details` only when demoing the RAG internals.
- Do not publish `.env` or `.streamlit/secrets.toml`.
- Rotate any API key that has appeared in screenshots, terminal output, or editor selections before pushing publicly.

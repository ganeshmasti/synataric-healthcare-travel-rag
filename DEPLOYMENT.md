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

3. Push the app to GitHub:

```bash
git add .
git commit -m "Prepare Streamlit cloud deployment"
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

- Keep the public app on the user-facing `Ask Navigator` page by default.
- Use `Show technical details` only when demoing the RAG internals.
- Do not publish `.env` or `.streamlit/secrets.toml`.

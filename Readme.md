# AI Lawyer RAG

RAG-powered Q&A over uploaded legal PDFs using FAISS + LangChain for retrieval and Groq for answers. Embeddings come from a lightweight local sentence-transformer that works on Streamlit Cloud.

## Prerequisites
- Python 3.10+
- Groq API key

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment:
   - Copy `.env.example` to `.env` and fill values, or set in shell:
   ```powershell
   $Env:GROQ_API_KEY = "<your_key>"
   $Env:EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   # Optional: explicitly choose a Groq model
   $Env:GROQ_MODEL = "llama-3.3-70b-versatile"
   ```

## Run
```powershell
streamlit run frontend.py
# or
streamlit run main.py
```

## Features
- Upload a PDF and build a fresh FAISS index per session
- Top-K slider to control retrieved chunks
- Token-aware context construction to respect model limits
- Source citations with chunk previews and page info
- Robust Groq model selection and automatic fallback on decommission

## Tips
- For fast, reliable retrieval, keep `EMBEDDING_MODEL` set to `sentence-transformers/all-MiniLM-L6-v2`.
- Large PDFs: reduce `Top K` and ask specific questions to improve focus.

## Troubleshooting
- First run may take longer because the embedding model is downloaded automatically.
- Groq decommission errors: set `GROQ_MODEL` to a supported model or rely on automatic selection.
- Memory errors with large models: use lightweight embedding models; avoid chat models for embeddings.

## Streamlit Cloud Deployment
1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create a new app and point it to `main.py`.
3. Add `GROQ_API_KEY` in the app secrets or environment settings.
4. Optional: add `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` if you want to override the default.

Notes:
- Uploaded PDFs and generated FAISS indexes are stored on the app filesystem and may reset after a redeploy or restart on hosted platforms.
- For long-term persistence, move PDFs and vector indexes to cloud storage or a managed vector database.
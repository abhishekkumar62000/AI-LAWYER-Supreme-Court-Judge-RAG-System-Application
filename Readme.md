<img width="1920" height="1024" alt="1" src="https://github.com/user-attachments/assets/cf7dd6f2-e9ba-47a2-b519-02756bb3d33d" />
<img width="1908" height="1020" alt="4" src="https://github.com/user-attachments/assets/93a20245-999b-4a1f-8ebd-153cd7fe2670" />

# App Demo:-

https://github.com/user-attachments/assets/ce4e461a-1195-4a27-809a-ec256a26e744

# ⚖️ AI Lawyer: Supreme Court Judge RAG System
### *Revolutionizing Legal Intelligence with Hybrid RAG & Courtroom Simulation*

---

## 🏛️ Project Overview
The **AI Lawyer Supreme Court Judge RAG System** is an advanced, multi-modal Legal AI platform designed to bridge the gap between complex legal data and actionable judicial intelligence. Unlike standard chatbots, this system features a dual-interface architecture tailored for both **Lawyers** (Strategy & Research) and **Supreme Court Judges** (Analysis & Bench Support).

### 🎯 The Problem We Solve
Legal professionals are overwhelmed by "Information Density." Analyzing thousands of pages of case law, finding contradictions in witness statements, and predicting judicial outcomes are manual, error-prone tasks. 
- **Our Solution:** We provide an **"AI Bench Partner"** that sifts through evidence, detects factual clashes, and simulates courtroom pressure in real-time.

---

## 🚀 Outstanding Features

### 1. 🔍 Hybrid Intelligence RAG
*   **Contextual Retrieval:** Combines **Vector Similarity (FAISS)** with **Lexical Search (BM25)** to ensure that both legal concepts and specific statutory keywords are found.
*   **Multi-Document Clash Detection:** A specialized agent that scans multiple PDFs to find "The Sleeping Truth"—contradictions between witness statements or evidence.

### 2. 👨‍⚖️ Supreme Court Judge Mode
*   **Cause List Intelligence:** AI-powered daily hearing tracker with automated matter summarization.
*   **Order Drafting Studio:** Rapidly generate structured Interim/Final orders with automated citation of CPC, CrPC, and IPC.
*   **Bench Notes Generator:** Instant internal briefings for presiding judges to prepare for high-stakes oral arguments.

### 3. 🧠 Trial Strategy Suite
*   **Witness Cross-Exam Coach:** Generates tactical "Leading Questions" based on case evidence to trap witnesses in inconsistencies.
*   **Case Strength Gauge:** An animated, data-driven visualizer that predicts the "Win Probability" of a legal petition.
*   **Precedent Reliability Analyzer:** Instantly verify if a cited judgment is **Binding**, **Persuasive**, or has been **Questioned** by larger benches.

### 🎭 4. Virtual Courtroom Simulator
*   **Realistic Advocacy Training:** Present arguments to an AI Judge that responds in "Judge Sahib" persona.
*   **Bench Tension Meter:** A revolutionary UI feature that visually maps the "Judge's Skepticism" in real-time based on the quality of your argument.

---

## 🏗️ Technical Architecture & Workflow

### **The Technology Stack**
- **Frontend:** Streamlit with a custom **Dark Neon Theme** for premium UX.
- **Orchestration:** LangChain (Memory, Prompt Engineering, RAG Chains).
- **Brain:** Groq & DeepSeek (Llama-3.3-70b/DeepSeek-R1) for low-latency reasoning.
- **Vector DB:** FAISS (Local persistent vector storage).
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local processing, zero data leak).

### **How it Works (Step-by-Step)**
1.  **Ingestion:** PDFs are parsed using `PDFPlumber` and split into semantic chunks.
2.  **Indexing:** Chunks are vectorized and stored in a local FAISS index via a manifest-driven system.
3.  **The "Judge's Lens":** When a query is made, the system retrieves context and applies a **Judicial Prompting Layer** to ensure professional, unbiased, and statutory-accurate responses.

---

## 🛠️ Installation & Deployment

### **Streamlit Cloud Deployment**
This repository is **deployment-ready** for Streamlit Cloud.
1. Sync this repo to your [Streamlit Cloud](https://share.streamlit.io/).
2. Add your `GROQ_API_KEY` to the **Secrets** panel.
3. The app is optimized via `requirements.txt` to include all necessary Torch/FAISS dependencies automatically.

### **Local Setup**
```bash
# Clone the repo
git clone https://github.com/abhishekkumar62000/AI-LAWYER-Supreme-Court-Judge-RAG-System-Application.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

---

## 📜 Objectives
- **Precision:** 99% accuracy in legal citation and context retrieval.
- **Portability:** Lightweight enough to run on Streamlit Cloud but powerful enough for enterprise legal firms.
- **Innovation:** Moving AI from a "Search Utility" to a "Tactical Partner" in the courtroom.

---
*Created with ❤️ for the Legal Community by Abhishek Kumar.*
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

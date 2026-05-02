import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS  # type: ignore
from groq import Groq, BadRequestError
import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from langdetect import detect, LangDetectException
from pyvis.network import Network
from rank_bm25 import BM25Okapi
import base64
import urllib.parse
import re
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
import zipfile
from embeddings_provider import get_embedding_model, resolve_embedding_model_name, DEFAULT_EMBEDDING_MODEL



custom_prompt_template = """
You are an expert AI Legal Assistant with deep knowledge of legal documents and regulations.

Your task is to provide a comprehensive, detailed answer to the user's question based on the context provided.

Instructions:
- Analyze the context carefully and provide a thorough, well-structured answer
- Use specific details, citations, and quotes from the context when relevant
- If the context contains article numbers or section references, cite them
- Provide explanations in clear, professional language
- If you need to clarify or add legal context based on the documents, do so
- If the answer is not in the context, clearly state that
- Structure your response with proper paragraphs for readability

Conversation Summary:
{memory}

User Question:
{question}

Relevant Context from Legal Documents:
{context}

Detailed Answer:
"""

EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
FAISS_DB_ROOT = "vectorstore/db_faiss"
MANIFEST_PATH = os.path.join(FAISS_DB_ROOT, "manifest.json")
PDFS_DIR = "pdfs/"

DEFAULT_NEWS_SOURCES = [
    "https://www.livelaw.in/rss/topstories.xml",
    "https://www.barandbench.com/feed",
    "https://www.scobserver.in/feed/",
]

st.set_page_config(page_title="AI Lawyer RAG", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

# 🎨 ADVANCED DARK NEON THEME UI/UX
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    
    /* ===== GLOBAL DARK THEME ===== */
    :root {
        --neon-cyan: #00f3ff;
        --neon-purple: #b026ff;
        --neon-pink: #ff006e;
        --neon-green: #39ff14;
        --neon-yellow: #ffea00;
        --dark-bg: #0a0e27;
        --dark-card: #1a1f3a;
        --dark-hover: #252b4a;
        --text-primary: #e8eaf6;
        --text-secondary: #9fa8da;
        --glass-bg: rgba(26, 31, 58, 0.85);
    }
    
    /* Background & Main Container */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d35 50%, #0f1528 100%);
        background-attachment: fixed;
    }
    
    .main {
        background: transparent;
        padding: 1rem 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Background Gradient */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 50%, rgba(0, 243, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(176, 38, 255, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 40% 20%, rgba(255, 0, 110, 0.03) 0%, transparent 40%);
        pointer-events: none;
        animation: pulse 15s ease-in-out infinite;
        z-index: 0;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    /* ===== TYPOGRAPHY ===== */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(0, 243, 255, 0.3);
    }
    
    h1 {
        font-size: 3rem !important;
        background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-purple) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.3); }
    }
    
    p, div, span, label {
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif;
    }

    /* ===== JUDGE MODE STYLES ===== */
    .judge-header {
        background: linear-gradient(135deg, rgba(255, 234, 0, 0.08), rgba(0, 243, 255, 0.1));
        border: 1px solid rgba(255, 234, 0, 0.35);
        border-radius: 18px;
        padding: 2rem 1.5rem;
        box-shadow: 0 0 40px rgba(255, 234, 0, 0.08);
    }

    .judge-title {
        font-size: 3.2rem;
        margin: 0.5rem 0 0.2rem;
        color: #ffea00;
        font-weight: 900;
        text-shadow: 0 0 14px rgba(255, 234, 0, 0.6), 0 0 24px rgba(0, 243, 255, 0.35);
        letter-spacing: 0.5px;
    }

    .judge-subtitle {
        font-size: 1.05rem;
        color: #9fa8da;
        margin-top: 0.4rem;
        letter-spacing: 1px;
    }

    .bench-chip {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 14px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid rgba(0, 243, 255, 0.35);
        color: #00f3ff;
        background: rgba(0, 243, 255, 0.12);
        margin: 6px 6px 0 0;
    }

    .bench-card {
        background: rgba(26, 31, 58, 0.6);
        border: 1px solid rgba(0, 243, 255, 0.2);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.25);
    }

    .court-strip {
        background: linear-gradient(90deg, rgba(255, 234, 0, 0.2), rgba(0, 243, 255, 0.2));
        border-radius: 10px;
        padding: 0.6rem 1rem;
        color: #e8eaf6;
        font-weight: 700;
        letter-spacing: 0.8px;
        border: 1px solid rgba(255, 234, 0, 0.3);
        text-align: center;
    }

    .docket-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        background: rgba(26, 31, 58, 0.5);
        border-left: 4px solid rgba(255, 234, 0, 0.6);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
    }

    .order-box {
        background: rgba(10, 14, 39, 0.8);
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        color: #e8eaf6;
        box-shadow: inset 0 0 10px rgba(0, 243, 255, 0.12);
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1128 0%, #1a1f3a 100%);
        border-right: 2px solid rgba(0, 243, 255, 0.2);
        box-shadow: 5px 0 30px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary);
    }
    
    /* ===== GLASSMORPHISM CARDS ===== */
    .element-container, .stMarkdown, [data-testid="stExpander"] {
        backdrop-filter: blur(10px);
        background: var(--glass-bg);
        border-radius: 15px;
        border: 1px solid rgba(0, 243, 255, 0.1);
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    [data-testid="stExpander"] {
        border: 1px solid rgba(0, 243, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stExpander"]:hover {
        border-color: var(--neon-cyan);
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.3);
        transform: translateY(-2px);
    }
    
    /* ===== NEON BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 243, 255, 0.2) 0%, rgba(176, 38, 255, 0.2) 100%);
        border: 2px solid var(--neon-cyan);
        border-radius: 12px;
        color: var(--text-primary) !important;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(0, 243, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.5s, height 0.5s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 243, 255, 0.4) 0%, rgba(176, 38, 255, 0.4) 100%);
        border-color: var(--neon-purple);
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.6), 0 0 50px rgba(176, 38, 255, 0.4);
        transform: translateY(-3px) scale(1.02);
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }
    
    /* ===== NEON INPUTS ===== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background: rgba(26, 31, 58, 0.8) !important;
        border: 2px solid rgba(0, 243, 255, 0.4) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 0.8rem !important;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--neon-cyan) !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.4), inset 0 2px 8px rgba(0, 0, 0, 0.3) !important;
        background: rgba(26, 31, 58, 1) !important;
    }
    
    /* ===== METRICS CARDS ===== */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, var(--neon-cyan) 0%, var(--neon-green) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stMetric"] {
        background: var(--glass-bg);
        border: 1px solid rgba(0, 243, 255, 0.3);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 30px rgba(0, 243, 255, 0.4);
        border-color: var(--neon-cyan);
    }
    
    /* ===== SLIDER ===== */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--neon-cyan) 0%, var(--neon-purple) 100%) !important;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.5);
    }
    
    .stSlider > div > div > div > div {
        background-color: var(--neon-cyan) !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.8);
    }
    
    /* ===== CHAT MESSAGES ===== */
    .stChatMessage {
        background: var(--glass-bg) !important;
        border: 1px solid rgba(0, 243, 255, 0.2);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        border-color: var(--neon-cyan);
        box-shadow: 0 0 25px rgba(0, 243, 255, 0.3);
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: var(--glass-bg);
        border: 2px dashed rgba(0, 243, 255, 0.4);
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--neon-cyan);
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.3);
        background: rgba(26, 31, 58, 1);
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, var(--neon-cyan) 50%, transparent 100%);
        margin: 2rem 0;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dark-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--neon-cyan) 0%, var(--neon-purple) 100%);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 243, 255, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--neon-purple) 0%, var(--neon-pink) 100%);
        box-shadow: 0 0 20px rgba(176, 38, 255, 0.8);
    }
    
    /* ===== SUCCESS/ERROR/WARNING BOXES ===== */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stSuccess {
        background: rgba(57, 255, 20, 0.1);
        border-color: var(--neon-green);
        box-shadow: 0 0 20px rgba(57, 255, 20, 0.3);
    }
    
    .stError {
        background: rgba(255, 0, 110, 0.1);
        border-color: var(--neon-pink);
        box-shadow: 0 0 20px rgba(255, 0, 110, 0.3);
    }
    
    .stWarning {
        background: rgba(255, 234, 0, 0.1);
        border-color: var(--neon-yellow);
        box-shadow: 0 0 20px rgba(255, 234, 0, 0.3);
    }
    
    .stInfo {
        background: rgba(0, 243, 255, 0.1);
        border-color: var(--neon-cyan);
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.3);
    }
    
    /* ===== DOWNLOAD BUTTON ===== */
    .stDownloadButton > button {
        background: linear-gradient(135deg, rgba(57, 255, 20, 0.2) 0%, rgba(0, 243, 255, 0.2) 100%);
        border: 2px solid var(--neon-green);
        box-shadow: 0 0 15px rgba(57, 255, 20, 0.4);
    }
    
    .stDownloadButton > button:hover {
        border-color: var(--neon-cyan);
        box-shadow: 0 0 30px rgba(57, 255, 20, 0.6), 0 0 50px rgba(0, 243, 255, 0.4);
    }
    
    /* ===== LOADING SPINNER ===== */
    .stSpinner > div {
        border-color: var(--neon-cyan) transparent transparent transparent !important;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.6);
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--glass-bg);
        border: 2px solid rgba(0, 243, 255, 0.3);
        border-radius: 10px;
        color: var(--text-secondary);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 243, 255, 0.1);
        border-color: var(--neon-cyan);
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.4);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 243, 255, 0.3) 0%, rgba(176, 38, 255, 0.3) 100%);
        border-color: var(--neon-cyan);
        color: var(--text-primary) !important;
        box-shadow: 0 0 25px rgba(0, 243, 255, 0.5);
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes slideInFromLeft {
        0% { transform: translateX(-100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Apply animations */
    .element-container {
        animation: fadeIn 0.5s ease-in;
    }
    
    /* ===== CUSTOM UTILITY CLASSES ===== */
    .neon-box {
        background: var(--glass-bg);
        border: 2px solid var(--neon-cyan);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .neon-box:hover {
        box-shadow: 0 0 50px rgba(0, 243, 255, 0.5);
        transform: scale(1.02);
    }
    
    .pulse-animation {
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        h1 { font-size: 2rem !important; }
        .main { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set. Please set it in environment or .env file.")
groq_client = Groq(api_key=groq_api_key)

def resolve_groq_model(client: Groq) -> str:
    env_model = os.environ.get("GROQ_MODEL")
    if env_model:
        return env_model
    try:
        models = client.models.list()
        available_ids = [m.id for m in getattr(models, "data", [])]
        preferred = os.environ.get(
            "GROQ_MODEL_PREFERENCE",
            "llama-3.3-70b-versatile, llama-3.2-11b-text-preview, llama-3.1-8b-instant, mixtral-8x7b-32768, gemma2-9b-it",
        )
        prefs = [m.strip() for m in preferred.split(",") if m.strip()]
        for p in prefs:
            if p in available_ids:
                return p
        for mid in available_ids:
            if "llama" in mid:
                return mid
        if available_ids:
            return available_ids[0]
    except Exception:
        # If listing models fails, return a conservative default that often exists.
        pass
    return "llama-3.2-11b-text-preview"

GROQ_MODEL = resolve_groq_model(groq_client)

def _ensure_dirs():
    os.makedirs(PDFS_DIR, exist_ok=True)
    os.makedirs(FAISS_DB_ROOT, exist_ok=True)

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def load_manifest() -> List[Dict]:
    _ensure_dirs()
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_manifest(entries: List[Dict]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

def upsert_manifest_entry(entry: Dict):
    entries = load_manifest()
    # Remove any existing entry with same doc_id
    entries = [e for e in entries if e.get("doc_id") != entry.get("doc_id")]
    entries.append(entry)
    save_manifest(entries)

def delete_manifest_entry(doc_id: str):
    entries = load_manifest()
    entries = [e for e in entries if e.get("doc_id") != doc_id]
    save_manifest(entries)

def upload_pdf(file) -> Dict:
    """Save uploaded PDF and return metadata entry (without FAISS yet)."""
    _ensure_dirs()
    raw = file.getbuffer()
    doc_id = _sha256_bytes(raw)
    safe_name = file.name.replace("/", "_").replace("\\", "_")
    pdf_path = os.path.join(PDFS_DIR, f"{doc_id}_{safe_name}")
    with open(pdf_path, "wb") as f:
        f.write(raw)
    entry = {
        "doc_id": doc_id,
        "name": safe_name,
        "size": len(raw),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pdf_path": pdf_path,
        "db_path": os.path.join(FAISS_DB_ROOT, doc_id),
        "embed_model": EMBEDDING_MODEL,
    }
    upsert_manifest_entry(entry)
    return entry


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks


def create_vector_store(db_faiss_path: str, text_chunks, embedding_model_name: str):
    os.makedirs(db_faiss_path, exist_ok=True)
    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(embedding_model_name))
    faiss_db.save_local(db_faiss_path)
    return faiss_db

def load_vector_store(db_faiss_path: str, embedding_model_name: str) -> Optional[FAISS]:
    try:
        return FAISS.load_local(
            db_faiss_path,
            get_embedding_model(embedding_model_name),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def retrieve_docs(faiss_db, query):
    # Default retrieval using MMR for diversity
    try:
        return faiss_db.max_marginal_relevance_search(query, k=10, fetch_k=40)
    except Exception:
        return faiss_db.similarity_search(query)

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]

def hybrid_rerank(faiss_db, query: str, top_k: int) -> List:
    """Hybrid retrieval: vector (MMR) + lexical BM25 + lightweight LLM scoring.
    Operates over vector candidates to keep latency acceptable.
    """
    candidates = retrieve_docs(faiss_db, query)
    if not candidates:
        return []
    corpus = [d.page_content for d in candidates]
    tokenized = [_tokenize(c) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    bm_scores = bm25.get_scores(_tokenize(query))

    # Lightweight LLM scoring
    def llm_score(q: str, chunk: str) -> float:
        prompt = (
            "Rate the relevance of the chunk to the query on a 0-1 scale. "
            "Reply with only a number.\n\nQuery:\n" + q + "\n\nChunk:\n" + chunk
        )
        try:
            comp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            text = comp.choices[0].message.content.strip()
            val = float(text)
            if val < 0:
                return 0.0
            if val > 1:
                return 1.0
            return val
        except Exception:
            return 0.5

    # Cap LLM scoring to first N for speed
    N = min(len(candidates), max(10, top_k * 3))
    llm_scores = [0.0] * len(candidates)
    for i in range(N):
        llm_scores[i] = llm_score(query, corpus[i])

    # Normalize BM25 scores
    if len(bm_scores) > 0:
        max_bm = max(bm_scores) or 1.0
        bm_norm = [s / max_bm for s in bm_scores]
    else:
        bm_norm = [0.0] * len(candidates)

    # Apply feedback weights if any
    def chunk_id_from_text(t: str) -> str:
        return hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()

    feedback = st.session_state.get("feedback", {})
    weights = []
    for i, text in enumerate(corpus):
        cid = chunk_id_from_text(text)
        w = 1.0
        fb = feedback.get(cid, {})
        if fb.get("pinned"):
            w += 0.2
        if fb.get("downvotes", 0) > 0:
            w -= min(0.2, 0.05 * fb.get("downvotes", 0))
        if fb.get("upvotes", 0) > 0:
            w += min(0.2, 0.05 * fb.get("upvotes", 0))
        weights.append(max(0.5, min(1.5, w)))

    # Final score: weighted blend
    final = []
    for i, d in enumerate(candidates):
        score = (0.5 * bm_norm[i] + 0.5 * llm_scores[i]) * weights[i]
        final.append((d, score, bm_norm[i], llm_scores[i], weights[i]))
    final.sort(key=lambda x: x[1], reverse=True)
    return final[:top_k]


def get_context(documents, max_chars: int = 12000):
    parts = []
    total = 0
    for doc in documents:
        text = doc.page_content
        if total + len(text) > max_chars:
            text = text[: max(0, max_chars - total)]
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(parts)

def summarize_history(messages: List[Dict]) -> str:
    """Create a short summary of the conversation history using Groq."""
    if not messages:
        return ""
    # Keep last ~6 exchanges for brevity
    recent = messages[-12:]
    text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent])
    prompt = f"Summarize the following conversation in 3-5 sentences focusing on the user's intent, constraints, and what has already been answered.\n\n{text}"
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception:
        return ""

def translate_text(text: str, target_lang: str) -> str:
    """Translate text using Groq; target_lang like 'en', 'fr'."""
    if not text:
        return text
    prompt = f"Translate the following text to {target_lang}. Keep legal terms precise.\n\n{text}"
    try:
        comp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return comp.choices[0].message.content
    except Exception:
        return text


def answer_query(documents, query, memory: str = ""):
    context = get_context(documents, max_chars=20000)  # Increased context size
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    final_prompt = prompt.format(question=query, context=context, memory=memory)
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.4,  # Increased for more detailed responses
            max_tokens=2048,  # Allow longer answers
        )
        return completion.choices[0].message.content
    except BadRequestError as e:
        # If the model is decommissioned or invalid, resolve a new one and retry once.
        msg = getattr(e, "message", str(e))
        if "model" in msg and ("decommissioned" in msg or "not found" in msg):
            fallback_model = resolve_groq_model(groq_client)
            completion = groq_client.chat.completions.create(
                model=fallback_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.4,
                max_tokens=2048,
            )
            return completion.choices[0].message.content
        raise

def clash_detection_agent(documents: List):
    """Detect contradictions and inconsistencies across multiple documents."""
    context = get_context(documents, max_chars=30000)
    prompt = f"""
    You are a Senior Discovery Agent. Analyze the following legal documents for CLASHES, CONTRADICTIONS, and INCONSISTENCIES.
    
    Look for:
    1. Direct contradictions between witness statements.
    2. Date/Time discrepancies.
    3. Financial or factual inconsistencies across different documents.
    4. Divergent claims by the same party.

    Context:
    {context}

    Return a structured list of clashes found. If none, state 'No significant clashes detected.'
    """
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in Clash Detection: {str(e)}"

def cross_examination_coach(documents: List, witness_name: str):
    """Generate pressure-point questions for a specific witness based on evidence."""
    context = get_context(documents, max_chars=20000)
    prompt = f"""
    You are an Expert Trial Lawyer and Cross-Examination Coach. 
    Analyze the evidence regarding Witness: {witness_name}.
    
    Generate:
    1. Five high-pressure 'Leading Questions' to trap the witness.
    2. Potential 'Escape Routes' (Deflections) the witness might use.
    3. Counter-moves for those deflections based on the evidence.

    Evidence Context:
    {context}
    """
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in Coach: {str(e)}"

def suggest_followups(question: str, answer: str) -> List[str]:
    prompt = (
        "Based on the user's question and the assistant's answer, "
        "propose three concise follow-up questions that would help clarify legal context, cite relevant articles, or refine the scope. "
        "Return them as a bullet list without numbering.\n\n"
        f"Question: {question}\n\nAnswer: {answer}"
    )
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = completion.choices[0].message.content
        lines = [l.strip("- ") for l in text.splitlines() if l.strip()]
        # Keep top 3 suggestions
        return [l for l in lines if l][:3]
    except Exception:
        return []

def extract_graph(answer: str, sources: List) -> Dict:
    """Extract a simple knowledge graph (entities, relations) from the answer and sources."""
    src_text = "\n\n".join([getattr(s, "page_content", "") for s in sources])
    schema = (
        "Return JSON with keys 'nodes' and 'edges'. 'nodes' is a list of objects {id, label, type}. "
        "'edges' is a list of objects {source, target, label}. Keep it concise and legally meaningful."
    )
    prompt = f"Build a knowledge graph from the answer and sources. {schema}\n\nAnswer:\n{answer}\n\nSources:\n{src_text}"
    try:
        comp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = comp.choices[0].message.content
        data = json.loads(text)
        return data
    except Exception:
        return {"nodes": [], "edges": []}

def render_graph(graph_data: Dict):
    net = Network(height="500px", width="100%", notebook=False, directed=True)
    for n in graph_data.get("nodes", []):
        net.add_node(n.get("id", n.get("label", "?")), label=n.get("label", ""), title=n.get("type", ""))
    for e in graph_data.get("edges", []):
        net.add_edge(e.get("source", ""), e.get("target", ""), title=e.get("label", ""))
    html = net.generate_html("graph.html")
    st.components.v1.html(html, height=520, scrolling=True)

def _pdf_path_to_data_url(pdf_path: str) -> Optional[str]:
    try:
        with open(pdf_path, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode("ascii")
        return f"data:application/pdf;base64,{b64}"
    except Exception:
        return None

def render_pdf_viewer(pdf_path: str, page: Optional[int] = None, search: Optional[str] = None, height: int = 700):
    data_url = _pdf_path_to_data_url(pdf_path)
    if not data_url:
        st.warning("Unable to load PDF source for viewer.")
        return
    file_param = urllib.parse.quote(data_url, safe="")
    hash_params = []
    if page:
        hash_params.append(f"page={int(page)}")
    if search:
        # Limit search length to avoid huge URLs
        q = urllib.parse.quote(str(search)[:200])
        hash_params.append(f"search={q}")
        hash_params.append("phrase=true")
    hash_str = ("#" + "&".join(hash_params)) if hash_params else ""
    # Use official PDF.js viewer hosted on GitHub Pages
    viewer = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={file_param}{hash_str}"
    st.components.v1.iframe(viewer, height=height)


# ==================== NEW FEATURES ====================

# FEATURE 1: Legal Report Generator
def generate_legal_report(qa_pairs: List[Tuple[str, str, List]], doc_name: str, format_type: str = "html") -> str:
    """Generate professional legal research report from Q&A session."""
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Legal Research Report - {doc_name}</title>
        <style>
            body {{ font-family: 'Georgia', serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 20px; color: #333; }}
            .header {{ text-align: center; border-bottom: 3px solid #2c3e50; padding-bottom: 20px; margin-bottom: 30px; }}
            .header h1 {{ color: #2c3e50; margin: 0; font-size: 2.2em; }}
            .header .subtitle {{ color: #7f8c8d; margin-top: 10px; font-style: italic; }}
            .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin-bottom: 30px; }}
            .qa-section {{ margin: 30px 0; page-break-inside: avoid; }}
            .question {{ background: #3498db; color: white; padding: 15px; border-radius: 5px 5px 0 0; font-weight: bold; font-size: 1.1em; }}
            .answer {{ background: #f8f9fa; padding: 20px; border: 1px solid #dee2e6; border-radius: 0 0 5px 5px; }}
            .sources {{ background: #fff3cd; padding: 15px; margin-top: 10px; border-left: 4px solid #ffc107; font-size: 0.9em; }}
            .sources h4 {{ margin: 0 0 10px 0; color: #856404; }}
            .source-item {{ margin: 5px 0; padding: 5px; }}
            .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 2px solid #ecf0f1; color: #7f8c8d; font-size: 0.9em; }}
            .page-break {{ page-break-after: always; }}
            @media print {{ body {{ margin: 20px; }} .page-break {{ page-break-after: always; }} }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>⚖️ Legal Research Report</h1>
            <div class="subtitle">AI-Powered Document Analysis</div>
        </div>
        
        <div class="metadata">
            <strong>Document Analyzed:</strong> {doc_name}<br>
            <strong>Report Generated:</strong> {timestamp}<br>
            <strong>Total Queries:</strong> {len(qa_pairs)}<br>
            <strong>Research Method:</strong> RAG (Retrieval-Augmented Generation) with Hybrid Search
        </div>
        
        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">Research Findings</h2>
    """
    
    for idx, (question, answer, sources) in enumerate(qa_pairs, 1):
        html_content += f"""
        <div class="qa-section">
            <div class="question">Q{idx}: {question}</div>
            <div class="answer">
                <p>{answer.replace(chr(10), '<br>')}</p>
            </div>
            <div class="sources">
                <h4>📚 Sources Referenced:</h4>
        """
        for src_idx, src in enumerate(sources[:3], 1):
            html_content += f'<div class="source-item">• Source {src_idx}: {src}</div>'
        html_content += "</div></div>"
    
    html_content += f"""
        <div class="footer">
            <p><strong>CONFIDENTIAL LEGAL RESEARCH</strong></p>
            <p>This report was generated using AI-assisted legal research tools.<br>
            Please verify all citations and consult with qualified legal counsel.</p>
            <p>Generated by AI Lawyer RAG System | {timestamp}</p>
        </div>
    </body>
    </html>
    """
    return html_content


# FEATURE 2: Smart Entity Extractor
def extract_legal_entities(text: str) -> Dict:
    """Extract legal entities, dates, amounts, parties using LLM."""
    prompt = f"""Analyze this legal text and extract key entities in JSON format.

Extract:
- parties: List of people/organizations mentioned
- dates: Important dates (format: YYYY-MM-DD or description)
- amounts: Money amounts with currency
- locations: Places, jurisdictions
- case_numbers: Case or docket numbers
- articles: Article/section numbers referenced

Text:
{text[:4000]}

Return ONLY valid JSON with these keys. Example:
{{"parties": ["John Doe", "ABC Corp"], "dates": ["2024-01-15"], "amounts": ["$50,000"], "locations": ["New York"], "case_numbers": ["2024-CV-12345"], "articles": ["Article 5", "Section 12"]}}

JSON:"""
    
    try:
        comp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        result = comp.choices[0].message.content.strip()
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except Exception as e:
        return {}


def build_timeline(entities: Dict) -> List[Dict]:
    """Build chronological timeline from extracted dates."""
    timeline = []
    dates = entities.get("dates", [])
    for date in dates:
        timeline.append({"date": date, "event": f"Event on {date}"})
    return sorted(timeline, key=lambda x: x.get("date", ""))


# FEATURE 3: Advanced Search Functions
def search_in_documents(query: str, doc_filter: Optional[str] = None, 
                       date_from: Optional[str] = None, date_to: Optional[str] = None) -> List:
    """Advanced search with filters across documents."""
    results = []
    manifest = load_manifest()
    
    for entry in manifest:
        # Apply document filter
        if doc_filter and doc_filter not in entry.get("name", ""):
            continue
        
        # Apply date filter
        created = entry.get("created_at", "")
        if date_from and created < date_from:
            continue
        if date_to and created > date_to:
            continue
        
        # Load and search
        model_name = resolve_embedding_model_name(entry.get("embed_model"))
        faiss_db = load_vector_store(entry["db_path"], model_name)
        if faiss_db:
            docs = faiss_db.similarity_search(query, k=3)
            for doc in docs:
                results.append({
                    "doc_name": entry.get("name"),
                    "content": doc.page_content[:300],
                    "metadata": getattr(doc, "metadata", {})
                })
    
    return results


def save_search_query(query: str, filters: Dict):
    """Save search query for quick access."""
    if "saved_searches" not in st.session_state:
        st.session_state["saved_searches"] = []
    st.session_state["saved_searches"].append({
        "query": query,
        "filters": filters,
        "timestamp": datetime.now().isoformat()
    })


# FEATURE 4: Document Tagging
def add_tag_to_document(doc_id: str, tag: str):
    """Add tag to document in manifest."""
    entries = load_manifest()
    for entry in entries:
        if entry.get("doc_id") == doc_id:
            tags = entry.get("tags", [])
            if tag not in tags:
                tags.append(tag)
            entry["tags"] = tags
            break
    save_manifest(entries)


def get_documents_by_tag(tag: str) -> List[Dict]:
    """Get all documents with specific tag."""
    entries = load_manifest()
    return [e for e in entries if tag in e.get("tags", [])]


# FEATURE 5: Export Utilities
def create_download_link(content: str, filename: str, file_type: str = "html") -> str:
    """Create download link for generated reports."""
    b64 = base64.b64encode(content.encode()).decode()
    mime_types = {
        "html": "text/html",
        "txt": "text/plain",
        "json": "application/json"
    }
    mime = mime_types.get(file_type, "text/plain")
    return f'<a href="data:{mime};base64,{b64}" download="{filename}">📥 Download {filename}</a>'


# ==================== ADVANCED FEATURES ====================

# FEATURE 6: AI Contract Review & Risk Assessment Engine
def analyze_contract_risks(text: str, doc_type: str = "contract") -> Dict:
    """
    Advanced contract risk analysis using AI.
    Returns risk assessment, obligations, missing clauses, and recommendations.
    """
    prompt = f"""You are an expert legal contract analyst. Analyze this {doc_type} and provide a comprehensive risk assessment.

Analyze for:
1. HIGH RISK items (unlimited liability, unusual penalties, unfair terms)
2. MEDIUM RISK items (long notice periods, restrictive covenants, ambiguous terms)
3. LOW RISK items (standard clauses)
4. OBLIGATIONS (all "must", "shall", "required" commitments)
5. MISSING clauses (force majeure, confidentiality, termination, etc.)
6. RED FLAGS (unusual language, hidden costs, automatic renewals)

Document Text:
{text[:6000]}

Return ONLY valid JSON:
{{
  "overall_risk_score": "HIGH|MEDIUM|LOW",
  "risk_summary": "Brief 2-3 sentence summary",
  "high_risks": [
    {{"clause": "description", "location": "page/section", "issue": "why risky", "recommendation": "what to do"}}
  ],
  "medium_risks": [
    {{"clause": "description", "location": "page/section", "issue": "concern"}}
  ],
  "low_risks": [
    {{"clause": "description", "location": "page/section"}}
  ],
  "obligations": [
    {{"party": "who", "obligation": "what must be done", "deadline": "when", "penalty": "consequence if not done"}}
  ],
  "missing_clauses": ["clause_name", "clause_name"],
  "red_flags": [
    {{"flag": "issue description", "severity": "HIGH|MEDIUM|LOW", "recommendation": "action"}}
  ],
  "compliance_issues": ["issue1", "issue2"],
  "key_terms": {{
    "termination_notice_days": 0,
    "liability_cap": "none|limited|amount",
    "non_compete_months": 0,
    "governing_law": "jurisdiction"
  }}
}}

JSON:"""
    
    try:
        comp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=3000,
        )
        result = comp.choices[0].message.content.strip()
        # Extract JSON
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}
    except Exception as e:
        return {"error": str(e), "overall_risk_score": "UNKNOWN"}


def generate_risk_report_html(risk_data: Dict, doc_name: str) -> str:
    """Generate beautiful HTML report for risk assessment."""
    if not risk_data or "error" in risk_data:
        return "<p>Unable to generate risk report.</p>"
    
    risk_colors = {"HIGH": "#dc3545", "MEDIUM": "#ffc107", "LOW": "#28a745"}
    overall_risk = risk_data.get("overall_risk_score", "UNKNOWN")
    risk_color = risk_colors.get(overall_risk, "#6c757d")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Risk Assessment - {doc_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 20px auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; }}
            .risk-badge {{ display: inline-block; padding: 8px 20px; border-radius: 20px; font-weight: bold; 
                          background: {risk_color}; color: white; font-size: 1.2em; }}
            .section {{ margin: 25px 0; padding: 20px; border-left: 4px solid #667eea; background: #f8f9fa; border-radius: 5px; }}
            .risk-item {{ margin: 15px 0; padding: 15px; border-radius: 5px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .risk-high {{ border-left: 5px solid #dc3545; }}
            .risk-medium {{ border-left: 5px solid #ffc107; }}
            .risk-low {{ border-left: 5px solid #28a745; }}
            .obligation {{ background: #e3f2fd; padding: 12px; margin: 8px 0; border-radius: 5px; border-left: 3px solid #2196f3; }}
            .red-flag {{ background: #ffebee; padding: 12px; margin: 8px 0; border-radius: 5px; border-left: 3px solid #f44336; }}
            .missing {{ color: #d32f2f; font-weight: 600; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th {{ background: #667eea; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
            .recommendation {{ background: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px; 
                             border-left: 3px solid #856404; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛡️ Contract Risk Assessment Report</h1>
            <p><strong>Document:</strong> {doc_name}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
            <p><strong>Overall Risk Level:</strong> <span class="risk-badge">{overall_risk}</span></p>
        </div>
        
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <p style="font-size: 1.1em; line-height: 1.6;">{risk_data.get('risk_summary', 'No summary available.')}</p>
        </div>
    """
    
    # High Risks
    high_risks = risk_data.get("high_risks", [])
    if high_risks:
        html += '<div class="section"><h2>🚨 HIGH RISK ITEMS</h2>'
        for risk in high_risks:
            html += f"""
            <div class="risk-item risk-high">
                <h3>⚠️ {risk.get('clause', 'Unknown Clause')}</h3>
                <p><strong>Location:</strong> {risk.get('location', 'Not specified')}</p>
                <p><strong>Issue:</strong> {risk.get('issue', 'Not specified')}</p>
                <div class="recommendation">
                    <strong>💡 Recommendation:</strong> {risk.get('recommendation', 'Review with legal counsel')}
                </div>
            </div>
            """
        html += '</div>'
    
    # Medium Risks
    medium_risks = risk_data.get("medium_risks", [])
    if medium_risks:
        html += '<div class="section"><h2>⚡ MEDIUM RISK ITEMS</h2>'
        for risk in medium_risks:
            html += f"""
            <div class="risk-item risk-medium">
                <h3>⚠ {risk.get('clause', 'Unknown Clause')}</h3>
                <p><strong>Location:</strong> {risk.get('location', 'Not specified')}</p>
                <p><strong>Concern:</strong> {risk.get('issue', 'Not specified')}</p>
            </div>
            """
        html += '</div>'
    
    # Obligations
    obligations = risk_data.get("obligations", [])
    if obligations:
        html += '<div class="section"><h2>📋 CONTRACTUAL OBLIGATIONS</h2>'
        for obl in obligations:
            html += f"""
            <div class="obligation">
                <strong>Party:</strong> {obl.get('party', 'Not specified')}<br>
                <strong>Must Do:</strong> {obl.get('obligation', 'Not specified')}<br>
                <strong>Deadline:</strong> {obl.get('deadline', 'Not specified')}<br>
                <strong>Penalty:</strong> {obl.get('penalty', 'Not specified')}
            </div>
            """
        html += '</div>'
    
    # Red Flags
    red_flags = risk_data.get("red_flags", [])
    if red_flags:
        html += '<div class="section"><h2>🚩 RED FLAGS</h2>'
        for flag in red_flags:
            html += f"""
            <div class="red-flag">
                <strong>⚠️ {flag.get('flag', 'Issue')}</strong> 
                [Severity: {flag.get('severity', 'UNKNOWN')}]<br>
                <strong>Action:</strong> {flag.get('recommendation', 'Review carefully')}
            </div>
            """
        html += '</div>'
    
    # Missing Clauses
    missing = risk_data.get("missing_clauses", [])
    if missing:
        html += '<div class="section"><h2>❌ MISSING CLAUSES</h2><ul>'
        for clause in missing:
            html += f'<li class="missing">• {clause}</li>'
        html += '</ul></div>'
    
    # Key Terms Table
    key_terms = risk_data.get("key_terms", {})
    if key_terms:
        html += '''<div class="section"><h2>🔑 KEY TERMS SUMMARY</h2>
        <table>
            <tr><th>Term</th><th>Value</th></tr>'''
        for key, value in key_terms.items():
            html += f'<tr><td>{key.replace("_", " ").title()}</td><td>{value}</td></tr>'
        html += '</table></div>'
    
    html += '''
        <div class="section" style="text-align: center; color: #666;">
            <p><strong>DISCLAIMER:</strong> This is an AI-generated risk assessment. 
            Always consult with qualified legal counsel before making decisions.</p>
        </div>
    </body>
    </html>
    '''
    return html


# FEATURE 7: Smart Legal Citation Validator & Research Assistant
def extract_citations(text: str) -> List[Dict]:
    """Extract legal citations from text using pattern matching and AI."""
    # Common citation patterns
    patterns = [
        r'(\d+)\s+U\.S\.\s+(\d+)',  # US Supreme Court
        r'(\d+)\s+F\.\s*(?:2d|3d|4th|App\.)\s+(\d+)',  # Federal cases
        r'(\d+)\s+S\.\s*Ct\.\s+(\d+)',  # Supreme Court Reporter
        r'(\w+)\s+v\.\s+(\w+)',  # Case names
        r'(\d+)\s+U\.S\.C\.\s*§\s*(\d+)',  # US Code
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            citations.append({
                "text": match.group(0),
                "type": "case" if "v." in match.group(0) else "statute",
                "position": match.span()
            })
    
    # Use AI to extract more complex citations
    prompt = f"""Extract ALL legal citations from this text. Include case names, statute references, and regulatory citations.

Text:
{text[:4000]}

Return JSON array:
[
  {{"citation": "Brown v. Board of Education, 347 U.S. 483 (1954)", "type": "case", "year": 1954}},
  {{"citation": "42 U.S.C. § 1983", "type": "statute", "year": null}}
]

JSON:"""
    
    try:
        comp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        result = comp.choices[0].message.content.strip()
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            ai_citations = json.loads(json_match.group())
            citations.extend(ai_citations)
    except Exception:
        pass
    
    return citations


def validate_citation(citation: str) -> Dict:
    """
    Validate a legal citation and provide metadata.
    Note: In production, this would connect to legal databases like Westlaw, LexisNexis, or Case.Law API
    """
    # Simulate validation (in real app, would use legal APIs)
    validation = {
        "citation": citation,
        "status": "unknown",
        "is_valid": True,
        "precedential_value": "unknown",
        "cited_by_count": 0,
        "year": None,
        "court": "unknown",
        "subsequent_history": [],
        "related_cases": [],
        "scholar_link": None,
        "bluebook_format": citation,
    }
    
    # Try to extract year
    year_match = re.search(r'\((\d{4})\)', citation)
    if year_match:
        validation["year"] = year_match.group(1)
    
    # Generate Google Scholar link
    search_query = citation.replace(" ", "+")
    validation["scholar_link"] = f"https://scholar.google.com/scholar?q={search_query}"
    
    # Use AI to analyze citation
    prompt = f"""Analyze this legal citation and provide information about its validity and importance.

Citation: {citation}

Return JSON:
{{
  "status": "good_law|overruled|modified|questioned",
  "is_valid": true|false,
  "court": "court name",
  "precedential_value": "binding|persuasive|low",
  "summary": "brief summary of what this case/statute is about",
  "key_holding": "main legal principle",
  "warnings": "any issues with using this citation"
}}

JSON:"""
    
    try:
        comp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        result = comp.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            ai_data = json.loads(json_match.group())
            validation.update(ai_data)
    except Exception:
        pass
    
    return validation


def generate_citation_report(citations: List[Dict], validations: List[Dict], doc_name: str) -> str:
    """Generate HTML report for citation analysis."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Citation Analysis - {doc_name}</title>
        <style>
            body {{ font-family: 'Georgia', serif; max-width: 1000px; margin: 20px auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; border-radius: 10px; }}
            .citation-card {{ margin: 20px 0; padding: 20px; border-radius: 8px; background: white; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .status-good {{ border-left: 5px solid #28a745; }}
            .status-overruled {{ border-left: 5px solid #dc3545; }}
            .status-questioned {{ border-left: 5px solid #ffc107; }}
            .status-unknown {{ border-left: 5px solid #6c757d; }}
            .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }}
            .badge-success {{ background: #d4edda; color: #155724; }}
            .badge-danger {{ background: #f8d7da; color: #721c24; }}
            .badge-warning {{ background: #fff3cd; color: #856404; }}
            .links {{ margin-top: 10px; }}
            .links a {{ margin-right: 15px; color: #007bff; text-decoration: none; }}
            .links a:hover {{ text-decoration: underline; }}
            h2 {{ color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📚 Legal Citation Analysis Report</h1>
            <p><strong>Document:</strong> {doc_name}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
            <p><strong>Citations Found:</strong> {len(citations)}</p>
        </div>
        
        <h2>📖 Citation Details</h2>
    """
    
    for idx, (citation, validation) in enumerate(zip(citations, validations), 1):
        status = validation.get("status", "unknown").replace("_", " ").title()
        status_class = f"status-{validation.get('status', 'unknown').split('_')[0]}"
        
        is_valid = validation.get("is_valid", False)
        badge_class = "badge-success" if is_valid else "badge-danger"
        badge_text = "✅ Valid" if is_valid else "❌ Invalid"
        
        html += f"""
        <div class="citation-card {status_class}">
            <h3>Citation #{idx}</h3>
            <p style="font-size: 1.1em; font-weight: 600;">{citation.get('citation', 'Unknown')}</p>
            <p><span class="badge {badge_class}">{badge_text}</span> 
               <span class="badge badge-warning">Status: {status}</span></p>
            
            {f'<p><strong>Court:</strong> {validation.get("court", "Unknown")}</p>' if validation.get("court") else ''}
            {f'<p><strong>Year:</strong> {validation.get("year", "Unknown")}</p>' if validation.get("year") else ''}
            {f'<p><strong>Summary:</strong> {validation.get("summary", "No summary available")}</p>' if validation.get("summary") else ''}
            {f'<p><strong>Key Holding:</strong> {validation.get("key_holding", "Not available")}</p>' if validation.get("key_holding") else ''}
            {f'<p style="color: #d32f2f;"><strong>⚠️ Warning:</strong> {validation.get("warnings", "")}</p>' if validation.get("warnings") else ''}
            
            <div class="links">
                {f'<a href="{validation.get("scholar_link", "#")}" target="_blank">🔍 Google Scholar</a>' if validation.get("scholar_link") else ''}
                <a href="#">📘 Bluebook Format</a>
                <a href="#">🔗 Related Cases</a>
            </div>
        </div>
        """
    
    html += '''
        <div style="text-align: center; margin-top: 40px; color: #666;">
            <p><strong>NOTE:</strong> Citation validation uses AI analysis. For authoritative verification, 
            consult official legal databases like Westlaw, LexisNexis, or Shepard's Citations.</p>
        </div>
    </body>
    </html>
    '''
    return html


# ==================== GAME-CHANGING FEATURES (8-12) ====================

# FEATURE 8: Auto Legal Document Generator
def generate_legal_document(doc_type: str, details: Dict) -> str:
    """Generate professional legal documents"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = f"""You are an expert legal document drafter. Generate a professional, legally sound {doc_type} document.

User Details:
{json.dumps(details, indent=2)}

Create a complete, court-ready document with:
- Proper legal format and structure
- Relevant legal provisions and citations
- Professional language
- All necessary sections
- Date, signatures, and verification

Generate the complete document now:"""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=3000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating document: {str(e)}"


# FEATURE 9: Case Strength Analyzer
def analyze_case_strength(case_details: str, case_type: str) -> Dict:
    """Analyze case and predict win probability"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = f"""You are a senior legal analyst. Analyze this {case_type} case and predict winning probability.

Case Details:
{case_details}

Provide analysis in JSON format:
{{
    "win_probability": 75,
    "confidence_level": "HIGH",
    "strength_factors": ["Strong evidence", "Clear liability"],
    "weakness_factors": ["Missing witness", "Delayed filing"],
    "evidence_analysis": {{
        "current_evidence": "Documents, photos, etc.",
        "evidence_strength": "STRONG",
        "missing_evidence": ["Police report", "Medical records"]
    }},
    "legal_precedents": [
        {{"case_name": "ABC vs XYZ", "relevance": "Similar facts", "outcome": "Won"}}
    ],
    "opponent_position": {{
        "likely_defense": "Denial of liability",
        "defense_strength": "WEAK"
    }},
    "strategic_recommendations": ["Collect missing evidence", "File within limitation"],
    "timeline_estimate": "6-12 months",
    "cost_estimate": "Rs. 50,000 - 1,00,000",
    "settlement_advice": "Strong case, proceed to trial",
    "action_plan": [
        {{"step": 1, "action": "Gather all evidence", "priority": "HIGH"}},
        {{"step": 2, "action": "File complaint", "priority": "HIGH"}}
    ]
}}

Be realistic based on Indian legal system:"""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=2500
        )
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "Could not parse analysis", "raw": content}
    except Exception as e:
        return {"error": str(e)}


# FEATURE 10: Court Filing Guide
def get_court_filing_guide(case_type: str, case_value: float, location: str) -> Dict:
    """Provide step-by-step court filing guidance"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = f"""Provide complete step-by-step guide to file a {case_type} case in India.

Case Type: {case_type}
Case Value: Rs. {case_value:,.0f}
Location: {location}

Return JSON format:
{{
    "court_type": "District Consumer Forum / Civil Court / etc.",
    "jurisdiction": "Explanation of jurisdiction",
    "required_documents": [
        {{"name": "Complaint/Plaint", "description": "Main petition", "mandatory": true}},
        {{"name": "Affidavit", "description": "Sworn statement", "mandatory": true}}
    ],
    "court_fees": {{
        "filing_fee": 5000,
        "stamp_duty": 500,
        "miscellaneous": 300,
        "total": 5800,
        "payment_mode": "Cash/DD/Online"
    }},
    "step_by_step_procedure": [
        {{"step": 1, "title": "Draft Documents", "description": "Prepare complaint/plaint", "tips": ["Use proper format", "Include all facts"]}},
        {{"step": 2, "title": "Get Documents Notarized", "description": "Affidavit notarization", "tips": ["Bring ID proof"]}},
        {{"step": 3, "title": "Pay Court Fees", "description": "Pay at court counter", "tips": ["Keep receipt safe"]}}
    ],
    "timeline": {{
        "filing_time": "Same day",
        "first_hearing": "30-45 days",
        "case_duration": "6-18 months"
    }},
    "common_mistakes": ["Incomplete documents", "Wrong court", "Missing signatures"],
    "helpful_tips": ["File within limitation period", "Keep all copies", "Arrive early on hearing"],
    "what_to_say_at_counter": "I want to file a {case_type} case. Here are my documents.",
    "hearing_preparation": {{
        "what_to_wear": "Formal/traditional attire",
        "court_etiquette": ["Stand when judge enters", "Address as Your Honor", "No mobile phones"],
        "how_to_present": "Speak clearly, stick to facts, be respectful"
    }}
}}

Base on practical Indian court procedures:"""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=3000
        )
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "Could not parse guide", "raw": content}
    except Exception as e:
        return {"error": str(e)}


# FEATURE 11: Enhanced AI Chatbot with Advanced Features
def chatbot_conversation(user_message: str, conversation_history: List[Dict], context_docs: List = None, language: str = "English") -> Dict:
    """Super Enhanced conversational AI lawyer with multi-language support"""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    doc_context = ""
    if context_docs:
        doc_context = "\n\nRelevant Legal Context:\n" + "\n".join([doc.page_content[:500] for doc in context_docs[:3]])
    
    # Language-specific system prompts
    language_instructions = {
        "English": "Respond in professional English",
        "Hindi": "Respond in simple Hindi (Devanagari script). Use common legal terms with English equivalents in brackets.",
        "Hinglish": "Respond in Hinglish (mix of Hindi and English). Be conversational and easy to understand."
    }
    
    messages = [{"role": "system", "content": f"""You are an expert AI lawyer providing FREE legal consultation to common people in India.

Your Role:
- Act like a SENIOR ADVOCATE with 20+ years experience
- Ask 2-3 SPECIFIC clarifying follow-up questions
- Provide PERSONALIZED advice based on user's specific situation
- Cite relevant Indian laws, sections, IPC, CrPC, CPC, and precedents
- Be empathetic, understanding, and supportive
- Explain complex legal terms in SIMPLE language
- Suggest PRACTICAL next steps with timelines
- Remember the ENTIRE conversation context
- Warn about LIMITATION PERIODS and important deadlines
- Calculate compensation/damages when applicable
- Suggest whether to settle or fight

Communication Style:
- Professional yet FRIENDLY and approachable
- Ask SPECIFIC questions to understand case better
- Provide ACTIONABLE, step-by-step advice with priorities
- Suggest when to get written documentation/evidence
- Explain consequences and all available options clearly
- Use examples and analogies for clarity
- Add urgency indicators (🔴 HIGH PRIORITY, 🟡 MEDIUM, 🟢 LOW)

Language: {language_instructions.get(language, "English")}

{doc_context}

IMPORTANT: End your response with 2-3 natural follow-up questions the user might ask, marked with [FOLLOWUP]."""}]
    
    # Add recent conversation history
    for msg in conversation_history[-10:]:
        messages.append(msg)
    
    # Add current message
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.6,
            max_tokens=1500
        )
        ai_response = response.choices[0].message.content
        
        # Extract follow-up questions
        followup_questions = []
        if "[FOLLOWUP]" in ai_response:
            parts = ai_response.split("[FOLLOWUP]")
            ai_response = parts[0].strip()
            if len(parts) > 1:
                followup_text = parts[1].strip()
                followup_questions = [q.strip() for q in re.findall(r'[^.!?]+\?', followup_text)]
        else:
            questions = re.findall(r'[^.!?]+\?', ai_response)
            followup_questions = questions[:3] if questions else []
        
        # Detect urgency level
        urgency = "MEDIUM"
        if any(word in user_message.lower() for word in ["urgent", "emergency", "immediate", "asap", "deadline"]):
            urgency = "HIGH"
        elif any(word in user_message.lower() for word in ["query", "information", "curious", "general"]):
            urgency = "LOW"
        
        # Generate contextual quick replies
        quick_replies = [
            "What documents do I need?",
            "What are the next steps?",
            "How much time do I have?",
            "What will it cost?",
            "Can I handle this myself?"
        ]
        
        return {
            "response": ai_response,
            "followup_questions": followup_questions,
            "urgency": urgency,
            "quick_replies": quick_replies,
            "conversation_length": len(conversation_history),
            "language": language
        }
    except Exception as e:
        return {
            "error": str(e), 
            "response": "I apologize, I'm having trouble processing your request. Please try again or rephrase your question.",
            "followup_questions": [],
            "urgency": "MEDIUM",
            "quick_replies": []
        }


# FEATURE 12: Generate Conversation Summary
def generate_conversation_summary(conversation_history: List[Dict]) -> str:
    """Generate professional summary of legal consultation"""
    if not conversation_history:
        return "No conversation yet."
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    conv_text = "\n\n".join([f"{'User' if m['role']=='user' else 'AI Lawyer'}: {m['content']}" 
                             for m in conversation_history])
    
    prompt = f"""Summarize this legal consultation in professional format:

{conv_text}

Provide concise summary with:
1. MAIN LEGAL ISSUE
2. KEY FACTS
3. ADVICE PROVIDED
4. RECOMMENDED ACTIONS
5. TIMELINE/DEADLINES
6. NEXT STEPS"""
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except:
        return "Summary generation failed."


# FEATURE 13: Visual Case Strength Gauge
def generate_case_strength_visual(win_probability: int) -> str:
    """Generate animated HTML gauge for win probability"""
    if win_probability >= 70:
        color, label, icon = "#39ff14", "STRONG CASE", "🟢"
    elif win_probability >= 50:
        color, label, icon = "#FFD700", "MODERATE CASE", "🟡"
    else:
        color, label, icon = "#ff006e", "WEAK CASE", "🔴"
    
    return f"""
    <div style="text-align: center; margin: 2rem 0;">
        <div style="font-size: 5rem; margin-bottom: 1rem; animation: pulse 2s infinite;">{icon}</div>
        <div style="position: relative; width: 300px; height: 300px; margin: 0 auto;">
            <svg width="300" height="300" viewBox="0 0 300 300">
                <circle cx="150" cy="150" r="120" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="30"/>
                <circle cx="150" cy="150" r="120" fill="none" stroke="{color}" stroke-width="30"
                        stroke-dasharray="{win_probability * 7.54} 754" stroke-linecap="round"
                        transform="rotate(-90 150 150)" style="filter: drop-shadow(0 0 10px {color});">
                    <animate attributeName="stroke-dasharray" from="0 754" to="{win_probability * 7.54} 754" 
                             dur="2s" fill="freeze"/>
                </circle>
            </svg>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                <div style="font-size: 4rem; font-weight: 900; color: {color};">{win_probability}%</div>
                <div style="font-size: 1.2rem; color: #9fa8da; margin-top: 0.5rem;">WIN PROBABILITY</div>
            </div>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin-top: 1.5rem;">{label}</div>
    </div>
    """

def generate_bench_tension_visual(tension_level: int) -> str:
    """Generate a 'Bench Tension Meter' visual gauge"""
    # Inverse color: low tension (green), high tension (red)
    if tension_level <= 30:
        color, label = "#39ff14", "FRIENDLY BENCH"
    elif tension_level <= 70:
        color, label = "#FFea00", "MODERATE SKEPTICISM"
    else:
        color, label = "#ff006e", "HOSTILE BENCH / HIGH TENSION"
    
    return f"""
    <div style="text-align: center; margin: 1rem 0; padding: 1rem; background: rgba(0,0,0,0.3); border-radius: 15px; border: 1px solid {color}44;">
        <div style="font-size: 0.9rem; color: #9fa8da; margin-bottom: 0.5rem; letter-spacing: 2px;">BENCH TENSION METER</div>
        <div style="position: relative; width: 100%; height: 20px; background: rgba(255,255,255,0.05); border-radius: 10px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
            <div style="width: {tension_level}%; height: 100%; background: linear-gradient(90deg, #39ff14 0%, #FFea00 50%, #ff006e 100%); transition: width 1s ease-in-out; position: relative;">
                <div style="position: absolute; right: 0; top: 0; width: 4px; height: 100%; background: white; box-shadow: 0 0 10px white;"></div>
            </div>
        </div>
        <div style="font-size: 1.1rem; font-weight: 800; color: {color}; margin-top: 0.8rem; text-shadow: 0 0 10px {color}66;">{label}</div>
    </div>
    """

def precedent_reliability_analyzer(precedent_text: str):
    """Analyze if a cited judgment is Binding, Persuasive, or Questioned."""
    prompt = f"""
    You are a Constitutional Law Expert. Analyze the following precedent/judgment for Reliability.
    
    Text: {precedent_text}

    Determine:
    1. Status: (Binding / Persuasive / Questioned / Overruled)
    2. Reliability Score: (0-100)
    3. Reason: Why is it ranked this way? (e.g., 'Supreme Court verdict', 'Dissenting opinion', 'Larger bench exists')

    Return as a concise report.
    """
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in Reliability Analyzer: {str(e)}"


# ==================== NEW REVOLUTIONARY FEATURES (14-18) ====================

# FEATURE 14: Contract Analyzer & Risk Detector
def analyze_contract_risk(contract_text: str, contract_type: str) -> Dict:
    """Analyze contract for risky clauses and unfair terms"""
    try:
        prompt = f"""You are a senior contract lawyer. Analyze this {contract_type} contract and identify ALL risky, unfair, or problematic clauses.

CONTRACT TEXT:
{contract_text}

Provide a comprehensive risk analysis in JSON format:
{{
    "risk_score": <0-100, where 100 is extremely risky>,
    "overall_verdict": "<HIGH RISK/MEDIUM RISK/LOW RISK> - <short verdict>",
    "red_flags": [
        {{
            "clause_number": "<e.g., 5.2>",
            "clause_text": "<exact problematic text>",
            "risk_level": "HIGH",
            "explanation": "<why this is risky>",
            "impact": "<what could go wrong>",
            "suggestion": "<how to fix it>"
        }}
    ],
    "yellow_flags": [<same structure for medium-risk items>],
    "safe_clauses": ["<list of fair clauses>"],
    "missing_protections": ["<what should be added>"],
    "negotiation_strategy": "<key points to negotiate>",
    "comparison_with_standard": "<how far from standard contract>",
    "recommended_changes": [
        {{"clause": "<number>", "change_to": "<suggested text>"}}
    ],
    "financial_risk": "<potential loss amount>"
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extract JSON from markdown if present
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(result_text)
    
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


# FEATURE 15: Court Hearing Script Generator
def generate_hearing_script(
    hearing_type: str,
    case_facts: str,
    evidence_list: str,
    opponent_claims: str,
    court_type: str
) -> Dict:
    """Generate professional court hearing dialogue script"""
    try:
        prompt = f"""You are a senior advocate preparing a client for court hearing. Generate a COMPLETE, PROFESSIONAL script for what to say in court.

HEARING TYPE: {hearing_type}
COURT TYPE: {court_type}
CASE FACTS: {case_facts}
YOUR EVIDENCE: {evidence_list}
OPPONENT'S CLAIMS: {opponent_claims}

Generate a comprehensive hearing script in JSON format:
{{
    "opening_statement": "<professional opening line>",
    "evidence_presentation": [
        {{
            "document": "<name of document>",
            "what_to_say": "<exact words to use>",
            "anticipated_objection": "<possible objection>",
            "counter_response": "<how to respond>"
        }}
    ],
    "examination_questions": ["<questions to ask your witness>"],
    "cross_examination_prep": {{
        "likely_questions": ["<questions opponent may ask>"],
        "best_answers": ["<how to respond>"]
    }},
    "objection_handling": {{
        "hearsay": "<how to object>",
        "irrelevant": "<how to object>",
        "leading": "<how to object>"
    }},
    "judge_interactions": [
        {{
            "likely_question": "<what judge may ask>",
            "answer": "<your response>",
            "followup": "<if judge asks more>"
        }}
    ],
    "closing_argument": "<final submission>",
    "dos_and_donts": ["<courtroom etiquette>"],
    "emergency_responses": {{
        "if_judge_angry": "<how to apologize>",
        "if_opponent_lying": "<how to counter>",
        "if_confused": "<how to ask clarification>"
    }},
    "time_estimate": "<duration>",
    "confidence_tips": ["<presentation tips>"]
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(result_text)
    
    except Exception as e:
        return {"error": f"Script generation failed: {str(e)}"}


# FEATURE 16: Evidence Organizer & Case File Builder
def organize_evidence(case_type: str, claims_made: str, evidence_description: str) -> Dict:
    """Organize evidence intelligently and create case file"""
    try:
        prompt = f"""You are a legal case manager. Organize the evidence for a {case_type} case.

CLAIMS MADE: {claims_made}
EVIDENCE AVAILABLE: {evidence_description}

Provide intelligent evidence organization in JSON format:
{{
    "evidence_by_category": {{
        "documentary": [
            {{
                "serial_no": 1,
                "type": "<invoice/contract/etc>",
                "description": "<what it proves>",
                "relevance": "<why important>",
                "supports_claim": "<which claim>",
                "certification_needed": "<what certification>",
                "strength": "STRONG/MODERATE/WEAK"
            }}
        ],
        "photographic": [<similar structure>],
        "electronic": [<similar structure>]
    }},
    "evidence_timeline": [
        {{"date": "<date>", "event": "<what happened>", "proof": "<document>"}}
    ],
    "evidence_strength_map": {{
        "strong_evidence": ["<list>"],
        "weak_evidence": ["<list>"],
        "missing_evidence": ["<what's missing>"]
    }},
    "claim_proof_linking": {{
        "<Claim 1>": ["<proofs>"],
        "<Claim 2>": ["<proofs>"]
    }},
    "missing_critical_evidence": [
        {{
            "evidence_type": "<what's missing>",
            "why_critical": "<importance>",
            "how_to_get": "<advice>",
            "urgency": "HIGH/MEDIUM/LOW"
        }}
    ],
    "suggestions": ["<improvement suggestions>"],
    "case_file_structure": {{
        "cover_page": "Case details",
        "index": "<document list>",
        "total_documents": <number>,
        "total_pages": <estimated>
    }},
    "estimated_case_strength": "<rating>/10"
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(result_text)
    
    except Exception as e:
        return {"error": f"Evidence organization failed: {str(e)}"}


# FEATURE 17: Settlement Calculator & Negotiation Strategist
def calculate_settlement_amount(
    case_type: str,
    actual_loss: float,
    evidence_strength: int,
    opponent_liability: int,
    case_duration_months: int
) -> Dict:
    """Calculate fair settlement and generate negotiation strategy"""
    try:
        prompt = f"""You are a settlement negotiation expert. Calculate fair settlement amount and strategy.

CASE TYPE: {case_type}
ACTUAL LOSS: ₹{actual_loss:,.0f}
EVIDENCE STRENGTH: {evidence_strength}/100
OPPONENT LIABILITY: {opponent_liability}/100
CASE DURATION: {case_duration_months} months

Provide comprehensive settlement analysis in JSON format:
{{
    "settlement_analysis": {{
        "worst_case_scenario": {{
            "amount": <minimum amount>,
            "explanation": "<why this is minimum>",
            "probability": "<percentage>"
        }},
        "realistic_settlement": {{
            "amount": <expected amount>,
            "breakdown": {{
                "actual_loss": <amount>,
                "compensation": <amount>,
                "interest": <amount>,
                "litigation_costs": <amount>
            }},
            "probability": "<percentage>"
        }},
        "best_case_scenario": {{
            "amount": <maximum amount>,
            "explanation": "<why this is possible>",
            "probability": "<percentage>"
        }}
    }},
    "negotiation_strategy": {{
        "opening_demand": <amount to start with>,
        "walk_away_point": <minimum acceptable>,
        "counter_offer_responses": {{
            "if_they_offer_low": "<response>",
            "if_they_offer_medium": "<response>",
            "if_they_offer_acceptable": "<response>"
        }},
        "timing_advice": "<when to settle>",
        "leverage_points": ["<your advantages>"],
        "pressure_tactics": ["<negotiation tactics>"]
    }},
    "cost_benefit_analysis": {{
        "if_settle_now": {{
            "amount_received": <amount>,
            "time_saved": "<duration>",
            "costs_avoided": <amount>,
            "net_benefit": <amount>
        }},
        "if_fight_full_trial": {{
            "possible_award": "<range>",
            "time_required": "<duration>",
            "additional_costs": <amount>,
            "winning_probability": "<percentage>",
            "expected_value": <calculated amount>,
            "risk": "<downside>"
        }},
        "recommendation": "<SETTLE or FIGHT>"
    }},
    "settlement_deed_points": ["<key clauses to include>"]
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        return json.loads(result_text)
    
    except Exception as e:
        return {"error": f"Settlement calculation failed: {str(e)}"}


# FEATURE 18: Legal Timeline & Limitation Period Checker
def check_legal_timeline(case_type: str, incident_date: str, current_stage: str = "Not filed") -> Dict:
    """Calculate all legal deadlines and limitation periods"""
    try:
        from datetime import datetime, timedelta
        
        incident_dt = datetime.strptime(incident_date, "%Y-%m-%d")
        current_dt = datetime.now()
        
        prompt = f"""You are a legal timeline expert. Calculate limitation periods and deadlines for a {case_type} case.

INCIDENT DATE: {incident_date}
CURRENT DATE: {current_dt.strftime("%Y-%m-%d")}
CURRENT STAGE: {current_stage}

Provide comprehensive timeline analysis in JSON format:
{{
    "limitation_analysis": {{
        "case_type": "<type>",
        "applicable_law": "<law name and section>",
        "limitation_period": "<X years/months>",
        "last_date_to_file": "<YYYY-MM-DD>",
        "days_remaining": <number (negative if expired)>,
        "status": "SAFE/URGENT/TIME-BARRED",
        "urgency_level": "LOW/MEDIUM/HIGH/CRITICAL",
        "zone": "GREEN/YELLOW/RED/BLACK",
        "alternative_options": ["<if time-barred, what can be done>"]
    }},
    "upcoming_deadlines": [
        {{
            "deadline_type": "<type>",
            "due_date": "<YYYY-MM-DD>",
            "days_remaining": <number>,
            "urgency": "LOW/MEDIUM/HIGH",
            "consequence_if_missed": "<what happens>",
            "action_required": "<what to do>"
        }}
    ],
    "case_specific_timelines": {{
        "notice_period": "<if applicable>",
        "reply_deadline": "<if applicable>",
        "filing_window": "<period to file case>"
    }},
    "important_dates": {{
        "next_action_date": "<YYYY-MM-DD>",
        "critical_deadline": "<YYYY-MM-DD>"
    }},
    "smart_reminders": [
        {{"days_before": <number>, "message": "<reminder text>"}}
    ],
    "limitation_extensions": {{
        "grounds_for_extension": ["<possible grounds>"],
        "condonation_available": <true/false>,
        "success_probability": "<if delay, chance of condonation>"
    }},
    "visual_countdown": {{
        "days_left": <number>,
        "percentage_time_left": "<percentage>",
        "display_message": "<urgency message>"
    }}
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(result_text)
        
        # Add calculated values
        days_passed = (current_dt - incident_dt).days
        result["days_since_incident"] = days_passed
        
        return result
    
    except Exception as e:
        return {"error": f"Timeline check failed: {str(e)}"}


# FEATURE 19: Case Outcome Predictor
def predict_case_outcome(
    case_type: str,
    case_facts: str,
    evidence_strength: int,
    court_level: str,
    desired_relief: str
) -> Dict:
    """Predict possible case outcomes and risks"""
    try:
        prompt = f"""You are a senior legal analyst. Predict realistic outcomes for this case.

CASE TYPE: {case_type}
COURT LEVEL: {court_level}
EVIDENCE STRENGTH: {evidence_strength}/100
DESIRED RELIEF: {desired_relief}
FACTS: {case_facts}

Return JSON:
{{
    "outcome_probability": <0-100>,
    "likely_outcomes": ["<outcome 1>", "<outcome 2>"],
    "risk_factors": ["<risk 1>", "<risk 2>"],
    "strengthening_steps": ["<step 1>", "<step 2>"],
    "settlement_advice": "<when to settle and why>",
    "confidence_note": "<why this prediction>"
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2500
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Outcome prediction failed: {str(e)}"}


# FEATURE 20: Legal Strategy Planner
def build_legal_strategy(case_type: str, case_facts: str, current_stage: str, constraints: str) -> Dict:
    """Build a step-by-step legal strategy"""
    try:
        prompt = f"""You are a litigation strategist. Create a step-by-step legal plan.

CASE TYPE: {case_type}
CURRENT STAGE: {current_stage}
CONSTRAINTS: {constraints}
FACTS: {case_facts}

Return JSON:
{{
    "strategy_steps": [
        {{"step": "<action>", "purpose": "<why>", "timeline": "<when>"}}
    ],
    "documents_needed": ["<doc 1>", "<doc 2>"],
    "timeline_overview": "<summary timeline>",
    "risk_traps": ["<pitfall 1>", "<pitfall 2>"],
    "next_best_action": "<most urgent step>"
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2500
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Strategy planning failed: {str(e)}"}


# FEATURE 21: Legal Fees & Cost Estimator
def estimate_legal_costs(
    case_type: str,
    court_level: str,
    hearings_count: int,
    document_pages: int,
    travel_distance_km: int
) -> Dict:
    """Estimate legal costs for a case"""
    try:
        prompt = f"""You are a legal cost estimator. Provide realistic cost ranges in INR.

CASE TYPE: {case_type}
COURT LEVEL: {court_level}
EXPECTED HEARINGS: {hearings_count}
DOCUMENT PAGES: {document_pages}
TRAVEL DISTANCE: {travel_distance_km} km

Return JSON:
{{
    "court_fee_estimate": {{"min": <amount>, "max": <amount>, "notes": "<why>"}},
    "lawyer_fee_range": {{"min": <amount>, "max": <amount>, "notes": "<why>"}},
    "misc_costs": {{"filing": <amount>, "copies": <amount>, "travel": <amount>, "misc": <amount>}},
    "total_range": {{"min": <amount>, "max": <amount>}},
    "cost_saving_tips": ["<tip 1>", "<tip 2>"]
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Cost estimation failed: {str(e)}"}


# FEATURE 22: Simple-Language Legal Explainer
def explain_in_simple_language(document_text: str, language: str) -> Dict:
    """Explain legal text in simple language"""
    try:
        if language == "Both":
            prompt = f"""You are a legal explainer. Convert the text into simple language for a non-lawyer.
Provide BOTH English and Hindi outputs.

TEXT: {document_text}

Return JSON:
{{
    "english": {{
        "simple_summary": "<short summary>",
        "key_points": ["<point 1>", "<point 2>"],
        "action_steps": ["<step 1>", "<step 2>"],
        "risk_notes": ["<risk 1>", "<risk 2>"],
        "questions_to_ask": ["<question 1>", "<question 2>"]
    }},
    "hindi": {{
        "simple_summary": "<short summary>",
        "key_points": ["<point 1>", "<point 2>"],
        "action_steps": ["<step 1>", "<step 2>"],
        "risk_notes": ["<risk 1>", "<risk 2>"],
        "questions_to_ask": ["<question 1>", "<question 2>"]
    }}
}}"""
        else:
            target_lang = "English" if language == "English" else "Hindi"
            prompt = f"""You are a legal explainer. Convert the text into simple language for a non-lawyer.

TARGET LANGUAGE: {target_lang}
TEXT: {document_text}

Return JSON:
{{
    "simple_summary": "<short summary>",
    "key_points": ["<point 1>", "<point 2>"],
    "action_steps": ["<step 1>", "<step 2>"],
    "risk_notes": ["<risk 1>", "<risk 2>"],
    "questions_to_ask": ["<question 1>", "<question 2>"]
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2500
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Simple explanation failed: {str(e)}"}


# FEATURE 23: Opposition Reply Generator
def generate_opposition_reply(opponent_notice: str, your_facts: str, evidence_list: str, desired_outcome: str) -> Dict:
    """Draft a reply to opponent's notice or claims"""
    try:
        prompt = f"""You are a legal drafting expert. Create a professional reply.

OPPONENT NOTICE/CLAIMS: {opponent_notice}
YOUR FACTS: {your_facts}
EVIDENCE: {evidence_list}
DESIRED OUTCOME: {desired_outcome}

Return JSON:
{{
    "reply_draft": "<full reply draft>",
    "key_denials": ["<denial 1>", "<denial 2>"],
    "counter_demands": ["<demand 1>", "<demand 2>"],
    "annexures": ["<document list>"],
    "send_checklist": ["<sending steps>"]
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3000
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Reply generation failed: {str(e)}"}


# FEATURE 24: Legal News Fetcher
def fetch_legal_news(rss_urls: List[str], max_items: int = 12) -> Dict:
    """Fetch legal news from RSS/Atom feeds"""
    items: List[Dict] = []
    errors: List[str] = []
    for url in rss_urls:
        if not url.strip():
            continue
        try:
            resp = requests.get(url.strip(), timeout=8)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            source_name = urllib.parse.urlparse(url).netloc or url

            channel = root.find("channel")
            if channel is not None:
                for item in channel.findall("item"):
                    title = (item.findtext("title") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    pub = (item.findtext("pubDate") or "").strip()
                    if title:
                        items.append({"title": title, "link": link, "published": pub, "source": source_name})
            else:
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                    link_el = entry.find("atom:link", ns)
                    link = link_el.get("href") if link_el is not None else ""
                    pub = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
                    if not pub:
                        pub = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
                    if title:
                        items.append({"title": title, "link": link, "published": pub, "source": source_name})
        except Exception as e:
            errors.append(f"{url} -> {str(e)}")

    return {"items": items[:max_items], "errors": errors}


# FEATURE 25: Legal News Summarizer
def summarize_legal_news(items: List[Dict], language: str) -> Dict:
    """Summarize legal news headlines with AI"""
    if not items:
        return {"error": "No news items to summarize."}

    headlines = "\n".join([f"- {i.get('title')} ({i.get('source')})" for i in items])

    def run_summary(target_lang: str) -> str:
        prompt = f"""Summarize these legal news headlines into 5-7 bullet points.
Use {target_lang}. Add a short 'Impact' line for ordinary people.

HEADLINES:\n{headlines}

Return plain text."""
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()

    try:
        if language == "Both":
            return {
                "english": run_summary("English"),
                "hindi": run_summary("Hindi")
            }
        return {"summary": run_summary(language)}
    except Exception as e:
        return {"error": f"News summary failed: {str(e)}"}


# FEATURE 26: Judge Sahib AI Response
def judge_sahib_response(case_query: str, language: str) -> Dict:
    """Generate judge-style guidance"""
    try:
        target_lang = "English" if language == "English" else "Hindi" if language == "Hindi" else "English"
        prompt = f"""You are an experienced judge-like legal reasoning assistant. Provide calm, structured, and strict guidance.
Do not claim to be a real judge. Add a short disclaimer line.
Respond in {target_lang}.

CASE QUERY: {case_query}

Return a structured response with:
1) Key Issues
2) Judge's Observations
3) What Evidence Matters
4) Likely Directions
5) Next Steps
6) Disclaimer (one line)
"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )
        return {"response": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": f"Judge response failed: {str(e)}"}


# FEATURE 27: Bench Notes Generator
def generate_bench_notes(case_query: str, language: str) -> Dict:
    """Generate concise judge-style bench notes"""
    try:
        target_lang = "English" if language == "English" else "Hindi"
        prompt = f"""You are preparing bench notes for a judge. Keep it concise and structured.
Respond in {target_lang}.

CASE QUERY: {case_query}

Return JSON:
{{
    "issues_for_determination": ["<issue 1>", "<issue 2>"],
    "key_facts": ["<fact 1>", "<fact 2>"],
    "evidence_to_watch": ["<evidence 1>", "<evidence 2>"],
    "questions_for_counsel": ["<question 1>", "<question 2>"]
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Bench notes failed: {str(e)}"}


# FEATURE 28: Precedent Snapshot
def generate_precedent_snapshot(case_query: str, language: str) -> Dict:
    """Generate a high-level precedent snapshot (no real citation claims)"""
    try:
        target_lang = "English" if language == "English" else "Hindi"
        prompt = f"""Provide a high-level precedent snapshot based on the case topic.
Do not claim exact case citations. Use generic categories only.
Respond in {target_lang}.

CASE QUERY: {case_query}

Return JSON:
{{
    "precedent_themes": ["<theme 1>", "<theme 2>"],
    "likely_principles": ["<principle 1>", "<principle 2>"],
    "search_keywords": ["<keyword 1>", "<keyword 2>"],
    "risk_caveat": "<short caveat>"
}}"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1200
        )

        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Precedent snapshot failed: {str(e)}"}


# FEATURE 30: Daily Bench Digest
def generate_daily_bench_digest(items: List[Dict], language: str) -> Dict:
    """Generate a daily bench digest from news items"""
    if not items:
        return {"error": "No news items available for digest."}

    headlines = "\n".join([f"- {i.get('title', '')} ({i.get('source', '')})" for i in items])
    target_lang = "English" if language == "English" else "Hindi"
    prompt = f"""You are preparing a daily bench digest for a Supreme Court bench.
Use {target_lang}.

Provide:
1) 5-7 bullet highlights
2) "Impact for Bench" in 2-3 lines
3) "Action Items" in 3 bullets

HEADLINES:
{headlines}

Return plain text."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=1200
        )
        return {"digest": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": f"Bench digest failed: {str(e)}"}


# FEATURE 31: Order Drafting Studio
def draft_court_order(order_type: str, case_facts: str, relief: str, language: str) -> Dict:
    """Draft a structured court order"""
    target_lang = "English" if language == "English" else "Hindi"
    prompt = f"""Draft a concise Supreme Court style order in {target_lang}.

ORDER TYPE: {order_type}
RELIEF SOUGHT: {relief}
CASE FACTS: {case_facts}

Use headings:
1) Facts
2) Issues
3) Findings
4) Directions
5) Compliance Date
6) Disclaimer (one line)

Return plain text."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
            max_tokens=1800
        )
        return {"order": response.choices[0].message.content.strip()}
    except Exception as e:
        return {"error": f"Order draft failed: {str(e)}"}


# FEATURE 32: Hearing Flow Simulator
def generate_hearing_flow(case_summary: str, stage: str, role: str, language: str) -> Dict:
    """Generate a hearing flow with judge questions and expected points"""
    target_lang = "English" if language == "English" else "Hindi"
    prompt = f"""Create a hearing flow for a {stage} hearing.
Role: {role}. Respond in {target_lang}.

CASE SUMMARY:
{case_summary}

Return JSON:
{{
  "steps": [
    {{
      "judge_question": "...",
      "expected_points": ["...", "..."],
      "likely_objection": "...",
      "tip": "..."
    }}
  ]
}}"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1600
        )
        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Hearing flow failed: {str(e)}"}


def score_hearing_response(judge_question: str, response_text: str, language: str) -> Dict:
    """Score the user's response for hearing practice"""
    target_lang = "English" if language == "English" else "Hindi"
    prompt = f"""Evaluate the lawyer's response in {target_lang}.
Give a score out of 10 and 3 improvement tips.

JUDGE QUESTION: {judge_question}
LAWYER RESPONSE: {response_text}

Return JSON:
{{
  "score": <0-10>,
  "strengths": ["..."],
  "improvements": ["..."],
  "bench_comment": "..."
}}"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=900
        )
        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Scoring failed: {str(e)}"}


# FEATURE 33: Precedent Heatmap
def build_precedent_heatmap(snapshot: Dict) -> List[Tuple[str, int]]:
    """Build simple heatmap scores for precedent keywords"""
    keywords = snapshot.get("search_keywords", []) if isinstance(snapshot, dict) else []
    scores: List[Tuple[str, int]] = []
    base = 100
    for idx, kw in enumerate(keywords):
        score = max(20, base - idx * 15)
        scores.append((kw, score))
    return scores


# FEATURE 34: Client Journey Wizard Plan
def generate_client_journey_plan(problem_text: str, category: str, urgency: str, language: str,
                                 location: str, timeline: str, desired_outcome: str) -> Dict:
    """Generate a step-by-step client journey plan in Hinglish/English"""
    target_lang = "Hinglish" if language == "Hinglish" else "English"
    prompt = f"""You are a legal intake assistant for first-time clients.
Use {target_lang}. Keep it simple with light legal terms.

PROBLEM CATEGORY: {category}
URGENCY: {urgency}
LOCATION: {location}
TIMELINE: {timeline}
DESIRED OUTCOME: {desired_outcome}
CLIENT PROBLEM:
{problem_text}

Return JSON:
{{
  "simple_summary": "...",
  "likely_sections": ["..."] ,
  "rights_overview": ["..."] ,
  "immediate_actions": ["..."] ,
  "evidence_checklist": ["..."] ,
  "questions_for_lawyer": ["..."] ,
  "lawyer_call_script": "...",
  "recommended_features": ["Document Generator", "Legal Notice", "Case Analyzer", "AI Lawyer Chat"]
}}"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1400
        )
        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Client plan failed: {str(e)}"}


# ========================================
# NEW FEATURE: VIRTUAL COURT SIMULATOR 🎭
# ========================================
def start_virtual_court_simulation(case_type: str, role: str, difficulty: str, case_facts: str, language: str) -> Dict:
    """Initialize a virtual court simulation with AI judge"""
    target_lang = "English" if language == "English" else "Hindi"
    
    prompt = f"""You are simulating a Court hearing for training purposes.
CASE TYPE: {case_type}
YOUR ROLE: {role} (you are the lawyer)
DIFFICULTY: {difficulty}
LANGUAGE: {target_lang}

CASE FACTS:
{case_facts}

Generate the opening of a court hearing. You are the Judge.
Start with: "The Hon'ble Court is in session."

Then ask a challenging opening question about the case.
Include a bench tip for how to answer effectively.

Return JSON:
{{
  "judge_opening": "...",
  "judge_question": "...",
  "expected_points": ["...", "...", "..."],
  "bench_tip": "...",
  "likely_objection": "..."
}}"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=1200
        )
        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Simulation start failed: {str(e)}"}


def evaluate_court_argument(judge_question: str, user_argument: str, case_type: str, difficulty: str, language: str) -> Dict:
    """Evaluate user's court argument comprehensively"""
    target_lang = "English" if language == "English" else "Hindi"
    
    prompt = f"""Evaluate this lawyer's court argument in {target_lang}.

CASE TYPE: {case_type}
DIFFICULTY: {difficulty}
JUDGE'S QUESTION: {judge_question}
LAWYER'S ARGUMENT: {user_argument}

Score 0-10. Provide:
1. Score with reasoning
2. Strengths (2-3 bullets)
3. Weaknesses (2-3 bullets)
4. Missed opportunities
5. What opposing counsel might object to
6. Judge's likely reaction
7. Next question the judge will ask
8. Bench Tension Score (0-100, where 100 is high tension/hostile and 0 is smooth/friendly)

Return JSON:
{{
  "score": <0-10>,
  "tension_score": <0-100>,
  "reasoning": "...",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "missed_opportunity": "...",
  "opposing_objection": "...",
  "judge_reaction": "...",
  "next_question": "..."
}}"""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1400
        )
        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


# =========================================
# NEW FEATURE: LIVE CASE STATUS TRACKER 🔍
# =========================================
def get_case_status_timeline(case_number: str) -> Dict:
    """Generate realistic case status with timeline from database simulation"""
    # Simulated e-Courts data (in production, this would query real e-Courts API)
    mock_cases = {
        "CRI-456/2024": {
            "parties": "State of India vs. Ramesh Kumar",
            "filing_date": "2024-01-15",
            "current_stage": "Arguments",
            "next_hearing": "2026-02-28",
            "judge": "Hon'ble Justice Sharma",
            "court": "Supreme Court, New Delhi",
            "status_color": "🔴",
            "stages_completed": ["Admission", "First Hearing", "Evidence"],
            "stages_remaining": ["Arguments", "Final Submissions", "Reserved", "Judgment"],
            "progress_percent": 40,
            "documents_filed": 8,
            "last_order": "Adjourned for arguments",
            "counsel_for": "Senior Advocate Mishra",
            "counsel_against": "Solicitor General of India"
        },
        "CIV-789/2023": {
            "parties": "Property Developer Ltd vs. Homebuyers Association",
            "filing_date": "2023-05-20",
            "current_stage": "Evidence",
            "next_hearing": "2026-03-15",
            "judge": "Hon'ble Justice Rao",
            "court": "Delhi High Court",
            "status_color": "🟡",
            "stages_completed": ["Admission", "First Hearing"],
            "stages_remaining": ["Evidence", "Arguments", "Judgment"],
            "progress_percent": 35,
            "documents_filed": 24,
            "last_order": "Document production extended by 2 weeks",
            "counsel_for": "Advocate Pandey",
            "counsel_against": "Senior Advocate Agarwal"
        },
        "TAX-333/2022": {
            "parties": "Revenue Department vs. ABC Manufacturing",
            "filing_date": "2022-11-10",
            "current_stage": "Reserved",
            "next_hearing": "Awaiting Judgment",
            "judge": "Hon'ble Justice Iyer",
            "court": "Supreme Court, New Delhi",
            "status_color": "🟢",
            "stages_completed": ["Admission", "Arguments", "Final Hearing"],
            "stages_remaining": ["Judgment"],
            "progress_percent": 85,
            "documents_filed": 15,
            "last_order": "Matter reserved for judgment",
            "counsel_for": "Solicitor General",
            "counsel_against": "Senior Advocate Verma"
        }
    }
    
    if case_number in mock_cases:
        return mock_cases[case_number]
    else:
        return {
            "error": f"Case {case_number} not found in system",
            "suggestion": "Try: CRI-456/2024, CIV-789/2023, or TAX-333/2022"
        }


# ====================================
# PHASE 1 FEATURES (3 Quick Wins)
# ====================================

# FEATURE 27: Smart Legal News Categorizer (Enhanced)
def categorize_legal_news(items: List[Dict]) -> Dict:
    """Categorize legal news by court type, topic area, and urgency"""
    if not items:
        return {"categorized": {}, "error": "No items to categorize"}
    
    headlines = "\n".join([f"{i['title']}" for i in items])
    
    prompt = f"""Categorize these legal news headlines into:
1. COURT TYPE: Supreme Court, High Court, District Court, Tribunal, Other
2. TOPIC: Criminal Law, Civil Law, Constitutional, Corporate, Tax, Family, Property, Consumer, Labor, Other
3. URGENCY: 🔴 CRITICAL (affects many cases), 🟡 IMPORTANT (landmark ruling), 🟢 UPDATE (informational)
4. KEY IMPACT: One-line impact for lawyers/judges
5. ACTION REQUIRED: Yes/No

HEADLINES:
{headlines}

Return JSON with structure:
{{
    "categorized_news": [
        {{"title": "...", "court": "...", "topic": "...", "urgency": "🔴/🟡/🟢", "impact": "...", "action_required": true/false}}
    ]
}}
Keep it concise and actionable."""

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        result_text = response.choices[0].message.content.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        return json.loads(result_text)
    except Exception as e:
        return {"error": f"Categorization failed: {str(e)}", "categorized_news": []}


# FEATURE 28: Bail Application Generator
def generate_bail_application(case_details: str, personal_details: str, legal_grounds: List[str]) -> Dict:
    """Generate a professional bail application with legal citations"""
    try:
        grounds_text = "\n".join([f"{i+1}. {g}" for i, g in enumerate(legal_grounds)])
        
        prompt = f"""You are an expert legal draftsman. Generate a PROFESSIONAL, COURT-READY BAIL APPLICATION.

CASE DETAILS:
{case_details}

PERSONAL DETAILS:  
{personal_details}

GROUNDS FOR BAIL:
{grounds_text}

Generate a complete bail application with:
1. CAPTION (Case number, parties, court)
2. PRAYER (What relief is sought)
3. GROUNDS (Numbered points - 3-5 para each)
4. LEGAL CITATIONS (S. 437/439 CrPC, relevant precedents)
5. PRAYER FOR RELIEF

Use formal legal language. Make it ready to file in court.
Include actual legal sections and precedent names.
Return as plain text (not markdown)."""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3000
        )
        
        application_text = response.choices[0].message.content.strip()
        return {
            "success": True,
            "application": application_text,
            "estimated_length": len(application_text.split()) // 250  # Pages
        }
    except Exception as e:
        return {"success": False, "error": f"Generation failed: {str(e)}"}


# FEATURE 29: Instant Legal Advice (10 Seconds)
def instant_legal_advice(scenario: str, language: str = "English") -> Dict:
    """Provide instant legal advice for emergency scenarios (10 second response)"""
    try:
        emergency_scenarios = {
            "arrest_no_warrant": "Police arrested without warrant - what are your rights?",
            "search_no_warrant": "Police searching house without warrant - legal?",
            "bail_conditions": "Can bail conditions be extremely harsh?",
            "dowry_harassment": "Facing dowry harassment - immediate steps?",
            "property_dispute": "Land dispute escalating - what to do?",
            "contract_breach": "Business partner breached contract - remedies?",
            "accident_damage": "Vehicle accident caused injury - liability?",
            "cheque_bounce": "Cheque bounced - legal consequences?",
            "defective_product": "Purchased defective product - legal action?",
            "tenant_rights": "Landlord illegally evicting - tenant rights?",
            "divorce_child": "Divorce custody - how to get custody?",
            "sexual_harassment": "Facing sexual harassment at work?",
            "workplace_injury": "Injured at workplace - compensation?",
            "online_fraud": "Victim of online scam - what to do?",
            "inheritance_dispute": "Fighting for rightful inheritance?",
        }
        
        # Determine scenario type
        scenario_type = "general"
        for key, desc in emergency_scenarios.items():
            if any(word in scenario.lower() for word in key.split('_')):
                scenario_type = key
                break
        
        target_lang = "English" if language == "English" else "Hindi"
        
        prompt = f"""You are an emergency legal advisor. Provide QUICK, ACTIONABLE legal guidance in {target_lang}.

EMERGENCY: {scenario}

Respond EXACTLY in this format:
1. IMMEDIATE ACTION: [1-2 key steps to take TODAY]
2. LEGAL BASIS: [Relevant section/law - be specific]
3. DO NOT: [Things that will worsen the case]
4. NEXT STEP: [What to do in next 24-48 hours]
5. TIMELINE: [How much time you have - CRITICAL if limited]
6. ESTIMATED COST: [Rough lawyer fees]

Keep each section 1-2 lines max. Be direct and confident.
Use simple language. Respond in {target_lang}."""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Very low temp for consistent advice
            max_tokens=500
        )
        
        advice_text = response.choices[0].message.content.strip()
        return {
            "success": True,
            "scenario_type": scenario_type,
            "advice": advice_text,
            "urgency": "🔴 URGENT" if any(x in scenario.lower() for x in ["arrest", "police", "harassment"]) else "🟡 IMPORTANT"
        }
    except Exception as e:
        return {"success": False, "error": f"Advice generation failed: {str(e)}"}


if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""
if "selected_doc_id" not in st.session_state:
    st.session_state["selected_doc_id"] = None
if "feedback" not in st.session_state:
    st.session_state["feedback"] = {}
if "qa_history" not in st.session_state:
    st.session_state["qa_history"] = []
if "saved_searches" not in st.session_state:
    st.session_state["saved_searches"] = []
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"
if "current_sources" not in st.session_state:
    st.session_state["current_sources"] = []
if "risk_assessment" not in st.session_state:
    st.session_state["risk_assessment"] = None
if "citations_data" not in st.session_state:
    st.session_state["citations_data"] = None
if "case_analysis" not in st.session_state:
    st.session_state["case_analysis"] = None
if "generated_document" not in st.session_state:
    st.session_state["generated_document"] = None
if "filing_guide" not in st.session_state:
    st.session_state["filing_guide"] = None
if "chatbot_history" not in st.session_state:
    st.session_state["chatbot_history"] = []
if "chatbot_mode" not in st.session_state:
    st.session_state["chatbot_mode"] = False
if "chatbot_language" not in st.session_state:
    st.session_state["chatbot_language"] = "English"
if "show_quick_replies" not in st.session_state:
    st.session_state["show_quick_replies"] = True

# New Features Session State (14-18)
if "contract_analysis" not in st.session_state:
    st.session_state["contract_analysis"] = None
if "hearing_script" not in st.session_state:
    st.session_state["hearing_script"] = None
if "evidence_organization" not in st.session_state:
    st.session_state["evidence_organization"] = None
if "settlement_calculation" not in st.session_state:
    st.session_state["settlement_calculation"] = None
if "timeline_check" not in st.session_state:
    st.session_state["timeline_check"] = None

# App Configuration Mode
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "Lawyer Mode"
if "enable_advanced_tools" not in st.session_state:
    st.session_state["enable_advanced_tools"] = False
if "enable_legal_news" not in st.session_state:
    st.session_state["enable_legal_news"] = False
if "enable_judge_ai" not in st.session_state:
    st.session_state["enable_judge_ai"] = False

# New Features Session State (19-26)
if "outcome_prediction" not in st.session_state:
    st.session_state["outcome_prediction"] = None
if "strategy_plan" not in st.session_state:
    st.session_state["strategy_plan"] = None
if "cost_estimate" not in st.session_state:
    st.session_state["cost_estimate"] = None
if "simple_explanation" not in st.session_state:
    st.session_state["simple_explanation"] = None
if "opposition_reply" not in st.session_state:
    st.session_state["opposition_reply"] = None
if "news_items" not in st.session_state:
    st.session_state["news_items"] = []
if "news_summary" not in st.session_state:
    st.session_state["news_summary"] = None
if "judge_chat_history" not in st.session_state:
    st.session_state["judge_chat_history"] = []
if "enable_bench_notes" not in st.session_state:
    st.session_state["enable_bench_notes"] = True
if "enable_precedent_snapshot" not in st.session_state:
    st.session_state["enable_precedent_snapshot"] = True
if "bench_notes_result" not in st.session_state:
    st.session_state["bench_notes_result"] = None
if "precedent_snapshot_result" not in st.session_state:
    st.session_state["precedent_snapshot_result"] = None
if "bench_log" not in st.session_state:
    st.session_state["bench_log"] = []
if "court_no" not in st.session_state:
    st.session_state["court_no"] = "Court No. 1"
if "bench_name" not in st.session_state:
    st.session_state["bench_name"] = "Hon'ble Justice"
if "matter_type" not in st.session_state:
    st.session_state["matter_type"] = "Criminal"
if "session_status" not in st.session_state:
    st.session_state["session_status"] = "In Session"
if "cause_list" not in st.session_state:
    st.session_state["cause_list"] = []
if "daily_digest" not in st.session_state:
    st.session_state["daily_digest"] = None
if "order_draft" not in st.session_state:
    st.session_state["order_draft"] = None
if "hearing_sim" not in st.session_state:
    st.session_state["hearing_sim"] = {"steps": [], "current": 0}
if "hearing_feedback" not in st.session_state:
    st.session_state["hearing_feedback"] = None
if "precedent_heatmap" not in st.session_state:
    st.session_state["precedent_heatmap"] = []

# Phase 1 Features Session State
if "categorized_news" not in st.session_state:
    st.session_state["categorized_news"] = None
if "bail_application" not in st.session_state:
    st.session_state["bail_application"] = None
if "instant_advice" not in st.session_state:
    st.session_state["instant_advice"] = None

# NEW FEATURES: Virtual Court Simulator & Case Tracker
if "court_simulation" not in st.session_state:
    st.session_state["court_simulation"] = None
if "sim_current_turn" not in st.session_state:
    st.session_state["sim_current_turn"] = 0
if "sim_score_history" not in st.session_state:
    st.session_state["sim_score_history"] = []
if "case_tracker" not in st.session_state:
    st.session_state["case_tracker"] = {}
if "selected_case" not in st.session_state:
    st.session_state["selected_case"] = None

# Client Journey Wizard
if "wizard_step" not in st.session_state:
    st.session_state["wizard_step"] = 1
if "wizard_result" not in st.session_state:
    st.session_state["wizard_result"] = None
if "wizard_data" not in st.session_state:
    st.session_state["wizard_data"] = {}

# FEATURE WIZARD STATE (All 10 Features)
if "doc_gen_step" not in st.session_state:
    st.session_state["doc_gen_step"] = 0
if "case_analyzer_step" not in st.session_state:
    st.session_state["case_analyzer_step"] = 0
if "legal_notice_step" not in st.session_state:
    st.session_state["legal_notice_step"] = 0
if "court_filing_step" not in st.session_state:
    st.session_state["court_filing_step"] = 0
if "contracts_step" not in st.session_state:
    st.session_state["contracts_step"] = 0
if "court_script_step" not in st.session_state:
    st.session_state["court_script_step"] = 0
if "evidence_step" not in st.session_state:
    st.session_state["evidence_step"] = 0
if "settlement_step" not in st.session_state:
    st.session_state["settlement_step"] = 0
if "timeline_step" not in st.session_state:
    st.session_state["timeline_step"] = 0

# Feature data storage
if "feature_data" not in st.session_state:
    st.session_state["feature_data"] = {}
if "feature_results" not in st.session_state:
    st.session_state["feature_results"] = {}

# Sidebar: Document Library & New Features
with st.sidebar:
    # Enhanced Sidebar Header
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 1rem; 
                 background: linear-gradient(135deg, rgba(0, 243, 255, 0.1) 0%, rgba(176, 38, 255, 0.1) 100%);
                 border-radius: 15px; border: 2px solid rgba(0, 243, 255, 0.3);">
        <h2 style="margin: 0; background: linear-gradient(135deg, #00f3ff 0%, #b026ff 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   background-clip: text; font-size: 1.8rem; font-weight: 900;">
            ⚙️ CONTROL PANEL
        </h2>
        <p style="margin: 0.5rem 0 0 0; color: #9fa8da; font-size: 0.85rem; font-weight: 600;">
            AI Legal Intelligence Hub
        </p>
    </div>
    """, unsafe_allow_html=True)

    # App Configuration Mode
    st.markdown('<h4 style="color: #00f3ff; margin-bottom: 0.5rem;">🧭 Configuration Mode</h4>', unsafe_allow_html=True)
    mode_choice = st.selectbox(
        "Select Mode",
        ["Lawyer Mode", "Supreme Court Judge Mode"],
        index=0 if st.session_state["app_mode"] == "Lawyer Mode" else 1,
        key="mode_choice"
    )
    st.session_state["app_mode"] = mode_choice

    if mode_choice == "Lawyer Mode":
        st.session_state["enable_advanced_tools"] = False
        st.session_state["enable_legal_news"] = False
        st.session_state["enable_judge_ai"] = False
        st.session_state["enable_bench_notes"] = False
        st.session_state["enable_precedent_snapshot"] = False
    else:
        st.session_state["enable_advanced_tools"] = False
        st.session_state["enable_legal_news"] = True
        st.session_state["enable_judge_ai"] = True
        st.session_state["enable_bench_notes"] = True
        st.session_state["enable_precedent_snapshot"] = True
    
    if mode_choice == "Lawyer Mode":
        # Theme Toggle (Feature 4) - Enhanced
        st.markdown('<h4 style="color: #00f3ff; margin-bottom: 0.5rem;">🎨 Interface Theme</h4>', unsafe_allow_html=True)
        theme_col1, theme_col2 = st.columns(2)
        with theme_col1:
            if st.button("🌙 Dark Mode", use_container_width=True):
                st.session_state["theme_mode"] = "dark"
        with theme_col2:
            if st.button("☀️ Light Mode", use_container_width=True):
                st.session_state["theme_mode"] = "light"
        
        st.divider()
        
        # Document Library - Enhanced
        st.markdown('<h4 style="color: #00f3ff; margin-bottom: 1rem;">📚 Document Library</h4>', unsafe_allow_html=True)
        manifest = load_manifest()
        
        # Tag Filter (Feature 5)
        all_tags = set()
        for e in manifest:
            all_tags.update(e.get("tags", []))
        
        if all_tags:
            selected_tag = st.selectbox("Filter by Tag", ["All"] + sorted(list(all_tags)))
            if selected_tag != "All":
                manifest = [e for e in manifest if selected_tag in e.get("tags", [])]
        
        options = {f"{e.get('name')} ({e.get('doc_id')[:8]})": e.get("doc_id") for e in manifest}
        selected_label = st.selectbox("Select indexed document", list(options.keys()) or ["(none)"])
        selected_doc_id = options.get(selected_label)
        st.session_state["selected_doc_id"] = selected_doc_id
        
        # Document Tags (Feature 5)
        if selected_doc_id:
            entry = next((e for e in load_manifest() if e.get("doc_id") == selected_doc_id), None)
            if entry:
                current_tags = entry.get("tags", [])
                st.write(f"**Tags:** {', '.join(current_tags) if current_tags else 'No tags'}")
                new_tag = st.text_input("Add tag:", key="new_tag_input")
                if st.button("➕ Add Tag") and new_tag:
                    add_tag_to_document(selected_doc_id, new_tag)
                    st.success(f"Tag '{new_tag}' added!")
                    st.rerun()

        uploaded_file_sidebar = st.file_uploader("Add PDF to library", type="pdf", accept_multiple_files=False)
        if uploaded_file_sidebar is not None:
            if st.button("Index PDF"):
                entry = upload_pdf(uploaded_file_sidebar)
                docs = load_pdf(entry["pdf_path"])
                chunks = create_chunks(docs)
                create_vector_store(entry["db_path"], chunks, resolve_embedding_model_name(entry.get("embed_model")))
                st.success(f"Indexed: {entry['name']}")
                manifest = load_manifest()  # refresh

        if selected_doc_id:
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Delete selected index", type="secondary"):
                    # Remove index directory and manifest entry
                    entry = next((e for e in load_manifest() if e.get("doc_id") == selected_doc_id), None)
                    if entry and os.path.isdir(entry["db_path"]):
                        try:
                            for root, dirs, files in os.walk(entry["db_path"], topdown=False):
                                for name in files:
                                    os.remove(os.path.join(root, name))
                                for name in dirs:
                                    os.rmdir(os.path.join(root, name))
                            os.rmdir(entry["db_path"])
                        except Exception:
                            pass
                    delete_manifest_entry(selected_doc_id)
                    st.session_state["selected_doc_id"] = None
                    st.success("Index deleted.")
                    st.rerun()
            with col_b:
                if st.button("Rebuild index"):
                    entry = next((e for e in load_manifest() if e.get("doc_id") == selected_doc_id), None)
                    if entry:
                        docs = load_pdf(entry["pdf_path"])
                        chunks = create_chunks(docs)
                        create_vector_store(entry["db_path"], chunks, resolve_embedding_model_name(entry.get("embed_model")))
                        st.success("Index rebuilt.")
                        st.rerun()
        
        st.divider()
        
        # Advanced Search (Feature 3)
        with st.expander("🔍 Advanced Search"):
            st.write("**Search across all documents**")
            search_query = st.text_input("Search query:", key="adv_search")
            doc_filter = st.text_input("Document name filter:", key="doc_filter")
            
            if st.button("🔎 Search All Docs"):
                if search_query:
                    results = search_in_documents(search_query, doc_filter)
                    st.write(f"**Found {len(results)} results**")
                    for r in results[:5]:
                        st.write(f"📄 {r['doc_name']}")
                        st.write(f"{r['content']}...")
                        st.write("---")
                    
                    # Save search
                    if st.button("💾 Save this search"):
                        save_search_query(search_query, {"doc_filter": doc_filter})
                        st.success("Search saved!")
        
        # Saved Searches
        if st.session_state["saved_searches"]:
            with st.expander("📌 Saved Searches"):
                for idx, saved in enumerate(st.session_state["saved_searches"][-5:]):
                    if st.button(f"🔍 {saved['query'][:30]}...", key=f"saved_{idx}"):
                        st.session_state["user_query"] = saved['query']
                        st.rerun()
        
        st.divider()
        
        # NEW FEATURE 6: Contract Risk Assessment - Enhanced
        st.markdown('<h4 style="color: #ff006e; margin-top: 2rem; margin-bottom: 1rem;">🛡️ Risk Assessment Engine</h4>', unsafe_allow_html=True)
        if selected_doc_id and st.button("🔍 Analyze Contract Risks", use_container_width=True, type="primary"):
            entry = next((e for e in load_manifest() if e.get("doc_id") == selected_doc_id), None)
            if entry:
                with st.spinner("Analyzing contract for risks..."):
                    # Load document text
                    docs = load_pdf(entry["pdf_path"])
                    full_text = "\n\n".join([doc.page_content for doc in docs[:10]])  # First 10 pages
                    
                    # Perform risk analysis
                    risk_data = analyze_contract_risks(full_text, doc_type="contract")
                    st.session_state["risk_assessment"] = risk_data
                    
                    if risk_data and "overall_risk_score" in risk_data:
                        st.success(f"✅ Analysis complete! Risk Level: {risk_data.get('overall_risk_score', 'UNKNOWN')}")
                    else:
                        st.error("Unable to complete risk analysis")
        
        # FEATURE: Discovery Agent (Clash Detection)
        st.markdown('<h4 style="color: #00f3ff; margin-top: 2rem; margin-bottom: 1rem;">🔍 Discovery Agent</h4>', unsafe_allow_html=True)
        if selected_doc_id and st.button("🎭 Detect Document Clashes", use_container_width=True):
            entry = next((e for e in load_manifest() if e.get("doc_id") == selected_doc_id), None)
            if entry:
                with st.spinner("Scanning for contradictions..."):
                    docs = load_pdf(entry["pdf_path"])
                    clashes = clash_detection_agent(docs)
                    st.session_state["feature_results"]["clash"] = clashes
                    st.rerun()

        # FEATURE: Cross-Examination Coach
        st.markdown('<h4 style="color: #b026ff; margin-top: 1.5rem; margin-bottom: 1rem;">🧠 Witness Coach</h4>', unsafe_allow_html=True)
        witness_name = st.text_input("Witness Name:", key="witness_name_input")
        if selected_doc_id and witness_name and st.button("🎙️ Generate X-Exam Strategy", use_container_width=True):
            entry = next((e for e in load_manifest() if e.get("doc_id") == selected_doc_id), None)
            if entry:
                with st.spinner(f"Coaching for {witness_name}..."):
                    docs = load_pdf(entry["pdf_path"])
                    strategy = cross_examination_coach(docs, witness_name)
                    st.session_state["feature_results"]["witness_coach"] = strategy
                    st.rerun()

        # FEATURE: Precedent Reliability
        st.markdown('<h4 style="color: #39ff14; margin-top: 1.5rem; margin-bottom: 1rem;">⚖️ Precedent Analyzer</h4>', unsafe_allow_html=True)
        if selected_doc_id and st.button("🛡️ Verify Precedent Reliability", use_container_width=True):
            entry = next((e for e in load_manifest() if e.get("doc_id") == selected_doc_id), None)
            if entry:
                with st.spinner("Analyzing precedent reliability..."):
                    docs = load_pdf(entry["pdf_path"])
                    text = "\n".join([d.page_content for d in docs[:5]])
                    reliability = precedent_reliability_analyzer(text)
                    st.session_state["feature_results"]["reliability"] = reliability
                    st.rerun()

        # Display results if any
        for fkey, fval in st.session_state["feature_results"].items():
            if fval:
                with st.expander(f"📌 Results for {fkey.capitalize()}"):
                    st.markdown(fval)
                    if st.button(f"Clear {fkey}", key=f"clear_{fkey}"):
                        st.session_state["feature_results"][fkey] = None
                        st.rerun()

        # Display risk assessment summary in sidebar
        if st.session_state.get("risk_assessment"):
            risk_data = st.session_state["risk_assessment"]
            if "overall_risk_score" in risk_data:
                risk_score = risk_data.get("overall_risk_score", "UNKNOWN")
                risk_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                st.write(f"{risk_colors.get(risk_score, '⚪')} **Risk Level:** {risk_score}")
                
                if st.button("📊 View Full Risk Report"):
                    st.session_state["show_risk_report"] = True
                
                if st.button("📥 Download Risk Report"):
                    doc_name = selected_label if selected_label else "Document"
                    risk_html = generate_risk_report_html(risk_data, doc_name)
                    st.download_button(
                        label="⬇️ Download HTML Report",
                        data=risk_html,
                        file_name=f"risk_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
        
        st.divider()
        
        # NEW FEATURE 7: Citation Validator - Enhanced
        st.markdown('<h4 style="color: #39ff14; margin-top: 2rem; margin-bottom: 1rem;">📚 Citation Validator</h4>', unsafe_allow_html=True)
        if selected_doc_id and st.button("🔎 Extract & Validate Citations", use_container_width=True, type="primary"):
            entry = next((e for e in load_manifest() if e.get("doc_id") == selected_doc_id), None)
            if entry:
                with st.spinner("Extracting and validating citations..."):
                    # Load document text
                    docs = load_pdf(entry["pdf_path"])
                    full_text = "\n\n".join([doc.page_content for doc in docs[:10]])
                    
                    # Extract citations
                    citations = extract_citations(full_text)
                    
                    # Validate each citation
                    validations = []
                    for citation in citations[:10]:  # Limit to 10 for performance
                        validation = validate_citation(citation.get("citation", ""))
                        validations.append(validation)
                    
                    st.session_state["citations_data"] = {
                        "citations": citations,
                        "validations": validations
                    }
                    
                    st.success(f"✅ Found {len(citations)} citations!")
        
        # Display citation summary
        if st.session_state.get("citations_data"):
            cit_data = st.session_state["citations_data"]
            citations = cit_data.get("citations", [])
            validations = cit_data.get("validations", [])
            
            st.write(f"📖 **Citations Found:** {len(citations)}")
            
            # Count valid citations
            valid_count = sum(1 for v in validations if v.get("is_valid", False))
            st.write(f"✅ **Valid:** {valid_count} | ❓ **Unknown:** {len(validations) - valid_count}")
            
            if st.button("📋 View Citation Report"):
                st.session_state["show_citation_report"] = True
            
            if st.button("📥 Download Citation Report"):
                doc_name = selected_label if selected_label else "Document"
                citation_html = generate_citation_report(citations, validations, doc_name)
                st.download_button(
                    label="⬇️ Download HTML Report",
                    data=citation_html,
                    file_name=f"citation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
        
        st.divider()
        
        # Report Generator (Feature 1)
        if st.session_state["qa_history"]:
            st.subheader("📄 Generate Report")
            if st.button("📥 Export Legal Report (HTML)"):
                doc_name = selected_label if selected_label else "Multiple Documents"
                report_html = generate_legal_report(
                    st.session_state["qa_history"],
                    doc_name
                )
                st.download_button(
                    label="⬇️ Download Report",
                    data=report_html,
                    file_name=f"legal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
                st.success("Report generated! Click to download.")
            
            if st.button("🗑️ Clear Q&A History"):
                st.session_state["qa_history"] = []
                st.session_state["messages"] = []
                st.success("History cleared!")
                st.rerun()
    
    # Developer Card with Image and Social Links (Bottom of Sidebar)
    st.divider()
    
    def get_base64_image(img_path):
        try:
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except:
            return None

    dev_img_b64 = get_base64_image("pic.jpg")
    
    if dev_img_b64:
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; border-radius: 20px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.3); margin-bottom: 1.5rem;">
            <img src="data:image/jpeg;base64,{dev_img_b64}" style="width: 130px; border-radius: 15px; border: 2px solid #00d2ff; margin-bottom: 15px;">
            <h4 style="margin: 0; color: #ffffff; font-size: 1.1rem;">Abhishek ❤️ Yadav</h4>
            <p style="margin: 5px 0 15px 0; font-size: 0.85rem; color: #00d2ff; font-weight: bold;">Full Stack AI Developer</p>
            <div style="display: flex; justify-content: center; gap: 15px;">
                <a href="https://abhi-yadav.vercel.app/" target="_blank"><img src="https://img.icons8.com/color/48/000000/portfolio.png" width="32"></a>
                <a href="https://www.linkedin.com/in/abhishek-kumar-807853375/" target="_blank"><img src="https://img.icons8.com/color/48/000000/linkedin.png" width="32"></a>
                <a href="https://github.com/abhishekkumar62000" target="_blank"><img src="https://img.icons8.com/fluent/48/000000/github.png" width="32"></a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border-radius: 20px; background: rgba(0, 243, 255, 0.05); border: 1px solid rgba(0, 243, 255, 0.2); box-shadow: 0 4px 15px rgba(0,0,0,0.3); margin-bottom: 1.5rem;">
            <div style="width: 130px; height: 130px; border-radius: 15px; border: 2px solid #00d2ff; margin: 0 auto 15px auto; display: flex; align-items: center; justify-content: center; background: rgba(0, 243, 255, 0.1);">
                <p style="color: #00d2ff; font-size: 0.9rem;">📸 Add pic.jpg</p>
            </div>
            <h4 style="margin: 0; color: #ffffff; font-size: 1.1rem;">Abhishek ❤️ Yadav</h4>
            <p style="margin: 5px 0 15px 0; font-size: 0.85rem; color: #00d2ff; font-weight: bold;">Full Stack AI Developer</p>
            <div style="display: flex; justify-content: center; gap: 15px;">
                <a href="https://abhi-yadav.vercel.app/" target="_blank"><img src="https://img.icons8.com/color/48/000000/portfolio.png" width="32"></a>
                <a href="https://www.linkedin.com/in/abhishek-kumar-807853375/" target="_blank"><img src="https://img.icons8.com/color/48/000000/linkedin.png" width="32"></a>
                <a href="https://github.com/abhishekkumar62000" target="_blank"><img src="https://img.icons8.com/fluent/48/000000/github.png" width="32"></a>
            </div>
        </div>
        """, unsafe_allow_html=True)


# Judge Mode UI (no lawyer features)
if st.session_state.get("app_mode") == "Supreme Court Judge Mode":
    st.markdown("""
    <div class="judge-header" style="text-align: center;">
        <div class="court-strip">IN THE SUPREME COURT OF INDIA • VIRTUAL BENCH</div>
        <h1 class="judge-title">👨‍⚖️ Supreme Court Judge.ai</h1>
        <p class="judge-subtitle">🏛️ Supreme Court Intelligence for News, Bench Notes, and Judge Guidance 🏛️</p>
        <div>
            <span class="bench-chip">Bench Focus</span>
            <span class="bench-chip" style="border-color: rgba(176, 38, 255, 0.5); color: #b026ff; background: rgba(176, 38, 255, 0.12);">Structured Orders</span>
            <span class="bench-chip" style="border-color: rgba(57, 255, 20, 0.5); color: #39ff14; background: rgba(57, 255, 20, 0.12);">Live Legal News</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='bench-card'>", unsafe_allow_html=True)
    col_bench1, col_bench2, col_bench3 = st.columns(3)
    with col_bench1:
        st.selectbox(
            "Court Number",
            ["Court No. 1", "Court No. 2", "Court No. 3", "Court No. 4"],
            index=["Court No. 1", "Court No. 2", "Court No. 3", "Court No. 4"].index(st.session_state["court_no"]),
            key="court_no"
        )
        st.selectbox(
            "Matter Type",
            ["Criminal", "Civil", "Constitutional", "Corporate", "Tax", "Family"],
            index=["Criminal", "Civil", "Constitutional", "Corporate", "Tax", "Family"].index(st.session_state["matter_type"]),
            key="matter_type"
        )
    with col_bench2:
        st.text_input(
            "Bench / Judge Name",
            value=st.session_state["bench_name"],
            key="bench_name"
        )
        st.selectbox(
            "Session Status",
            ["In Session", "Rising", "Reserved", "Adjourned"],
            index=["In Session", "Rising", "Reserved", "Adjourned"].index(st.session_state["session_status"]),
            key="session_status"
        )
    with col_bench3:
        now_ts = datetime.now().strftime("%A, %d %B %Y • %I:%M %p")
        st.markdown(f"**Court Clock:** {now_ts}")
        st.markdown(f"**Status:** {st.session_state['session_status']}")
        st.markdown(f"**Bench:** {st.session_state['bench_name']}")
    st.markdown("</div>", unsafe_allow_html=True)

    docket_col1, docket_col2 = st.columns([2, 1])
    with docket_col1:
        st.markdown("<div class='docket-row'><strong>Now Hearing</strong><span>Public Interest • {}</span></div>".format(st.session_state["matter_type"]), unsafe_allow_html=True)
        st.markdown("<div class='docket-row'><strong>Next Matter</strong><span>Admission • Court No. {}</span></div>".format(st.session_state["court_no"].split()[-1]), unsafe_allow_html=True)
    with docket_col2:
        bench_note = st.text_input("Bench Memo (1 line)", key="bench_memo")
        if st.button("Add to Bench Log", use_container_width=True):
            if bench_note:
                st.session_state["bench_log"].append(f"{datetime.now().strftime('%I:%M %p')} - {bench_note}")
            else:
                st.warning("Enter a memo before adding.")

    if st.session_state["bench_log"]:
        st.markdown("**Bench Log (Today):**")
        for entry in st.session_state["bench_log"][-5:]:
            st.markdown(f"- {entry}")

    judge_tabs = [
        "🏛️ Courtroom Intelligence",
        "📰 Smart News Aggregator",
        "🆓 Bail Application Generator", 
        "⚡ Instant Legal Advice",
        "👨‍⚖️ Judge Sahib AI"
    ]
    if st.session_state.get("enable_bench_notes"):
        judge_tabs.append("🗒️ Bench Notes")
    if st.session_state.get("enable_precedent_snapshot"):
        judge_tabs.append("📚 Precedent Snapshot")

    judge_feature_tabs = st.tabs(judge_tabs)
    judge_index = 0

    # ========== TAB 1: COURTROOM INTELLIGENCE ==========
    with judge_feature_tabs[judge_index]:
        st.markdown('<h3 style="color: #ffea00;">🏛️ Courtroom Intelligence Suite</h3>', unsafe_allow_html=True)
        st.info("💡 Live bench tools: cause list, order drafting, hearing flow, precedent heatmap, daily bench digest, virtual court simulator & case tracker.")

        ci_tabs = st.tabs([
            "📋 Cause-List & Bench Board",
            "✍️ Order Drafting Studio",
            "🎙️ Hearing Flow Simulator",
            "📊 Precedent Heatmap",
            "🗞️ Daily Bench Digest",
            "🎭 Virtual Court Simulator",
            "🔍 Live Case Status Tracker"
        ])

        # Cause-List & Bench Board
        with ci_tabs[0]:
            st.markdown("<div class='bench-card'>", unsafe_allow_html=True)
            st.markdown("**Bench Board**")
            board_col1, board_col2, board_col3 = st.columns(3)
            with board_col1:
                st.metric("Total Matters", len(st.session_state["cause_list"]))
            with board_col2:
                st.metric("Now Hearing", st.session_state["cause_list"][0]["title"] if st.session_state["cause_list"] else "-" )
            with board_col3:
                st.metric("Next Matter", st.session_state["cause_list"][1]["title"] if len(st.session_state["cause_list"]) > 1 else "-" )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### Add Matter to Cause List")
            col_case1, col_case2, col_case3 = st.columns(3)
            with col_case1:
                cl_title = st.text_input("Case Title", key="cl_title")
                cl_case_no = st.text_input("Case No", key="cl_case_no")
            with col_case2:
                cl_stage = st.selectbox("Stage", ["Admission", "Hearing", "Arguments", "Reserved", "Pronounced"], key="cl_stage")
                cl_time = st.text_input("Time", placeholder="10:30 AM", key="cl_time")
            with col_case3:
                cl_court = st.selectbox("Court", ["Court No. 1", "Court No. 2", "Court No. 3", "Court No. 4"], key="cl_court")
                if st.button("➕ Add Matter", use_container_width=True, key="cl_add_btn"):
                    if cl_title and cl_case_no:
                        st.session_state["cause_list"].append({
                            "title": cl_title,
                            "case_no": cl_case_no,
                            "stage": cl_stage,
                            "time": cl_time or "TBD",
                            "court": cl_court
                        })
                    else:
                        st.warning("Enter case title and case number.")

            if st.session_state["cause_list"]:
                st.markdown("### Cause List")
                for idx, item in enumerate(st.session_state["cause_list"], 1):
                    st.markdown(
                        f"<div class='docket-row'><strong>{idx}. {item['case_no']} - {item['title']}</strong>"
                        f"<span>{item['stage']} • {item['time']} • {item['court']}</span></div>",
                        unsafe_allow_html=True
                    )

                st.markdown("### Matter Timeline")
                matter_choices = [f"{m['case_no']} • {m['title']}" for m in st.session_state["cause_list"]]
                selected_matter = st.selectbox("Select Matter", matter_choices, key="cl_matter_select")
                sel_idx = matter_choices.index(selected_matter)
                sel_matter = st.session_state["cause_list"][sel_idx]
                stage_progress = {
                    "Admission": 20,
                    "Hearing": 45,
                    "Arguments": 70,
                    "Reserved": 85,
                    "Pronounced": 100
                }
                st.markdown(f"**Current Stage:** {sel_matter['stage']}")
                st.progress(stage_progress.get(sel_matter["stage"], 10) / 100)
                st.caption("Timeline shows approximate progress for the selected matter.")
                col_manage1, col_manage2 = st.columns(2)
                with col_manage1:
                    move_idx = st.number_input("Move to Top (No.)", min_value=1, max_value=len(st.session_state["cause_list"]), value=1, step=1, key="cl_move_idx")
                    if st.button("⬆️ Move to Top", use_container_width=True, key="cl_move_btn"):
                        idx = int(move_idx) - 1
                        if 0 <= idx < len(st.session_state["cause_list"]):
                            item = st.session_state["cause_list"].pop(idx)
                            st.session_state["cause_list"].insert(0, item)
                with col_manage2:
                    del_idx = st.number_input("Remove (No.)", min_value=1, max_value=len(st.session_state["cause_list"]), value=1, step=1, key="cl_del_idx")
                    if st.button("🗑️ Remove", use_container_width=True, key="cl_del_btn"):
                        idx = int(del_idx) - 1
                        if 0 <= idx < len(st.session_state["cause_list"]):
                            st.session_state["cause_list"].pop(idx)

        # Order Drafting Studio
        with ci_tabs[1]:
            st.markdown("### ✍️ Order Drafting Studio")
            order_language = st.selectbox("Language", ["English", "Hindi"], key="order_language")
            order_type = st.selectbox("Order Type", ["Interim Order", "Final Order", "Bail Order", "Transfer Order"], key="order_type")
            relief_template = st.selectbox(
                "Relief Template",
                ["Custom", "Stay order", "Grant bail", "Issue notice", "Transfer case", "Interim protection"],
                key="order_relief_template"
            )
            relief_hint = "Example: Stay order till next hearing" if relief_template == "Custom" else relief_template
            order_relief = st.text_input("Relief Sought", value=st.session_state.get("order_relief", ""), placeholder=relief_hint, key="order_relief")
            compliance_date = st.date_input("Compliance Date (if any)", key="order_compliance_date")
            order_facts = st.text_area("Case Facts", height=180, key="order_facts")
            if st.button("⚖️ Draft Order", use_container_width=True, type="primary", key="order_draft_btn"):
                if order_facts and order_relief:
                    with st.spinner("Drafting court order..."):
                        facts_with_meta = f"{order_facts}\n\nCompliance Date Preference: {compliance_date}"
                        st.session_state["order_draft"] = draft_court_order(order_type, facts_with_meta, order_relief, order_language)
                else:
                    st.error("Provide case facts and relief sought.")

            if st.session_state.get("order_draft"):
                draft = st.session_state["order_draft"]
                if "error" not in draft:
                    st.markdown("<div class='order-box'>", unsafe_allow_html=True)
                    st.write(draft.get("order", ""))
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.download_button(
                        "📥 Download Order",
                        draft.get("order", ""),
                        file_name=f"court_order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error(draft.get("error"))

        # Hearing Flow Simulator
        with ci_tabs[2]:
            st.markdown("### 🎙️ Hearing Flow Simulator")
            sim_language = st.selectbox("Language", ["English", "Hindi"], key="sim_language")
            sim_stage = st.selectbox("Hearing Stage", ["Admission", "Bail", "Arguments", "Final Hearing"], key="sim_stage")
            sim_role = st.selectbox("Your Role", ["Petitioner Counsel", "Respondent Counsel"], key="sim_role")
            sim_summary = st.text_area("Case Summary", height=160, key="sim_summary")

            if st.button("🎬 Generate Hearing Flow", use_container_width=True, type="primary", key="sim_gen_btn"):
                if sim_summary and len(sim_summary) > 40:
                    with st.spinner("Building hearing flow..."):
                        st.session_state["hearing_sim"] = generate_hearing_flow(sim_summary, sim_stage, sim_role, sim_language)
                        st.session_state["hearing_sim"]["current"] = 0
                        st.session_state["hearing_feedback"] = None
                else:
                    st.error("Please provide a longer case summary (min 40 chars).")

            sim_data = st.session_state.get("hearing_sim")
            steps = sim_data.get("steps", []) if isinstance(sim_data, dict) else []
            current_idx = sim_data.get("current", 0) if isinstance(sim_data, dict) else 0

            if steps:
                step = steps[min(current_idx, len(steps) - 1)]
                st.markdown(f"**Question {current_idx + 1} of {len(steps)}**")
                st.progress((current_idx + 1) / len(steps))
                st.markdown("<div class='order-box'>", unsafe_allow_html=True)
                st.markdown(f"**Judge Question:** {step.get('judge_question', '')}")
                st.markdown("**Expected Points:**")
                for p in step.get("expected_points", []):
                    st.write(f"• {p}")
                st.markdown(f"**Likely Objection:** {step.get('likely_objection', '')}")
                st.markdown(f"**Bench Tip:** {step.get('tip', '')}")
                st.markdown("</div>", unsafe_allow_html=True)

                response_text = st.text_area("Your Response", height=120, key="sim_response")
                col_sim1, col_sim2 = st.columns(2)
                with col_sim1:
                    if st.button("✅ Score Response", use_container_width=True, key="sim_score_btn"):
                        if response_text:
                            st.session_state["hearing_feedback"] = score_hearing_response(
                                step.get("judge_question", ""), response_text, sim_language
                            )
                        else:
                            st.warning("Enter your response to score.")
                with col_sim2:
                    if st.button("➡️ Next Question", use_container_width=True, key="sim_next_btn"):
                        if current_idx + 1 < len(steps):
                            st.session_state["hearing_sim"]["current"] = current_idx + 1
                            st.session_state["hearing_feedback"] = None
                if st.button("🔁 Reset Simulation", use_container_width=True, key="sim_reset_btn"):
                    st.session_state["hearing_sim"]["current"] = 0
                    st.session_state["hearing_feedback"] = None

                feedback = st.session_state.get("hearing_feedback")
                if isinstance(feedback, dict) and "error" not in feedback:
                    st.markdown("**Score:**")
                    st.success(f"{feedback.get('score', 0)}/10")
                    st.markdown("**Strengths:**")
                    for s in feedback.get("strengths", []):
                        st.write(f"• {s}")
                    st.markdown("**Improvements:**")
                    for i in feedback.get("improvements", []):
                        st.write(f"• {i}")
                    st.info(feedback.get("bench_comment", ""))
                elif isinstance(feedback, dict) and "error" in feedback:
                    st.error(feedback.get("error"))

        # Precedent Heatmap
        with ci_tabs[3]:
            st.markdown("### 📊 Precedent Heatmap")
            st.caption("Visualize which precedent keywords carry the highest weight today.")

            heat_source = st.radio(
                "Build from",
                ["Use last snapshot", "Generate new snapshot"],
                horizontal=True,
                key="heat_source"
            )
            heat_language = st.selectbox("Language", ["English", "Hindi"], key="heat_language")

            if heat_source == "Generate new snapshot":
                heat_query = st.text_area("Case topic", height=120, key="heat_query")
                if st.button("🔥 Build Heatmap", use_container_width=True, key="heat_build_btn"):
                    if heat_query and len(heat_query) > 30:
                        with st.spinner("Generating snapshot..."):
                            snap = generate_precedent_snapshot(heat_query, heat_language)
                            st.session_state["precedent_snapshot_result"] = snap
                            if "error" not in snap:
                                st.session_state["precedent_heatmap"] = build_precedent_heatmap(snap)
                    else:
                        st.error("Provide a longer case topic (min 30 chars).")
            else:
                if st.button("🔥 Build Heatmap from Last Snapshot", use_container_width=True, key="heat_last_btn"):
                    snap = st.session_state.get("precedent_snapshot_result")
                    if isinstance(snap, dict) and "error" not in snap:
                        st.session_state["precedent_heatmap"] = build_precedent_heatmap(snap)
                    else:
                        st.warning("No valid snapshot found.")

            heatmap = st.session_state.get("precedent_heatmap", [])
            if heatmap:
                st.markdown("**Top Keywords (Today):**")
                for kw, score in heatmap:
                    st.markdown(f"<span class='bench-chip'>{kw}</span>", unsafe_allow_html=True)
                    st.progress(score / 100)

        # Daily Bench Digest
        with ci_tabs[4]:
            st.markdown("### 🗞️ Daily Bench Digest")
            digest_lang = st.selectbox("Language", ["English", "Hindi"], key="digest_lang")
            st.caption("Generates a morning brief for the bench from latest legal news.")
            if st.button("📰 Generate Digest", use_container_width=True, type="primary", key="digest_btn"):
                if st.session_state.get("news_items"):
                    with st.spinner("Preparing bench digest..."):
                        st.session_state["daily_digest"] = generate_daily_bench_digest(
                            st.session_state["news_items"],
                            digest_lang
                        )
                else:
                    st.warning("Fetch news first in the News tab.")

            digest = st.session_state.get("daily_digest")
            if isinstance(digest, dict) and "error" not in digest:
                st.markdown("<div class='order-box'>", unsafe_allow_html=True)
                st.write(digest.get("digest", ""))
                st.markdown("</div>", unsafe_allow_html=True)
                st.success("Bench digest prepared for today's sitting.")
                st.download_button(
                    "📥 Download Digest",
                    digest.get("digest", ""),
                    file_name=f"bench_digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            elif isinstance(digest, dict) and "error" in digest:
                st.error(digest.get("error"))

        # ===== NEW TAB 6: VIRTUAL COURT SIMULATOR 🎭 =====
        with ci_tabs[5]:
            st.markdown('<h3 style="color: #39ff14;">🎭 Virtual Court Simulator</h3>', unsafe_allow_html=True)
            st.markdown("**Practice court arguments in realistic simulations. Get instant feedback and scoring.**")
            st.divider()

            # Setup Section
            col_setup1, col_setup2, col_setup3 = st.columns(3)
            with col_setup1:
                sim_case_type = st.selectbox(
                    "Case Type",
                    ["Criminal Trial", "Bail Hearing", "Civil Appeal", "Consumer Complaint"],
                    key="sim_case_type"
                )
            with col_setup2:
                sim_role = st.selectbox(
                    "Your Role",
                    ["Petitioner Counsel", "Respondent Counsel"],
                    key="sim_role_select"
                )
            with col_setup3:
                sim_difficulty = st.selectbox(
                    "Difficulty",
                    ["Easy (District Court)", "Medium (High Court)", "Expert (Supreme Court)"],
                    key="sim_difficulty"
                )

            sim_language = st.selectbox("Language", ["English", "Hindi"], key="sim_language_vcs")
            sim_facts = st.text_area(
                "Case Facts (or use default)",
                height=150,
                placeholder="Describe your case in detail...",
                key="sim_facts_input"
            )

            if st.button("🎬 Start Simulation", use_container_width=True, type="primary", key="sim_start_btn"):
                if not sim_facts or len(sim_facts) < 50:
                    st.error("Please provide detailed case facts (minimum 50 characters)")
                else:
                    with st.spinner("Preparing court simulation..."):
                        sim_result = start_virtual_court_simulation(
                            sim_case_type, sim_role, sim_difficulty, sim_facts, sim_language
                        )
                        if "error" not in sim_result:
                            st.session_state["court_simulation"] = sim_result
                            st.session_state["sim_current_turn"] = 0
                        else:
                            st.error(sim_result.get("error"))

            # Live Simulation Display
            sim_data = st.session_state.get("court_simulation")
            if isinstance(sim_data, dict) and "error" not in sim_data:
                st.markdown("<div class='order-box' style='background: rgba(57, 255, 20, 0.08); border-left: 4px solid #39ff14;'>", unsafe_allow_html=True)
                judge_opening = sim_data.get('judge_opening', "The Hon'ble Court is now in session")
                st.markdown(f"**⚖️ {judge_opening}**")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("### Judge's Question:")
                st.info(sim_data.get("judge_question", ""))

                st.markdown("### Expected Points to Cover:")
                for point in sim_data.get("expected_points", []):
                    st.write(f"✓ {point}")

                st.markdown("### Your Response:")
                user_response = st.text_area(
                    "Type your court argument here",
                    height=200,
                    key="vcs_user_response"
                )

                col_vcs1, col_vcs2 = st.columns(2)
                with col_vcs1:
                    if st.button("⚖️ Score My Argument", use_container_width=True, key="vcs_score_btn"):
                        if user_response and len(user_response) > 20:
                            with st.spinner("Judge is evaluating your argument..."):
                                eval_result = evaluate_court_argument(
                                    sim_data.get("judge_question", ""),
                                    user_response,
                                    sim_case_type,
                                    sim_difficulty,
                                    sim_language
                                )
                                if "error" not in eval_result:
                                    st.session_state["sim_score_history"].append(eval_result.get("score", 0))
                                    st.session_state["court_simulation"]["evaluation"] = eval_result
                        else:
                            st.error("Please write a more substantial response (20+ characters)")

                with col_vcs2:
                    if st.button("🔁 Reset Simulation", use_container_width=True, key="vcs_reset_btn"):
                        st.session_state["court_simulation"] = None
                        st.session_state["sim_current_turn"] = 0
                        st.session_state["sim_score_history"] = []
                        st.rerun()

                # Display Evaluation if Available
                eval_data = sim_data.get("evaluation")
                if isinstance(eval_data, dict) and "error" not in eval_data:
                    st.divider()
                    st.markdown("### 📊 Evaluation Results")

                    # NEW: Bench Tension Meter
                    tension_score = eval_data.get("tension_score", 50)
                    st.markdown(generate_bench_tension_visual(tension_score), unsafe_allow_html=True)

                    score = eval_data.get("score", 0)
                    score_color = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
                    st.markdown(f"### {score_color} Score: {score}/10")
                    st.progress(score / 10)

                    st.markdown(f"**Reasoning:** {eval_data.get('reasoning', '')}")

                    col_str, col_weak = st.columns(2)
                    with col_str:
                        st.markdown("### ✅ Strengths:")
                        for strength in eval_data.get("strengths", []):
                            st.write(f"• {strength}")

                    with col_weak:
                        st.markdown("### ⚠️ Weaknesses:")
                        for weakness in eval_data.get("weaknesses", []):
                            st.write(f"• {weakness}")

                    st.markdown("### 💡 Missed Opportunity:")
                    st.write(eval_data.get("missed_opportunity", ""))

                    st.markdown("### ⚡ Opposing Counsel's Likely Objection:")
                    st.warning(eval_data.get("opposing_objection", ""))

                    st.markdown("### 🧑‍⚖️ Judge's Likely Reaction:")
                    st.info(eval_data.get("judge_reaction", ""))

                    st.markdown("### ❓ Next Judge Question:")
                    with st.container(border=True):
                        st.write(eval_data.get("next_question", ""))

                    # Progress Analytics
                    if len(st.session_state["sim_score_history"]) > 1:
                        st.divider()
                        st.markdown("### 📈 Your Progress")
                        st.line_chart(st.session_state["sim_score_history"])

        # ===== NEW TAB 7: LIVE CASE STATUS TRACKER 🔍 =====
        with ci_tabs[6]:
            st.markdown('<h3 style="color: #00f3ff;">🔍 Live Case Status Tracker</h3>', unsafe_allow_html=True)
            st.markdown("**Real-time case tracking from Indian courts. Check status, timeline, and documents.**")
            st.divider()

            # Case Input Section
            col_tracker1, col_tracker2 = st.columns([3, 1])
            with col_tracker1:
                case_input = st.text_input(
                    "Enter Case Number",
                    placeholder="Example: CRI-456/2024",
                    help="Try: CRI-456/2024, CIV-789/2023, or TAX-333/2022",
                    key="case_number_input"
                )
            with col_tracker2:
                if st.button("🔍 Search", use_container_width=True, key="case_search_btn"):
                    if case_input:
                        case_data = get_case_status_timeline(case_input)
                        st.session_state["selected_case"] = case_data
                    else:
                        st.error("Enter a case number")

            # Display Case Details
            case = st.session_state.get("selected_case")
            if isinstance(case, dict) and "error" not in case:
                # Case Header
                st.markdown(f"<div class='bench-card'>", unsafe_allow_html=True)
                col_h1, col_h2, col_h3 = st.columns(3)
                with col_h1:
                    st.metric("Case No", case.get("parties", "").split("vs.")[0][:20])
                with col_h2:
                    st.metric("Current Stage", case.get("current_stage", ""))
                with col_h3:
                    st.metric("Progress", f"{case.get('progress_percent', 0)}%")
                st.markdown("</div>", unsafe_allow_html=True)

                # Case Overview
                st.markdown("### 📋 Case Details")
                col_cdet1, col_cdet2 = st.columns(2)
                with col_cdet1:
                    st.write(f"**Parties:** {case.get('parties', 'N/A')}")
                    st.write(f"**Court:** {case.get('court', 'N/A')}")
                    st.write(f"**Judge:** {case.get('judge', 'N/A')}")
                with col_cdet2:
                    st.write(f"**Filed:** {case.get('filing_date', 'N/A')}")
                    st.write(f"**Status:** {case.get('status_color', '')} {case.get('current_stage', 'N/A')}")
                    st.write(f"**Next Hearing:** {case.get('next_hearing', 'N/A')}")

                # Case Timeline
                st.markdown("### 📅 Case Timeline & Progress")
                st.progress(case.get("progress_percent", 0) / 100)

                # Stages Completed vs Remaining
                col_timeline1, col_timeline2 = st.columns(2)
                with col_timeline1:
                    st.markdown("**✅ Completed Stages:**")
                    for stage in case.get("stages_completed", []):
                        st.write(f"✓ {stage}")
                with col_timeline2:
                    st.markdown("**⏳ Remaining Stages:**")
                    for stage in case.get("stages_remaining", []):
                        st.write(f"○ {stage}")

                # Legal Representation
                st.markdown("### 👥 Legal Representation")
                col_counsel1, col_counsel2 = st.columns(2)
                with col_counsel1:
                    st.write(f"**For Petitioner/Plaintiff:**\n{case.get('counsel_for', 'Not assigned')}")
                with col_counsel2:
                    st.write(f"**For Respondent/Defendant:**\n{case.get('counsel_against', 'Not assigned')}")

                # Documents & Orders
                st.markdown("### 📄 Documents & Orders")
                col_doc1, col_doc2 = st.columns(2)
                with col_doc1:
                    st.metric("Documents Filed", case.get("documents_filed", 0))
                with col_doc2:
                    last_order = case.get("last_order", "No orders yet")
                    st.info(f"**Last Order:** {last_order}")

                # Export Options
                st.divider()
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    if st.button("📥 Download Case Summary", use_container_width=True, key="case_download_btn"):
                        summary = f"""CASE STATUS REPORT
════════════════════════════════════════

Case Number: {case.get('parties', '')}
Court: {case.get('court', '')}
Judge: {case.get('judge', '')}
Filing Date: {case.get('filing_date', '')}

CURRENT STATUS: {case.get('current_stage', '')} ({case.get('progress_percent', 0)}% complete)

PARTIES:
{case.get('parties', '')}

TIMELINE:
Completed: {', '.join(case.get('stages_completed', []))}
Remaining: {', '.join(case.get('stages_remaining', []))}

LEGAL REPRESENTATION:
Petitioner: {case.get('counsel_for', '')}
Respondent: {case.get('counsel_against', '')}

DOCUMENTS FILED: {case.get('documents_filed', 0)}
LAST ORDER: {case.get('last_order', '')}
NEXT HEARING: {case.get('next_hearing', '')}

Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"""
                        st.download_button(
                            "📥 Download",
                            summary,
                            file_name=f"case_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                with col_export2:
                    if st.button("🔔 Set Hearing Alert", use_container_width=True, key="case_alert_btn"):
                        st.success(f"✅ Alert set for {case.get('next_hearing', 'next hearing')}")

            elif isinstance(case, dict) and "error" in case:
                st.error(f"❌ {case.get('error', 'Case not found')}")
                st.info(case.get('suggestion', 'Try searching with a different case number'))

            # Demo Cases
            st.divider()
            st.markdown("### 🎯 Demo Cases to Try:")
            demo_cases = {
                "CRI-456/2024": "Criminal case (Murder) - Arguments stage",
                "CIV-789/2023": "Civil case (Property) - Evidence stage",
                "TAX-333/2022": "Tax case - Reserved for judgment"
            }
            for case_no, desc in demo_cases.items():
                if st.button(f"{case_no}: {desc}", use_container_width=True, key=f"demo_{case_no}"):
                    case_data = get_case_status_timeline(case_no)
                    st.session_state["selected_case"] = case_data
                    st.rerun()

    judge_index += 1

    # ========== TAB 2: SMART NEWS AGGREGATOR ==========
    with judge_feature_tabs[judge_index]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(0, 243, 255, 0.1), rgba(176, 38, 255, 0.1)); 
                    padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(0, 243, 255, 0.3);">
            <h3 style="color: #00f3ff; margin: 0;">📰 Smart Legal News Aggregator</h3>
            <p style="color: #9fa8da; margin: 0.5rem 0 0;">Auto-categorizes news by Court Type, Topic Area & Urgency</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='bench-card'>", unsafe_allow_html=True)
        st.markdown("**Daily Bench Brief**")
        st.caption("One-screen briefing for the sitting bench.")
        brief_col1, brief_col2, brief_col3 = st.columns(3)
        with brief_col1:
            st.metric("📌 Items", len(st.session_state.get("news_items", [])))
        with brief_col2:
            crit_count = 0
            cat_data = st.session_state.get("categorized_news")
            if isinstance(cat_data, dict):
                crit_count = sum(1 for n in cat_data.get("categorized_news", []) if n.get("urgency") == "🔴 CRITICAL")
            st.metric("🔴 Critical", crit_count)
        with brief_col3:
            imp_count = 0
            if isinstance(cat_data, dict):
                imp_count = sum(1 for n in cat_data.get("categorized_news", []) if n.get("urgency") == "🟡 IMPORTANT")
            st.metric("🟡 Important", imp_count)
        st.markdown("</div>", unsafe_allow_html=True)

        news_language = st.selectbox("Language", ["English", "Hindi", "Both"], key="judge_news_language")
        sources_text = st.text_area(
            "News Sources (one per line)",
            value="\n".join(DEFAULT_NEWS_SOURCES),
            height=80,
            key="judge_news_sources"
        )

        col_fetch1, col_fetch2 = st.columns(2)
        with col_fetch1:
            if st.button("🔄 FETCH LIVE NEWS", use_container_width=True, type="primary", key="judge_news_fetch"):
                urls = [u.strip() for u in sources_text.split("\n") if u.strip()]
                with st.spinner("🔍 Fetching from legal news sources..."):
                    news_result = fetch_legal_news(urls, max_items=15)
                    st.session_state["news_items"] = news_result.get("items", [])
                    errors = news_result.get("errors", [])
                    if errors:
                        st.warning("⚠️ Some sources failed:\n" + "\n".join(errors))
                    else:
                        st.success(f"✅ Fetched {len(st.session_state['news_items'])} news items!")

        with col_fetch2:
            if st.button("🧠 CATEGORIZE & SUMMARIZE", use_container_width=True, key="judge_news_categorize_btn"):
                if st.session_state.get("news_items"):
                    with st.spinner("⏳ AI analyzing and categorizing news..."):
                        cat_result = categorize_legal_news(st.session_state["news_items"])
                        st.session_state["categorized_news"] = cat_result
                else:
                    st.warning("⚠️ Fetch news first!")

        # Display Categorized News
        if st.session_state.get("categorized_news"):
            cat_data = st.session_state["categorized_news"]
            if isinstance(cat_data, dict) and "error" not in cat_data:
                st.markdown("---")
                st.markdown("### 📊 Categorized Legal News")
                
                # Filter by category
                col_filter1, col_filter2, col_filter3 = st.columns(3)
                with col_filter1:
                    filter_court = st.selectbox(
                        "Filter by Court",
                        ["All"] + list(set([item.get("court", "Other") for item in cat_data.get("categorized_news", [])]))
                    )
                with col_filter2:
                    filter_topic = st.selectbox(
                        "Filter by Topic",
                        ["All"] + list(set([item.get("topic", "Other") for item in cat_data.get("categorized_news", [])]))
                    )
                with col_filter3:
                    filter_urgency = st.selectbox(
                        "Filter by Urgency",
                        ["All", "🔴 CRITICAL", "🟡 IMPORTANT", "🟢 UPDATE"]
                    )
                
                st.markdown("---")
                
                # Display filtered news
                for idx, news_item in enumerate(cat_data.get("categorized_news", []), 1):
                    if filter_court != "All" and news_item.get("court") != filter_court:
                        continue
                    if filter_topic != "All" and news_item.get("topic") != filter_topic:
                        continue
                    if filter_urgency != "All" and news_item.get("urgency") != filter_urgency:
                        continue
                    
                    urgency = news_item.get("urgency", "🟢")
                    court = news_item.get("court", "Other")
                    topic = news_item.get("topic", "General")
                    
                    st.markdown(f"""
                    <div style="background: rgba(26, 31, 58, 0.6); padding: 1.2rem; border-radius: 10px; 
                                border-left: 4px solid {'#ff0000' if urgency == '🔴 CRITICAL' else '#ffea00' if urgency == '🟡 IMPORTANT' else '#00f3ff'};
                                margin-bottom: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: start;">
                            <div>
                                <h4 style="color: #ffea00; margin: 0 0 0.5rem 0;">{news_item.get('title', 'Untitled')}</h4>
                                <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 0.5rem;">
                                    <span style="background: rgba(0, 243, 255, 0.15); padding: 4px 12px; border-radius: 8px; font-size: 0.85rem; color: #00f3ff;">
                                        🏛️ {court}
                                    </span>
                                    <span style="background: rgba(176, 38, 255, 0.15); padding: 4px 12px; border-radius: 8px; font-size: 0.85rem; color: #b026ff;">
                                        📋 {topic}
                                    </span>
                                    <span style="background: rgba(57, 255, 20, 0.15); padding: 4px 12px; border-radius: 8px; font-size: 0.85rem; color: #39ff14;">
                                        {urgency}
                                    </span>
                                </div>
                                <p style="color: #9fa8da; margin: 0.5rem 0;">💡 <strong>Impact:</strong> {news_item.get('impact', 'N/A')}</p>
                                <p style="color: #9fa8da; margin: 0.5rem 0;">
                                    {'✅ <strong>Action Required</strong> - Review & apply to your cases' if news_item.get('action_required') else '📌 For reference only'}
                                </p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            elif isinstance(cat_data, dict) and "error" in cat_data:
                st.error(f"❌ {cat_data.get('error', 'Unknown error occurred')}")

    judge_index += 1

    # ========== TAB 2: BAIL APPLICATION GENERATOR ==========
    with judge_feature_tabs[judge_index]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(57, 255, 20, 0.1), rgba(0, 243, 255, 0.1)); 
                    padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(57, 255, 20, 0.3);">
            <h3 style="color: #39ff14; margin: 0;">🆓 Bail Application Generator</h3>
            <p style="color: #9fa8da; margin: 0.5rem 0 0;">Professional, court-ready bail applications in minutes - Saves ₹15,000-50,000!</p>
        </div>
        """, unsafe_allow_html=True)

        st.info("⚡ Step-by-step bail application wizard. All responses automatically formatted for court filing.")

        # Step 1: Case Details
        with st.expander("📋 STEP 1: Case Details", expanded=True):
            st.markdown("**Enter your case information:**")
            col_bail1, col_bail2 = st.columns(2)
            with col_bail1:
                fir_no = st.text_input("FIR Number", placeholder="e.g., 456/2024", key="bail_fir_no")
                police_station = st.text_input("Police Station", placeholder="e.g., Civil Lines", key="bail_ps")
            with col_bail2:
                sections = st.text_input("Sections of Law", placeholder="e.g., 420, 467, 471 IPC", key="bail_sections")
                arrest_date = st.date_input("Date of Arrest", key="bail_arrest_date")

        # Step 2: Personal Details
        with st.expander("👤 STEP 2: Personal Details", expanded=True):
            st.markdown("**About the accused:**")
            col_personal1, col_personal2 = st.columns(2)
            with col_personal1:
                name = st.text_input("Full Name", placeholder="Accused name", key="bail_name")
                age = st.number_input("Age", min_value=18, max_value=100, step=1, key="bail_age")
                occupation = st.selectbox("Occupation", ["Shopkeeper", "Farmer", "Laborer", "Employee", "Student", "Business owner", "Other"], key="bail_occupation")
            with col_personal2:
                family_status = st.selectbox("Family Status", ["Married", "Single", "Divorced", "Widowed"], key="bail_family")
                dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1, key="bail_dependents")
                medical = st.text_area("Medical Issues (if any)", placeholder="e.g., Diabetes, Hypertension", height=60, key="bail_medical")

        # Step 3: Criminal History & Character
        with st.expander("📝 STEP 3: Background Check", expanded=True):
            st.markdown("**Criminal record & character:**")
            col_char1, col_char2 = st.columns(2)
            with col_char1:
                criminal_record = st.selectbox("Previous Criminal Record", ["No", "Yes - Acquitted", "Yes - Convicted"], key="bail_record")
                if criminal_record != "No":
                    record_details = st.text_area("Details of previous cases", height=80, key="bail_record_details")
            with col_char2:
                community_ties = st.multiselect(
                    "Community Ties",
                    ["Home ownership", "Local business", "Family in area", "Employment history", "Social work"],
                    key="bail_ties"
                )

        # Step 4: Bail Grounds (AI Suggested)
        with st.expander("⚖️ STEP 4: Bail Grounds", expanded=True):
            st.markdown("**Select applicable grounds (or let AI suggest):**")
            
            default_grounds = [
                "No criminal history/clean record",
                "Willing to cooperate with investigation",
                "Strong community ties (family, job, home)",
                "Not a flight risk (stable residence)",
                "Trial will take 2-3 years (bail preferable to jail)",
                "Parity with co-accused (others got bail)",
                "Age & medical condition (elderly/sick)",
                "Sole breadwinner of family",
                "Presumption of innocence until proven guilty",
                "Investigation is complete"
            ]
            
            bail_grounds = st.multiselect(
                "Select Bail Grounds",
                default_grounds,
                default=[default_grounds[0], default_grounds[1], default_grounds[3]],
                key="bail_grounds"
            )

        # Generate Button
        col_gen1, col_gen2 = st.columns([2, 1])
        with col_gen1:
            if st.button("📝 GENERATE BAIL APPLICATION", use_container_width=True, type="primary", key="bail_generate_btn"):
                if not all([fir_no, police_station, sections, name]):
                    st.error("❌ Please fill in mandatory fields: FIR No, Police Station, Sections, Name")
                else:
                    with st.spinner("⏳ AI is drafting your bail application... (generating now)"):
                        case_details = f"""FIR No: {fir_no}
Police Station: {police_station}
Sections: {sections}
Arrest Date: {arrest_date}"""
                        
                        personal_details = f"""Name: {name}
Age: {age} years
Occupation: {occupation}
Family Status: {family_status}
Dependents: {dependents}
Medical Issues: {medical if medical else 'None'}
Criminal Record: {criminal_record}
Community Ties: {', '.join(community_ties) if community_ties else 'Multiple'}"""
                        
                        st.session_state["bail_application"] = generate_bail_application(
                            case_details, personal_details, bail_grounds
                        )
                    
                    if st.session_state.get("bail_application", {}).get("success"):
                        st.success("✅ Bail application generated successfully!")

        with col_gen2:
            bail_app_data = st.session_state.get("bail_application")
            if bail_app_data and isinstance(bail_app_data, dict) and bail_app_data.get("success"):
                st.metric("📄 Pages", f"~{bail_app_data.get('estimated_length', 5)} pages")

        # Display Generated Application
        if st.session_state.get("bail_application"):
            bail_app = st.session_state["bail_application"]
            if isinstance(bail_app, dict) and bail_app.get("success"):
                st.markdown("---")
                st.markdown("### 📄 Your Bail Application (Ready to File)")
                
                with st.container():
                    st.text_area(
                        "Application Text",
                        value=bail_app.get("application", ""),
                        height=400,
                        disabled=True,
                        key="bail_app_display"
                    )
                
                col_action1, col_action2, col_action3 = st.columns(3)
                with col_action1:
                    download_name = st.session_state.get("bail_name", "Accused")
                    download_fir = st.session_state.get("bail_fir_no", "FIR")
                    st.download_button(
                        "📥 Download as .txt",
                        bail_app.get("application", ""),
                        file_name=f"Bail_Application_{download_name}_{download_fir}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col_action2:
                    st.button("📋 Copy to Clipboard", use_container_width=True, key="bail_copy_btn")
                with col_action3:
                    st.button("🔄 Regenerate", use_container_width=True, key="bail_regen_btn")
            elif isinstance(bail_app, dict) and not bail_app.get("success"):
                st.error(f"❌ {bail_app.get('error', 'Error generating bail application')}")

    judge_index += 1

    # ========== TAB 3: INSTANT LEGAL ADVICE ==========
    with judge_feature_tabs[judge_index]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(255, 234, 0, 0.1), rgba(255, 0, 110, 0.1)); 
                    padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255, 234, 0, 0.3);">
            <h3 style="color: #ffea00; margin: 0;">⚡ Instant Legal Advice (10 Second Response)</h3>
            <p style="color: #9fa8da; margin: 0.5rem 0 0;">Emergency legal guidance for critical situations - Available 24/7!</p>
        </div>
        """, unsafe_allow_html=True)

        st.warning("🚨 Use this for EMERGENCY scenarios only. For formal consultation, consult a licensed lawyer.")

        # Quick Scenario Selection
        st.markdown("<h4 style='color: #00f3ff;'>Quick Scenarios (Click to auto-fill):</h4>", unsafe_allow_html=True)
        
        scenarios_col1, scenarios_col2, scenarios_col3 = st.columns(3)
        quick_scenarios = {
            "Arrest without warrant": "Police arrested me without a warrant. What are my legal rights?",
            "Police search": "Police searching my home without warrant - is this legal?",
            "Bail conditions": "Can bail conditions be extremely harsh? How to challenge?",
            "Dowry harassment": "My wife's family demanding dowry. What immediate steps?",
            "Property dispute": "Fighting over ancestor's land property. What's my right?"
        }
        
        advice_scenario = None
        
        with scenarios_col1:
            if st.button("🚔 Arrest without warrant", use_container_width=True):
                advice_scenario = quick_scenarios["Arrest without warrant"]
        
        with scenarios_col2:
            if st.button("🔍 Police search", use_container_width=True):
                advice_scenario = quick_scenarios["Police search"]
        
        with scenarios_col3:
            if st.button("⚖️ Bail conditions", use_container_width=True):
                advice_scenario = quick_scenarios["Bail conditions"]

        st.markdown("---")
        
        # Or type custom scenario
        scenario_input = st.text_area(
            "Or describe your emergency situation:",
            value=advice_scenario or "",
            height=100,
            placeholder="Describe what happened. Be brief and specific.",
            key="instant_advice_input"
        )

        advice_language = st.selectbox("Response Language", ["English", "Hindi"], key="instant_advice_language")

        # Generate Instant Advice
        if st.button("⚡ GET INSTANT ADVICE (10 SECONDS)", use_container_width=True, type="primary", key="instant_advice_btn"):
            if not scenario_input or len(scenario_input) < 20:
                st.error("❌ Please describe your situation (at least 20 characters)")
            else:
                with st.spinner("⚡ AI Judge analyzing your emergency... (10 seconds)"):
                    st.session_state["instant_advice"] = instant_legal_advice(scenario_input, advice_language)
                
                if st.session_state.get("instant_advice", {}).get("success"):
                    st.success("✅ Advice generated!")

        # Display Instant Advice
        if st.session_state.get("instant_advice"):
            advice = st.session_state["instant_advice"]
            if isinstance(advice, dict) and advice.get("success"):
                st.markdown("---")
                
                urgency = advice.get("urgency", "🟡 IMPORTANT")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(176, 38, 255, 0.15), rgba(0, 243, 255, 0.15));
                            padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255, 234, 0, 0.3);">
                    <h3 style="color: #ffea00; margin-top: 0;">{urgency} LEGAL ADVICE</h3>
                    <hr style="border-color: rgba(0, 243, 255, 0.3);">
                    <div style="color: #e8eaf6; line-height: 1.8; font-size: 1rem;">
                        {advice.get('advice', '').replace(chr(10), '<br/>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                col_action1, col_action2 = st.columns(2)
                with col_action1:
                    st.button("🆘 Contact Emergency Helpline", use_container_width=True, key="advice_helpline")
                with col_action2:
                    st.button("🔄 Ask Another Question", use_container_width=True, key="advice_another")
            elif isinstance(advice, dict) and not advice.get("success"):
                st.error(f"❌ {advice.get('error', 'Error generating advice')}")

    judge_index += 1

    # ========== TAB 4: JUDGE SAHIB AI ==========
    with judge_feature_tabs[judge_index]:
        st.markdown('<h3 style="color: #00f3ff;">👨‍⚖️ Judge Sahib AI (Supreme Court Style)</h3>', unsafe_allow_html=True)
        st.info("💡 Judge-like reasoning with strict, high-level guidance.")

        judge_language = st.selectbox("Response Language", ["English", "Hindi"], key="judge_language_mode")
        judge_style = st.selectbox(
            "Output Style",
            ["Bench Guidance", "Draft Order"],
            key="judge_output_style"
        )
        judge_query = st.text_area("Describe your case", height=200, key="judge_query_mode")

        if st.button("⚖️ GET JUDGE OPINION", use_container_width=True, type="primary", key="judge_opinion_btn"):
            if judge_query and len(judge_query) > 40:
                with st.spinner("Judge Sahib is analyzing..."):
                    if judge_style == "Draft Order":
                        prompt_prefix = "Draft a concise court order with headings: Facts, Issues, Findings, Directions, Compliance Date, Disclaimer."
                        judge_resp = judge_sahib_response(f"{prompt_prefix}\n\n{judge_query}", judge_language)
                    else:
                        judge_resp = judge_sahib_response(judge_query, judge_language)
                    st.session_state["judge_chat_history"].append({
                        "query": judge_query,
                        "response": judge_resp,
                        "style": judge_style
                    })
            else:
                st.error("❌ Please provide at least 40 characters")

        if st.session_state.get("judge_chat_history"):
            st.markdown("### 🧑‍⚖️ Judge Sahib Responses")
            for idx, item in enumerate(reversed(st.session_state["judge_chat_history"][-5:]), 1):
                resp = item.get("response", {})
                if "error" in resp:
                    st.error(resp.get("error"))
                else:
                    style_label = item.get("style", "Bench Guidance")
                    st.markdown(f"**Case #{idx} ({style_label})**")
                    st.markdown(f"<div class='order-box'>{resp.get('response', '')}</div>", unsafe_allow_html=True)

    judge_index += 1

    # Bench Notes Tab
    if st.session_state.get("enable_bench_notes"):
        with judge_feature_tabs[judge_index]:
            st.markdown('<h3 style="color: #00f3ff;">🗒️ Bench Notes</h3>', unsafe_allow_html=True)
            st.info("💡 Quick judge-style notes for case hearing.")

            bench_language = st.selectbox("Language", ["English", "Hindi"], key="bench_language")
            bench_query = st.text_area("Case summary", height=180, key="bench_query")

            if st.button("📝 GENERATE BENCH NOTES", use_container_width=True, type="primary", key="bench_notes_btn"):
                if bench_query and len(bench_query) > 40:
                    with st.spinner("Preparing bench notes..."):
                        st.session_state["bench_notes_result"] = generate_bench_notes(bench_query, bench_language)
                else:
                    st.error("❌ Please provide at least 40 characters")

            if st.session_state.get("bench_notes_result"):
                notes = st.session_state["bench_notes_result"]
                if "error" not in notes:
                    st.markdown("<div class='order-box'>", unsafe_allow_html=True)
                    st.markdown("**Issues for Determination:**")
                    for item in notes.get("issues_for_determination", []):
                        st.write(f"• {item}")
                    st.markdown("**Key Facts:**")
                    for item in notes.get("key_facts", []):
                        st.write(f"• {item}")
                    st.markdown("**Evidence to Watch:**")
                    for item in notes.get("evidence_to_watch", []):
                        st.write(f"• {item}")
                    st.markdown("**Questions for Counsel:**")
                    for item in notes.get("questions_for_counsel", []):
                        st.write(f"• {item}")
                    st.markdown("</div>", unsafe_allow_html=True)

                    notes_text = "\n".join([
                        "Issues:",
                        *[f"- {i}" for i in notes.get("issues_for_determination", [])],
                        "",
                        "Key Facts:",
                        *[f"- {i}" for i in notes.get("key_facts", [])],
                        "",
                        "Evidence to Watch:",
                        *[f"- {i}" for i in notes.get("evidence_to_watch", [])],
                        "",
                        "Questions for Counsel:",
                        *[f"- {i}" for i in notes.get("questions_for_counsel", [])],
                    ])
                    st.download_button(
                        "📥 Download Bench Notes",
                        notes_text,
                        file_name=f"bench_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error(notes.get("error"))

        judge_index += 1

    # Precedent Snapshot Tab
    if st.session_state.get("enable_precedent_snapshot"):
        with judge_feature_tabs[judge_index]:
            st.markdown('<h3 style="color: #00f3ff;">📚 Precedent Snapshot</h3>', unsafe_allow_html=True)
            st.info("💡 High-level themes to guide precedent research.")

            prec_language = st.selectbox("Language", ["English", "Hindi"], key="prec_language")
            prec_query = st.text_area("Case topic / issue", height=160, key="prec_query")

            if st.button("📚 GENERATE SNAPSHOT", use_container_width=True, type="primary", key="prec_snapshot_btn"):
                if prec_query and len(prec_query) > 30:
                    with st.spinner("Generating precedent snapshot..."):
                        st.session_state["precedent_snapshot_result"] = generate_precedent_snapshot(prec_query, prec_language)
                else:
                    st.error("❌ Please provide at least 30 characters")

            if st.session_state.get("precedent_snapshot_result"):
                snap = st.session_state["precedent_snapshot_result"]
                if "error" not in snap:
                    st.markdown("<div class='order-box'>", unsafe_allow_html=True)
                    st.markdown("**Precedent Themes:**")
                    for item in snap.get("precedent_themes", []):
                        st.write(f"• {item}")
                    st.markdown("**Likely Principles:**")
                    for item in snap.get("likely_principles", []):
                        st.write(f"• {item}")
                    st.markdown("**Search Keywords:**")
                    for item in snap.get("search_keywords", []):
                        st.write(f"• {item}")
                    st.info(snap.get("risk_caveat", ""))
                    st.markdown("</div>", unsafe_allow_html=True)

                    snap_text = "\n".join([
                        "Precedent Themes:",
                        *[f"- {i}" for i in snap.get("precedent_themes", [])],
                        "",
                        "Likely Principles:",
                        *[f"- {i}" for i in snap.get("likely_principles", [])],
                        "",
                        "Search Keywords:",
                        *[f"- {i}" for i in snap.get("search_keywords", [])],
                        "",
                        f"Risk Caveat: {snap.get('risk_caveat', '')}",
                    ])
                    st.download_button(
                        "📥 Download Snapshot",
                        snap_text,
                        file_name=f"precedent_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.error(snap.get("error"))

    st.divider()
    st.stop()

# 🎨 ENHANCED INTERACTIVE HEADER
st.markdown("""
<div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
    <h1 style="font-size: 3.5rem; margin: 0; background: linear-gradient(135deg, #00f3ff 0%, #b026ff 50%, #ff006e 100%); 
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                background-clip: text; animation: glow 3s ease-in-out infinite; 
                text-shadow: none; font-weight: 900;">
        ⚖️ AI LAWYER RAG
    </h1>
    <p style="font-size: 1.3rem; color: #9fa8da; margin-top: 0.5rem; font-weight: 300; letter-spacing: 2px;">
        🔮 PROFESSIONAL LEGAL INTELLIGENCE SYSTEM 🔮
    </p>
    <div style="display: flex; justify-content: center; gap: 15px; margin-top: 1.5rem; flex-wrap: wrap;">
        <span style="background: rgba(0, 243, 255, 0.15); padding: 8px 20px; border-radius: 20px; 
                     border: 1px solid rgba(0, 243, 255, 0.4); color: #00f3ff; font-size: 0.9rem; font-weight: 600;">
            ⚡ AI-Powered
        </span>
        <span style="background: rgba(176, 38, 255, 0.15); padding: 8px 20px; border-radius: 20px; 
                     border: 1px solid rgba(176, 38, 255, 0.4); color: #b026ff; font-size: 0.9rem; font-weight: 600;">
            🛡️ Risk Analysis
        </span>
        <span style="background: rgba(57, 255, 20, 0.15); padding: 8px 20px; border-radius: 20px; 
                     border: 1px solid rgba(57, 255, 20, 0.4); color: #39ff14; font-size: 0.9rem; font-weight: 600;">
            📚 Citation Validator
        </span>
        <span style="background: rgba(255, 0, 110, 0.15); padding: 8px 20px; border-radius: 20px; 
                     border: 1px solid rgba(255, 0, 110, 0.4); color: #ff006e; font-size: 0.9rem; font-weight: 600;">
            🔍 Smart Search
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Stats Dashboard with Enhanced Visuals
st.markdown('<div style="margin: 2rem 0;">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(0, 243, 255, 0.1) 0%, rgba(0, 243, 255, 0.05) 100%); 
                 border: 2px solid rgba(0, 243, 255, 0.3); border-radius: 15px; padding: 1.5rem; 
                 box-shadow: 0 0 20px rgba(0, 243, 255, 0.2); transition: all 0.3s ease; text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📚</div>
        <div style="font-size: 2rem; font-weight: 900; color: #00f3ff; margin-bottom: 0.3rem;">
            {}</div>
        <div style="font-size: 0.9rem; color: #9fa8da; font-weight: 600;">DOCUMENTS</div>
    </div>
    """.format(len(load_manifest())), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(176, 38, 255, 0.1) 0%, rgba(176, 38, 255, 0.05) 100%); 
                 border: 2px solid rgba(176, 38, 255, 0.3); border-radius: 15px; padding: 1.5rem; 
                 box-shadow: 0 0 20px rgba(176, 38, 255, 0.2); transition: all 0.3s ease; text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💬</div>
        <div style="font-size: 2rem; font-weight: 900; color: #b026ff; margin-bottom: 0.3rem;">
            {}</div>
        <div style="font-size: 0.9rem; color: #9fa8da; font-weight: 600;">QUERIES</div>
    </div>
    """.format(len(st.session_state.get("messages", [])) // 2), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(57, 255, 20, 0.1) 0%, rgba(57, 255, 20, 0.05) 100%); 
                 border: 2px solid rgba(57, 255, 20, 0.3); border-radius: 15px; padding: 1.5rem; 
                 box-shadow: 0 0 20px rgba(57, 255, 20, 0.2); transition: all 0.3s ease; text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📝</div>
        <div style="font-size: 2rem; font-weight: 900; color: #39ff14; margin-bottom: 0.3rem;">
            {}</div>
        <div style="font-size: 0.9rem; color: #9fa8da; font-weight: 600;">Q&A PAIRS</div>
    </div>
    """.format(len(st.session_state.get("qa_history", []))), unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255, 0, 110, 0.1) 0%, rgba(255, 0, 110, 0.05) 100%); 
                 border: 2px solid rgba(255, 0, 110, 0.3); border-radius: 15px; padding: 1.5rem; 
                 box-shadow: 0 0 20px rgba(255, 0, 110, 0.2); transition: all 0.3s ease; text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔖</div>
        <div style="font-size: 2rem; font-weight: 900; color: #ff006e; margin-bottom: 0.3rem;">
            {}</div>
        <div style="font-size: 0.9rem; color: #9fa8da; font-weight: 600;">SAVED SEARCHES</div>
    </div>
    """.format(len(st.session_state.get("saved_searches", []))), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# ==================== 🧭 CLIENT JOURNEY WIZARD ====================
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <h2 style="font-size: 2.3rem; background: linear-gradient(135deg, #00f3ff 0%, #ffea00 100%);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                font-weight: 900;">
        🧭 CLIENT JOURNEY WIZARD (Step-by-Step)
    </h2>
    <p style="color: #9fa8da; font-size: 1rem; margin-top: 0.5rem;">
        New client? Follow these simple steps. We will guide you like a lawyer.
    </p>
</div>
""", unsafe_allow_html=True)

wizard_step = st.session_state["wizard_step"]
st.progress(wizard_step / 4)

if wizard_step == 1:
    st.markdown("### Step 1: Select Your Problem Type")
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        wiz_category = st.selectbox(
            "Problem Category",
            ["Criminal", "Family", "Property", "Consumer", "Employment", "Cyber Fraud", "Money/Loan", "Accident", "Business", "Other"],
            key="wiz_category"
        )
    with col_w2:
        wiz_language = st.selectbox("Language", ["Hinglish", "English"], key="wiz_language")
    with col_w3:
        wiz_urgency = st.selectbox("Urgency", ["Low", "Medium", "High"], key="wiz_urgency")

    if st.button("Next →", use_container_width=True, key="wiz_next_1"):
        st.session_state["wizard_data"].update({
            "category": wiz_category,
            "language": wiz_language,
            "urgency": wiz_urgency
        })
        st.session_state["wizard_step"] = 2
        st.rerun()

elif wizard_step == 2:
    st.markdown("### Step 2: Describe What Happened (Simple Words)")
    st.caption("Explain like you are speaking to a lawyer. No legal terms needed.")
    wiz_problem = st.text_area("Your Problem", height=160, key="wiz_problem")
    col_w21, col_w22, col_w23 = st.columns(3)
    with col_w21:
        wiz_location = st.text_input("City/State", key="wiz_location")
    with col_w22:
        wiz_timeline = st.selectbox("When did it happen?", ["Today", "This week", "This month", "Older"], key="wiz_timeline")
    with col_w23:
        wiz_outcome = st.text_input("What result do you want?", key="wiz_outcome")

    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("← Back", use_container_width=True, key="wiz_back_2"):
            st.session_state["wizard_step"] = 1
            st.rerun()
    with col_nav2:
        if st.button("Next →", use_container_width=True, key="wiz_next_2"):
            st.session_state["wizard_data"].update({
                "problem": wiz_problem,
                "location": wiz_location,
                "timeline": wiz_timeline,
                "desired_outcome": wiz_outcome
            })
            st.session_state["wizard_step"] = 3
            st.rerun()

elif wizard_step == 3:
    st.markdown("### Step 3: Generate Your Legal Path")
    st.caption("We will suggest sections, actions, and how to talk to a lawyer.")
    if st.button("⚖️ Generate My Legal Path", use_container_width=True, type="primary", key="wiz_generate"):
        data = st.session_state.get("wizard_data", {})
        if not data.get("problem"):
            st.error("Please complete Step 2 first.")
        else:
            with st.spinner("Preparing your step-by-step plan..."):
                st.session_state["wizard_result"] = generate_client_journey_plan(
                    data.get("problem", ""),
                    data.get("category", "Other"),
                    data.get("urgency", "Medium"),
                    data.get("language", "Hinglish"),
                    data.get("location", ""),
                    data.get("timeline", ""),
                    data.get("desired_outcome", "")
                )
            st.session_state["wizard_step"] = 4
            st.rerun()

    if st.button("← Back", use_container_width=True, key="wiz_back_3"):
        st.session_state["wizard_step"] = 2
        st.rerun()

else:
    st.markdown("### Step 4: Your Step-by-Step Lawyer Plan")
    result = st.session_state.get("wizard_result")
    if isinstance(result, dict) and "error" not in result:
        st.markdown("<div class='order-box'>", unsafe_allow_html=True)
        st.markdown("**Simple Summary:**")
        st.write(result.get("simple_summary", ""))

        st.markdown("**Likely Sections (basic):**")
        for item in result.get("likely_sections", []):
            st.write(f"• {item}")

        st.markdown("**Your Rights (short):**")
        for item in result.get("rights_overview", []):
            st.write(f"• {item}")

        st.markdown("**Immediate Actions (Step-wise):**")
        for item in result.get("immediate_actions", []):
            st.write(f"• {item}")

        st.markdown("**Evidence Checklist:**")
        for item in result.get("evidence_checklist", []):
            st.write(f"• {item}")

        st.markdown("**Questions to Ask a Lawyer:**")
        for item in result.get("questions_for_lawyer", []):
            st.write(f"• {item}")

        st.markdown("**Lawyer Call Script (Hinglish):**")
        st.write(result.get("lawyer_call_script", ""))

        st.markdown("**Recommended Features to Use Next:**")
        for item in result.get("recommended_features", []):
            st.write(f"✅ {item}")
        st.markdown("</div>", unsafe_allow_html=True)

        download_text = "\n".join([
            "CLIENT LEGAL PATH", "",
            "Summary:", result.get("simple_summary", ""), "",
            "Likely Sections:", *[f"- {i}" for i in result.get("likely_sections", [])], "",
            "Immediate Actions:", *[f"- {i}" for i in result.get("immediate_actions", [])], "",
            "Evidence Checklist:", *[f"- {i}" for i in result.get("evidence_checklist", [])], "",
            "Lawyer Call Script:", result.get("lawyer_call_script", "")
        ])
        st.download_button(
            "📥 Download Client Plan",
            download_text,
            file_name=f"client_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.error(result.get("error", "Unable to generate plan"))

    col_reset1, col_reset2 = st.columns(2)
    with col_reset1:
        if st.button("🔁 Start Over", use_container_width=True, key="wiz_restart"):
            st.session_state["wizard_step"] = 1
            st.session_state["wizard_result"] = None
            st.session_state["wizard_data"] = {}
            st.rerun()
    with col_reset2:
        if st.button("← Edit Answers", use_container_width=True, key="wiz_edit"):
            st.session_state["wizard_step"] = 2
            st.rerun()

st.divider()

# ==================== 🔥 GAME-CHANGING FEATURES UI 🔥 ====================
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <h2 style="font-size: 2.5rem; background: linear-gradient(135deg, #ff006e 0%, #00f3ff 100%); 
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                font-weight: 900;">
        🔥 FREE LEGAL SERVICES - SAVE LAKHS IN LAWYER FEES! 🔥
    </h2>
    <p style="color: #9fa8da; font-size: 1.1rem; margin-top: 0.5rem;">Professional Legal Assistance - Absolutely FREE</p>
</div>
""", unsafe_allow_html=True)

base_tabs = [
    "📝 Document Generator",
    "🎯 Case Analyzer",
    "⚡ Legal Notice",
    "🗂️ Court Filing",
    "🤖 AI Lawyer Chat",
    "📋 Contract Analyzer",
    "⚖️ Court Script",
    "📂 Evidence Organizer",
    "💰 Settlement Calculator",
    "⏰ Timeline Checker",
]

extra_tabs = []
if st.session_state.get("enable_advanced_tools"):
    extra_tabs.append("🚀 Advanced Tools")
if st.session_state.get("enable_legal_news"):
    extra_tabs.append("📰 Legal News")
if st.session_state.get("enable_judge_ai"):
    extra_tabs.append("👨‍⚖️ Judge Sahib AI")

feature_tabs = st.tabs(base_tabs + extra_tabs)

# TAB 1: Document Generator - STEP-WISE WIZARD
with feature_tabs[0]:
    st.markdown('<h3 style="color: #00f3ff;">📝 Legal Document Generator - Step by Step</h3>', unsafe_allow_html=True)
    st.info("💡 Create professional legal documents like a real lawyer would! Let's do it step-by-step.")
    
    # Initialize document wizard state
    if "doc_gen_step" not in st.session_state:
        st.session_state["doc_gen_step"] = 0
    if "doc_gen_data" not in st.session_state:
        st.session_state["doc_gen_data"] = {}

    # STEP 1: Choose Document Type
    if st.session_state["doc_gen_step"] == 0:
        st.markdown("### 📋 Step 1: What kind of document do you need?")
        st.progress(0.25, text="Step 1 of 3")
        
        doc_options = {
            "🏢 Consumer Complaint": "consumer_complaint",
            "⚡ Legal Notice": "legal_notice",
            "🏠 Rent Agreement": "rent_agreement",
            "🚨 FIR Draft": "fir_draft",
            "📋 Affidavit": "affidavit"
        }
        
        selected_doc = st.radio("Choose Document Type", list(doc_options.keys()), 
                               help="Pick the legal document you need")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➡️ Next", use_container_width=True, type="primary"):
                st.session_state["doc_gen_data"]["doc_type"] = doc_options[selected_doc]
                st.session_state["doc_gen_data"]["doc_type_display"] = selected_doc
                st.session_state["doc_gen_step"] = 1
                st.rerun()
    
    # STEP 2: Fill in Details (ENHANCED WITH LIVE VALIDATION)
    elif st.session_state["doc_gen_step"] == 1:
        doc_type = st.session_state["doc_gen_data"]["doc_type"]
        st.progress(0.50, text="Step 2 of 3: Fill Your Details")
        st.markdown(f"### 📝 Step 2: Tell us about your {st.session_state['doc_gen_data']['doc_type_display']}")
        
        # FORM VALIDATION STATE
        st.markdown("**📋 Complete all fields marked with * to continue:**")
        
        if doc_type == "consumer_complaint":
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Your Full Name *", placeholder="e.g., Rajesh Kumar")
                phone = st.text_input("Your Phone *", placeholder="10-digit number")
            with col2:
                address = st.text_area("Your Address *", height=80, placeholder="Full address with pin code")
                email = st.text_input("Your Email (optional)", placeholder="your@email.com")
            
            company_name = st.text_input("Company/Person You're Complaining Against *", placeholder="e.g., XYZ Electronics Pvt Ltd")
            issue = st.text_area("What happened? Describe the problem clearly *", height=150, 
                                placeholder="Be specific: What product/service did you buy? What went wrong? When did it happen?")
            amount = st.number_input("Compensation Amount You Want (₹) *", min_value=1000, value=50000, step=1000)
            
            # LIVE VALIDATION FEEDBACK
            st.markdown("---")
            st.markdown("### ✅ Form Completion Status")
            
            validation_col1, validation_col2, validation_col3 = st.columns(3)
            
            with validation_col1:
                if name and phone and address:
                    st.success("✅ Your Details: Complete")
                else:
                    missing = []
                    if not name: missing.append("Name")
                    if not phone: missing.append("Phone")
                    if not address: missing.append("Address")
                    st.warning(f"⚠️ Missing: {', '.join(missing)}")
            
            with validation_col2:
                if company_name and issue and amount > 0:
                    st.success("✅ Complaint Details: Complete")
                else:
                    missing = []
                    if not company_name: missing.append("Respondent Name")
                    if not issue: missing.append("Issue Description")
                    if amount == 0: missing.append("Amount")
                    st.warning(f"⚠️ Missing: {', '.join(missing)}")
            
            with validation_col3:
                st.metric("Complaint Amount", f"₹{amount:,}")
            
            st.session_state["doc_gen_data"].update({
                "complainant_name": name, "complainant_phone": phone, "complainant_address": address,
                "complainant_email": email, "respondent_name": company_name, "issue": issue, "amount": amount
            })
        
        elif doc_type == "legal_notice":
            col1, col2 = st.columns(2)
            with col1:
                your_name = st.text_input("Your Full Name *", placeholder="Your full name")
                your_address = st.text_area("Your Address *", height=80, placeholder="Complete address")
            with col2:
                recipient = st.text_input("Send Notice To *", placeholder="Person/Company name")
                recipient_address = st.text_area("Their Address *", height=80, placeholder="Their address")
            
            subject = st.text_input("Subject of Notice *", placeholder="e.g., Payment Recovery for Loan Default")
            issue = st.text_area("What's the issue? Explain clearly *", height=150, placeholder="Describe what happened and why they owe you")
            amount = st.number_input("Amount You're Claiming (₹) *", min_value=1000, value=100000, step=1000)
            deadline = st.number_input("How many days should they have to respond? *", min_value=5, value=15, step=1)
            
            # LIVE VALIDATION FEEDBACK
            st.markdown("---")
            st.markdown("### ✅ Form Completion Status")
            
            validation_col1, validation_col2 = st.columns(2)
            
            with validation_col1:
                if your_name and your_address and recipient and recipient_address:
                    st.success("✅ Sender & Recipient Details: Complete")
                else:
                    missing = []
                    if not your_name: missing.append("Your Name")
                    if not your_address: missing.append("Your Address")
                    if not recipient: missing.append("Recipient Name")
                    if not recipient_address: missing.append("Recipient Address")
                    st.warning(f"⚠️ Missing: {', '.join(missing)}")
            
            with validation_col2:
                if subject and issue and amount > 0 and deadline > 0:
                    st.success(f"✅ Notice Details: Complete (₹{amount:,}, {deadline} days)")
                else:
                    missing = []
                    if not subject: missing.append("Subject")
                    if not issue: missing.append("Issue")
                    if amount == 0: missing.append("Amount")
                    if deadline == 0: missing.append("Deadline")
                    st.warning(f"⚠️ Missing: {', '.join(missing)}")
            
            st.session_state["doc_gen_data"].update({
                "your_name": your_name, "your_address": your_address, "recipient": recipient,
                "recipient_address": recipient_address, "subject": subject, "issue": issue,
                "amount": amount, "deadline": deadline
            })
        
        elif doc_type == "fir_draft":
            col1, col2 = st.columns(2)
            with col1:
                complainant = st.text_input("Your Full Name *", placeholder="Your name")
                age = st.number_input("Your Age *", min_value=18, max_value=120, value=30)
                phone = st.text_input("Your Phone *", placeholder="10-digit number")
            with col2:
                police_station = st.text_input("Which Police Station? *", placeholder="e.g., Koramangala Police Station")
                district = st.text_input("District *", placeholder="e.g., Bangalore / Mumbai")
                state = st.text_input("State *", placeholder="e.g., Karnataka")
            
            incident = st.text_area("Describe what happened *", height=150,
                                   placeholder="What happened? Who was involved? When and where? Be detailed.")
            ipc = st.text_area("IPC Sections (if you know them)", height=80, placeholder="e.g., IPC 323, 504, 506 (or leave blank if unsure)")
            
            # LIVE VALIDATION FEEDBACK
            st.markdown("---")
            st.markdown("### ✅ Form Completion Status")
            
            validation_col1, validation_col2 = st.columns(2)
            
            with validation_col1:
                if complainant and age > 0 and phone:
                    st.success(f"✅ Your Details: Complete (Age {age})")
                else:
                    missing = []
                    if not complainant: missing.append("Name")
                    if age == 0: missing.append("Age")
                    if not phone: missing.append("Phone")
                    st.warning(f"⚠️ Missing: {', '.join(missing)}")
            
            with validation_col2:
                if police_station and district and state and incident:
                    st.success(f"✅ Incident Details: Complete ({district}, {state})")
                else:
                    missing = []
                    if not police_station: missing.append("Police Station")
                    if not district: missing.append("District")
                    if not state: missing.append("State")
                    if not incident: missing.append("Incident Description")
                    st.warning(f"⚠️ Missing: {', '.join(missing)}")
            
            st.session_state["doc_gen_data"].update({
                "complainant": complainant, "age": age, "phone": phone, "police_station": police_station,
                "district": district, "state": state, "incident": incident, "ipc": ipc or "To be determined"
            })
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="back_1"):
                st.session_state["doc_gen_step"] = 0
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Generate Document", use_container_width=True, type="primary", key="next_1"):
                st.session_state["doc_gen_step"] = 2
                st.rerun()
    
    # STEP 3: Generate & Preview (ENHANCED WITH PROFESSIONAL FORMATTING)
    elif st.session_state["doc_gen_step"] == 2:
        st.progress(0.75, text="Step 3 of 3: Review & Download")
        st.markdown("### ✅ Step 3: Your Document is Ready!")
        
        with st.spinner("⚙️ Generating your professional document..."):
            doc_type = st.session_state["doc_gen_data"]["doc_type"]
            data = st.session_state["doc_gen_data"]
            
            # Generate document based on type
            if doc_type == "consumer_complaint":
                document = f"""COMPLAINT UNDER CONSUMER PROTECTION ACT, 2019

TO THE DISTRICT CONSUMER COMMISSION

COMPLAINT NO: {datetime.now().strftime('%Y%m%d%H%M%S')}

COMPLAINANT DETAILS:
Name: {data.get('complainant_name', 'N/A')}
Address: {data.get('complainant_address', 'N/A')}
Phone: {data.get('complainant_phone', 'N/A')}
Email: {data.get('complainant_email', 'N/A')}

RESPONDENT DETAILS:
Name/Company: {data.get('respondent_name', 'N/A')}
Address: Registered office

COMPLAINT:
The Complainant hereby lodges this complaint against the Respondent for deficiency in service/product quality.

FACTS OF THE CASE:
{data.get('issue', 'N/A')}

The above act/omission amounts to an unfair trade practice and deficiency in service as per Consumer Protection Act, 2019.

RELIEFS SOUGHT:
1. Refund/Compensation: Rs. {data.get('amount', 50000):,}/-
2. Mental agony and harassment damages: Rs. {data.get('amount', 50000)//5:,}/-
3. Litigation cost: Rs. 5,000/-

Total Claim: Rs. {data.get('amount', 50000) + data.get('amount', 50000)//5 + 5000:,}/-

WHEREFORE, the Complainant prays that this Commission may be pleased to:
1. Direct the Respondent to refund the entire amount paid
2. Award compensation for mental agony and suffering
3. Award litigation cost and interest as per law

Dated: {datetime.now().strftime('%B %d, %Y')}

Signature of Complainant
({data.get('complainant_name', 'Complainant')})
"""
            
            elif doc_type == "legal_notice":
                document = f"""LEGAL NOTICE
[IMPORTANT: This is a formal legal document. Consult a lawyer before sending.]

DATED: {datetime.now().strftime('%B %d, %Y')}

TO
{data.get('recipient', 'Respondent')}
{data.get('recipient_address', 'Address')}

RE: LEGAL NOTICE FOR {data.get('subject', 'COMPLIANCE').upper()}

DEAR SIR/MADAM,

This Office has been instructed by our client {data.get('your_name', 'Client')}, residing at {data.get('your_address', 'Address')}, to serve you with this legal notice.

1. FACTS AND CIRCUMSTANCES:
{data.get('issue', 'Issue details')}

2. LEGAL GROUNDS:
The act/omission mentioned above is in violation of contractual obligations and applicable laws of India.

3. FORMAL DEMAND:
You are hereby demanded to:
✓ Pay Rs. {data.get('amount', 100000):,}/- within {data.get('deadline', 15)} days from the date of receipt
✓ Comply with all contractual obligations
✓ Take necessary corrective action as required

4. CONSEQUENCES OF NON-COMPLIANCE:
If you fail to comply with this notice within {data.get('deadline', 15)} days, your client shall be liable for:
• Principal Amount: Rs. {data.get('amount', 100000):,}/-
• Interest at applicable rates
• Legal costs and damages
• Criminal prosecution if applicable

You are required to acknowledge receipt of this notice within 7 days and confirm full compliance within {data.get('deadline', 15)} days.

Dated: {datetime.now().strftime('%B %d, %Y')}

FOR AND ON BEHALF OF
{data.get('your_name', 'Your Name')}
{data.get('your_address', 'Address')}
"""
            
            elif doc_type == "fir_draft":
                document = f"""FIRST INFORMATION REPORT (FIR) DRAFT

Police Station: {data.get('police_station', 'N/A')}
District: {data.get('district', 'N/A')}
State: {data.get('state', 'N/A')}
Date of Report: {datetime.now().strftime('%B %d, %Y')}

DETAILS OF COMPLAINANT:
Name: {data.get('complainant', 'N/A')}
Age: {data.get('age', 'N/A')} years
Address: N/A
Phone: {data.get('phone', 'N/A')}

INCIDENT DETAILS:
Date of Incident: {datetime.now().strftime('%B %d, %Y')}
Time of Incident: As stated
Place of Incident: As stated
District: {data.get('district', 'N/A')}

DETAILED DESCRIPTION OF INCIDENT:
{data.get('incident', 'N/A')}

APPLICABLE IPC SECTIONS:
{data.get('ipc', 'To be determined by investigating officer')}

WITNESSES:
To be identified and recorded during police inquiry

EVIDENCE AVAILABLE:
As described in the complaint

Hereby submit this report for registration of FIR.

Signature of Complainant
({data.get('complainant', 'N/A')})
"""
            
            else:
                document = "Document generation in progress..."
            
            st.session_state["doc_gen_data"]["generated_doc"] = document
        
        # Display Document with Professional Formatting
        st.success("✅ Document Generated Successfully! Click to preview and customize below.")
        st.markdown("---")
        
        # Tabs for preview and edit
        tab1, tab2, tab3 = st.tabs(["📖 Professional Preview", "✏️ Edit & Customize", "📊 Document Info"])
        
        with tab1:
            st.markdown(f"""
            <div style="background: #ffffff; color: #000000; padding: 40px; border-radius: 10px; 
                         font-family: 'Times New Roman', serif; line-height: 2.0; 
                         max-height: 700px; overflow-y: auto; border: 3px solid #00f3ff; 
                         box-shadow: 0 0 20px rgba(0,243,255,0.3); page-break-after: always;">
                <pre style="white-space: pre-wrap; font-family: 'Times New Roman', serif; font-size: 14px; letter-spacing: 0.5px;">{document}</pre>
            </div>
            """, unsafe_allow_html=True)
            st.caption("💡 Tip: Use Ctrl+P or Cmd+P to print this document directly from your browser")
        
        with tab2:
            st.markdown("**Edit your document if needed (changes will reflect in downloads):**")
            edited = st.text_area("Edit your document here:", value=document, height=450, key="edit_doc")
            
            col_edit1, col_edit2 = st.columns(2)
            with col_edit1:
                if st.button("💾 Save Changes", type="secondary", use_container_width=True):
                    st.session_state["doc_gen_data"]["generated_doc"] = edited
                    st.success("✅ Changes saved! Download buttons will use updated version.")
            
            with col_edit2:
                if st.button("↩️ Reset to Original", use_container_width=True):
                    st.session_state["doc_gen_data"]["generated_doc"] = document
                    st.info("Reset to original version")
        
        with tab3:
            st.markdown("**📋 Document Information:**")
            doc_info_col1, doc_info_col2 = st.columns(2)
            
            with doc_info_col1:
                st.metric("Document Type", st.session_state['doc_gen_data']['doc_type_display'])
                st.metric("Word Count", len(document.split()))
                st.metric("Character Count", len(document))
            
            with doc_info_col2:
                st.metric("Generated Date", datetime.now().strftime("%B %d, %Y"))
                st.metric("Generated Time", datetime.now().strftime("%I:%M %p"))
                st.metric("Document Pages", f"~{max(1, len(document) // 3000)}")
        
        # Download Options with Professional Formatting
        st.markdown("---")
        st.markdown("### 📥 Download Your Document")
        
        col1, col2, col3, col4 = st.columns(4)
        doc_name = st.session_state["doc_gen_data"]["doc_type_display"].replace(" ", "_")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get the current edited document
        final_document = st.session_state["doc_gen_data"].get("generated_doc", document)
        
        with col1:
            st.download_button(
                "📄 TXT Format",
                final_document,
                file_name=f"{doc_name}_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            html_version = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{doc_name}</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 2.0;
            padding: 40px;
            max-width: 900px;
            margin: auto;
            color: #000;
            background: #fff;
        }}
        h1, h2, h3 {{
            text-align: center;
            margin-top: 20px;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        @media print {{
            body {{ padding: 0; }}
            a {{ text-decoration: none; color: #000; }}
        }}
    </style>
</head>
<body>
    <pre>{final_document}</pre>
    <footer style="text-align: center; margin-top: 50px; color: #999; font-size: 12px;">
        <p>Generated by AI Lawyer RAG System | {datetime.now().strftime('%d/%m/%Y at %H:%M')}</p>
    </footer>
</body>
</html>"""
            st.download_button(
                "🌐 HTML (Print-Ready)",
                html_version,
                file_name=f"{doc_name}_{timestamp}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col3:
            # Create Word-compatible format
            word_version = f"""{final_document}


---
Generated by AI Lawyer RAG System
Date: {datetime.now().strftime('%d/%m/%Y at %H:%M')}
"""
            st.download_button(
                "📋 Word (.docx)",
                word_version,
                file_name=f"{doc_name}_{timestamp}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
        
        with col4:
            if st.button("🔄 Create Another", use_container_width=True, key="new_doc"):
                st.session_state["doc_gen_step"] = 0
                st.session_state["doc_gen_data"] = {}
                st.rerun()
        
        # Navigation back
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("⬅️ Back to Edit", use_container_width=True, key="back_2"):
                st.session_state["doc_gen_step"] = 1
                st.session_state["doc_gen_step"] = 1
                st.rerun()

# TAB 2: Case Analyzer - STEP-WISE WIZARD
with feature_tabs[1]:
    st.markdown('<h3 style="color: #00f3ff;">🎯 Case Strength Analyzer - Step by Step</h3>', unsafe_allow_html=True)
    st.info("💡 Let AI predict your chances of winning! Answer our questions step-by-step.")
    
    # Initialize Case Analyzer state
    if "case_analyzer_step" not in st.session_state:
        st.session_state["case_analyzer_step"] = 0
    if "case_analyzer_data" not in st.session_state:
        st.session_state["case_analyzer_data"] = {}
    
    # STEP 1: Choose Case Type
    if st.session_state["case_analyzer_step"] == 0:
        st.progress(0.25, text="Step 1 of 4: Choose Your Case Type")
        st.markdown("### 📋 Step 1: What kind of case is it?")
        
        case_type_options = {
            "🏢 Consumer Complaint": "consumer",
            "⚖️ Civil (Money/Property)": "civil",
            "🚨 Criminal Complaint": "criminal",
            "👨‍👩‍👧 Family (Marriage/Custody)": "family",
            "🏠 Property Dispute": "property",
            "💼 Labour Dispute": "labour",
            "🔴 Cheque Bounce / Financial": "financial"
        }
        
        selected_type = st.radio("Choose Case Type", list(case_type_options.keys()),
                                help="Select the category that best describes your case")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="case_next_0"):
                st.session_state["case_analyzer_data"]["case_type"] = case_type_options[selected_type]
                st.session_state["case_analyzer_data"]["case_type_display"] = selected_type
                st.session_state["case_analyzer_step"] = 1
                st.rerun()
    
    # STEP 2: Gather Case Details
    elif st.session_state["case_analyzer_step"] == 1:
        st.progress(0.50, text="Step 2 of 4: Tell Us About Your Case")
        st.markdown("### 📝 Step 2: Explain your situation in detail")
        
        st.markdown("""
        **To get the best analysis, please explain:**
        1. **Timeline:** When did this happen? (dates/timeframe)
        2. **People involved:** Who are the main parties?
        3. **The issue:** What exactly is the dispute about?
        4. **Your evidence:** What proof do you have? (documents, witnesses, etc.)
        5. **Their position:** What would the other side say?
        """)
        
        case_details = st.text_area(
            "Describe your case in detail *", 
            height=250,
            placeholder="""Example: I bought a washing machine in January 2024 from ABC Electronics. 
It stopped working after 3 months. I have the receipt, warranty card, and photos. 
The company refuses to repair or refund. I want Rs. 25,000 compensation."""
        )
        
        has_documents = st.radio("Do you have supporting documents? *", 
                                ["Yes (bills, receipts, emails, contracts, etc.)",
                                 "Some documents",
                                 "No documents yet"])
        
        st.session_state["case_analyzer_data"].update({
            "case_details": case_details,
            "has_documents": has_documents
        })
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="case_back_1"):
                st.session_state["case_analyzer_step"] = 0
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Analyze Now", use_container_width=True, type="primary", key="case_next_1"):
                if case_details.strip():
                    st.session_state["case_analyzer_step"] = 2
                    st.rerun()
                else:
                    st.error("Please describe your case to proceed")
    
    # STEP 3: AI Analysis Running
    elif st.session_state["case_analyzer_step"] == 2:
        st.progress(0.75, text="Step 3 of 4: Analyzing Your Case...")
        st.markdown("### ⚙️ Step 3: Our AI is analyzing your case...")
        
        with st.spinner("🧠 Powerful AI is analyzing your case details, checking similar cases, and calculating win probability..."):
            case_type = st.session_state["case_analyzer_data"]["case_type"]
            case_details = st.session_state["case_analyzer_data"]["case_details"]
            
            # Simulate or call actual analysis function
            try:
                analysis = analyze_case_strength(case_details, case_type)
                st.session_state["case_analyzer_data"]["analysis"] = analysis
                st.session_state["case_analyzer_step"] = 3
                st.rerun()
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                if st.button("Retry Analysis"):
                    st.rerun()
    
    # STEP 4: Display Results
    elif st.session_state["case_analyzer_step"] == 3:
        st.progress(0.95, text="Step 4 of 4: Your Analysis Report")
        st.markdown("### 📊 Step 4: Your Case Analysis Report")
        
        analysis = st.session_state["case_analyzer_data"].get("analysis", {})
        
        if "error" not in analysis:
            # Win Probability
            st.markdown("---")
            win_prob = analysis.get("win_probability", 50)
            
            # Visual representation
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(0, 243, 255, 0.2) 0%, rgba(57, 255, 20, 0.2) 100%);
                             border: 3px solid #00f3ff; border-radius: 15px; padding: 2rem; text-align: center;">
                    <div style="font-size: 3rem; font-weight: 700; color: #39ff14;">{win_prob}%</div>
                    <div style="font-size: 1.2rem; color: #00f3ff; font-weight: 600;">WINNING CHANCES</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                confidence = analysis.get("confidence_level", "MEDIUM")
                conf_text = "🟢 HIGH - Very reliable prediction" if confidence == "HIGH" else \
                            "🟡 MEDIUM - Reasonable prediction" if confidence == "MEDIUM" else "🔴 LOW - Use with caution"
                
                st.markdown(f"""
                <div style="background: rgba(100, 255, 218, 0.1); border-left: 5px solid #00f3ff; 
                             border-radius: 10px; padding: 1.5rem;">
                    <strong style="color: #00f3ff;">Confidence Level</strong>
                    <br>{conf_text}
                    <br><br>
                    <small style="color: #9fa8da;">This prediction is based on your case details, 
                    available evidence, and analysis of similar cases.</small>
                </div>
                """, unsafe_allow_html=True)
            
            # Strengths & Weaknesses
            st.markdown("---")
            st.markdown("### 💪 Your Case Strengths vs Concerns")
            
            col_s, col_w = st.columns(2)
            
            with col_s:
                st.markdown("#### ✅ What's Working For You")
                strengths = analysis.get("strength_factors", [])
                if strengths:
                    for idx, strength in enumerate(strengths, 1):
                        st.success(f"**{idx}.** {strength}")
                else:
                    st.info("Build a stronger case by gathering more evidence")
            
            with col_w:
                st.markdown("#### ⚠️ Areas to Watch Out For")
                weaknesses = analysis.get("weakness_factors", [])
                if weaknesses:
                    for idx, weakness in enumerate(weaknesses, 1):
                        st.warning(f"**{idx}.** {weakness}")
                else:
                    st.success("No major concerns identified!")
            
            # Strategic Actions
            st.markdown("---")
            st.markdown("### 🎯 What You Should Do Next")
            st.markdown("> These recommendations are in order of priority. Complete them step-by-step.\n")
            
            for i, rec in enumerate(analysis.get("strategic_recommendations", []), 1):
                st.info(f"**Action {i}:** {rec}")
            
            # Cost & Timeline
            st.markdown("---")
            st.markdown("### 📋 Important Information")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("⏱️ Expected Duration", 
                         analysis.get('timeline_estimate', '6-12 months'))
            
            with metric_col2:
                st.metric("💰 Estimated Cost",
                         analysis.get('cost_estimate', '₹5,000-₹50,000'))
            
            with metric_col3:
                settlement = "Consider" if analysis.get('settlement_advice') and 'settle' in analysis.get('settlement_advice', '').lower() else "Not Recommended"
                st.metric("💬 Settlement", settlement)
            
            # Download Report
            st.markdown("---")
            st.markdown("### 📥 Download Your Report")
            
            report_content = f"""CASE STRENGTH ANALYSIS REPORT
Generated: {datetime.now().strftime('%B %d, %Y')}
Case Type: {st.session_state['case_analyzer_data'].get('case_type_display', 'N/A')}

═══════════════════════════════════════════════════════════════

WIN PROBABILITY: {win_prob}%
CONFIDENCE LEVEL: {confidence}

═══════════════════════════════════════════════════════════════

YOUR STRENGTHS:
{chr(10).join(f'✓ {s}' for s in analysis.get('strength_factors', []))}

WEAKNESSES/CONCERNS:
{chr(10).join(f'✗ {w}' for w in analysis.get('weakness_factors', []))}

RECOMMENDED ACTIONS:
{chr(10).join(f'{i}. {r}' for i, r in enumerate(analysis.get('strategic_recommendations', []), 1))}

TIMELINE: {analysis.get('timeline_estimate', 'N/A')}
COST: {analysis.get('cost_estimate', 'N/A')}
SETTLEMENT ADVICE: {analysis.get('settlement_advice', 'N/A')}

═══════════════════════════════════════════════════════════════
"""
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            with col_dl1:
                st.download_button(
                    "📄 Download Report (TXT)",
                    report_content,
                    file_name=f"case_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                    use_container_width=True
                )
            
            with col_dl2:
                if st.button("🔄 Analyze Another Case", use_container_width=True):
                    st.session_state["case_analyzer_step"] = 0
                    st.session_state["case_analyzer_data"] = {}
                    st.rerun()
            
            with col_dl3:
                if st.button("⬅️ Try Different Details", use_container_width=True):
                    st.session_state["case_analyzer_step"] = 1
                    st.rerun()
            
            # Expert Tips
            with st.expander("💡 Tips to Strengthen Your Case"):
                st.markdown("""
                **Immediate Actions:**
                - ✅ Document everything (photos, emails, messages, letters)
                - ✅ Get witness statements written down
                - ✅ Check if there are laws protecting you
                - ✅ Calculate exact damages/losses
                
                **Before Filing:**
                - 📋 Collect all original documents
                - 📋 Make copies (keep originals safe)
                - 📋 Get a lawyer's opinion
                - 📋 Understand court procedures
                
                **During Legal Action:**
                - 🎯 Respond to all notices within time
                - 🎯 Keep your lawyer updated
                - 🎯 Be honest about your case
                - 🎯 Only share information with your lawyer
                """)
        else:
            st.error(f"Error in analysis: {analysis.get('error', 'Unknown error')}")
            if st.button("Retry"):
                st.session_state["case_analyzer_step"] = 2
                st.rerun()

# TAB 3: Legal Notice - STEP-WISE WIZARD
with feature_tabs[2]:
    st.markdown('<h3 style="color: #00f3ff;">⚡ Legal Notice Generator - Step by Step</h3>', unsafe_allow_html=True)
    st.info("💡 Send professional legal notice without a lawyer! Let's do it step-by-step.")
    
    # Initialize state
    if "legal_notice_step" not in st.session_state:
        st.session_state["legal_notice_step"] = 0
    if "legal_notice_data" not in st.session_state:
        st.session_state["legal_notice_data"] = {}
    
    # STEP 1: Choose Notice Type
    if st.session_state["legal_notice_step"] == 0:
        st.progress(0.20, text="Step 1 of 5: Choose Notice Type")
        st.markdown("### 📋 Step 1: What kind of notice do you need?")
        
        notice_options = {
            "💰 Cheque Bounce / Payment Default": "cheque_bounce",
            "💵 Money Recovery / Loan Default": "money_recovery",
            "🏠 Property Dispute / Encroachment": "property",
            "🏢 Service Deficiency / Non-Performance": "service",
            "📋 Breach of Contract": "contract",
            "🏘️ Eviction Notice / Unauthorized Occupancy": "eviction"
        }
        
        selected_notice = st.radio("Select Notice Type", list(notice_options.keys()),
                                  help="Choose the type that matches your situation")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="notice_next_0"):
                st.session_state["legal_notice_data"]["notice_type"] = notice_options[selected_notice]
                st.session_state["legal_notice_data"]["notice_type_display"] = selected_notice
                st.session_state["legal_notice_step"] = 1
                st.rerun()
    
    # STEP 2: Your Details
    elif st.session_state["legal_notice_step"] == 1:
        st.progress(0.40, text="Step 2 of 5: Your Details")
        st.markdown("### 👤 Step 2: Tell us about yourself")
        
        col1, col2 = st.columns(2)
        with col1:
            your_full_name = st.text_input("Your Full Name *", placeholder="Your legal name")
            your_address = st.text_area("Your Address *", height=80, placeholder="Complete residential address")
        with col2:
            phone = st.text_input("Your Phone Number *", placeholder="10-digit number")
            email = st.text_input("Your Email (optional)", placeholder="your@email.com")
        
        st.session_state["legal_notice_data"].update({
            "your_name": your_full_name,
            "your_address": your_address,
            "phone": phone,
            "email": email
        })
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="notice_back_1"):
                st.session_state["legal_notice_step"] = 0
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="notice_next_1"):
                if your_full_name and your_address:
                    st.session_state["legal_notice_step"] = 2
                    st.rerun()
                else:
                    st.error("Please fill all marked fields")
    
    # STEP 3: Recipient Details
    elif st.session_state["legal_notice_step"] == 2:
        st.progress(0.60, text="Step 3 of 5: Recipient Details")
        st.markdown("### 📬 Step 3: Who do you want to send notice to?")
        
        col1, col2 = st.columns(2)
        with col1:
            recipient_name = st.text_input("Recipient Full Name *", placeholder="Person/Company name")
            recipient_address = st.text_area("Recipient Address *", height=80, placeholder="Complete address of recipient")
        with col2:
            recipient_phone = st.text_input("Recipient Phone (optional)", placeholder="Their contact number")
            recipient_email = st.text_input("Recipient Email (optional)", placeholder="Their email")
        
        st.session_state["legal_notice_data"].update({
            "recipient_name": recipient_name,
            "recipient_address": recipient_address,
            "recipient_phone": recipient_phone,
            "recipient_email": recipient_email
        })
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="notice_back_2"):
                st.session_state["legal_notice_step"] = 1
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="notice_next_2"):
                if recipient_name and recipient_address:
                    st.session_state["legal_notice_step"] = 3
                    st.rerun()
                else:
                    st.error("Please fill recipient details")
    
    # STEP 4: Issue Details
    elif st.session_state["legal_notice_step"] == 3:
        st.progress(0.80, text="Step 4 of 5: Describe the Issue")
        st.markdown("### 📝 Step 4: What's the issue? Describe clearly")
        
        st.markdown("""**Help:** Be specific about:
- What exactly happened? (When? How?)
- What is the other person supposed to do?
- What amount of money is involved? (if any)
- What evidence do you have? (emails, receipts, witnesses)
""")
        
        issue_description = st.text_area("Describe the issue in detail *", height=200,
            placeholder="""Example: Mr. Sharma took a loan of ₹2,00,000 from me on 1st Jan 2024 
with promise to repay by 31st March 2024. He signed an agreement. 
I have the signed agreement and bank transfer receipt as proof. 
He has not repaid despite multiple reminders and is avoiding my calls.""")
        
        amount = st.number_input("Amount Involved (₹) if any", min_value=0, value=0, step=100)
        
        response_deadline = st.number_input("How many days should they have to comply? *", 
                                           min_value=5, max_value=60, value=15, step=1)
        
        st.session_state["legal_notice_data"].update({
            "issue": issue_description,
            "amount": amount,
            "deadline": response_deadline
        })
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="notice_back_3"):
                st.session_state["legal_notice_step"] = 2
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Generate", use_container_width=True, type="primary", key="notice_next_3"):
                if issue_description:
                    st.session_state["legal_notice_step"] = 4
                    st.rerun()
                else:
                    st.error("Please describe the issue")
    
    # STEP 5: Preview & Download
    elif st.session_state["legal_notice_step"] == 4:
        st.progress(1.0, text="Step 5 of 5: Review & Send")
        st.markdown("### ✅ Step 5: Your Notice is Ready!")
        
        notice_type = st.session_state["legal_notice_data"]["notice_type"]
        data = st.session_state["legal_notice_data"]
        
        # Generate notice
        with st.spinner("⚙️ Generating professional legal notice..."):
            notice_content = f"""LEGAL NOTICE
[सूचना / NOTICE]

Date: {datetime.now().strftime('%B %d, %Y')}

TO
{data.get('recipient_name', 'Recipient')}
{data.get('recipient_address', 'Address')}

RE: LEGAL NOTICE FOR {data.get('notice_type_display', '').split('/')[-1].upper().strip()}

DEAR {data.get('recipient_name', 'Sir/Madam')},

This legal notice is being served on you by {data.get('your_name', 'Client')}, to bring the following matter to your attention.

═════════════════════════════════════════════════════════════════

FACTS AND CIRCUMSTANCES:

{data.get('issue', 'Issue details')}

═════════════════════════════════════════════════════════════════

LEGAL GROUNDS:

Your act/omission is in violation of:
• Indian Contract Act, 1872
• Applicable laws of India
• Terms mutually agreed between us

═════════════════════════════════════════════════════════════════

FORMAL DEMAND:

You are hereby demanded to:

1. **IMMEDIATE ACTION:** Comply with the above within {data.get('deadline', 15)} days from the date of receipt of this notice

2. **FINANCIAL LIABILITY:** 
   - Principal Amount: ₹{data.get('amount', 0):,.0f}/-
   - Interest @ 18% per annum: ₹{(data.get('amount', 0) * 0.18 * data.get('deadline', 15) / 365):,.0f}/-
   - Litigation Cost: ₹5,000/-
   - Total Liability: ₹{data.get('amount', 0) + (data.get('amount', 0) * 0.18 * data.get('deadline', 15) / 365) + 5000:,.0f}/-

═════════════════════════════════════════════════════════════════

CONSEQUENCES OF NON-COMPLIANCE:

If you fail to comply within {data.get('deadline', 15)} days, I shall pursue the following legal remedies:

✗ File Civil Suit for recovery in District Court (Jurisdiction: {data.get('your_address', '').split(',')[-1].strip()})
✗ Criminal prosecution under applicable laws
✗ Recovery through courts with enhanced damages
✗ You shall be liable for all court costs, witness expenses, and legal fees
✗ Your property may be attached/seized
✗ Your reputation and credit rating will be affected

═════════════════════════════════════════════════════════════════

IMPORTANT NOTES:

⚠️ This notice is a final attempt to settle the matter amicably.

⚠️ Your acknowledgment receipt is important evidence. Please keep it safe.

⚠️ If you do not respond within {data.get('deadline', 15)} days, legal proceedings will start without further notice.

⚠️ You are expected to comply either by:
   • Direct payment to: {data.get('your_name', 'Claimant')}
   • Phone: {data.get('phone', 'Contact number')}
   • Email: {data.get('email', 'Email')}

═════════════════════════════════════════════════════════════════

This notice is issued under Indian law and should be taken seriously.

Date: {datetime.now().strftime('%B %d, %Y')}

Signature
{data.get('your_name', 'Claimant Name')}
Address: {data.get('your_address', 'Address')}
Phone: {data.get('phone', 'Phone')}
Email: {data.get('email', 'Email')}

═════════════════════════════════════════════════════════════════

AFFIDAVIT (To be sworn before Notary Public):

I, {data.get('your_name', 'Name')}, son/daughter of ________, aged _____ years, 
hereby declare that the facts stated above are true and correct to my knowledge 
and belief. I declare that I am not actuated by any improper motive.

Deponent

[To be notarized]
"""
            st.session_state["legal_notice_data"]["generated_notice"] = notice_content
        
        # Display tabs
        tab1, tab2 = st.tabs(["📄 Preview Notice", "📧 How to Send"])
        
        with tab1:
            st.markdown(f"""
            <div style="background: #ffffff; color: #000; padding: 30px; border-radius: 10px; 
                         font-family: 'Times New Roman', serif; line-height: 1.8; 
                         max-height: 700px; overflow-y: auto; border: 3px solid #00f3ff; box-shadow: 0 0 20px rgba(0,243,255,0.3);">
                <pre style="white-space: pre-wrap; font-family: 'Times New Roman', serif; color: #000;">{notice_content}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ Notice is formal, legally sound, and ready to send!")
        
        with tab2:
            st.markdown("### 📧 How to Send Your Legal Notice")
            
            delivery = st.selectbox("Select Delivery Method:", 
                ["Registered Post AD (Recommended)", "Speed Post with AD", "Email + Registered Post", "Courier (Tracked)"])
            
            st.info(f"""
### Steps to Send via {delivery}:

1. **📄 Print the Notice**
   - Print 2 copies (both for you and recipient)
   - Use good quality paper
   - Print clearly and completely
   - Don't use old/faded printer

2. **✍️ Sign Attest the Notice**
   - Sign both copies in blue ink
   - Date each copy
   - Get it notarized (Optional but recommended) - Cost: ₹100-500

3. **📮 Send by Registered Post**
   - Go to nearest post office
   - Pick {delivery} service
   - Pay ₹{100 if 'Registered' in delivery else 150 if 'Speed' in delivery else 200}/-
   - Get Receipt + ACS acknowledgment (KEEP SAFE!)
   - Keep photocopy of receipt

4. **📋 Keep Records**
   - Original receipt with ACS
   - Copy of notice sent
   - Proof of delivery
   - Your correspondence notes

**⏰ Timeline:**
- Notice reaches in 5-7 working days
- Start counting {data.get('deadline', 15)} days from acknowledgment date
- Deadline is crucial for legal action

**💡 Cost:** ₹100-300 (Post office fee)
**⚖️ Evidence:** Receipt is legal proof of notice sent
            """)
            
            if st.button("📋 Print Checklist for Sending"):
                st.markdown(f"""
### ✅ NOTICE SENDING CHECKLIST

**Before Sending:**
- [ ] Read the notice completely (check all details)
- [ ] Verify recipient name and address
- [ ] Verify amount and deadline
- [ ] Make 2 copies
- [ ] Check printer works well
- [ ] Gather documents

**While Sending:**
- [ ] Print on quality paper (A4 white)
- [ ] Sign both copies clearly
- [ ] Write date: {datetime.now().strftime('%B %d, %Y')}
- [ ] Go to nearest post office
- [ ] Ask for Registered Post AD
- [ ] Fill Address label clearly
- [ ] Pay postal fee
- [ ] Get receipt
- [ ] Get ACS (Advice of Delivery) slip
- [ ] Take photos of receipt
- [ ] Keep receipt in safe place

**After Sending:**
- [ ] Note the postal registration number
- [ ] Wait for acknowledgment (5-7 days)
- [ ] Count {data.get('deadline', 15)} days from acknowledgment
- [ ] Take screenshot of acknowledgment
- [ ] Store all originals safely
- [ ] If no response on deadline → Start legal action

**If They Don't Reply:**
- Go to Court Filing tab
- File civil suit
- Attach notice + receipt (proof of attempt)
- This strengthens your case!

**Cost Summary:**
- PostalFee: ₹50-150
- Notarization (optional): ₹100-500
- Total: ₹150-650 (vs ₹5,000-20,000 with lawyer!)
                """)
        
        # Download options
        st.markdown("---")
        st.markdown("### 📥 Download Your Legal Notice")
        
        col1, col2, col3 = st.columns(3)
        timestamp = datetime.now().strftime('%Y%m%d')
        
        with col1:
            st.download_button(
                "📄 Download TXT",
                notice_content,
                file_name=f"legal_notice_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            html_version = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>body {{font-family:'Times New Roman',serif;line-height:1.8;padding:2rem;max-width:900px;margin:auto;color:#000;}}</style>
</head><body><pre style="white-space:pre-wrap;">{notice_content}</pre></body></html>"""
            st.download_button(
                "🌐 Download HTML",
                html_version,
                file_name=f"legal_notice_{timestamp}.html",
                mime="text/html",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 Create Another", use_container_width=True):
                st.session_state["legal_notice_step"] = 0
                st.session_state["legal_notice_data"] = {}
                st.rerun()
        
        # Back button
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="notice_back_4"):
                st.session_state["legal_notice_step"] = 3
                st.rerun()
        
        # Tips expander
        with st.expander("💡 Important Tips"):
            st.markdown("""
### Legal Tips for Notice:

✅ **DO:**
- Send via registered post (creates legal proof)
- Be clear and specific about demands
- Include correct legal references
- Spell recipient name correctly
- Keep copy for yourself
- Take photos of receipt
- Follow up if no response

❌ **DON'T:**
- Use threatening or abusive language
- Make false claims
- Set unrealistic deadlines (less than 7-15 days)
- Forget to sign the notice
- Send by regular post (no proof)
- Change details after sending
- Lose the postal receipt

### After Notice Sent:
1. Wait for acknowledgment (5-7 days)
2. If response → Negotiate settlement
3. If no response after deadline → File court case
4. The notice receipt becomes strong evidence
5. Courts look favorably on parties who tried settlement first

### Expected Outcomes:
- **Best Case:** 60% settle after notice
- **Medium Case:** Negotiate terms
- **Worst Case:** Take to court (notice helps your case!)
            """)



# TAB 4: Court Filing - ENHANCED STEP-WISE WIZARD
with feature_tabs[3]:
    st.markdown('<h3 style="color: #00f3ff;">🗂️ Court Filing Assistant - Step by Step</h3>', unsafe_allow_html=True)
    st.info("💡 Complete guided process to file your case WITHOUT a lawyer! Follow 5 simple steps.")
    
    # Initialize state
    if "court_filing_step" not in st.session_state:
        st.session_state["court_filing_step"] = 0
    if "court_filing_data" not in st.session_state:
        st.session_state["court_filing_data"] = {}
    
    # STEP 0: Case Selection
    if st.session_state["court_filing_step"] == 0:
        st.progress(0.20, text="Step 1 of 5: Select Your Case Type")
        st.markdown("### 📋 Step 1: What type of case do you want to file?")
        
        case_types = {
            "🛍️ Consumer Complaint": "consumer",
            "💰 Cheque Bounce Notice": "cheque",
            "🏠 Property Dispute": "property",
            "💼 Employment / Wrongful Termination": "employment",
            "📝 Contract Breach / Non-Performance": "contract",
            "👨‍⚖️ Civil Suit (Other)": "civil",
            "🚨 Criminal Complaint": "criminal",
            "👪 Family Court (Divorce/Custody)": "family"
        }
        
        selected_case = st.radio("Select Case Type", list(case_types.keys()),
                                help="Choose the category that best matches your situation")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="filing_next_0"):
                st.session_state["court_filing_data"]["case_type"] = case_types[selected_case]
                st.session_state["court_filing_data"]["case_type_display"] = selected_case
                st.session_state["court_filing_step"] = 1
                st.rerun()
    
    # STEP 1: Case Details
    elif st.session_state["court_filing_step"] == 1:
        st.progress(0.40, text="Step 2 of 5: Case Value & Location")
        st.markdown("### 💵 Step 2: What is your case value and location?")
        
        col1, col2 = st.columns(2)
        with col1:
            case_value = st.number_input(
                "Total Claim Amount (₹) *",
                min_value=1000,
                value=100000,
                step=10000,
                help="Total amount you're claiming or dispute value")
        
        with col2:
            location = st.text_input(
                "City/District *",
                value=st.session_state["court_filing_data"].get("location", ""),
                placeholder="E.g., Mumbai, Delhi, Bangalore",
                help="Where do you want to file?")
        
        opponent_details = st.text_area(
            "Opponent Details (if known)",
            value=st.session_state["court_filing_data"].get("opponent_details", ""),
            placeholder="Name, address, phone number of the person you're suing",
            height=100
        )
        
        st.session_state["court_filing_data"].update({
            "case_value": case_value,
            "location": location,
            "opponent_details": opponent_details
        })
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="filing_back_1"):
                st.session_state["court_filing_step"] = 0
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="filing_next_1"):
                if location:
                    st.session_state["court_filing_step"] = 2
                    st.rerun()
                else:
                    st.error("❌ Please enter your location")
    
    # STEP 2: Evidence Collection
    elif st.session_state["court_filing_step"] == 2:
        st.progress(0.60, text="Step 3 of 5: Your Evidence & Documents")
        st.markdown("### 📄 Step 3: What evidence do you have?")
        
        st.markdown("**Select which evidence you have:**")
        
        col1, col2 = st.columns(2)
        with col1:
            has_written_proof = st.checkbox("✅ Written proof (invoices, agreements, etc.)", 
                                           value=st.session_state["court_filing_data"].get("has_written_proof", False))
            has_photos = st.checkbox("✅ Photos/Screenshots",
                                    value=st.session_state["court_filing_data"].get("has_photos", False))
            has_messages = st.checkbox("✅ WhatsApp/Email messages",
                                      value=st.session_state["court_filing_data"].get("has_messages", False))
        
        with col2:
            has_witnesses = st.checkbox("✅ Witness statements",
                                       value=st.session_state["court_filing_data"].get("has_witnesses", False))
            has_bank_proof = st.checkbox("✅ Bank/payment proof",
                                        value=st.session_state["court_filing_data"].get("has_bank_proof", False))
            has_official_docs = st.checkbox("✅ Official documents from authorities",
                                           value=st.session_state["court_filing_data"].get("has_official_docs", False))
        
        st.session_state["court_filing_data"].update({
            "has_written_proof": has_written_proof,
            "has_photos": has_photos,
            "has_messages": has_messages,
            "has_witnesses": has_witnesses,
            "has_bank_proof": has_bank_proof,
            "has_official_docs": has_official_docs
        })
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="filing_back_2"):
                st.session_state["court_filing_step"] = 1
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="filing_next_2"):
                st.session_state["court_filing_step"] = 3
                st.rerun()
    
    # STEP 3: Requirement Checklist
    elif st.session_state["court_filing_step"] == 3:
        st.progress(0.80, text="Step 4 of 5: Required Documents")
        st.markdown("### 📋 Step 4: Which documents do you need?")
        
        data = st.session_state["court_filing_data"]
        case_type = data.get("case_type", "consumer")
        case_value = data.get("case_value", 100000)
        
        # Determine required documents
        st.markdown("**Essential Documents for Your Filing:**")
        
        documents_needed = {
            "Original Complaint/Petition": True,
            "Copies of Complaint (2-5 sets)": True,
            "Affidavit in support of claim": True,
            "Proof of identity (Voter ID/Aadhaar)": True,
            "Proof of address (utility bill/lease)": True,
            "Supporting evidence (originals)": data.get("has_written_proof", False),
            "Photographs/Screenshots (printed)": data.get("has_photos", False),
            "Witness statements": data.get("has_witnesses", False),
            "Bank statements/proof of payment": data.get("has_bank_proof", False),
            f"Stamp paper (value: {min(500, case_value//1000)*100})": True,
            "Court fee receipt": True,
        }
        
        doc_col1, doc_col2 = st.columns(2)
        with doc_col1:
            st.markdown("**🔴 MANDATORY:**")
            for doc, needed in documents_needed.items():
                if needed and "Photographs" not in doc and "Witness" not in doc and "Bank" not in doc:
                    st.checkbox(f"✅ {doc}", value=True, disabled=True)
        
        with doc_col2:
            st.markdown("**🟡 IF APPLICABLE:**")
            for doc, needed in documents_needed.items():
                if "Photographs" in doc or "Witness" in doc or "Bank" in doc:
                    st.checkbox(f"✅ {doc}", value=needed, disabled=True)
        
        # Court Fee Calculator
        st.markdown("---")
        st.markdown("### 💰 Court Filing Fees")
        
        # Simple fee calculation based on case value
        filing_fee = max(500, min(5000, case_value // 10000 * 500))
        stamp_duty = case_value // 100
        miscellaneous = 200
        total_fee = filing_fee + stamp_duty + miscellaneous
        
        col_fee1, col_fee2, col_fee3, col_fee4 = st.columns(4)
        with col_fee1:
            st.metric("📄 Filing Fee", f"₹{filing_fee:,}")
        with col_fee2:
            st.metric("🏷️ Stamp Duty", f"₹{stamp_duty:,}")
        with col_fee3:
            st.metric("📎 Misc. Charges", f"₹{miscellaneous:,}")
        with col_fee4:
            st.metric("💵 TOTAL", f"₹{total_fee:,}", delta="Pay at counter")
        
        st.session_state["court_filing_data"]["total_fee"] = total_fee
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="filing_back_3"):
                st.session_state["court_filing_step"] = 2
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Next - Generate Guide", use_container_width=True, type="primary", key="filing_next_3"):
                st.session_state["court_filing_step"] = 4
                st.rerun()
    
    # STEP 4: Results & Filing Guide
    elif st.session_state["court_filing_step"] == 4:
        st.progress(1.0, text="Step 5 of 5: Your Filing Guide Ready!")
        st.markdown("### ✅ Step 5: Your Complete Filing Guide")
        
        data = st.session_state["court_filing_data"]
        
        # Summary Card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0, 243, 255, 0.15) 0%, rgba(176, 38, 255, 0.15) 100%); 
                     border: 3px solid #00f3ff; border-radius: 15px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <h3 style="color: #39ff14; margin: 0;">✅ YOU'RE READY TO FILE!</h3>
            <p style="color: #9fa8da; margin: 0.5rem 0;"><strong>Case Type:</strong> {data.get('case_type_display', 'N/A')}</p>
            <p style="color: #9fa8da; margin: 0.5rem 0;"><strong>Claim Amount:</strong> ₹{data.get('case_value', 0):,}</p>
            <p style="color: #9fa8da; margin: 0.5rem 0;"><strong>Filing Location:</strong> {data.get('location', 'N/A')}</p>
            <p style="color: #9fa8da; margin: 0;"><strong>Total Fee to Pay:</strong> ₹{data.get('total_fee', 0):,}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Timeline
        st.markdown("### ⏱️ Timeline")
        col_timeline1, col_timeline2, col_timeline3, col_timeline4 = st.columns(4)
        with col_timeline1:
            st.markdown("""<div style="background: rgba(57, 255, 20, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem;">📝</div>
                <div style="font-weight: 700; color: #39ff14;">Filing</div>
                <div style="color: #9fa8da;">1-2 days</div>
            </div>""", unsafe_allow_html=True)
        
        with col_timeline2:
            st.markdown("""<div style="background: rgba(255, 215, 0, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem;">⚖️</div>
                <div style="font-weight: 700; color: #FFD700;">Get Case No.</div>
                <div style="color: #9fa8da;">1-7 days</div>
            </div>""", unsafe_allow_html=True)
        
        with col_timeline3:
            st.markdown("""<div style="background: rgba(0, 243, 255, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem;">📮</div>
                <div style="font-weight: 700; color: #00f3ff;">Notice Served</div>
                <div style="color: #9fa8da;">15-30 days</div>
            </div>""", unsafe_allow_html=True)
        
        with col_timeline4:
            st.markdown("""<div style="background: rgba(176, 38, 255, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 2rem;">🎯</div>
                <div style="font-weight: 700; color: #B026FF;">Hearing</div>
                <div style="color: #9fa8da;">30-60 days</div>
            </div>""", unsafe_allow_html=True)
        
        # Step by Step Instructions
        st.markdown("---")
        st.markdown("### 📝 Filing Procedure (5 Steps)")
        
        with st.expander("✅ Step 1: Prepare Your Documents", expanded=True):
            st.markdown("""
            1. **Print complaint in 4-5 sets** - One original + 3-5 copies
            2. **Get affidavit made** - From nearest advocate (₹500-1000)
            3. **Attach all proof** - Photos, messages, documents in order
            4. **Get stamp paper** - Value as per court requirements
            5. **Check all copies** - Ensure all are signed and dated
            """)
        
        with st.expander("✅ Step 2: Visit the Court"):
            st.markdown(f"""
            **Which Court:** District/Consumer Forum in {data.get('location', 'your location')}
            
            **Filing Counter Hours:** 10 AM - 1 PM, 2 PM - 4 PM (Mon-Fri)
            
            **What to carry:**
            - Original complaint + copies
            - Court fee (₹{data.get('total_fee', 0):,})
            - All supporting documents
            - Identification proof
            - Address proof
            """)
        
        with st.expander("✅ Step 3: Submit Documents at Counter"):
            st.markdown("""
            **Say this at court:** 
            
            "I want to file a case. Here is my complaint, supporting documents, and court fee of ₹[amount]. Please tell me what to do next."
            
            Court staff will:
            - Check if all documents are there
            - Give you case number
            - Tell you first hearing date
            - Keep one set of documents
            """)
        
        with st.expander("✅ Step 4: Serve Notice to Opponent"):
            st.markdown("""
            **Within 2-3 days after filing:**
            - Court sends notice to opponent
            - By post or personally
            - Opponent gets 30 days to respond
            
            You will be informed about hearing date.
            """)
        
        with st.expander("✅ Step 5: Appear on Hearing Date"):
            st.markdown("""
            **On hearing date:**
            - Arrive 30 minutes early
            - Carry all original documents
            - Wear formal clothes
            - Stand when judge enters
            - Speak respectfully
            - Answer truthfully
            """)
        
        # Essential Tips
        st.markdown("---")
        with st.expander("💡 Pro Tips to Succeed"):
            col_pro1, col_pro2 = st.columns(2)
            
            with col_pro1:
                st.success("""
                **✅ DO THIS:**
                - Be truthful in every statement
                - Keep all documents organized
                - Reach court early
                - Listen carefully to judge
                - Speak clearly and slowly
                - Be polite to everyone
                """)
            
            with col_pro2:
                st.error("""
                **❌ DON'T DO THIS:**
                - Lie or hide information
                - Lose your temper in court
                - Talk irrelevantly
                - Interrupt the judge
                - Bring unnecessary documents
                - Argue with the court staff
                """)
        
        # Download Guide
        st.markdown("---")
        st.markdown("### 📥 Download Your Filing Guide")
        
        guide_text = f"""COURT FILING COMPLETE GUIDE
Generated: {datetime.now().strftime('%B %d, %Y')}

YOUR DETAILS:
- Case Type: {data.get('case_type_display', 'N/A')}
- Claim Amount: ₹{data.get('case_value', 0):,}
- Location: {data.get('location', 'N/A')}
- Total Court Fee: ₹{data.get('total_fee', 0):,}

DOCUMENTS YOU NEED:
- Complaint (original + 4 copies)
- Affidavit in support
- Identification proof
- Address proof
- All supporting evidence
- Stamp paper

TIMELINE:
1. Filing: 1-2 days
2. Get Case Number: 1-7 days
3. Notice to Opponent: 15-30 days
4. First Hearing: 30-60 days

COURT FEES BREAKDOWN:
- Filing Fee: ₹{max(500, min(5000, data.get('case_value', 0) // 10000 * 500)):,}
- Stamp Duty: ₹{data.get('case_value', 0) // 100:,}
- Miscellaneous: ₹200
- TOTAL: ₹{data.get('total_fee', 0):,}

Important: Take this guide when visiting the court!
Keep track of all dates and case number.
"""
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            st.download_button(
                "📥 Download Guide",
                guide_text,
                file_name=f"court_filing_guide_{datetime.now().strftime('%Y%m%d')}.txt",
                use_container_width=True
            )
        
        with col_dl2:
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state["court_filing_step"] = 0
                st.session_state["court_filing_data"] = {}
                st.rerun()
        
        with col_dl3:
            if st.button("⬅️ Modify Answers", use_container_width=True):
                st.session_state["court_filing_step"] = 0
                st.rerun()


# TAB 5: AI Lawyer Chatbot - Guided Intake
with feature_tabs[4]:
    st.markdown('<h3 style="color: #00f3ff;">🤖 AI Lawyer Chatbot - Guided Intake</h3>', unsafe_allow_html=True)
    st.info("💡 New client? Start with a quick intake, then chat with a lawyer-style AI.")

    if "chat_intake_step" not in st.session_state:
        st.session_state["chat_intake_step"] = 0
    if "chat_intake_data" not in st.session_state:
        st.session_state["chat_intake_data"] = {}
    if "show_quick_replies" not in st.session_state:
        st.session_state["show_quick_replies"] = True

    if not st.session_state.get("chatbot_mode"):
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, rgba(0, 243, 255, 0.1) 0%, rgba(176, 38, 255, 0.1) 100%);
                     border-radius: 15px; border: 2px solid rgba(0, 243, 255, 0.3);">
            <div style="font-size: 4.2rem; margin-bottom: 1rem;">⚖️🤖</div>
            <p style="font-size: 1.4rem; color: #00f3ff; font-weight: 700;">AI LEGAL CONSULTANT</p>
            <p style="font-size: 1.05rem; color: #9fa8da;">Tell us your issue first. We will prepare your case summary.</p>
        </div>
        """, unsafe_allow_html=True)

        # Step 1: Category + language + urgency
        if st.session_state["chat_intake_step"] == 0:
            st.progress(0.33, text="Step 1 of 3: Case Category")
            st.markdown("### 📋 Step 1: What kind of issue is this?")

            category = st.selectbox(
                "Issue Category",
                ["Criminal", "Family", "Property", "Consumer", "Employment", "Cyber Fraud", "Money/Loan", "Accident", "Business", "Other"],
                key="chat_intake_category"
            )
            language = st.selectbox("Language", ["Hinglish", "English", "Hindi"], key="chat_intake_language")
            urgency = st.selectbox("Urgency", ["Low", "Medium", "High"], key="chat_intake_urgency")

            col_i1, col_i2 = st.columns([1, 4])
            with col_i1:
                if st.button("➡️ Next", use_container_width=True, type="primary", key="chat_intake_next_0"):
                    st.session_state["chat_intake_data"].update({
                        "category": category,
                        "language": language,
                        "urgency": urgency
                    })
                    st.session_state["chat_intake_step"] = 1
                    st.rerun()
            with col_i2:
                if st.button("Skip Intake →", use_container_width=True, key="chat_intake_skip"):
                    st.session_state["chatbot_mode"] = True
                    st.session_state["chatbot_history"] = []
                    st.rerun()

        # Step 2: Details
        elif st.session_state["chat_intake_step"] == 1:
            st.progress(0.66, text="Step 2 of 3: Case Details")
            st.markdown("### 📝 Step 2: Explain your issue in simple words")
            problem = st.text_area("What happened?", height=160, key="chat_intake_problem")
            location = st.text_input("City/State", key="chat_intake_location")
            desired = st.text_input("What do you want to achieve?", key="chat_intake_desired")

            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                if st.button("⬅️ Back", use_container_width=True, key="chat_intake_back_1"):
                    st.session_state["chat_intake_step"] = 0
                    st.rerun()
            with col_i2:
                st.empty()
            with col_i3:
                if st.button("➡️ Next", use_container_width=True, type="primary", key="chat_intake_next_1"):
                    st.session_state["chat_intake_data"].update({
                        "problem": problem,
                        "location": location,
                        "desired": desired
                    })
                    st.session_state["chat_intake_step"] = 2
                    st.rerun()

        # Step 3: Summary + start chat
        else:
            st.progress(1.0, text="Step 3 of 3: Start Consultation")
            st.markdown("### ✅ Step 3: Your Case Summary")
            data = st.session_state.get("chat_intake_data", {})

            summary_text = (
                f"Category: {data.get('category', 'N/A')}\n"
                f"Urgency: {data.get('urgency', 'N/A')}\n"
                f"Location: {data.get('location', 'N/A')}\n"
                f"Desired Outcome: {data.get('desired', 'N/A')}\n"
                f"Problem Summary: {data.get('problem', 'N/A')}"
            )

            st.markdown(f"""
            <div style="background: rgba(0, 243, 255, 0.1); border-radius: 10px; padding: 1rem; border: 1px solid rgba(0, 243, 255, 0.3);">
                <pre style="white-space: pre-wrap; margin: 0; color: #e8eaf6;">{summary_text}</pre>
            </div>
            """, unsafe_allow_html=True)

            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1:
                if st.button("⬅️ Edit", use_container_width=True, key="chat_intake_edit"):
                    st.session_state["chat_intake_step"] = 1
                    st.rerun()
            with col_i2:
                st.empty()
            with col_i3:
                if st.button("🚀 Start Chat", use_container_width=True, type="primary", key="chat_intake_start"):
                    language = data.get("language", "English")
                    with st.spinner("Starting your consultation..."):
                        response_data = chatbot_conversation(
                            summary_text,
                            [],
                            None,
                            language=language
                        )
                        ai_response = response_data.get("response", "Please ask your question.")
                        st.session_state["chatbot_history"] = [
                            {"role": "user", "content": summary_text},
                            {"role": "assistant", "content": ai_response}
                        ]
                        st.session_state["last_followups"] = response_data.get("followup_questions", [])
                        st.session_state["last_urgency"] = response_data.get("urgency", data.get("urgency", "MEDIUM")).upper()
                        st.session_state["chatbot_language"] = language
                        st.session_state["chatbot_mode"] = True
                    st.rerun()

    else:
        # ADVANCED CHATBOT INTERFACE
        st.markdown("---")

        # Language Selector & Settings
        col_lang1, col_lang2, col_lang3 = st.columns([2, 2, 1])
        with col_lang1:
            language = st.selectbox("🌐 Language", ["English", "Hindi", "Hinglish"],
                                   index=["English", "Hindi", "Hinglish"].index(st.session_state.get("chatbot_language", "English")))
            st.session_state["chatbot_language"] = language
        with col_lang2:
            urgency_display = {"HIGH": "🔴 Urgent", "MEDIUM": "🟡 Normal", "LOW": "🟢 General"}
            st.info(f"**Status:** {urgency_display.get(st.session_state.get('last_urgency', 'MEDIUM'), '🟡 Normal')}")
        with col_lang3:
            st.metric("💬 Messages", len(st.session_state.get("chatbot_history", [])))

        # Conversation Display with Enhanced UI
        st.markdown("""
        <div style="background: rgba(26, 31, 58, 0.5); border-radius: 15px; padding: 1rem; margin: 1rem 0;
                     max-height: 500px; overflow-y: auto; border: 1px solid rgba(0, 243, 255, 0.2);">
        """, unsafe_allow_html=True)

        for idx, msg in enumerate(st.session_state.get("chatbot_history", [])):
            if msg["role"] == "user":
                st.chat_message("user", avatar="👤").write(msg["content"])
            else:
                st.chat_message("assistant", avatar="⚖️").write(msg["content"])

                # Show follow-up suggestions for last AI message
                if idx == len(st.session_state["chatbot_history"]) - 1 and st.session_state.get("last_followups"):
                    st.markdown("**💡 You might also ask:**")
                    for fq in st.session_state.get("last_followups", [])[:3]:
                        if st.button(f"❓ {fq[:60]}...", key=f"followup_{idx}_{fq[:20]}", use_container_width=True):
                            st.session_state["chatbot_history"].append({"role": "user", "content": fq})
                            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # Quick Reply Buttons
        if st.session_state.get("show_quick_replies") and st.session_state.get("chatbot_history"):
            st.markdown("#### ⚡ Quick Questions:")
            qr_cols = st.columns(5)
            quick_questions = [
                "📄 Documents needed?",
                "⏱️ Time required?",
                "💰 Costs?",
                "📋 Next steps?",
                "⚖️ Can I do it myself?"
            ]
            for idx, (col, q) in enumerate(zip(qr_cols, quick_questions)):
                with col:
                    if st.button(q, key=f"qr_{idx}", use_container_width=True):
                        full_q = q.replace("📄 ", "").replace("⏱️ ", "").replace("💰 ", "").replace("📋 ", "").replace("⚖️ ", "")
                        st.session_state["chatbot_history"].append({"role": "user", "content": full_q})
                        st.rerun()

        # Chat Input with Voice Support Hint
        st.markdown("""
        <div style="padding: 0.5rem; background: rgba(0, 243, 255, 0.1); border-radius: 10px; margin: 1rem 0; border: 1px solid rgba(0, 243, 255, 0.3);">
            <p style="margin: 0; color: #9fa8da; font-size: 0.9rem;">
                💡 <strong>Tip:</strong> Ask detailed questions for better answers. You can use Hindi/Hinglish!
            </p>
        </div>
        """, unsafe_allow_html=True)

        chat_input = st.chat_input("💬 Type your legal question here...")

        if chat_input:
            st.session_state["chatbot_history"].append({"role": "user", "content": chat_input})

            with st.spinner("🧠 AI Lawyer analyzing your question..."):
                # Get context from documents if available
                selected_id = st.session_state.get("selected_doc_id")
                context_docs = None
                if selected_id:
                    entry = next((e for e in load_manifest() if e.get("doc_id") == selected_id), None)
                    if entry:
                        faiss_db = load_vector_store(entry["db_path"], resolve_embedding_model_name(entry.get("embed_model")))
                        if faiss_db:
                            context_docs = retrieve_docs(faiss_db, chat_input)

                # Get enhanced response
                response_data = chatbot_conversation(
                    chat_input,
                    st.session_state["chatbot_history"],
                    context_docs,
                    language=st.session_state.get("chatbot_language", "English")
                )

                ai_response = response_data.get("response", "I apologize, please try again.")
                st.session_state["chatbot_history"].append({"role": "assistant", "content": ai_response})

                # Store follow-up questions and urgency
                st.session_state["last_followups"] = response_data.get("followup_questions", [])
                st.session_state["last_urgency"] = response_data.get("urgency", "MEDIUM")

            st.rerun()

        # Advanced Control Panel
        st.markdown("---")
        st.markdown("### 🎛️ Control Panel")

        col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)

        with col_ctrl1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state["chatbot_history"] = []
                st.session_state["last_followups"] = []
                st.rerun()

        with col_ctrl2:
            if st.button("📊 Get Summary", use_container_width=True):
                if st.session_state.get("chatbot_history"):
                    with st.spinner("Generating summary..."):
                        summary = generate_conversation_summary(st.session_state["chatbot_history"])
                        st.session_state["chat_summary"] = summary
                    st.success("✅ Summary generated!")

        with col_ctrl3:
            if st.button("💾 Save Chat", use_container_width=True):
                if st.session_state.get("chatbot_history"):
                    chat_text = f"AI LAWYER CONSULTATION - {datetime.now().strftime('%B %d, %Y %I:%M %p')}\n"
                    chat_text += "="*60 + "\n\n"
                    chat_text += "\n\n".join([
                        f"{'[USER]' if m['role']=='user' else '[AI LAWYER]'}\n{m['content']}"
                        for m in st.session_state["chatbot_history"]
                    ])

                    if st.session_state.get("chat_summary"):
                        chat_text += "\n\n" + "="*60 + "\n"
                        chat_text += "CONSULTATION SUMMARY\n" + "="*60 + "\n"
                        chat_text += st.session_state["chat_summary"]

                    st.download_button(
                        "📥 Download Transcript",
                        chat_text,
                        file_name=f"legal_consultation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

        with col_ctrl4:
            if st.button("❌ Exit Chatbot", use_container_width=True, type="secondary"):
                st.session_state["chatbot_mode"] = False
                st.session_state["chat_intake_step"] = 0
                st.rerun()

        # Display Summary if Generated
        if st.session_state.get("chat_summary"):
            with st.expander("📋 Consultation Summary", expanded=True):
                st.markdown(st.session_state["chat_summary"])

        # Legal Disclaimer
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; background: rgba(255, 0, 110, 0.1); border-radius: 10px; border: 1px solid rgba(255, 0, 110, 0.3);">
            <p style="margin: 0; color: #ff006e; font-size: 0.9rem;">
                ⚠️ <strong>Disclaimer:</strong> This AI consultation is for informational purposes only.
                For official legal advice, please consult a licensed advocate.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==================== NEW REVOLUTIONARY FEATURES TABS (6-10) ====================

# TAB 6: Contract Analyzer & Risk Detector - STEP-WISE WIZARD
with feature_tabs[5]:
    st.markdown('<h3 style="color: #00f3ff;">📋 Contract Analyzer & Risk Detector - Step by Step</h3>', unsafe_allow_html=True)
    st.info("💡 Upload any contract and get instant risk analysis with RED FLAGS highlighted! 4 simple steps.")
    
    # Initialize state
    if "contract_step" not in st.session_state:
        st.session_state["contract_step"] = 0
    if "contract_data" not in st.session_state:
        st.session_state["contract_data"] = {}
    
    # STEP 0: Contract Type Selection
    if st.session_state["contract_step"] == 0:
        st.progress(0.25, text="Step 1 of 4: Contract Type")
        st.markdown("### 📋 Step 1: What type of contract is it?")
        
        contract_types = {
            "🏠 Rental/Lease Agreement": "rental",
            "💼 Employment Contract": "employment",
            "💰 Loan/Credit Agreement": "loan",
            "🤝 Partnership Deed": "partnership",
            "⚙️ Service Agreement": "service",
            "🏢 Sale Agreement (Property)": "sale",
            "📝 Other Contract": "other"
        }
        
        selected_type = st.radio("Select Contract Type", list(contract_types.keys()),
                                help="Choose the contract type")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="contract_next_0"):
                st.session_state["contract_data"]["contract_type"] = contract_types[selected_type]
                st.session_state["contract_data"]["contract_type_display"] = selected_type
                st.session_state["contract_step"] = 1
                st.rerun()
    
    # STEP 1: Contract Input Method
    elif st.session_state["contract_step"] == 1:
        st.progress(0.50, text="Step 2 of 4: Provide Contract")
        st.markdown("### 📄 Step 2: How would you like to provide the contract?")
        
        input_method = st.radio(
            "Select input method:",
            ["📋 Paste existing contract text", "✍️ Describe contract terms", "📝 Type key clauses"],
            horizontal=True
        )
        
        if input_method == "📋 Paste existing contract text":
            contract_text = st.text_area(
                "Paste Your Contract Text *",
                value=st.session_state["contract_data"].get("contract_text", ""),
                height=250,
                placeholder="Paste the full contract text here...",
                help="Include all clauses and terms"
            )
            st.caption("💡 Tip: Include all clauses for accurate analysis")
        elif input_method == "✍️ Describe contract terms":
            contract_text = st.text_area(
                "Describe the Key Terms *",
                value=st.session_state["contract_data"].get("contract_text", ""),
                height=250,
                placeholder="Describe main terms, payment, duration, penalties, termination clauses, etc.",
                help="Summary of key terms"
            )
        else:
            contract_text = st.text_area(
                "Key Clauses *",
                value=st.session_state["contract_data"].get("contract_text", ""),
                height=250,
                placeholder="List the important clauses: payment terms, payment duration, penalties, etc.",
                help="Key clauses to analyze"
            )
        
        st.session_state["contract_data"]["contract_text"] = contract_text
        
        # Navigation with sample button
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="contract_back_1"):
                st.session_state["contract_step"] = 0
                st.rerun()
        with col2:
            if st.button("📄 Load Sample", use_container_width=True):
                st.session_state["contract_data"]["contract_text"] = """RENTAL AGREEMENT

This agreement dated 01/01/2026 between Ram Kumar (Landlord) and Tenant.

Property: Flat 203, XYZ Apartment, Delhi
Rent: ₹25,000 per month
Security Deposit: ₹2,00,000

Clause 1: Rent payment due by 5th of every month
Clause 2: Landlord can terminate anytime without notice
Clause 3: Security deposit is non-refundable
Clause 4: Tenant responsible for all repairs including structural
Clause 5: Rent increases by 20% every 6 months
"""
                st.rerun()
        with col3:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="contract_next_1"):
                if contract_text and len(contract_text) > 30:
                    st.session_state["contract_step"] = 2
                    st.rerun()
                else:
                    st.error("❌ Please provide contract text (minimum 30 characters)")
    
    # STEP 2: Additional Details
    elif st.session_state["contract_step"] == 2:
        st.progress(0.75, text="Step 3 of 4: Additional Context")
        st.markdown("### ℹ️ Step 3: Tell us more about your situation")
        
        col1, col2 = st.columns(2)
        with col1:
            your_role = st.selectbox(
                "What is your role? *",
                ["Landlord/Service Provider", "Tenant/Employee/Customer", "Both parties equally", "Not sure"],
                help="Are you the service provider or recipient?"
            )
        
        with col2:
            contract_value = st.number_input(
                "Estimated Contract Value (₹) *",
                min_value=0,
                value=st.session_state["contract_data"].get("contract_value", 100000),
                step=10000,
                help="Total value or annual value of contract"
            )
        
        concerns = st.text_area(
            "Your Main Concerns (Optional)",
            value=st.session_state["contract_data"].get("concerns", ""),
            height=120,
            placeholder="What worries you about this contract? Any specific terms you're unsure about?",
            help="This helps us focus the analysis"
        )
        
        st.session_state["contract_data"].update({
            "your_role": your_role,
            "contract_value": contract_value,
            "concerns": concerns
        })
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="contract_back_2"):
                st.session_state["contract_step"] = 1
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Analyze Now", use_container_width=True, type="primary", key="contract_next_2"):
                st.session_state["contract_step"] = 3
                st.rerun()
    
    # STEP 3: Results & Analysis
    elif st.session_state["contract_step"] == 3:
        st.progress(1.0, text="Step 4 of 4: Risk Analysis Complete")
        st.markdown("### 📊 Step 4: Your Contract Risk Analysis")
        
        data = st.session_state["contract_data"]
        contract_text = data.get("contract_text", "")
        
        # Analyze
        with st.spinner("🧠 AI Lawyer analyzing your contract... (30-60 seconds)"):
            progress = st.progress(0)
            status = st.empty()
            
            status.text("🔍 Scanning contract clauses...")
            progress.progress(25)
            time.sleep(0.3)
            
            status.text("⚠️ Detecting risky terms...")
            progress.progress(50)
            time.sleep(0.3)
            
            status.text("📋 Identifying missing protections...")
            progress.progress(75)
            
            analysis = analyze_contract_risk(contract_text, data.get("contract_type", "other"))
            
            progress.progress(100)
            status.text("✅ Analysis Complete!")
        
        if "error" not in analysis:
            st.balloons()
            
            risk_score = analysis.get("risk_score", 50)
            verdict = analysis.get("overall_verdict", "Analysis Complete")
            
            # Risk Gauge
            if risk_score >= 70:
                gauge_color, gauge_icon, recommendation = "#ff006e", "🔴", "🚨 HIGH RISK - Don't Sign!"
            elif risk_score >= 40:
                gauge_color, gauge_icon, recommendation = "#FFD700", "🟡", "⚠️ MEDIUM RISK - Negotiate changes"
            else:
                gauge_color, gauge_icon, recommendation = "#39ff14", "🟢", "✅ LOW RISK - Safe to sign"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba({
                '255, 0, 110' if gauge_color == '#ff006e' else '255, 215, 0' if gauge_color == '#FFD700' else '57, 255, 20'
            }, 0.15) 0%, rgba({
                '255, 0, 110' if gauge_color == '#ff006e' else '255, 215, 0' if gauge_color == '#FFD700' else '57, 255, 20'
            }, 0.05) 100%); 
                         border: 3px solid {gauge_color}; border-radius: 15px; padding: 2rem; text-align: center; margin-bottom: 2rem;">
                <div style="font-size: 5rem;">{gauge_icon}</div>
                <div style="font-size: 3rem; font-weight: 900; color: {gauge_color};">RISK SCORE: {risk_score}/100</div>
                <div style="font-size: 1.3rem; color: {gauge_color}; margin-top: 1rem; font-weight: 700;">{recommendation}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Red Flags
            if analysis.get("red_flags"):
                st.markdown("### 🔴 CRITICAL RED FLAGS")
                for idx, flag in enumerate(analysis["red_flags"], 1):
                    with st.expander(f"🚨 RED FLAG {idx}: {flag.get('clause_number', 'Clause')} - **{flag.get('risk_level', 'CRITICAL')}**", expanded=(idx <= 2)):
                        st.error(f"**Clause:** {flag.get('clause_text', 'N/A')}")
                        st.write(f"**⚠️ Why Risky:** {flag.get('explanation', 'N/A')}")
                        st.write(f"**💥 Impact:** {flag.get('impact', 'N/A')}")
                        st.success(f"**✅ Fix:** {flag.get('suggestion', 'N/A')}")
            
            # Yellow Flags
            if analysis.get("yellow_flags"):
                st.markdown("### 🟡 YELLOW FLAGS - Review These")
                for idx, flag in enumerate(analysis["yellow_flags"], 1):
                    with st.expander(f"⚠️ Warning {idx}: {flag.get('clause_number', 'Clause')}"):
                        st.warning(f"**Clause:** {flag.get('clause_text', 'N/A')}")
                        st.write(f"**Issue:** {flag.get('explanation', 'N/A')}")
                        st.write(f"**Suggestion:** {flag.get('suggestion', 'N/A')}")
            
            # Missing Protections
            if analysis.get("missing_protections"):
                st.markdown("### 📋 MISSING PROTECTIONS - ADD THESE!")
                st.info("**For your safety, the contract should include:**")
                for idx, protection in enumerate(analysis["missing_protections"], 1):
                    st.write(f"**{idx}.** {protection}")
            
            # Safe Clauses
            if analysis.get("safe_clauses"):
                with st.expander("✅ Safe Clauses (Standard Terms)"):
                    for clause in analysis["safe_clauses"]:
                        st.success(f"✓ {clause}")
            
            # Negotiation Tips
            if analysis.get("negotiation_strategy"):
                with st.expander("💡 Negotiation Tips"):
                    st.markdown(f"**{analysis['negotiation_strategy']}**")
            
            # Financial Risk
            if analysis.get("financial_risk"):
                st.error(f"💰 **Financial Risk:** {analysis['financial_risk']}")
            
            # Download Report
            st.markdown("---")
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                report = f"""CONTRACT RISK ANALYSIS - {data.get('contract_type_display', 'Contract')}
Generated: {datetime.now().strftime('%B %d, %Y')}

RISK SCORE: {risk_score}/100
VERDICT: {verdict}

RED FLAGS ({len(analysis.get('red_flags', []))}):
{chr(10).join(f"{i}. {f.get('clause_number')}: {f.get('explanation')}" for i, f in enumerate(analysis.get('red_flags', []), 1))}

YELLOW FLAGS ({len(analysis.get('yellow_flags', []))}):
{chr(10).join(f"{i}. {f.get('clause_number')}: {f.get('explanation')}" for i, f in enumerate(analysis.get('yellow_flags', []), 1))}

MISSING PROTECTIONS:
{chr(10).join(f"• {m}" for m in analysis.get('missing_protections', []))}

FINANCIAL RISK: {analysis.get('financial_risk', 'N/A')}
"""
                st.download_button(
                    "📥 Download Report",
                    report,
                    file_name=f"contract_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                    use_container_width=True
                )
            
            with col_dl2:
                if st.button("🔄 Analyze Another", use_container_width=True):
                    st.session_state["contract_step"] = 0
                    st.session_state["contract_data"] = {}
                    st.rerun()
            
            with col_dl3:
                if st.button("✏️ Edit Input", use_container_width=True):
                    st.session_state["contract_step"] = 1
                    st.rerun()
        
        else:
            st.error(f"❌ Analysis Error: {analysis.get('error')}")


# TAB 7: Court Hearing Script Generator - STEP-WISE WIZARD
with feature_tabs[6]:
    st.markdown('<h3 style="color: #00f3ff;">⚖️ Court Hearing Script - Step by Step</h3>', unsafe_allow_html=True)
    st.info("💡 Get exact dialogue for court. Follow 3 steps for a realistic script.")

    if "court_script_step" not in st.session_state:
        st.session_state["court_script_step"] = 0
    if "court_script_data" not in st.session_state:
        st.session_state["court_script_data"] = {}

    # Step 1: Hearing type and court
    if st.session_state["court_script_step"] == 0:
        st.progress(0.33, text="Step 1 of 3: Hearing Setup")
        st.markdown("### 📋 Step 1: Select hearing and court")

        hearing_type = st.selectbox("Hearing Type", [
            "First Hearing (Statement of Case)",
            "Evidence Recording (Examination-in-Chief)",
            "Cross-Examination",
            "Arguments",
            "Final Hearing"
        ], key="court_script_hearing")

        court_type = st.selectbox("Court Type", [
            "Consumer Forum",
            "District Court",
            "High Court",
            "Magistrate Court",
            "Family Court"
        ], key="court_script_court")

        user_role = st.selectbox("Your Role", ["Complainant/Plaintiff", "Respondent/Defendant", "Witness"], key="court_script_role")

        col_s1, col_s2 = st.columns([1, 4])
        with col_s1:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="court_script_next_0"):
                st.session_state["court_script_data"].update({
                    "hearing_type": hearing_type,
                    "court_type": court_type,
                    "user_role": user_role
                })
                st.session_state["court_script_step"] = 1
                st.rerun()

    # Step 2: Case details
    elif st.session_state["court_script_step"] == 1:
        st.progress(0.66, text="Step 2 of 3: Case Details")
        st.markdown("### 📝 Step 2: Enter your case details")

        case_facts = st.text_area(
            "Case Facts (What happened?)",
            value=st.session_state["court_script_data"].get("case_facts", ""),
            height=150,
            placeholder="Describe your case: Who? What? When? Where? Why?..."
        )

        evidence_list = st.text_area(
            "Evidence (What proof you have)",
            value=st.session_state["court_script_data"].get("evidence_list", ""),
            height=100,
            placeholder="List evidence: documents, photos, witnesses, bank statements, etc..."
        )

        opponent_claims = st.text_area(
            "Opponent's Claims (What they say)",
            value=st.session_state["court_script_data"].get("opponent_claims", ""),
            height=100,
            placeholder="What is the other side claiming or arguing?"
        )

        st.session_state["court_script_data"].update({
            "case_facts": case_facts,
            "evidence_list": evidence_list,
            "opponent_claims": opponent_claims
        })

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("⬅️ Back", use_container_width=True, key="court_script_back_1"):
                st.session_state["court_script_step"] = 0
                st.rerun()
        with col_s2:
            st.empty()
        with col_s3:
            if st.button("➡️ Generate", use_container_width=True, type="primary", key="court_script_next_1"):
                if case_facts and len(case_facts) > 30:
                    st.session_state["court_script_step"] = 2
                    st.rerun()
                else:
                    st.error("❌ Please provide case facts (minimum 30 characters)")

    # Step 3: Results
    else:
        st.progress(1.0, text="Step 3 of 3: Script Ready")
        st.markdown("### 📜 Step 3: Your Court Hearing Script")

        data = st.session_state["court_script_data"]
        hearing_type = data.get("hearing_type", "First Hearing (Statement of Case)")
        court_type = data.get("court_type", "District Court")
        case_facts = data.get("case_facts", "")
        evidence_list = data.get("evidence_list", "")
        opponent_claims = data.get("opponent_claims", "")

        if st.button("📜 Generate Script", use_container_width=True, type="primary", key="court_script_generate"):
            with st.spinner("Generating professional court script... (30-60 seconds)"):
                progress = st.progress(0)
                status = st.empty()

                status.text("⚖️ Analyzing case details...")
                progress.progress(20)
                time.sleep(0.3)

                status.text("📝 Drafting opening statement...")
                progress.progress(40)
                time.sleep(0.3)

                status.text("💬 Creating dialogue script...")
                progress.progress(60)

                script = generate_hearing_script(
                    hearing_type, case_facts, evidence_list, opponent_claims, court_type
                )

                progress.progress(80)
                status.text("✅ Finalizing script...")
                time.sleep(0.2)

                progress.progress(100)
                status.text("✅ Script Ready!")

                st.session_state["hearing_script"] = script
                st.balloons()

        if st.session_state.get("hearing_script"):
            script = st.session_state["hearing_script"]
            if "error" not in script:
                st.markdown("---")
                st.markdown('<h3 style="color: #39ff14; text-align: center;">📜 YOUR COURT HEARING SCRIPT</h3>', unsafe_allow_html=True)

                st.success("✅ Read this script before going to court. Practice it multiple times!")

                # Opening Statement
                if script.get("opening_statement"):
                    st.markdown("### 1️⃣ OPENING STATEMENT")
                    st.markdown(f"""
                    <div style="background: rgba(0, 243, 255, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #00f3ff; margin-bottom: 1rem;">
                        <p style="font-size: 1.1rem; line-height: 1.8; color: #e8eaf6; margin: 0;">
                            {script['opening_statement']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Evidence Presentation
                if script.get("evidence_presentation"):
                    st.markdown("### 2️⃣ EVIDENCE PRESENTATION")
                    for idx, ev in enumerate(script["evidence_presentation"], 1):
                        with st.expander(f"Evidence {idx}: {ev.get('document', 'Document')}", expanded=(idx <= 2)):
                            st.markdown("**📄 What to Say:**")
                            st.info(ev.get('what_to_say', 'N/A'))

                            if ev.get('anticipated_objection'):
                                st.markdown("**⚠️ Possible Objection:**")
                                st.warning(ev['anticipated_objection'])

                            if ev.get('counter_response'):
                                st.markdown("**✅ Your Response:**")
                                st.success(ev['counter_response'])

                # Judge Interactions
                if script.get("judge_interactions"):
                    st.markdown("### 3️⃣ RESPONDING TO JUDGE")
                    for idx, interaction in enumerate(script["judge_interactions"], 1):
                        st.markdown(f"**Judge May Ask:** {interaction.get('likely_question', 'N/A')}")
                        st.success(f"**You Say:** {interaction.get('answer', 'N/A')}")
                        if interaction.get('followup'):
                            st.info(f"**If Judge Asks More:** {interaction['followup']}")
                        st.markdown("---")

                # Closing Argument
                if script.get("closing_argument"):
                    st.markdown("### 4️⃣ CLOSING ARGUMENT")
                    st.markdown(f"""
                    <div style="background: rgba(57, 255, 20, 0.1); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #39ff14; margin-bottom: 1rem;">
                        <p style="font-size: 1.1rem; line-height: 1.8; color: #e8eaf6; margin: 0;">
                            {script['closing_argument']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Do's and Don'ts
                if script.get("dos_and_donts"):
                    with st.expander("✅ Do's and ❌ Don'ts in Court"):
                        for item in script["dos_and_donts"]:
                            if "DO:" in item or "✅" in item:
                                st.success(item)
                            else:
                                st.error(item)

                # Emergency Responses
                if script.get("emergency_responses"):
                    with st.expander("🚨 Emergency Responses (If Things Go Wrong)"):
                        emergency = script["emergency_responses"]
                        if emergency.get("if_judge_angry"):
                            st.error(f"**If Judge Gets Angry:** {emergency['if_judge_angry']}")
                        if emergency.get("if_opponent_lying"):
                            st.warning(f"**If Opponent Lies:** {emergency['if_opponent_lying']}")
                        if emergency.get("if_confused"):
                            st.info(f"**If You're Confused:** {emergency['if_confused']}")

                # Confidence Tips
                if script.get("confidence_tips"):
                    st.markdown("### 💪 Confidence Tips")
                    for tip in script["confidence_tips"]:
                        st.write(f"• {tip}")

                # Time Estimate
                if script.get("time_estimate"):
                    st.info(f"⏱️ **Estimated Duration:** {script['time_estimate']}")

                # Download Script
                st.markdown("---")
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                with col_dl1:
                    script_text = f"""COURT HEARING SCRIPT
Generated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}

HEARING TYPE: {hearing_type}
COURT: {court_type}

OPENING STATEMENT:
{script.get('opening_statement', 'N/A')}

EVIDENCE PRESENTATION:
{chr(10).join(f'{i}. {e.get("document")}: {e.get("what_to_say")}' for i, e in enumerate(script.get('evidence_presentation', []), 1))}

CLOSING ARGUMENT:
{script.get('closing_argument', 'N/A')}

TIME ESTIMATE: {script.get('time_estimate', 'N/A')}
"""
                    st.download_button(
                        "📄 Download Script",
                        script_text,
                        file_name=f"court_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                with col_dl2:
                    if st.button("🔄 Start Over", use_container_width=True):
                        st.session_state["court_script_step"] = 0
                        st.session_state["court_script_data"] = {}
                        st.session_state["hearing_script"] = None
                        st.rerun()

                with col_dl3:
                    if st.button("⬅️ Edit Inputs", use_container_width=True):
                        st.session_state["court_script_step"] = 1
                        st.rerun()
            else:
                st.error(f"❌ Script Generation Error: {script.get('error')}")


# TAB 8: Evidence Organizer & Case File Builder - STEP-WISE WIZARD
with feature_tabs[7]:
    st.markdown('<h3 style="color: #00f3ff;">📂 Evidence Organizer & Case File Builder - Step by Step</h3>', unsafe_allow_html=True)
    st.info("💡 AI organizes ALL your evidence and creates a professional case file! 4 simple steps.")
    
    # Initialize state
    if "evidence_step" not in st.session_state:
        st.session_state["evidence_step"] = 0
    if "evidence_data" not in st.session_state:
        st.session_state["evidence_data"] = {}
    
    # STEP 0: Case Type
    if st.session_state["evidence_step"] == 0:
        st.progress(0.25, text="Step 1 of 4: Case Type")
        st.markdown("### 📋 Step 1: What type of case is it?")
        
        case_types = {
            "🛍️ Consumer Complaint": "consumer",
            "💰 Cheque Bounce": "cheque",
            "🏠 Property Dispute": "property",
            "💼 Employment Dispute": "employment",
            "📝 Contract Breach": "contract",
            "⚖️ Civil Suit": "civil",
            "🚨 Criminal Case": "criminal",
            "👪 Family Court": "family"
        }
        
        selected_case = st.radio("Select Case Type", list(case_types.keys()),
                                help="Choose your case type")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="evidence_next_0"):
                st.session_state["evidence_data"]["case_type"] = case_types[selected_case]
                st.session_state["evidence_data"]["case_type_display"] = selected_case
                st.session_state["evidence_step"] = 1
                st.rerun()
    
    # STEP 1: Claims Definition
    elif st.session_state["evidence_step"] == 1:
        st.progress(0.50, text="Step 2 of 4: Your Claims")
        st.markdown("### ⚖️ Step 2: What are you claiming? (What do you want?)")
        
        claims = st.text_area(
            "List your claims *",
            value=st.session_state["evidence_data"].get("claims", ""),
            height=150,
            placeholder="""Example:
1. Refund of ₹50,000 for defective product
2. Compensation for mental agony: ₹25,000
3. Cost of investigation: ₹5,000
4. Total requested: ₹80,000
            
Or in property case:
1. Possession of property back
2. Rent compensation for 2 years: ₹5,00,000
3. Damages: ₹2,00,000
""",
            help="Be specific about what you're asking for"
        )
        
        st.session_state["evidence_data"]["claims"] = claims
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="evidence_back_1"):
                st.session_state["evidence_step"] = 0
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="evidence_next_1"):
                if claims and len(claims) > 20:
                    st.session_state["evidence_step"] = 2
                    st.rerun()
                else:
                    st.error("❌ Please describe your claims (minimum 20 characters)")
    
    # STEP 2: Evidence Collection
    elif st.session_state["evidence_step"] == 2:
        st.progress(0.75, text="Step 3 of 4: Your Evidence")
        st.markdown("### 📄 Step 3: What evidence do you have?")
        
        st.markdown("**Be as detailed as possible - list EVERYTHING you have:**")
        
        evidence_text = st.text_area(
            "Describe all your evidence *",
            value=st.session_state["evidence_data"].get("evidence", ""),
            height=300,
            placeholder="""Example for consumer case:
1. Purchase invoice dated 15/01/2024 - Product cost ₹50,000
2. WhatsApp chat with seller from 20/01/2024 - Complaint about defect
3. Photos of defective product - Screen not working (10 photos)
4. Email to customer care dated 25/01/2024 with complaint
5. Service center report - Confirms product defect
6. Bank statement showing payment
7. Witness 1: Friend who saw the problem (name: Raj Singh, phone: 98xxxxx)
8. Witness 2: My brother (name: Amit, phone: 99xxxxx)
9. Video recording showing product not working (1 minute 30 seconds)
10. Repair quote from technician - ₹8,000 to fix
""",
            help="Include: dates, amounts, names, locations, witnesses"
        )
        
        st.session_state["evidence_data"]["evidence"] = evidence_text
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="evidence_back_2"):
                st.session_state["evidence_step"] = 1
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Organize", use_container_width=True, type="primary", key="evidence_next_2"):
                if evidence_text and len(evidence_text) > 50:
                    st.session_state["evidence_step"] = 3
                    st.rerun()
                else:
                    st.error("❌ Please describe your evidence (minimum 50 characters)")
    
    # STEP 3: Results
    elif st.session_state["evidence_step"] == 3:
        st.progress(1.0, text="Step 4 of 4: Evidence Organization Complete")
        st.markdown("### 📊 Step 4: Your Organized Case File")
        
        data = st.session_state["evidence_data"]
        
        # Analyze with spinner
        with st.spinner("🧠 Organizing your evidence with AI... (30-45 seconds)"):
            progress = st.progress(0)
            status = st.empty()
            
            status.text("📋 Cataloging evidence...")
            progress.progress(25)
            time.sleep(0.3)
            
            status.text("🔗 Linking proofs to claims...")
            progress.progress(50)
            time.sleep(0.3)
            
            status.text("📊 Analyzing case strength...")
            progress.progress(75)
            
            organization = organize_evidence(
                data.get("case_type", "consumer"),
                data.get("claims", ""),
                data.get("evidence", "")
            )
            
            progress.progress(100)
            status.text("✅ Organization Complete!")
        
        if "error" not in organization:
            st.balloons()
            
            # Case File Summary
            strength = organization.get("estimated_case_strength", "MODERATE")
            strength_color = "#39ff14" if "STRONG" in strength else "#FFD700" if "MODERATE" in strength else "#ff006e"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(0, 243, 255, 0.15) 0%, rgba(176, 38, 255, 0.15) 100%); 
                         border: 3px solid {strength_color}; border-radius: 15px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <h3 style="color: {strength_color}; margin: 0;">📊 CASE FILE STRENGTH</h3>
                <p style="color: #9fa8da; margin: 0.5rem 0;"><strong>{strength}</strong></p>
                <p style="color: #9fa8da; margin: 0;">Your evidence is well-organized and ready for court</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Evidence Summary by Category
            if organization.get("evidence_by_category"):
                st.markdown("### 📋 Your Evidence Organized by Category")
                
                cat_tabs = st.tabs(["📄 Documentary", "📸 Photos/Video", "💬 Communications", "👥 Witnesses"])
                
                with cat_tabs[0]:  # Documentary
                    docs = organization["evidence_by_category"].get("documentary", [])
                    if docs:
                        for ev in docs:
                            strength_icon = "💪" if ev.get("strength") == "STRONG" else "⚠️" if ev.get("strength") == "MODERATE" else "⚠️"
                            st.markdown(f"**{ev.get('serial_no')}. {ev.get('type')}** {strength_icon}")
                            st.write(f"📝 {ev.get('description')}")
                            st.caption(f"Supports: {ev.get('supports_claim')}")
                            st.divider()
                    else:
                        st.info("No documentary evidence listed. Add purchase invoices, emails, agreements, etc.")
                
                with cat_tabs[1]:  # Photos/Video
                    photos = organization["evidence_by_category"].get("photographic", [])
                    if photos:
                        for ev in photos:
                            st.markdown(f"**Photo {len(photos)}:** {ev.get('description', 'Photographic evidence')}")
                            st.divider()
                    else:
                        st.info("No photos/videos listed. Add screenshots, photos of damage, video evidence, etc.")
                
                with cat_tabs[2]:  # Communications
                    comms = organization["evidence_by_category"].get("electronic", [])
                    if comms:
                        for ev in comms:
                            st.markdown(f"**{ev.get('description', 'Communication evidence')}**")
                            st.divider()
                    else:
                        st.info("No emails/messages listed. Add WhatsApp chats, emails, SMS, courier receipts, etc.")
                
                with cat_tabs[3]:  # Witnesses
                    st.markdown("**Gather written statements from:**")
                    for idx, witness in enumerate(docs + photos + comms, 1):
                        if "witness" in witness.get('description', '').lower():
                            st.write(f"{idx}. {witness.get('description')}")
            
            # Evidence Timeline
            if organization.get("evidence_timeline"):
                with st.expander("📅 Evidence Timeline (Chronological Order)", expanded=True):
                    for event in organization["evidence_timeline"]:
                        st.write(f"**{event.get('date')}:** {event.get('event')}")
                        st.caption(f"Proof: {event.get('proof')}")
                        st.divider()
            
            # Claim-Proof Mapping
            if organization.get("claim_proof_linking"):
                st.markdown("### 🔗 Your Claims vs Your Proofs")
                for claim, proofs in organization["claim_proof_linking"].items():
                    st.markdown(f"**📌 Claim:** {claim}")
                    for proof in proofs:
                        st.success(f"✅ **Proof:** {proof}")
                    st.divider()
            
            # Missing Evidence
            if organization.get("missing_critical_evidence"):
                st.markdown("### ⚠️ MISSING IMPORTANT EVIDENCE")
                for missing in organization["missing_critical_evidence"]:
                    urgency_icon = "🔴" if missing.get("urgency") == "HIGH" else "🟡"
                    with st.expander(f"{urgency_icon} {missing.get('urgency')}: {missing.get('evidence_type')}", 
                                    expanded=(missing.get("urgency") == "HIGH")):
                        st.error(f"**Why Important:** {missing.get('why_critical')}")
                        st.info(f"**How to Get It:** {missing.get('how_to_get')}")
            
            # Case File Statistics
            st.markdown("---")
            st.markdown("### 📊 Your Case File Statistics")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            
            with col_stat1:
                total_evidence = len(organization.get("evidence_by_category", {}).get("documentary", []))
                st.metric("📄 Documents", total_evidence)
            
            with col_stat2:
                total_photos = len(organization.get("evidence_by_category", {}).get("photographic", []))
                st.metric("📸 Photos/Videos", total_photos)
            
            with col_stat3:
                total_comms = len(organization.get("evidence_by_category", {}).get("electronic", []))
                st.metric("💬 Messages/Emails", total_comms)
            
            with col_stat4:
                st.metric("📋 Exhibits Ready", total_evidence + total_photos + total_comms)
            
            # Next Steps
            st.markdown("---")
            st.markdown("### 🎯 Next Steps")
            st.success("""
            1. **Print your case file** - Use the download button below
            2. **Get originals organized** - Keep all originals safe
            3. **Make copies** - 2-3 sets for filing in court
            4. **Get affidavit** - From a lawyer (₹500-1000)
            5. **File in court** - Use the Court Filing tab to submit
            """)
            
            # Download Report
            st.markdown("---")
            report_text = f"""EVIDENCE ORGANIZATION REPORT
Generated: {datetime.now().strftime('%B %d, %Y')}

CASE TYPE: {data.get('case_type_display', 'Case')}
CASE FILE STRENGTH: {strength}

YOUR CLAIMS:
{data.get('claims', 'N/A')}

EVIDENCE SUMMARY:
- Documentary: {len(organization.get('evidence_by_category', {}).get('documentary', []))} items
- Photographic: {len(organization.get('evidence_by_category', {}).get('photographic', []))} items  
- Electronic: {len(organization.get('evidence_by_category', {}).get('electronic', []))} items

EVIDENCE TIMELINE:
{chr(10).join(f"- {e.get('date')}: {e.get('event')} ({e.get('proof')})" for e in organization.get('evidence_timeline', []))}

CRITICAL MISSING EVIDENCE:
{chr(10).join(f"• {m.get('evidence_type')} - {m.get('how_to_get')}" for m in organization.get('missing_critical_evidence', []))}

CASE STRENGTH: {strength}
Next: File in court within limitation period!
"""
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                st.download_button(
                    "📥 Download Case File Report",
                    report_text,
                    file_name=f"case_file_{datetime.now().strftime('%Y%m%d')}.txt",
                    use_container_width=True
                )
            
            with col_dl2:
                if st.button("🔄 Organize Another", use_container_width=True):
                    st.session_state["evidence_step"] = 0
                    st.session_state["evidence_data"] = {}
                    st.rerun()
            
            with col_dl3:
                if st.button("✏️ Edit Evidence", use_container_width=True):
                    st.session_state["evidence_step"] = 2
                    st.rerun()
        
        else:
            st.error(f"❌ Organization Error: {organization.get('error')}")


# TAB 9: Settlement Calculator & Negotiation - STEP-WISE WIZARD
with feature_tabs[8]:
    st.markdown('<h3 style="color: #00f3ff;">💰 Settlement Calculator - Step by Step</h3>', unsafe_allow_html=True)
    st.info("💡 Calculate FAIR settlement range + Get negotiation strategy to maximize your compensation!")
    
    # Initialize state
    if "settlement_step" not in st.session_state:
        st.session_state["settlement_step"] = 0
    if "settlement_data" not in st.session_state:
        st.session_state["settlement_data"] = {}
    
    # STEP 1: Case Details
    if st.session_state["settlement_step"] == 0:
        st.progress(0.25, text="Step 1 of 4: Your Case")
        st.markdown("### 📋 Step 1: Tell us about your case")
        
        case_options = {
            "💰 Cheque Bounce / Payment Default": "cheque_bounce",
            "🛍️ Consumer Complaint (defective product/service)": "consumer",
            "🏠 Property Dispute / Encroachment": "property",
            "🚗 Accident Claim / Personal Injury": "accident",
            "💼 Employment Dispute / Wrongful Termination": "employment",
            "📄 Contract Breach / Non-Performance": "contract"
        }
        
        selected_case = st.radio("Choose Your Case Type", list(case_options.keys()),
                                help="Select the category matching your situation")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="settle_next_0"):
                st.session_state["settlement_data"]["case_type"] = case_options[selected_case]
                st.session_state["settlement_data"]["case_type_display"] = selected_case
                st.session_state["settlement_step"] = 1
                st.rerun()
    
    # STEP 2: Financial Details
    elif st.session_state["settlement_step"] == 1:
        st.progress(0.50, text="Step 2 of 4: Financial Details")
        st.markdown("### 💵 Step 2: How much money is involved?")
        
        col1, col2 = st.columns(2)
        with col1:
            actual_loss = st.number_input(
                "Your Actual Direct Loss/Damage (₹) *",
                min_value=1000,
                value=100000,
                step=10000,
                help="Direct financial loss you suffered (invoice amount, medical bills, salary lost, etc.)")
        
        with col2:
            indirect_loss = st.number_input(
                "Indirect Loss - Mental Agony/Loss of Time (₹)",
                min_value=0,
                value=10000,
                step=5000,
                help="Suffering, lost time, stress, business disruption")
        
        months_since = st.number_input(
            "How many months has this been going on? *",
            min_value=1,
            max_value=120,
            value=6,
            help="Duration of the problem/case")
        
        st.session_state["settlement_data"].update({
            "actual_loss": actual_loss,
            "indirect_loss": indirect_loss,
            "months_since": months_since
        })
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="settle_back_1"):
                st.session_state["settlement_step"] = 0
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="settle_next_1"):
                st.session_state["settlement_step"] = 2
                st.rerun()
    
    # STEP 3: Evidence & Liability (ENHANCED WITH LIVE FEEDBACK)
    elif st.session_state["settlement_step"] == 2:
        st.progress(0.75, text="Step 3 of 4: Evidence & Liability")
        st.markdown("### 📄 Step 3: How strong is your case?")
        
        # Get current values
        evidence = st.session_state["settlement_data"].get("evidence", 7)
        liability = st.session_state["settlement_data"].get("liability", 8)
        
        # LIVE SLIDER WITH REAL-TIME FEEDBACK
        st.markdown("**⚖️ Rating Sliders (Move them to see instant impact on settlement amounts):**")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            evidence = st.slider(
                "How Strong is Your Evidence? *",
                1, 10, evidence,
                help="1=No proof, 10=Solid documents/witnesses",
                key=f"evidence_slider_{id(st.session_state)}")
        with col2:
            st.metric("Rating", f"{evidence}/10", delta=None)
        st.caption("🔍 1 = No proof at all | 10 = Complete documents & witnesses")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            liability = st.slider(
                "How Much is the Other Side at Fault? *",
                1, 10, liability,
                help="1=It's your fault, 10=Completely their fault",
                key=f"liability_slider_{id(st.session_state)}")
        with col2:
            st.metric("Rating", f"{liability}/10", delta=None)
        st.caption("⚖️ 1 = You're at fault | 10 = They're completely guilty")
        
        st.session_state["settlement_data"].update({
            "evidence": evidence,
            "liability": liability
        })
        
        # LIVE IMPACT VISUALIZATION
        st.markdown("---")
        st.markdown("### 📊 Live Impact Preview (Real-time Calculation)")
        
        # Get other data
        actual_loss = st.session_state["settlement_data"].get("actual_loss", 100000)
        indirect = st.session_state["settlement_data"].get("indirect_loss", 10000)
        months = st.session_state["settlement_data"].get("months_since", 6)
        
        # Calculate live scenarios
        base_amount = actual_loss + indirect
        evidence_multiplier = 0.5 + (evidence / 10) * 0.5
        liability_multiplier = (liability / 10)
        monthly_interest = actual_loss * 0.015
        interest_amount = monthly_interest * months
        
        worst_case = (actual_loss * 0.4) + interest_amount
        realistic = (base_amount * evidence_multiplier * liability_multiplier) + interest_amount
        best_case = (base_amount * 1.2) + interest_amount
        win_probability = (liability * evidence / 100)
        
        # Show live metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "🔴 Worst Case",
                f"₹{worst_case:,.0f}",
                delta="If case fails",
                delta_color="off",
                help="If opponent wins or case is weak"
            )
        
        with metric_col2:
            st.metric(
                "🟡 Realistic",
                f"₹{realistic:,.0f}",
                delta="Most likely",
                delta_color="off",
                help="Fair settlement based on case strength"
            )
        
        with metric_col3:
            st.metric(
                "🟢 Best Case",
                f"₹{best_case:,.0f}",
                delta="If you win",
                delta_color="off",
                help="Full compensation + interest"
            )
        
        # Case Strength Indicator
        st.markdown("---")
        st.markdown("### 🎯 Your Case Strength Analysis")
        
        strength_col1, strength_col2 = st.columns([2, 3])
        
        with strength_col1:
            st.metric("Win Probability", f"{win_probability:.0f}%")
            
            # Visual strength indicator
            if win_probability >= 80:
                strength_color = "🟢 VERY STRONG"
                strength_msg = "Excellent chance to win!"
            elif win_probability >= 60:
                strength_color = "🟡 GOOD"
                strength_msg = "Decent settlement likely"
            elif win_probability >= 40:
                strength_color = "🟠 MODERATE"
                strength_msg = "Settlement worth considering"
            else:
                strength_color = "🔴 WEAK"
                strength_msg = "Risky to pursuing court case"
            
            st.markdown(f"**{strength_color}** - {strength_msg}")
        
        with strength_col2:
            # Create settlement range visualization
            import pandas as pd
            scenario_data = pd.DataFrame({
                "Scenario": ["Worst Case", "Realistic", "Best Case"],
                "Amount": [worst_case, realistic, best_case],
                "Color": ["#ff006e", "#FFD700", "#39ff14"]
            })
            
            st.bar_chart(
                scenario_data.set_index("Scenario")["Amount"],
                use_container_width=True,
                height=250
            )
        
        st.markdown("---")
        
        # Navigation with status
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="settle_back_2"):
                st.session_state["settlement_step"] = 1
                st.rerun()
        with col2:
            st.info(f"✅ Ready to calculate! (Evidence: {evidence}/10, Liability: {liability}/10)")
        with col3:
            if st.button("➡️ Calculate", use_container_width=True, type="primary", key="settle_next_2"):
                st.session_state["settlement_step"] = 3
                st.rerun()
    
    # STEP 4: Results & Strategy (ENHANCED WITH INTERACTIVE VISUALIZATIONS)
    elif st.session_state["settlement_step"] == 3:
        st.progress(1.0, text="Step 4 of 4: Settlement Results")
        st.markdown("### 💎 Step 4: Your Settlement Analysis")
        
        data = st.session_state["settlement_data"]
        
        # Calculations
        actual_loss = data.get("actual_loss", 100000)
        indirect = data.get("indirect_loss", 10000)
        evidence = data.get("evidence", 7)
        liability = data.get("liability", 8)
        months = data.get("months_since", 6)
        
        # Settlement Calculation Formula
        base_amount = actual_loss + indirect
        
        # Evidence multiplier
        evidence_multiplier = 0.5 + (evidence / 10) * 0.5  # 0.5 to 1.0
        
        # Liability multiplier
        liability_multiplier = (liability / 10)  # 0 to 1.0
        
        # Interest for months
        monthly_interest = actual_loss * 0.015  # 1.5% per month
        interest_amount = monthly_interest * months
        
        # Worst, Realistic, Best scenarios
        worst_case = (actual_loss * 0.4) + interest_amount
        realistic = (base_amount * evidence_multiplier * liability_multiplier) + interest_amount
        best_case = (base_amount * 1.2) + interest_amount
        
        # Court filing would cost
        court_cost = 15000
        lawyer_expected = 25000
        
        st.markdown("---")
        st.markdown("### 💰 Three Settlement Scenarios")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: rgba(255, 0, 110, 0.1); border: 3px solid #ff006e; padding: 20px; border-radius: 15px; text-align: center;">
                <div style="color: #ff006e; font-weight: 700; font-size: 1.1rem;">🔴 WORST CASE</div>
                <div style="font-size: 3rem; font-weight: 900; color: #ff006e; margin: 10px 0;">₹{worst_case:,.0f}</div>
                <div style="color: #9fa8da; font-size: 0.85rem;">What court might award if case is weak</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: rgba(255, 215, 0, 0.1); border: 3px solid #FFD700; padding: 20px; border-radius: 15px; text-align: center;">
                <div style="color: #FFD700; font-weight: 700; font-size: 1.1rem;">🟡 REALISTIC</div>
                <div style="font-size: 3rem; font-weight: 900; color: #FFD700; margin: 10px 0;">₹{realistic:,.0f}</div>
                <div style="color: #9fa8da; font-size: 0.85rem;">Fair settlement most likely</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: rgba(57, 255, 20, 0.1); border: 3px solid #39ff14; padding: 20px; border-radius: 15px; text-align: center;">
                <div style="color: #39ff14; font-weight: 700; font-size: 1.1rem;">🟢 BEST CASE</div>
                <div style="font-size: 3rem; font-weight: 900; color: #39ff14; margin: 10px 0;">₹{best_case:,.0f}</div>
                <div style="color: #9fa8da; font-size: 0.85rem;">If you win in court</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ENHANCED: Settlement Range Visualization
        st.markdown("---")
        st.markdown("### 📊 Settlement Range Visualization")
        
        import pandas as pd
        settlement_range = pd.DataFrame({
            "Scenario": ["Worst\n(40%)", "Realistic\n(Balanced)", "Best\n(120%)"],
            "Amount": [worst_case, realistic, best_case]
        })
        
        col_viz1, col_viz2 = st.columns([2, 1])
        with col_viz1:
            st.bar_chart(settlement_range.set_index("Scenario")["Amount"], use_container_width=True, height=250)
        
        with col_viz2:
            st.markdown(f"""
### Range Spread
**Difference:** ₹{best_case - worst_case:,.0f}
**Realistic Gap:** ₹{realistic - worst_case:,.0f}
**Upside Potential:** ₹{best_case - realistic:,.0f}

**Your Case Strength**
Evidence: {evidence}/10
Liability: {liability}/10
Win Chance: {liability * evidence / 100:.0f}%
            """)
        
        # Breakdown
        st.markdown("---")
        with st.expander("📋 How We Calculated This", expanded=True):
            st.markdown(f"""
### 💯 Calculation Breakdown

**Your Total Losses:**
| Item | Amount |
|------|--------|
| Direct Loss | ₹{actual_loss:,.0f} |
| Indirect Loss (Mental Agony) | ₹{indirect:,.0f} |
| Interest for {months} months @ 1.5%/month | ₹{interest_amount:,.0f} |
| **TOTAL** | **₹{base_amount + interest_amount:,.0f}** |

**Case Strength Factors:**
| Factor | Rating | Impact |
|--------|--------|--------|
| Evidence Strength | {evidence}/10 | {evidence_multiplier:.1%} weight |
| Opponent's Liability | {liability}/10 | {liability_multiplier:.1%} weight |
| Combined Win Chances | - | {(liability * evidence / 100):.0f}% |

**Settlement Scenarios:**
- **Worst Case (40% of loss + interest):** ₹{worst_case:,.0f}
- **Realistic (case strength adjusted):** ₹{realistic:,.0f}
- **Best Case (120% if you win):** ₹{best_case:,.0f}
            """)
        
        # ENHANCED: Settlement vs Court Comparison
        st.markdown("---")
        st.markdown("### ⚖️ Settlement vs Going to Court - Cost & Time Comparison")
        
        # Cost comparison data
        settlement_cost = 5000  # Legal notice/documentation
        court_total_cost = lawyer_expected + court_cost
        settlement_time_months = 2
        court_time_months = 36
        settlement_stress = 3
        court_stress = 9
        
        # Create comparison visualization
        comparison_data = {
            "Aspect": ["Total Cost", "Processing Time\n(Months)", "Stress Level\n(1-10)"],
            "Settlement": [settlement_cost, settlement_time_months, settlement_stress],
            "Court": [court_total_cost, court_time_months, court_stress]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("""
### 🤝 SETTLEMENT: Fast & Smart

**What You Get:**
✅ Money in 30-60 days
✅ Save ₹20,000+ in legal/court fees  
✅ Definite outcome (no appeals)
✅ Low stress & minimal time
✅ Relationship might be salvaged
✅ Quick closure

**Costs:**
- Legal notice prep: ₹2,000-5,000
- Negotiation (usually free)
- Documentation: ₹1,000-3,000
- **Total:** ₹5,000

**Time:** 2 months
**Stress:** Low (Level 3)
            """)
        
        with comp_col2:
            st.markdown(f"""
### ⚖️ COURT: Long & Expensive

**What You Get:**
✅ Potential full amount ₹{best_case:,.0f}
✅ Sets legal precedent
✅ Forces opponent accountability
❌ Takes 2-5 YEARS
❌ Uncertain outcome
❌ High emotional toll

**Costs:**
- Lawyer fees: ₹20,000-50,000
- Court filing fees: ₹1,000-5,000
- Your time cost: ₹{months * 8 * 1000:,.0f} (lost productivity)
- **Total:** ₹{court_total_cost:,.0f}+

**Time:** 36+ months
**Stress:** Very High (Level 9)
            """)
        
        # Recommendation
        st.markdown("---")
        st.markdown("### 🎯 Smart Recommendation")
        
        money_saved = court_total_cost - settlement_cost
        time_saved_years = (court_time_months - settlement_time_months) / 12
        
        st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(0, 243, 255, 0.1), rgba(176, 38, 255, 0.1)); border: 2px solid #00f3ff; border-radius: 15px; padding: 20px;">

💡 **Based on your case details:**

- **Best Settlement Offer to Accept:** ₹{realistic * 0.9:,.0f}+
- **Don't Go Below:** ₹{realistic * 0.7:,.0f}
- **Open Negotiation At:** ₹{realistic * 1.3:,.0f}

**Why settlement makes sense:**
- 💰 You save ₹{money_saved:,.0f} in court/lawyer costs
- ⏰ You get paid in {settlement_time_months} months vs {court_time_months} months ({time_saved_years:.1f} years saved!)
- 😌 Avoid {court_time_months} months of court stress

**When to fight in court:**
- If settlement offer < ₹{realistic * 0.6:,.0f}
- If opponent refuses to negotiate
- If this sets important legal precedent for others

</div>
        """, unsafe_allow_html=True)
        
        # Negotiation Strategy
        st.markdown("---")
        st.markdown("### 🎯 Your Winning Negotiation Strategy")
        
        opening = realistic * 1.3  # Ask 30% more to start
        walkaway = realistic * 0.7  # Don't accept less than 70%
        
        st.markdown(f"""
<div style="background: rgba(0, 243, 255, 0.1); border-left: 5px solid #00f3ff; padding: 20px; border-radius: 10px;">

**PHASE 1: OPENING DEMAND** 📈
<div style="font-size: 2.2rem; font-weight: 700; color: #00f3ff; margin: 10px 0;">₹{opening:,.0f}</div>
<span style="color: #9fa8da;">Start here (expect them to negotiate down) - Don't care if they reject</span>

---

**PHASE 2: TARGET SETTLEMENT** 🎯
<div style="font-size: 2.2rem; font-weight: 700; color: #FFD700; margin: 10px 0;">₹{realistic:,.0f}</div>
<span style="color: #9fa8da;">Your goal - this is truly FAIR for both sides</span>

---

**PHASE 3: WALK-AWAY POINT** ⛔
<div style="font-size: 2.2rem; font-weight: 700; color: #ff006e; margin: 10px 0;">₹{walkaway:,.0f}</div>
<span style="color: #9fa8da;">Absolute minimum - don't go below this or file court case</span>

</div>
        """, unsafe_allow_html=True)
        
        # Detailed tactics
        st.markdown("---")
        with st.expander("💡 Step-by-Step Negotiation Tactics & Scripts", expanded=True):
            st.markdown(f"""
### How to Negotiate Like a Pro (Detailed Steps)

#### **WEEK 1-2: Initial Offer**

Send professional email:
```
Subject: Settlement Proposal - [Your Case Title]

Dear [Opponent Name],

I propose a settlement of ₹{opening:,.0f} to resolve this matter within 30 days.

Attached: All supporting documents and evidence.

This settlement offer is valid for 30 days. After that, I'll file a court case 
which will cost both of us significantly more in time and money.

Regards,
[Your Name]
```

**Why this amount:** You ask 30% MORE so they have room to negotiate DOWN.

---

#### **WEEK 2-4: Negotiation**

**If they counter with low offer (< ₹{realistic * 0.8:,.0f}):**

Response:
```
Your offer of ₹{realistic * 0.5:,.0f} is 50% below my actual loss.

Here's the math:
- My direct loss: ₹{actual_loss:,.0f}
- Your liability: {liability}/10 (clear fault)
- Fair price: ₹{realistic:,.0f}

Going to court will cost you ₹{court_total_cost:,.0f}+ for {court_time_months} months.
I'm offering ₹{opening:,.0f} for fast resolution.

You have 7 days to counter.
```

**Key Phrases to Use:**
✅ "I have solid evidence"  
✅ "Court costs will be much higher"
✅ "This is my final offer"
✅ "I'm prepared to file immediately"
✅ "Your liability is clear"

**Never Say:**
❌ "I need the money urgently"
❌ "I'll accept anything at this point"
❌ "I'm not sure about my chances"
❌ First offer too low (anchors negotiation down)

---

#### **WEEK 4-6: Closing**

**If stuck between two numbers:**

Counter with average:
```
You offered: ₹{realistic * 0.5:,.0f}
I asked for: ₹{opening:,.0f}
Middle ground: ₹{(realistic * 0.5 + opening) / 2:,.0f}

This is my final offer. Yes or No required in 48 hours.
```

**If they still refuse:**
```
Your final chance: ₹{realistic:,.0f} (my true bottom line)
If you refuse, I file in court tomorrow.
You have 24 hours.
```

**Then actually file in court** (don't bluff!)

---

### Payment Terms to Negotiate

If they're short on cash, offer:
- 50% now, 50% in 30 days
- Monthly installments (but charge 2-3% monthly interest)
- Post-dated cheques
- Bank guarantee

**Golden Rule:** Get something in writing and signed!

---

### Critical Dos & Don'ts

**DO:**
✅ Stay professional in all communication
✅ Document everything (emails, calls, dates)
✅ Show confidence in your case
✅ Mention court costs repeatedly
✅ Set clear deadlines
✅ Get written final agreement
✅ Verify payment before closing

**DON'T:**
❌ Show desperation ("I really need this money")
❌ Accept first counter-offer
❌ Negotiate on phone (email/WhatsApp for proof)
❌ Miss deadlines you set
❌ Accept settlement without written agreement
❌ Bluff about court unless you'll actually file
❌ Share your walk-away amount
            """)
        
        # Download Report
        st.markdown("---")
        st.markdown("### 📥 Download Your Complete Settlement Report")
        
        report = f"""SETTLEMENT CALCULATION & NEGOTIATION REPORT
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

═══════════════════════════════════════════════════════════════════════════════

CASE DETAILS:
Type: {data.get('case_type_display', 'N/A')}
Direct Loss: ₹{actual_loss:,.0f}
Indirect Loss: ₹{indirect:,.0f}
Duration: {months} months
Evidence Strength: {evidence}/10
Opponent's Liability: {liability}/10

═══════════════════════════════════════════════════════════════════════════════

SETTLEMENT CALCULATION RESULTS:

Scenario 1 - WORST CASE:  ₹{worst_case:,.0f}
  (If case is weak and court decides unfavorably)

Scenario 2 - REALISTIC:   ₹{realistic:,.0f}
  (Fair settlement based on your case strength)

Scenario 3 - BEST CASE:   ₹{best_case:,.0f}
  (If you win completely in court)

Range Spread: ₹{best_case - worst_case:,.0f}

═══════════════════════════════════════════════════════════════════════════════

WINNING NEGOTIATION STRATEGY:

Phase 1 - OPENING DEMAND:   ₹{opening:,.0f}
  (Start here, expect them to negotiate down)

Phase 2 - TARGET GOAL:      ₹{realistic:,.0f}
  (Your realistic & fair target)

Phase 3 - WALK-AWAY POINT:  ₹{walkaway:,.0f}
  (Absolute minimum - don't go below this!)

Success Rate: {(liability * evidence / 100):.0f}% probability of winning in court

═══════════════════════════════════════════════════════════════════════════════

SETTLEMENT VS COURT COST ANALYSIS:

SETTLEMENT ROUTE:
  Total Cost: ₹{settlement_cost:,.0f}
  Time Required: {settlement_time_months} months
  Stress Level: Low
  Money You Keep: ₹{realistic - settlement_cost:,.0f}

COURT ROUTE:
  Total Cost: ₹{court_total_cost:,.0f}+
  Time Required: {court_time_months}+ months ({time_saved_years:.1f}+ years!)
  Stress Level: VERY HIGH
  Money You Keep: ₹{best_case - court_total_cost:,.0f} (if you win)
  Risk: Could lose and get nothing

SETTLEMENT SAVES YOU:
  💰 ₹{money_saved:,.0f} in costs
  ⏰ {(court_time_months - settlement_time_months)} months of your time
  😌 {court_stress - settlement_stress}0% less stress

═══════════════════════════════════════════════════════════════════════════════

NEGOTIATION TIMELINE:

Week 1:   Send opening offer of ₹{opening:,.0f}
Week 2-4: Negotiate & counter-offer
Week 4-6: Final settlement phase OR file court case
Month 2:  Get paid

═════════════════════════════════════════════════════════════════════════════════

RECOMMENDATION:

✅ ACCEPT any settlement between ₹{walkaway:,.0f} and ₹{realistic:,.0f}
❌ REJECT offers below ₹{walkaway:,.0f} - they're insulting
🟡 COUNTER all offers with negotiation tactics above
⚖️ FILE COURT if settlement fails (don't hesitate)

═════════════════════════════════════════════════════════════════════════════════

NEXT STEPS:

1. ✉️ Send formal settlement proposal within 48 hours
2. 📎 Attach all documents & evidence
3. ⏰ Set 30-day deadline in the letter
4. 📞 Follow up in 7 days if no response
5. 💬 Negotiate using the tactics in this report
6. ✍️ GET EVERYTHING IN WRITING (very important!)
7. 🏦 Verify payment before closing case
8. 📋 If they refuse → File court case within 1 week

═════════════════════════════════════════════════════════════════════════════════

Good luck! You have a {(liability * evidence / 100):.0f}% chance of winning.
A settlement of ₹{realistic:,.0f} is fair and recommended.

═════════════════════════════════════════════════════════════════════════════════
Report Generated by AI Lawyer RAG System
Date: {datetime.now().strftime('%d/%m/%Y')}
"""
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            st.download_button(
                "📥 Download Full Report (TXT)",
                report,
                file_name=f"settlement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )
        
        with col_dl2:
            if st.button("🔄 Recalculate", use_container_width=True):
                st.session_state["settlement_step"] = 0
                st.session_state["settlement_data"] = {}
                st.rerun()
        
        with col_dl3:
            if st.button("⬅️ Modify Inputs", use_container_width=True):
                st.session_state["settlement_step"] = 2
                st.rerun()


# TAB 10: Timeline Checker - STEP-WISE WIZARD
with feature_tabs[9]:
    st.markdown('<h3 style="color: #00f3ff;">⏰ Legal Timeline Checker - Step by Step</h3>', unsafe_allow_html=True)
    st.info("💡 Check if your case deadline has passed! Don't miss your chance to file in court.")
    
    # Initialize state
    if "timeline_step" not in st.session_state:
        st.session_state["timeline_step"] = 0
    if "timeline_data" not in st.session_state:
        st.session_state["timeline_data"] = {}
    
    # STEP 1: Case Type
    if st.session_state["timeline_step"] == 0:
        st.progress(0.33, text="Step 1 of 3: Your Case Type")
        st.markdown("### 📋 Step 1: What type of case is it?")
        
        case_types = {
            "🛍️ Consumer Complaint": "consumer",
            "💰 Cheque Bounce": "cheque_bounce",
            "💵 Money Recovery (Loan/Payment)": "money_recovery",
            "🏠 Property Dispute": "property",
            "🚨 Criminal Case": "criminal",
            "⚖️ Civil Suit": "civil",
            "💼 Labour Dispute": "labour",
            "📄 Breach of Contract": "contract"
        }
        
        selected_case = st.radio("Select Case Type", list(case_types.keys()),
                                help="Choose the type matching your situation")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➡️ Next", use_container_width=True, type="primary", key="time_next_0"):
                st.session_state["timeline_data"]["case_type"] = case_types[selected_case]
                st.session_state["timeline_data"]["case_type_display"] = selected_case
                st.session_state["timeline_step"] = 1
                st.rerun()
    
    # STEP 2: When Did Problem Happen
    elif st.session_state["timeline_step"] == 1:
        st.progress(0.66, text="Step 2 of 3: When Did It Happen?")
        st.markdown("### 📅 Step 2: When did the problem/incident happen?")
        
        problem_date = st.date_input(
            "Date of Incident/Problem *",
            value=datetime.now() - timedelta(days=180),
            help="When did the issue first occur?",
            max_value=datetime.now()
        )
        
        current_status = st.selectbox(
            "What have you done so far? *",
            [
                "❌ Nothing yet - just realized the issue",
                "📮 Sent legal notice",
                "📝 About to file in court",
                "⚖️ Already filed in court",
                "📊 Still gathering evidence"
            ],
            help="Select your current status"
        )
        
        st.session_state["timeline_data"].update({
            "problem_date": problem_date,
            "current_status": current_status
        })
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Back", use_container_width=True, key="time_back_1"):
                st.session_state["timeline_step"] = 0
                st.rerun()
        with col2:
            st.empty()
        with col3:
            if st.button("➡️ Check", use_container_width=True, type="primary", key="time_next_1"):
                st.session_state["timeline_step"] = 2
                st.rerun()
    
    # STEP 3: Results
    elif st.session_state["timeline_step"] == 2:
        st.progress(1.0, text="Step 3 of 3: Your Deadline Status")
        st.markdown("### ⏱️ Step 3: Your Deadline Status")
        
        data = st.session_state["timeline_data"]
        problem_date = data.get("problem_date", datetime.now())
        case_type = data.get("case_type", "consumer")
        
        # Calculate deadlines based on case type and Indian law
        limitation_periods = {
            "consumer": 2,  # 2 years from knowledge
            "cheque_bounce": 0.5,  # 90 days for notice only
            "money_recovery": 3,  # 3 years
            "property": 12,  # 12 years for title, 3 years for possession
            "criminal": 1,  # Varies but generally within a year
            "civil": 3,  # 3 years general
            "labour": 0.5,  # 1 year or 3 months as per case
            "contract": 3   # 3 years
        }
        
        limitation_days = limitation_periods.get(case_type, 3) * 365
        deadline = problem_date + timedelta(days=limitation_days)
        today = datetime.now().date()
        days_left = (deadline.date() - today).days
        
        # Determine status
        if days_left > 90:
            status = "SAFE"
            icon = "🟢"
            color = "#39ff14"
            message = "You have plenty of time to file!"
        elif days_left > 30:
            status = "CAUTION"
            icon = "🟡"
            color = "#FFD700"
            message = "File soon - don't delay!"
        elif days_left > 0:
            status = "URGENT"
            icon = "🔴"
            color = "#ff006e"
            message = "FILE IMMEDIATELY! Days are running out!"
        else:
            status = "TIME-BARRED"
            icon = "⚫"
            color = "#666"
            message = f"⚠️ YOUR DEADLINE HAS PASSED! ({abs(days_left)} days ago)"
        
        # Display Status
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(0, 243, 255, 0.15) 0%, rgba(176, 38, 255, 0.15) 100%); 
                     border: 4px solid {color}; border-radius: 20px; padding: 30px; text-align: center; margin: 20px 0;">
            <div style="font-size: 4rem; margin-bottom: 10px;">{icon}</div>
            <div style="font-size: 2.5rem; font-weight: 900; color: {color}; margin-bottom: 10px;">{status}</div>
            <div style="font-size: 1.3rem; color: #9fa8da;">{message}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Timeline Details
        st.markdown("---")
        st.markdown("### 📊 Deadline Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Problem Date", problem_date.strftime("%d %b %Y"))
        
        with col2:
            st.metric("Limit Period", f"{limitation_periods.get(case_type, 3)} Years")
        
        with col3:
            st.metric("Your Deadline", deadline.strftime("%d %b %Y"))
        
        with col4:
            st.metric("Days Remaining", f"{days_left} days" if days_left > 0 else "EXPIRED")
        
        # What to do next
        st.markdown("---")
        st.markdown("### 🎯 What You Should Do Next")
        
        if days_left > 90:
            st.success("""
### ✅ YOU HAVE TIME - Plan Properly

1. **This Month:**
   - Gather all documents and evidence
   - Take photos/screenshots of everything
   - Get witness statements in writing
   - Visit **Case Analyzer** to assess your chances

2. **Next Month:**
   - Send legal notice (if applicable)
   - Calculate settlement offer range
   - Go to **Settlement Calculator**

3. **Before Deadline:**
   - File court case or settle
   - Don't unnecessarily delay
            """)
        
        elif days_left > 30:
            st.warning("""
### ⚠️ CAUTION - SPEED UP!

1. **This Week:**
   - Gather ALL documents TODAY
   - Complete evidence collection
   - Finalize witness names/contact
   - **DO NOT DELAY**

2. **Next Week:**
   - Send legal notice immediately
   - Use **Legal Notice Generator**
   - File court case if no settlement

3. **Critical:**
   - Court filing takes 3-5 working days
   - Don't wait until last day!
            """)
        
        elif days_left > 0:
            st.error(f"""
### 🚨 URGENT - FILE TODAY!

**You have only {days_left} day(s) left!**

**IMMEDIATE ACTIONS (TODAY):**
1. Go to nearest court immediately
2. Use **Court Filing Tab** to prepare documents
3. Get help from court advocate
4. Submit petition TODAY

**DO NOT WAIT EVEN ONE MORE DAY!**

Every day lost puts you at risk!
            """)
        
        else:
            st.error(f"""
### ⚫ TIME-BARRED - Deadline Passed {abs(days_left)} Days Ago

**Your deadline for filing has EXPIRED!**

**You may have missed your chance, BUT alternative options exist:**

1. **Condonation of Delay** (Court might allow you to file late)
   - Requires strong reason (illness, government notice, etc.)
   - Success rate: 30-50%
   - Costs: Extra ₹5,000-10,000 in court fees

2. **Criminal Prosecution** (If applicable)
   - Cheque bounce cases: Can be filed anytime
   - Criminal cases: Different limitation periods

3. **Settle Out of Court** (Still possible)
   - Send notice even if late
   - Many settle without court

4. **Consult Experience Lawyer**
   - Needed for condonation petitions
   - Cost: ₹5,000-15,000

**Don't Give Up! Talk to a lawyer immediately for options.**
            """)
        
        # Important Information
        st.markdown("---")
        with st.expander("📋 Limitation Periods for All Case Types (India)", expanded=False):
            st.markdown("""
### Legal Limitation Periods in India

**Quick Reference:**

| Case Type | Limitation | Key Notes |
|-----------|-----------|-----------|
| Consumer | 2 Years | From knowledge of defect |
| Cheque Bounce | 90 Days | Legal notice period |
| Money Recovery | 3 Years | From due date |
| Property Title | 12 Years | From loss of possession |
| Property Possession | 3 Years | From deprivation |
| Criminal (General) | 1-2 Years | Varies by IPC section |
| Civil Suit | 3 Years | General period |
| Labour | 1-3 Years | Depends on claim type |
| Contract Breach | 3 Years | From breach date |

**Important:**
- Limitation runs from the DATE YOU KNEW about the issue
- Not from when it happened (sometimes)
- Sending notice before deadline can help preserve your rights
- Always consult lawyer for exact dates
            """)
        
        # Download Timeline Report
        st.markdown("---")
        st.markdown("### 📥 Download Your Timeline Report")
        
        report = f"""LEGAL TIMELINE ANALYSIS
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

═══════════════════════════════════════════════════════════════

CASE DETAILS:
Type: {data.get('case_type_display', 'N/A')}
Problem Date: {problem_date.strftime('%B %d, %Y')}
Current Status: {data.get('current_status', 'N/A')}

═══════════════════════════════════════════════════════════════

DEADLINE ANALYSIS:

Status: {status}
Days Remaining: {days_left}
Your Deadline: {deadline.strftime('%B %d, %Y')}
Limitation Period: {limitation_periods.get(case_type, 3)} Years

═══════════════════════════════════════════════════════════════

URGENCY LEVEL: {status}

{message}

═══════════════════════════════════════════════════════════════

IF TIME-BARRED:

Don't lose hope! You may still have options:

1. Condonation of Delay Application
   - File immediately with strong reason
   - Success rate: 30-50%
   - Extra cost: ₹5,000-10,000

2. Criminal Remedies (if applicable)
   - Cheque bounce can be filed anytime
   - Some criminal cases have different periods

3. Settlement (always possible)
   - Send notice and try to settle
   - Avoid costly court proceedings

4. Consult Lawyer
   - Needed for condonation petitions
   - Cost: ₹5,000-15,000 consultation

═══════════════════════════════════════════════════════════════

NEXT STEPS:

If Days > 90:
- Gather evidence slowly
- File before deadline
- Use AI Lawyer to prepare

If Days < 90:
- Speed up immediately
- File within next 30 days
- Start court filing process

If Time-Barred:
- Contact lawyer TODAY
- Explore condonation option
- Try settlement approach

═════════════════════════════════════════════════════════════════
"""
        
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        
        with col_dl1:
            st.download_button(
                "📥 Download Report",
                report,
                file_name=f"timeline_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                use_container_width=True
            )
        
        with col_dl2:
            if st.button("🔄 Check Different Date", use_container_width=True):
                st.session_state["timeline_step"] = 1
                st.rerun()
        
        with col_dl3:
            if st.button("⬅️ Start Over", use_container_width=True):
                st.session_state["timeline_step"] = 0
                st.session_state["timeline_data"] = {}
                st.rerun()



# TAB 11+: Advanced Tools (5 new features)
extra_index = len(base_tabs)
if st.session_state.get("enable_advanced_tools"):
    with feature_tabs[extra_index]:
        st.markdown('<h3 style="color: #00f3ff;">🚀 Advanced Legal Tools (New)</h3>', unsafe_allow_html=True)
        st.info("💡 Power tools to replace expensive lawyer consultations.")

        tool_tabs = st.tabs([
            "🎯 Outcome Predictor",
            "🗺️ Strategy Planner",
            "💸 Cost Estimator",
            "🗣️ Simple Explainer",
            "🛡️ Opposition Reply"
        ])

        # Outcome Predictor
        with tool_tabs[0]:
            st.markdown("### 🎯 Case Outcome Predictor")
            case_type_outcome = st.selectbox("Case Type", [
                "Consumer Complaint", "Civil Suit", "Cheque Bounce", "Property Dispute", "Employment Dispute"
            ], key="outcome_case_type")
            court_level = st.selectbox("Court Level", ["District Court", "High Court", "Consumer Forum", "Magistrate Court"], key="outcome_court_level")
            evidence_strength = st.slider("Evidence Strength (0-100)", 0, 100, 70, key="outcome_strength")
            desired_relief = st.text_input("Desired Relief (What you want)", "Compensation/Relief", key="outcome_relief")
            case_facts_outcome = st.text_area("Case Facts", height=140, key="outcome_facts")

            if st.button("🔮 PREDICT OUTCOME", use_container_width=True, type="primary"):
                if case_facts_outcome and len(case_facts_outcome) > 30:
                    with st.spinner("Predicting outcome..."):
                        st.session_state["outcome_prediction"] = predict_case_outcome(
                            case_type_outcome, case_facts_outcome, evidence_strength, court_level, desired_relief
                        )
                        st.balloons()
                else:
                    st.error("❌ Please enter case facts (min 30 characters)")

            if st.session_state.get("outcome_prediction"):
                outcome = st.session_state["outcome_prediction"]
                if "error" not in outcome:
                    st.success(f"✅ Outcome Probability: {outcome.get('outcome_probability', 'N/A')}%")
                    st.markdown("**Likely Outcomes:**")
                    for item in outcome.get("likely_outcomes", []):
                        st.write(f"• {item}")
                    st.markdown("**Risk Factors:**")
                    for item in outcome.get("risk_factors", []):
                        st.warning(item)
                    st.markdown("**Strengthening Steps:**")
                    for item in outcome.get("strengthening_steps", []):
                        st.success(item)
                    st.info(outcome.get("settlement_advice", ""))
                else:
                    st.error(outcome.get("error"))

        # Strategy Planner
        with tool_tabs[1]:
            st.markdown("### 🗺️ Legal Strategy Planner")
            strategy_case_type = st.selectbox("Case Type", [
                "Consumer Complaint", "Civil Suit", "Cheque Bounce", "Property Dispute", "Employment Dispute"
            ], key="strategy_case_type")
            current_stage = st.selectbox("Current Stage", [
                "Not filed", "Legal Notice Sent", "Case Filed", "Evidence Stage", "Arguments"
            ], key="strategy_stage")
            constraints = st.text_input("Constraints (time/money/other)", "Low budget, need fast result", key="strategy_constraints")
            strategy_facts = st.text_area("Case Facts", height=140, key="strategy_facts")

            if st.button("🧭 BUILD STRATEGY", use_container_width=True, type="primary"):
                if strategy_facts and len(strategy_facts) > 30:
                    with st.spinner("Building legal strategy..."):
                        st.session_state["strategy_plan"] = build_legal_strategy(
                            strategy_case_type, strategy_facts, current_stage, constraints
                        )
                        st.balloons()
                else:
                    st.error("❌ Please enter case facts (min 30 characters)")

            if st.session_state.get("strategy_plan"):
                plan = st.session_state["strategy_plan"]
                if "error" not in plan:
                    st.markdown("**Step-by-Step Plan:**")
                    for step in plan.get("strategy_steps", []):
                        st.write(f"**{step.get('step')}** — {step.get('purpose')} ({step.get('timeline')})")
                    st.markdown("**Documents Needed:**")
                    for doc in plan.get("documents_needed", []):
                        st.write(f"• {doc}")
                    st.warning("**Risk Traps:**")
                    for risk in plan.get("risk_traps", []):
                        st.warning(risk)
                    st.success(f"**Next Best Action:** {plan.get('next_best_action', 'N/A')}")
                else:
                    st.error(plan.get("error"))

        # Cost Estimator
        with tool_tabs[2]:
            st.markdown("### 💸 Legal Cost Estimator")
            cost_case_type = st.selectbox("Case Type", [
                "Consumer Complaint", "Civil Suit", "Cheque Bounce", "Property Dispute", "Employment Dispute"
            ], key="cost_case_type")
            cost_court_level = st.selectbox("Court Level", ["District Court", "High Court", "Consumer Forum", "Magistrate Court"], key="cost_court_level")
            hearings_count = st.slider("Expected Hearings", 1, 50, 8, key="cost_hearings")
            document_pages = st.slider("Document Pages", 10, 500, 80, key="cost_pages")
            travel_distance = st.slider("Travel Distance (km)", 0, 300, 20, key="cost_travel")

            if st.button("💰 ESTIMATE COSTS", use_container_width=True, type="primary"):
                with st.spinner("Estimating costs..."):
                    st.session_state["cost_estimate"] = estimate_legal_costs(
                        cost_case_type, cost_court_level, hearings_count, document_pages, travel_distance
                    )
                    st.balloons()

            if st.session_state.get("cost_estimate"):
                cost = st.session_state["cost_estimate"]
                if "error" not in cost:
                    st.success(
                        f"Total Range: ₹{cost.get('total_range', {}).get('min', 0):,.0f} - ₹{cost.get('total_range', {}).get('max', 0):,.0f}"
                    )
                    st.info(f"Court Fee: ₹{cost.get('court_fee_estimate', {}).get('min', 0):,.0f} - ₹{cost.get('court_fee_estimate', {}).get('max', 0):,.0f}")
                    st.info(f"Lawyer Fee: ₹{cost.get('lawyer_fee_range', {}).get('min', 0):,.0f} - ₹{cost.get('lawyer_fee_range', {}).get('max', 0):,.0f}")
                    st.markdown("**Cost Saving Tips:**")
                    for tip in cost.get("cost_saving_tips", []):
                        st.write(f"• {tip}")
                else:
                    st.error(cost.get("error"))

        # Simple Explainer
        with tool_tabs[3]:
            st.markdown("### 🗣️ Simple-Language Legal Explainer")
            explain_language = st.selectbox("Language", ["English", "Hindi", "Both"], key="explain_language")
            document_text = st.text_area("Paste legal text here", height=200, key="explain_text")

            if st.button("🧾 EXPLAIN SIMPLY", use_container_width=True, type="primary"):
                if document_text and len(document_text) > 50:
                    with st.spinner("Explaining in simple language..."):
                        st.session_state["simple_explanation"] = explain_in_simple_language(document_text, explain_language)
                else:
                    st.error("❌ Please paste at least 50 characters")

            if st.session_state.get("simple_explanation"):
                expl = st.session_state["simple_explanation"]
                if "error" not in expl:
                    if explain_language == "Both":
                        st.markdown("**English:**")
                        en = expl.get("english", {})
                        st.write(en.get("simple_summary", ""))
                        for item in en.get("key_points", []):
                            st.write(f"• {item}")
                        st.markdown("**Hindi:**")
                        hi = expl.get("hindi", {})
                        st.write(hi.get("simple_summary", ""))
                        for item in hi.get("key_points", []):
                            st.write(f"• {item}")
                    else:
                        st.markdown("**Simple Summary:**")
                        st.write(expl.get("simple_summary", ""))
                        st.markdown("**Key Points:**")
                        for item in expl.get("key_points", []):
                            st.write(f"• {item}")
                        st.markdown("**Action Steps:**")
                        for item in expl.get("action_steps", []):
                            st.success(item)
                        st.markdown("**Risks:**")
                        for item in expl.get("risk_notes", []):
                            st.warning(item)
                else:
                    st.error(expl.get("error"))

        # Opposition Reply
        with tool_tabs[4]:
            st.markdown("### 🛡️ Opposition Reply Generator")
            opponent_notice = st.text_area("Opponent Notice / Claims", height=120, key="reply_notice")
            your_facts_reply = st.text_area("Your Facts", height=120, key="reply_facts")
            evidence_list_reply = st.text_area("Evidence List", height=80, key="reply_evidence")
            desired_outcome_reply = st.text_input("Desired Outcome", "Withdraw notice / settle / compensation", key="reply_outcome")

            if st.button("✍️ GENERATE REPLY", use_container_width=True, type="primary"):
                if opponent_notice and your_facts_reply:
                    with st.spinner("Drafting reply..."):
                        st.session_state["opposition_reply"] = generate_opposition_reply(
                            opponent_notice, your_facts_reply, evidence_list_reply, desired_outcome_reply
                        )
                        st.balloons()
                else:
                    st.error("❌ Provide opponent notice and your facts")

            if st.session_state.get("opposition_reply"):
                reply = st.session_state["opposition_reply"]
                if "error" not in reply:
                    st.markdown("**Reply Draft:**")
                    st.text_area("Draft", reply.get("reply_draft", ""), height=240)
                    st.markdown("**Key Denials:**")
                    for d in reply.get("key_denials", []):
                        st.warning(d)
                    st.markdown("**Counter Demands:**")
                    for c in reply.get("counter_demands", []):
                        st.success(c)
                    st.markdown("**Annexures:**")
                    for a in reply.get("annexures", []):
                        st.write(f"• {a}")
                else:
                    st.error(reply.get("error"))

    extra_index += 1

# TAB 12: Legal News
if st.session_state.get("enable_legal_news"):
    with feature_tabs[extra_index]:
        st.markdown('<h3 style="color: #00f3ff;">📰 Legal News & Updates</h3>', unsafe_allow_html=True)
        st.info("💡 Live legal updates + AI summaries in English/Hindi.")

        news_language = st.selectbox("Summary Language", ["English", "Hindi", "Both"], key="news_language")
        sources_text = st.text_area(
            "News Sources (one per line)",
            value="\n".join(DEFAULT_NEWS_SOURCES),
            height=100,
            key="news_sources"
        )

        col_news1, col_news2 = st.columns(2)
        with col_news1:
            if st.button("🔄 Fetch Live News", use_container_width=True, type="primary"):
                urls = [u.strip() for u in sources_text.split("\n") if u.strip()]
                with st.spinner("Fetching legal news..."):
                    news_result = fetch_legal_news(urls, max_items=12)
                    st.session_state["news_items"] = news_result.get("items", [])
                    errors = news_result.get("errors", [])
                    if errors:
                        st.warning("Some sources failed:\n" + "\n".join(errors))

        with col_news2:
            if st.button("🧠 Generate AI Summary", use_container_width=True):
                with st.spinner("Summarizing news..."):
                    st.session_state["news_summary"] = summarize_legal_news(
                        st.session_state.get("news_items", []), news_language
                    )

        if st.session_state.get("news_items"):
            st.markdown("### 🗞️ Latest Headlines")
            for item in st.session_state["news_items"]:
                title = item.get("title", "Untitled")
                link = item.get("link", "")
                published = item.get("published", "")
                source = item.get("source", "")
                if link:
                    st.markdown(f"- **{title}** ([Link]({link})) — {source} {published}")
                else:
                    st.markdown(f"- **{title}** — {source} {published}")

        if st.session_state.get("news_summary"):
            summary = st.session_state["news_summary"]
            if "error" not in summary:
                st.markdown("### 📌 AI Summary")
                if news_language == "Both":
                    st.markdown("**English:**")
                    st.write(summary.get("english", ""))
                    st.markdown("**Hindi:**")
                    st.write(summary.get("hindi", ""))
                else:
                    st.write(summary.get("summary", ""))
            else:
                st.error(summary.get("error"))

    extra_index += 1

# TAB 13: Judge Sahib AI
if st.session_state.get("enable_judge_ai"):
    with feature_tabs[extra_index]:
        st.markdown('<h3 style="color: #00f3ff;">👨‍⚖️ Judge Sahib AI (Supreme Court Style)</h3>', unsafe_allow_html=True)
        st.info("💡 Judge-like reasoning with strict, high-level guidance.")

        judge_language = st.selectbox("Response Language", ["English", "Hindi"], key="judge_language")
        judge_query = st.text_area("Describe your case", height=200, key="judge_query")

        if st.button("⚖️ GET JUDGE OPINION", use_container_width=True, type="primary"):
            if judge_query and len(judge_query) > 40:
                with st.spinner("Judge Sahib is analyzing..."):
                    judge_resp = judge_sahib_response(judge_query, judge_language)
                    st.session_state["judge_chat_history"].append({
                        "query": judge_query,
                        "response": judge_resp
                    })
            else:
                st.error("❌ Please provide at least 40 characters")

        if st.session_state.get("judge_chat_history"):
            st.markdown("### 🧑‍⚖️ Judge Sahib Responses")
            for idx, item in enumerate(reversed(st.session_state["judge_chat_history"][-5:]), 1):
                resp = item.get("response", {})
                if "error" in resp:
                    st.error(resp.get("error"))
                else:
                    st.markdown(f"**Case #{idx}:**")
                    st.write(resp.get("response", ""))


st.divider()

# Enhanced File Uploader
st.markdown('<h3 style="color: #00f3ff; margin-bottom: 1rem;">📂 Upload Document</h3>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="pdf", accept_multiple_files=False, label_visibility="collapsed")

# Advanced Settings Panel
with st.expander("⚙️ Advanced Search Settings", expanded=False):
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown('<p style="color: #9fa8da; font-weight: 600;">🔢 Max Source Chunks (Top K)</p>', unsafe_allow_html=True)
        top_k = st.slider("", min_value=2, max_value=10, value=5, help="Number of document chunks to retrieve", label_visibility="collapsed")
    with col_s2:
        st.markdown('<p style="color: #9fa8da; font-weight: 600;">🎯 Search Mode</p>', unsafe_allow_html=True)
        search_mode = st.selectbox("", ["Hybrid Search (Recommended)", "Semantic Only", "Keyword Only"], label_visibility="collapsed")

# Enhanced Query Input
st.markdown('<h3 style="color: #00f3ff; margin-top: 2rem; margin-bottom: 1rem;">💭 Ask Your Legal Question</h3>', unsafe_allow_html=True)
user_query = st.text_area("", height=150, placeholder="🔍 Type your legal question here... (e.g., 'What are the key obligations in this contract?')", key="user_query", label_visibility="collapsed")

# Enhanced Action Button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    ask_question = st.button("🚀 ASK AI LAWYER", use_container_width=True, type="primary")

if ask_question:

    if not user_query:
        st.error("⚠️ Please enter a question to continue.")
    else:
        faiss_db = None
        docs_for_answer = None

        # Priority 1: Use selected indexed document from sidebar
        selected_id = st.session_state.get("selected_doc_id")
        if selected_id:
            entry = next((e for e in load_manifest() if e.get("doc_id") == selected_id), None)
            if entry:
                model_name = resolve_embedding_model_name(entry.get("embed_model"))
                faiss_db = load_vector_store(entry["db_path"], model_name)
                if faiss_db is None:
                    # Attempt rebuild if index missing
                    source_docs = load_pdf(entry["pdf_path"])
                    chunks = create_chunks(source_docs)
                    faiss_db = create_vector_store(entry["db_path"], chunks, model_name)
        # Priority 2: Session-only uploaded file (no persistence)
        elif uploaded_file:
            key = f"faiss::{uploaded_file.name}::{EMBEDDING_MODEL}"
            if key in st.session_state:
                faiss_db = st.session_state[key]
            else:
                temp_entry = upload_pdf(uploaded_file)  # also adds to manifest
                documents = load_pdf(temp_entry["pdf_path"])  # use persisted path to avoid temp issues
                text_chunks = create_chunks(documents)
                faiss_db = create_vector_store(temp_entry["db_path"], text_chunks, EMBEDDING_MODEL)
                st.session_state[key] = faiss_db
        else:
            st.error("Select a document from the sidebar or upload a PDF.")

        if faiss_db:
            # Multilingual: detect and translate query to English for retrieval
            try:
                q_lang = detect(user_query)
            except LangDetectException:
                q_lang = "en"
            query_for_retrieval = user_query
            if q_lang != "en":
                query_for_retrieval = translate_text(user_query, "en")

            ranked = hybrid_rerank(faiss_db, query_for_retrieval, top_k)
            retrieved_docs = [d for (d, _, _, _, _) in ranked]

            # Conversation memory
            st.session_state["messages"].append({"role": "user", "content": user_query})
            memory = summarize_history(st.session_state["messages"]) if len(st.session_state["messages"]) > 0 else ""
            response = answer_query(documents=retrieved_docs, query=query_for_retrieval, memory=memory)
            # Translate response back to original language if needed
            if q_lang != "en":
                response = translate_text(response, q_lang)
            st.session_state["messages"].append({"role": "assistant", "content": response})
            
            # Save to Q&A history for report generation
            source_refs = [f"Page {getattr(d, 'metadata', {}).get('page', '?')}" for d in retrieved_docs[:3]]
            st.session_state["qa_history"].append((user_query, response, source_refs))
            st.session_state["current_sources"] = retrieved_docs

            st.chat_message("user").write(user_query)
            st.chat_message("AI Lawyer").write(response)
            
            # NEW FEATURE: Entity Extraction & Timeline (Feature 2)
            with st.expander("🔍 Smart Entity Extraction & Analysis"):
                st.write("**Extracting legal entities from the answer and sources...**")
                combined_text = response + "\n\n" + "\n".join([d.page_content for d in retrieved_docs[:3]])
                entities = extract_legal_entities(combined_text)
                
                if entities:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if entities.get("parties"):
                            st.write("**👥 Parties Mentioned:**")
                            for party in entities.get("parties", [])[:5]:
                                st.write(f"• {party}")
                        
                        if entities.get("amounts"):
                            st.write("**💰 Monetary Amounts:**")
                            for amt in entities.get("amounts", [])[:5]:
                                st.write(f"• {amt}")
                        
                        if entities.get("locations"):
                            st.write("**📍 Locations/Jurisdictions:**")
                            for loc in entities.get("locations", [])[:5]:
                                st.write(f"• {loc}")
                    
                    with col2:
                        if entities.get("dates"):
                            st.write("**📅 Important Dates:**")
                            for date in entities.get("dates", [])[:5]:
                                st.write(f"• {date}")
                        
                        if entities.get("articles"):
                            st.write("**📜 Articles/Sections:**")
                            for art in entities.get("articles", [])[:5]:
                                st.write(f"• {art}")
                        
                        if entities.get("case_numbers"):
                            st.write("**⚖️ Case Numbers:**")
                            for case in entities.get("case_numbers", [])[:3]:
                                st.write(f"• {case}")
                    
                    # Timeline
                    if entities.get("dates"):
                        st.write("**📊 Timeline:**")
                        timeline = build_timeline(entities)
                        for event in timeline[:5]:
                            st.write(f"🔹 {event['date']}: {event['event']}")
                else:
                    st.info("No specific entities extracted. Try asking about specific legal cases or documents.")

            # Follow-up suggestions
            with st.expander("Follow-up suggestions"):
                suggestions = suggest_followups(user_query, response)
                for s in suggestions:
                    if st.button(s):
                        st.session_state["user_query"] = s
                        st.experimental_rerun()

            with st.expander(f"Sources (top {top_k})"):
                for i, tup in enumerate(ranked, start=1):
                    d, score, bm_s, llm_s, w = tup
                    md = getattr(d, "metadata", {}) or {}
                    page = md.get("page", md.get("source", "unknown"))
                    preview = (d.page_content[:750] + ("…" if len(d.page_content) > 750 else ""))
                    st.markdown(f"**{i}.** Page: {page} | Score: {score:.2f} (bm={bm_s:.2f}, llm={llm_s:.2f}, w={w:.2f})\n\n{preview}")

                    # Feedback controls
                    cid = hashlib.sha256(d.page_content.encode("utf-8", errors="ignore")).hexdigest()
                    cols = st.columns(4)
                    if cols[0].button("👍", key=f"up_{cid}_{i}"):
                        fb = st.session_state["feedback"].get(cid, {"upvotes": 0, "downvotes": 0, "pinned": False})
                        fb["upvotes"] = fb.get("upvotes", 0) + 1
                        st.session_state["feedback"][cid] = fb
                        st.success("Marked helpful")
                    if cols[1].button("👎", key=f"down_{cid}_{i}"):
                        fb = st.session_state["feedback"].get(cid, {"upvotes": 0, "downvotes": 0, "pinned": False})
                        fb["downvotes"] = fb.get("downvotes", 0) + 1
                        st.session_state["feedback"][cid] = fb
                        st.info("Marked less relevant")
                    if cols[2].button("📌 Pin", key=f"pin_{cid}_{i}"):
                        fb = st.session_state["feedback"].get(cid, {"upvotes": 0, "downvotes": 0, "pinned": False})
                        fb["pinned"] = True
                        st.session_state["feedback"][cid] = fb
                        st.success("Pinned for future queries")
                    # PDF viewer
                    src_path = md.get("source") or md.get("file_path")
                    page_num = md.get("page") if isinstance(md.get("page"), int) else None
                    if src_path and cols[3].button("📄 View", key=f"view_{cid}_{i}"):
                        render_pdf_viewer(src_path, page=page_num, search=user_query if user_query else d.page_content[:80])

            # Knowledge Graph
            with st.expander("Knowledge Graph"):
                graph = extract_graph(response, retrieved_docs)
                render_graph(graph)


# ==================== DISPLAY SECTIONS FOR NEW FEATURES ====================

st.divider()

# Display Full Risk Assessment Report
if st.session_state.get("show_risk_report") and st.session_state.get("risk_assessment"):
    st.header("🛡️ Contract Risk Assessment Report")
    
    risk_data = st.session_state["risk_assessment"]
    
    # Overall Risk Score
    overall_risk = risk_data.get("overall_risk_score", "UNKNOWN")
    risk_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    st.subheader(f"{risk_colors.get(overall_risk, '⚪')} Overall Risk Level: {overall_risk}")
    
    # Risk Summary
    st.info(f"**Executive Summary:** {risk_data.get('risk_summary', 'No summary available.')}")
    
    # High Risks
    high_risks = risk_data.get("high_risks", [])
    if high_risks:
        st.subheader("🚨 HIGH RISK ITEMS")
        for idx, risk in enumerate(high_risks, 1):
            with st.container():
                st.markdown(f"""
                <div style="border-left: 4px solid #dc3545; padding: 15px; background: #f8d7da; border-radius: 5px; margin: 10px 0;">
                    <h4>⚠️ Risk #{idx}: {risk.get('clause', 'Unknown')}</h4>
                    <p><strong>Location:</strong> {risk.get('location', 'Not specified')}</p>
                    <p><strong>Issue:</strong> {risk.get('issue', 'Not specified')}</p>
                    <p style="background: #fff3cd; padding: 10px; border-radius: 5px;">
                        <strong>💡 Recommendation:</strong> {risk.get('recommendation', 'Consult legal counsel')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Medium Risks
    medium_risks = risk_data.get("medium_risks", [])
    if medium_risks:
        with st.expander(f"⚡ MEDIUM RISK ITEMS ({len(medium_risks)})"):
            for idx, risk in enumerate(medium_risks, 1):
                st.markdown(f"""
                **{idx}. {risk.get('clause', 'Unknown')}**  
                📍 Location: {risk.get('location', 'Not specified')}  
                ⚠️ Concern: {risk.get('issue', 'Not specified')}
                """)
                st.divider()
    
    # Obligations
    obligations = risk_data.get("obligations", [])
    if obligations:
        with st.expander(f"📋 CONTRACTUAL OBLIGATIONS ({len(obligations)})"):
            for idx, obl in enumerate(obligations, 1):
                st.markdown(f"""
                <div style="background: #e3f2fd; padding: 12px; border-radius: 5px; border-left: 3px solid #2196f3; margin: 8px 0;">
                    <strong>Obligation #{idx}</strong><br>
                    👤 Party: {obl.get('party', 'Not specified')}<br>
                    ✅ Must Do: {obl.get('obligation', 'Not specified')}<br>
                    📅 Deadline: {obl.get('deadline', 'Not specified')}<br>
                    ⚠️ Penalty: {obl.get('penalty', 'Not specified')}
                </div>
                """, unsafe_allow_html=True)
    
    # Red Flags
    red_flags = risk_data.get("red_flags", [])
    if red_flags:
        with st.expander(f"🚩 RED FLAGS ({len(red_flags)})"):
            for idx, flag in enumerate(red_flags, 1):
                severity_color = {"HIGH": "#dc3545", "MEDIUM": "#ffc107", "LOW": "#28a745"}
                color = severity_color.get(flag.get('severity', 'MEDIUM'), "#6c757d")
                st.markdown(f"""
                <div style="background: #ffebee; padding: 12px; border-radius: 5px; border-left: 3px solid {color}; margin: 8px 0;">
                    <strong>🚩 {flag.get('flag', 'Issue')}</strong> [Severity: {flag.get('severity', 'UNKNOWN')}]<br>
                    💡 Action: {flag.get('recommendation', 'Review carefully')}
                </div>
                """, unsafe_allow_html=True)
    
    # Missing Clauses
    missing = risk_data.get("missing_clauses", [])
    if missing:
        with st.expander(f"❌ MISSING CLAUSES ({len(missing)})"):
            for clause in missing:
                st.markdown(f"- ❌ **{clause}**")
    
    # Key Terms
    key_terms = risk_data.get("key_terms", {})
    if key_terms:
        with st.expander("🔑 KEY TERMS SUMMARY"):
            cols = st.columns(2)
            items = list(key_terms.items())
            for idx, (key, value) in enumerate(items):
                with cols[idx % 2]:
                    st.metric(key.replace("_", " ").title(), value)
    
    if st.button("❌ Close Risk Report"):
        st.session_state["show_risk_report"] = False
        st.rerun()

st.divider()

# Display Full Citation Report
if st.session_state.get("show_citation_report") and st.session_state.get("citations_data"):
    st.header("📚 Legal Citation Analysis Report")
    
    cit_data = st.session_state["citations_data"]
    citations = cit_data.get("citations", [])
    validations = cit_data.get("validations", [])
    
    st.info(f"**Total Citations Found:** {len(citations)}")
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        valid_count = sum(1 for v in validations if v.get("is_valid", False))
        st.metric("✅ Valid Citations", valid_count)
    with col2:
        good_law = sum(1 for v in validations if v.get("status") == "good_law")
        st.metric("✔️ Good Law", good_law)
    with col3:
        overruled = sum(1 for v in validations if v.get("status") == "overruled")
        st.metric("❌ Overruled", overruled)
    
    st.divider()
    
    # Display each citation
    for idx, (citation, validation) in enumerate(zip(citations[:10], validations[:10]), 1):
        with st.container():
            status = validation.get("status", "unknown")
            status_emoji = {"good_law": "✅", "overruled": "❌", "modified": "⚠️", "questioned": "⚠️"}.get(status, "❓")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### {status_emoji} Citation #{idx}")
                st.markdown(f"**{citation.get('citation', 'Unknown')}**")
                
                if validation.get("summary"):
                    st.write(f"📝 {validation.get('summary')}")
                
                if validation.get("key_holding"):
                    st.info(f"**Key Holding:** {validation.get('key_holding')}")
                
                if validation.get("warnings"):
                    st.warning(f"⚠️ **Warning:** {validation.get('warnings')}")
            
            with col2:
                st.markdown(f"**Status:** {status.replace('_', ' ').title()}")
                if validation.get("year"):
                    st.markdown(f"**Year:** {validation.get('year')}")
                if validation.get("court"):
                    st.markdown(f"**Court:** {validation.get('court')}")
                if validation.get("precedential_value"):
                    st.markdown(f"**Precedential:** {validation.get('precedential_value').title()}")
            
            # Links
            if validation.get("scholar_link"):
                st.markdown(f"[🔍 Google Scholar]({validation.get('scholar_link')}) | [📘 Bluebook Format] | [🔗 Related Cases]")
            
            st.divider()
    
    if st.button("❌ Close Citation Report"):
        st.session_state["show_citation_report"] = False
        st.rerun()


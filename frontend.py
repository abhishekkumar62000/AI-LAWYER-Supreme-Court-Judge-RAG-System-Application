from rag_pipeline import answer_query, retrieve_docs
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import streamlit as st
from embeddings_provider import get_embedding_model, DEFAULT_EMBEDDING_MODEL

load_dotenv()
PDFS_DIR = 'pdfs/'
EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL

def upload_pdf(file):
    os.makedirs(PDFS_DIR, exist_ok=True)
    with open(PDFS_DIR + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(documents)

def create_vector_store(db_path: str, text_chunks, model_name: str):
    faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(model_name))
    faiss_db.save_local(db_path)
    return faiss_db

st.set_page_config(page_title="AI Lawyer RAG", page_icon="⚖️", layout="wide")

uploaded_file = st.file_uploader("Upload PDF",
                                 type="pdf",
                                 accept_multiple_files=False)


#Step2: Chatbot Skeleton (Question & Answer)

col1, col2 = st.columns([3, 1])
with col1:
    user_query = st.text_area("Enter your prompt: ", height=150 , placeholder= "Ask Anything!")
with col2:
    top_k = st.slider("Top K", 2, 10, 5)
    temperature = st.slider("Temp", 0.0, 1.0, 0.2)

ask_question = st.button("Ask AI Lawyer")

if ask_question:

    if uploaded_file and user_query:

        st.chat_message("user").write(user_query)

        # Save and index uploaded PDF
        upload_pdf(uploaded_file)
        documents = load_pdf(PDFS_DIR + uploaded_file.name)
        text_chunks = create_chunks(documents)
        db_path = "vectorstore/db_faiss"
        faiss_db = create_vector_store(db_path, text_chunks, EMBEDDING_MODEL)

        # RAG Pipeline
        retrieved_docs = retrieve_docs(faiss_db, user_query)[:top_k]
        response = answer_query(documents=retrieved_docs, query=user_query)
        st.chat_message("AI Lawyer").write(response)
        with st.expander(f"Sources (top {top_k})"):
            for i, d in enumerate(retrieved_docs, start=1):
                md = getattr(d, "metadata", {}) or {}
                page = md.get("page", md.get("source", "unknown"))
                preview = (d.page_content[:750] + ("…" if len(d.page_content) > 750 else ""))
                st.markdown(f"**{i}.** Page: {page}\n\n{preview}")
    
    else:
        if not uploaded_file:
            st.error("Please upload a PDF to index.")
        elif not user_query:
            st.error("Please enter a question.")
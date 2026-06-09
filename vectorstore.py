import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTORSTORE_DIR, PDF_PATH

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource(show_spinner="📄 Loading and indexing the handbook...")
def load_vectorstore(_api_key: str) -> Chroma:
    embeddings = get_embeddings()

    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        print("Loaded existing vector store from disk.")
        return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

    loader = PyPDFLoader(PDF_PATH)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(loader.load())
    print(f"Built vector store from {len(chunks)} chunks.")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )
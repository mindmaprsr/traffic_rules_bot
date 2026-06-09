import streamlit as st

VECTORSTORE_DIR = "./chroma_db_pdf"
PDF_PATH = "Drivers-Handbook.pdf"

def get_api_key() -> str:
    return st.secrets["OPENAI_API_KEY"]

def get_hf_token() -> str:
    return st.secrets["HF_TOKEN"]
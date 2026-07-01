import os
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_DIR = "./chroma_db_pdf"
PDF_PATH = "Drivers-Handbook.pdf"

def get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY env var not set")
    return key

def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var not set")
    return token

import os
from dotenv import load_dotenv

load_dotenv()

VECTORSTORE_DIR = "./chroma_db"
DATA_DIR = "./data"

def get_hf_token() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN env var not set")
    return token

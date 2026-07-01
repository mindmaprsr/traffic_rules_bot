import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import VECTORSTORE_DIR, DATA_DIR

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def load_vectorstore(country: str) -> Chroma:
    embeddings = get_embeddings()
    persist_dir = os.path.join(VECTORSTORE_DIR, country)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"[{country}] Loaded existing vector store from disk.")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=country,
        )

    pdf_path = os.path.join(DATA_DIR, country, "handbook.pdf")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No PDF found for country '{country}' at {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(loader.load())
    print(f"[{country}] Built vector store from {len(chunks)} chunks.")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=country,
    )

def get_available_countries() -> list[str]:
    if not os.path.exists(DATA_DIR):
        return []
    return [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and os.path.exists(os.path.join(DATA_DIR, d, "handbook.pdf"))
    ]

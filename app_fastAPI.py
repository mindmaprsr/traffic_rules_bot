import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config_fastAPI import get_api_key, get_hf_token
from vectorstore_fastAPI import load_vectorstore
from qa_chain_fastAPI import build_qa_chain
from utils import normalize_query

# Module-level singletons (replaces @st.cache_resource)
_qa_chain = None

@asynccontextmanager
async def lifespan(app_fastAPI: FastAPI):
    global _qa_chain
    vectorstore = load_vectorstore()
    _qa_chain = build_qa_chain(vectorstore, get_hf_token())
    yield

app = FastAPI(title="Traffic Rules Bot", lifespan=lifespan)

class QuestionRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    chunk_index: int
    page: str
    source_file: str
    content: str

class AnswerResponse(BaseModel):
    answer: str
    normalized_query: str | None  # None if no normalization happened
    source_chunks: list[SourceChunk]

@app.post("/ask", response_model=AnswerResponse)
def ask(body: QuestionRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    normalized = normalize_query(question)
    response = _qa_chain.invoke({"query": normalized})

    chunks = []
    for i, doc in enumerate(response["source_documents"]):
        raw_page = doc.metadata.get("page")
        page_label = f"Page {raw_page + 1}" if isinstance(raw_page, int) else "Page unknown"
        source_file = doc.metadata.get("source", "")
        content = doc.page_content.strip()
        chunks.append(SourceChunk(
            chunk_index=i + 1,
            page=page_label,
            source_file=os.path.basename(source_file),
            content=content[:500] + ("..." if len(content) > 500 else ""),
        ))

    return AnswerResponse(
        answer=response["result"],
        normalized_query=normalized if normalized != question else None,
        source_chunks=chunks,
    )

@app.get("/")
def root():
    return {"message": "Traffic Rules Bot API is running. Use POST /ask to query."}

@app.get("/health")
def health():
    return {"status": "ok"}

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import get_hf_token
from vectorstore import load_vectorstore, get_available_countries
from qa_chain import build_qa_chain
from utils import normalize_query

_chains: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chains
    countries = get_available_countries()
    if not countries:
        print("WARNING: No country PDFs found in /data. Add PDFs to data/<country>/handbook.pdf")
    for country in countries:
        vectorstore = load_vectorstore(country)
        _chains[country] = build_qa_chain(vectorstore, get_hf_token())
        print(f"Loaded chain for: {country}")
    yield

app = FastAPI(title="Traffic Rules Bot", lifespan=lifespan)


class QuestionRequest(BaseModel):
    country: str
    question: str

class SourceChunk(BaseModel):
    chunk_index: int
    page: str
    source_file: str
    content: str

class AnswerResponse(BaseModel):
    country: str
    answer: str
    normalized_query: str | None
    source_chunks: list[SourceChunk]


@app.get("/")
def root():
    return {"message": "Traffic Rules Bot API is running. Use POST /ask to query."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/countries")
def countries():
    return {"countries": get_available_countries()}

@app.post("/ask", response_model=AnswerResponse)
def ask(body: QuestionRequest):
    country = body.country.strip().lower()
    question = body.question.strip()

    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    if country not in _chains:
        available = list(_chains.keys())
        raise HTTPException(
            status_code=404,
            detail=f"No data loaded for country '{country}'. Available: {available}",
        )

    normalized = normalize_query(question)
    response = _chains[country].invoke({"query": normalized})

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
        country=country,
        answer=response["result"],
        normalized_query=normalized if normalized != question else None,
        source_chunks=chunks,
    )

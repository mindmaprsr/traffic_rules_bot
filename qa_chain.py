import streamlit as st
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from vectorstore import get_embeddings

_PROMPT_TEMPLATE = """You are an expert on German traffic rules and road safety.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the handbook to answer that."
Do NOT make up or infer rules that are not explicitly stated.

IMPORTANT: If the question asks for a specific number of items (e.g. "three", "five", "2"),
you MUST return exactly that many. Do not return more or fewer.

Context:
{context}

Question: {question}

Answer:"""

@st.cache_resource(show_spinner="🤖 Warming up the LLM...")
def build_qa_chain(_vectorstore: Chroma, _api_key: str) -> RetrievalQA:
    endpoint = HuggingFaceEndpoint(
        repo_id="MiniMaxAI/MiniMax-M2.5",
        provider="auto",
        max_new_tokens=512,
        do_sample=False,
        huggingfacehub_api_token=_api_key,
    )
    llm = ChatHuggingFace(llm=endpoint)

    retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "embedding": get_embeddings()},
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PromptTemplate(
            template=_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )},
    )
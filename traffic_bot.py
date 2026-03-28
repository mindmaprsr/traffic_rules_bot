import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

st.title("🚨🚦 Traffic Rules Chat App")

VECTORSTORE_DIR = "./chroma_db_pdf"
PDF_PATH = "Drivers-Handbook.pdf"


# ── Cached resources: only built once per session ────────────────────────────

@st.cache_resource(show_spinner="📄 Loading and indexing the handbook...")
def load_vectorstore() -> Chroma:
    """Load PDF, chunk it, embed, and persist — or reload from disk if already done."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        print("✅ Loaded existing vector store from disk.")
        return Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embeddings,
        )

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    print(f"🔨 Built vector store from {len(chunks)} chunks.")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )


@st.cache_resource(show_spinner="🤖 Warming up the LLM...")
def build_qa_chain(_vectorstore: Chroma) -> RetrievalQA:
    """Build the RetrievalQA chain once and reuse it."""
    template = """You are an expert on German traffic rules and road safety.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the handbook to answer that."
Do NOT make up or infer rules that are not explicitly stated.

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

    llm = OllamaLLM(model="llama3.2", temperature=0.1)

    retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20},
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def rewrite_query(question: str) -> str:
    """
    Normalise the query before it hits the vector store.

    Definite articles like 'the' signal a presupposed canonical answer to the
    embedding model, shifting the query vector away from general informational
    chunks. Rewriting strips that presupposition so retrieval stays robust
    regardless of how the user words their question.
    """
    llm = OllamaLLM(model="llama3.2", temperature=0)
    instruction = (
        "Rewrite the following question to be more generic and retrieval-friendly. "
        "Remove definite articles or phrasing that implies a single canonical answer exists. "
        "Return only the rewritten question, no explanation, no punctuation changes.\n\n"
        f"Question: {question}"
    )
    rewritten = llm.invoke(instruction).strip()
    print(f"Original : {question}")
    print(f"Rewritten: {rewritten}")
    return rewritten


# ── Build resources ───────────────────────────────────────────────────────────

vectorstore = load_vectorstore()
qa_chain = build_qa_chain(vectorstore)


# ── UI ────────────────────────────────────────────────────────────────────────

with st.form("qa_form"):
    question = st.text_area(
        "Ask a question about German traffic rules:",
        placeholder="e.g. What are the three basic traffic rules in Germany?",
    )
    submitted = st.form_submit_button("🔍 Ask")

if submitted:
    question = question.strip()
    if not question:
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Thinking..."):
            rewritten = rewrite_query(question)
            response = qa_chain.invoke(rewritten)

        st.subheader("Answer")
        st.write(response["result"])

        # Optional: show what the query was rewritten to (helpful for debugging)
        with st.expander("🔁 Rewritten query"):
            st.caption(rewritten)

        # Show source pages so users can verify answers against the handbook
        with st.expander("📚 Source pages used"):
            seen = set()
            for doc in response["source_documents"]:
                page = doc.metadata.get("page", "?")
                if page not in seen:
                    seen.add(page)
                    st.markdown(f"**Page {page}**")
                    st.caption(doc.page_content[:300] + "…")

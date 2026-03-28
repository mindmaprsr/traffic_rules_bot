import os
import re
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

# Number words that appear before a noun and carry a quantity constraint.
# "the three rules" → "three rules", but "the rule" is left alone.
_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|"
    "eleven|twelve|thirteen|fourteen|fifteen|"
    "twenty|thirty|forty|fifty|hundred"
)
_THE_BEFORE_NUMBER = re.compile(
    rf"\bthe\s+(?=\d+|(?:{_NUMBER_WORDS})\b)", re.IGNORECASE
)


def normalize_query(question: str) -> str:
    """
    Remove 'the' only when it precedes a number word or digit.

    'What are the three rules'  →  'What are three rules'
    'What is the speed limit'   →  unchanged  (no number follows 'the')

    This is deterministic, zero-latency, and cannot accidentally drop
    keywords the way an LLM rewriter can.
    """
    normalized = _THE_BEFORE_NUMBER.sub("", question).strip()
    # Collapse any double spaces left behind
    normalized = re.sub(r" {2,}", " ", normalized)
    if normalized != question:
        print(f"Original  : {question}")
        print(f"Normalized: {normalized}")
    return normalized


# ── Cached resources: only built once per session ────────────────────────────

@st.cache_resource(show_spinner="📄 Loading and indexing the handbook...")
def load_vectorstore() -> Chroma:
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
    template = """You are an expert on German traffic rules and road safety.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the handbook to answer that."
Do NOT make up or infer rules that are not explicitly stated.

IMPORTANT: If the question asks for a specific number of items (e.g. "three", "five", "2"),
you MUST return exactly that many. Do not return more or fewer.

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
            normalized = normalize_query(question)
            response = qa_chain.invoke(normalized)

        st.subheader("Answer")
        st.write(response["result"])

        if normalized != question:
            with st.expander("🔁 Normalized query"):
                st.caption(normalized)

        with st.expander("📚 Source pages used"):
            seen = set()
            for doc in response["source_documents"]:
                page = doc.metadata.get("page", "?")
                if page not in seen:
                    seen.add(page)
                    st.markdown(f"**Page {page}**")
                    st.caption(doc.page_content[:300] + "…")
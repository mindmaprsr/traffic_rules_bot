import os
import re
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

st.title("🚨🚦 Traffic Rules Chat App")

VECTORSTORE_DIR = "./chroma_db_pdf"
PDF_PATH = "Drivers-Handbook.pdf"

# ── Load API key from Streamlit secrets ──────────────────────────────────────

api_key = st.secrets["OPENAI_API_KEY"]
# os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
# os.environ["GOOGLE_GENAI_API_VERSION"] = "v1"
# ── Query normalisation ──────────────────────────────────────────────────────

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

    'What are the three rules'  ->  'What are three rules'
    'What is the speed limit'   ->  unchanged  (no number follows 'the')
    """
    normalized = _THE_BEFORE_NUMBER.sub("", question).strip()
    normalized = re.sub(r" {2,}", " ", normalized)
    if normalized != question:
        print(f"Original  : {question}")
        print(f"Normalized: {normalized}")
    return normalized


# ── Cached resources: only built once per session ────────────────────────────

@st.cache_resource(show_spinner="📄 Loading and indexing the handbook...")
def load_vectorstore(_api_key: str) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",  # or bge-base, bge-large
        model_kwargs={"device": "cpu"},  # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True}  # recommended for BGE
    )

    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        print("Loaded existing vector store from disk.")
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
    print(f"Built vector store from {len(chunks)} chunks.")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
    )


@st.cache_resource(show_spinner="🤖 Warming up the LLM...")
def build_qa_chain(_vectorstore: Chroma, _api_key: str) -> RetrievalQA:
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

    # llm = ChatOpenAI(model="gpt-4o", api_key=st.secrets["OPENAI_API_KEY"])
    endpoint = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    provider="auto",
    max_new_tokens=512,        # was 100 — too low, caused truncation/StopIteration
    do_sample=False,
    huggingfacehub_api_token=st.secrets["HF_TOKEN"],
    )
    llm = ChatHuggingFace(llm=endpoint)  # ← this wrapper fixes the StopIteration

    retriever = _vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "embedding": HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",  # or bge-base, bge-large
                 model_kwargs={"device": "cpu"},        # use "cuda" if you have a GPU
                encode_kwargs={"normalize_embeddings": True}  # recommended for BGE
            )
        },
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


# ── Build resources ───────────────────────────────────────────────────────────

vectorstore = load_vectorstore(api_key)
qa_chain = build_qa_chain(vectorstore, api_key)


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
            response = qa_chain.invoke({"query": normalized})

        st.subheader("Answer")
        st.write(response["result"])

        if normalized != question:
            with st.expander("🔁 Normalized query"):
                st.caption(normalized)

        with st.expander(f"📚 Source pages used ({len(response['source_documents'])} chunks)"):
            if not response["source_documents"]:
                st.info("No source documents were returned by the retriever.")
            else:
                for i, doc in enumerate(response["source_documents"]):
                    raw_page = doc.metadata.get("page", None)
                    page_label = f"Page {raw_page + 1}" if isinstance(raw_page, int) else "Page unknown"
                    source_file = doc.metadata.get("source", "")

                    with st.container(border=True):
                        st.markdown(f"**Chunk {i + 1} — {page_label}**")
                        if source_file:
                            st.caption(f"Source: {os.path.basename(source_file)}")
                        content = doc.page_content.strip()
                        st.text(content[:500] + ("..." if len(content) > 500 else ""))

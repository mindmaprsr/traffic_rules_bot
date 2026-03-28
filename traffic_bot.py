import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

import os

st.title("🚨🚦Traffic rules chat App")

# Load PDF
loader = PyPDFLoader("Drivers-Handbook.pdf")
documents = loader.load()
# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)
# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# Vector store 
vectorstore_directory="./chroma_db_pdf"

if os.path.exists(vectorstore_directory) and os.listdir(vectorstore_directory):
    print("Loading existing vector store.....")
    vectorstore = Chroma(persist_directory=vectorstore_directory, embedding_function = embeddings)
else:
    print("Building vector store.....")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=vectorstore_directory
    )

        
# Custom prompt template
template = """You are an expert on German traffic rules. Use the following context to answer the question.
If you don't know the answer, just say you don't know. Don't make up an answer.

Context: {context}

Question: {question}

Answer: """

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)
llm = OllamaLLM(model="llama3.2", temperature=0.1)
with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are three basic traffic rules in Germany?",
    )
    submitted = st.form_submit_button("Submit")
    qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
    if submitted:
        response = qa_chain.invoke(text)
        st.write(response['result'])
    
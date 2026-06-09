import os
import streamlit as st
from config import get_api_key, get_hf_token
from vectorstore import load_vectorstore
from qa_chain import build_qa_chain
from utils import normalize_query

st.title("🚨🚦 Traffic Rules Chat App")

api_key = get_api_key()
hf_token = get_hf_token()

vectorstore = load_vectorstore(api_key)
qa_chain = build_qa_chain(vectorstore, hf_token)

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
                    raw_page = doc.metadata.get("page")
                    page_label = f"Page {raw_page + 1}" if isinstance(raw_page, int) else "Page unknown"
                    source_file = doc.metadata.get("source", "")
                    with st.container(border=True):
                        st.markdown(f"**Chunk {i + 1} — {page_label}**")
                        if source_file:
                            st.caption(f"Source: {os.path.basename(source_file)}")
                        content = doc.page_content.strip()
                        st.text(content[:500] + ("..." if len(content) > 500 else ""))
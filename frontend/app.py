import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="Traffic Rules Bot", page_icon="🚦", layout="centered")
st.title("🚦 Traffic Rules Bot")
st.caption("Ask questions about traffic rules by country.")


@st.cache_data(ttl=60)
def fetch_countries() -> list[str]:
    try:
        resp = requests.get(f"{API_URL}/countries", timeout=5)
        resp.raise_for_status()
        return resp.json().get("countries", [])
    except Exception:
        return []


countries = fetch_countries()

if not countries:
    st.error("No country data available. Add PDFs to data/<country>/handbook.pdf and restart.")
    st.stop()

country = st.selectbox("Select country", sorted(countries))
question = st.text_input("Your question", placeholder="What is the speed limit on highways?")

if st.button("Ask", disabled=not question.strip()):
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={"country": country, "question": question},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()

                st.markdown("### Answer")
                st.write(data["answer"])

                if data.get("normalized_query"):
                    st.info(f"Query normalized to: _{data['normalized_query']}_")

                if data.get("source_chunks"):
                    with st.expander(f"Source chunks ({len(data['source_chunks'])})"):
                        for chunk in data["source_chunks"]:
                            st.markdown(
                                f"**Chunk {chunk['chunk_index']}** — {chunk['page']} "
                                f"| `{chunk['source_file']}`"
                            )
                            st.text(chunk["content"])
                            st.divider()
            else:
                detail = resp.json().get("detail", resp.text)
                st.error(f"Error {resp.status_code}: {detail}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the backend. Make sure the API is running.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

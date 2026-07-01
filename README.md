# Traffic Rules Bot

A RAG-based Q&A app for German traffic rules, powered by LangChain, ChromaDB, and HuggingFace.

---

## Sources

- Germany: https://adilbari.wordpress.com/wp-content/uploads/2015/07/md-guide-to-driving-in-germany.pdf
- https://www.gettingaroundgermany.info/zeichen2.shtml (free data)
- UK: https://www.gov.uk/browse/driving/highway-code-road-safety

---

## Running the FastAPI App

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Set up environment variables

Copy the example and fill in your keys:

```sh
cp .env .env.local   # or just edit .env directly
```

`.env` format:

```
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

> **Never commit `.env` to git.** It is already listed in `.gitignore`.

### 3. Start the server

```sh
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

### 4. Interactive API docs

Open `http://localhost:8000/docs` in your browser to explore and test the endpoints via Swagger UI.

### 5. Ask a question

```sh
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the three basic traffic rules in Germany?"}'
```

**Response format:**

```json
{
  "answer": "...",
  "normalized_query": "...",
  "source_chunks": [
    {
      "chunk_index": 1,
      "page": "Page 5",
      "source_file": "Drivers-Handbook.pdf",
      "content": "..."
    }
  ]
}
```

### Available endpoints

| Method | Path      | Description                        |
|--------|-----------|------------------------------------|
| POST   | `/ask`    | Ask a question about traffic rules |
| GET    | `/health` | Health check                       |

---

## Running with Docker

Build the image:

```sh
docker build -t trafficbot-app .
```

Run the container (pass env vars at runtime):

```sh
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e HF_TOKEN=your_token \
  trafficbot-app
```

---

## Running the Streamlit App (original)

```sh
streamlit run app.py
```

---

## Project Structure

```
traffic_rules_bot/
├── app.py                # FastAPI application
├── config_fastAPI.py     # Env-based config for FastAPI
├── config.py             # Streamlit-based config (original)
├── qa_chain_fastAPI.py   # QA chain (no Streamlit dependency)
├── qa_chain.py           # QA chain (original, Streamlit cached)
├── vectorstore.py        # ChromaDB vector store loader
├── utils.py              # Query normalisation
├── Drivers-Handbook.pdf  # Source PDF
├── chroma_db_pdf/        # Persisted vector store (auto-generated)
└── .env                  # API keys (never commit this)
```

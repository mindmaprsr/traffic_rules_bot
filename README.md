# Traffic Rules Bot

A RAG (Retrieval-Augmented Generation) app that answers questions about traffic rules by country. Ask it anything from speed limits to right-of-way rules and it will find the answer directly from the official handbook PDF — no hallucinations, no guessing.

Built with **FastAPI**, **Streamlit**, **LangChain**, **ChromaDB**, and **HuggingFace**.

---

## How it works

1. On startup, the backend reads the PDF handbooks from the `data/` folder
2. Each PDF is split into chunks and stored in a ChromaDB vector store (one per country)
3. When you ask a question, the backend finds the most relevant chunks and passes them to an LLM
4. The LLM answers using only the retrieved context — it will not make up rules
5. The Streamlit frontend provides a simple UI to select a country and ask questions

---

## Project structure

```
traffic_rules_bot/
├── backend/
│   ├── app.py            # FastAPI — /ask, /countries, /health endpoints
│   ├── vectorstore.py    # Loads and builds ChromaDB per country
│   ├── qa_chain.py       # LangChain RAG chain
│   ├── config.py         # Reads env vars
│   ├── utils.py          # Query normalisation
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py            # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   └── <country>/
│       └── handbook.pdf  # One PDF per country
├── docker-compose.yml
├── .env.example
└── .gitignore
```

---

## Adding a country

Drop the official traffic/driving handbook PDF into `data/<country>/handbook.pdf`:

```
data/
├── germany/
│   └── handbook.pdf
├── uk/
│   └── handbook.pdf
└── india/
    └── handbook.pdf
```

The country name becomes the key used in the UI dropdown and API requests. Restart the app after adding a new PDF.

**PDF sources:**
- Germany: https://adilbari.wordpress.com/wp-content/uploads/2015/07/md-guide-to-driving-in-germany.pdf
- UK: https://www.gov.uk/browse/driving/highway-code-road-safety

---

## Running with Docker Compose (recommended)

### 1. Get a HuggingFace token

Sign up at https://huggingface.co and create a token at Settings → Access Tokens.

### 2. Set up environment variables

```sh
cp .env.example .env
```

Edit `.env` and fill in your token:

```
HF_TOKEN=your_huggingface_token_here
```

### 3. Add at least one country PDF

```sh
mkdir -p data/germany
cp /path/to/your/handbook.pdf data/germany/handbook.pdf
```

### 4. Make sure Docker Desktop is running

Open **Docker Desktop** from your Applications folder and wait for the whale icon in the menu bar to stop animating before proceeding.

If you see this error:
```
/var/run/docker.sock: connect: no such file or directory
```
It means Docker Desktop is not running. Start it and wait until it's ready, then try again. If it's already open, use the menu bar icon → **Restart**.

### 5. Start the app

```sh
docker compose up --build
```

The first run will take a few minutes — it installs dependencies and indexes the PDFs.

### 6. Open the UI

Go to `http://localhost:8501` in your browser. Select a country, type a question, and hit **Ask**.

---

## API endpoints

The FastAPI backend is also directly accessible at `http://localhost:8000`.

| Method | Path          | Description                          |
|--------|---------------|--------------------------------------|
| GET    | `/`           | Check the API is running             |
| GET    | `/health`     | Health check (used by Docker)        |
| GET    | `/countries`  | List countries with loaded data      |
| POST   | `/ask`        | Ask a question                       |

### Example request

```sh
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"country": "germany", "question": "What is the speed limit on the Autobahn?"}'
```

### Example response

```json
{
  "country": "germany",
  "answer": "There is no general speed limit on the Autobahn...",
  "normalized_query": null,
  "source_chunks": [
    {
      "chunk_index": 1,
      "page": "Page 12",
      "source_file": "handbook.pdf",
      "content": "..."
    }
  ]
}
```

Interactive API docs (Swagger UI) are available at `http://localhost:8000/docs`.

---

## Stopping the app

```sh
docker compose down
```

To also delete the stored vector database:

```sh
docker compose down -v
```

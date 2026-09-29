# Agentic AI RAG Chatbot

A document-grounded question-answering application for the *Agentic AI* eBook. The backend uses FastAPI and LangGraph, with OpenAI embeddings and chat completions and Pinecone vector search. A Streamlit chat interface is also included.

## How it works

During ingestion, the eBook PDF is split into overlapping text chunks, embedded with OpenAI's `text-embedding-3-small`, and stored in a cosine-similarity Pinecone index. At query time, LangGraph retrieves the top matching chunks. Questions below the configured relevance threshold are refused; otherwise, the model is prompted to answer only from the retrieved text.

The API returns the answer, retrieved source chunks with page numbers and similarity scores, and a confidence score calculated from the top three retrieved similarities. Refusals return a confidence score of `0.0`.

## Requirements

- Python 3.10 or newer
- An OpenAI API key with access to embeddings and chat completions, and sufficient API billing/credits
- A Pinecone API key
- Internet access for the initial PDF download and hosted APIs

OpenAI and Pinecone usage may incur charges. The Pinecone index is created by the ingestion command if it does not already exist.

## Setup

Run these commands from the repository root. A virtual environment is recommended to isolate dependencies.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies and create the local configuration file:

```bash
# Windows Command Prompt
.venv\Scripts\activate

# macOS / Linux (use this instead of the Windows activation command)
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set real credentials in `.env`. Never commit `.env` or put live credentials in `.env.example`.

```dotenv
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=agentic-ai-index
```

On Windows Command Prompt, copy the template with:

```bat
copy .env.example .env
```

On macOS/Linux, use:

```bash
cp .env.example .env
```

## Ingest the eBook

Run ingestion once after configuring credentials. The script downloads the source PDF to `data/Ebook-Agentic-AI.pdf` if it is not already present, creates the configured Pinecone index when needed, and uploads the document embeddings.

```bash
python -m src.ingestion
```

To delete and rebuild the configured index, use `--recreate`. This operation removes the existing index contents before recreating it.

```bash
python -m src.ingestion --recreate
```

If the automatic download fails, place the PDF at `data/Ebook-Agentic-AI.pdf` and rerun ingestion. Ingestion requires working OpenAI credentials and available API quota.

## Run the API

```bash
uvicorn app:app --reload
```

- Health check: <http://127.0.0.1:8000/health>
- Interactive API docs: <http://127.0.0.1:8000/docs>

The API initializes its LangGraph workflow and connects to the configured Pinecone index at startup. Make sure ingestion has populated that index before asking document questions.

## Run the Streamlit interface

In a separate terminal, from the repository root with the environment activated:

```bash
streamlit run streamlit_app.py
```

## Chat API example

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Agentic AI according to the eBook?"}'
```

Example response shape:

```json
{
  "answer": "Answer grounded in the retrieved eBook passages.",
  "retrieved_chunks": [
    {"text": "Retrieved passage...", "page": 3, "similarity": 0.61}
  ],
  "confidence_score": 0.58
}
```

## Sample queries

With the API running and the index populated, run the included query examples:

```bash
python tests_sample_queries.py
```

The script sends benchmark questions to the local API and writes its results to `sample_results.json`. One out-of-scope question is included to exercise the refusal behavior.

## Configuration

Retrieval and chunking defaults are defined in `src/config.py`.

| Setting | Purpose |
| --- | --- |
| `MIN_RELEVANCE` | Minimum best-match similarity required to generate an answer |
| `TOP_K` | Number of chunks retrieved for a question |
| `CHUNK_SIZE` | Maximum text chunk size used during ingestion |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks |

After changing chunk size or overlap, rerun ingestion with `--recreate` to rebuild the index.

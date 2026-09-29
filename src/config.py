import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

# Pinecone serverless (free tier)
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
LLM_MODEL = "gpt-4o-mini"

# Chunking / retrieval
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
MIN_RELEVANCE = 0.30   # below this best-match score, the bot refuses to answer

# Data
PDF_PATH = "data/Ebook-Agentic-AI.pdf"
PDF_DRIVE_ID = "15VLphKcY23_fpYxN62UEQRri_psRVfP9"

REFUSAL_MESSAGE = "I cannot answer based on the provided document."


def validate():
    missing = [k for k, v in {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "PINECONE_API_KEY": PINECONE_API_KEY,
    }.items() if not v]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}. Check your .env file."
        )

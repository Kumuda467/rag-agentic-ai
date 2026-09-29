import os
import sys
import time
import argparse

import gdown
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config


def download_pdf():
    if os.path.exists(config.PDF_PATH):
        print(f"[ok] PDF already exists at {config.PDF_PATH}")
        return
    os.makedirs(os.path.dirname(config.PDF_PATH), exist_ok=True)
    print("[..] Downloading PDF from Google Drive...")
    gdown.download(id=config.PDF_DRIVE_ID, output=config.PDF_PATH, quiet=False)
    if not os.path.exists(config.PDF_PATH):
        sys.exit("Download failed. Download the PDF manually and save it as data/Ebook-Agentic-AI.pdf")


def ensure_index(pc: Pinecone, recreate: bool = False) -> bool:
    """Create the index if needed. Returns True if it already existed."""
    name = config.PINECONE_INDEX_NAME
    exists = name in pc.list_indexes().names()

    if exists and recreate:
        print(f"[..] Deleting existing index '{name}'")
        pc.delete_index(name)
        exists = False

    if not exists:
        print(f"[..] Creating index '{name}' (dim={config.EMBEDDING_DIM}, cosine)")
        pc.create_index(
            name=name,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
    return exists


def run_ingestion(recreate: bool = False):
    config.validate()
    download_pdf()

    # 1. Load
    docs = PyPDFLoader(config.PDF_PATH).load()
    print(f"[ok] Loaded {len(docs)} pages")

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[ok] Created {len(chunks)} chunks")

    # 3. Index
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existed = ensure_index(pc, recreate=recreate)
    if existed:
        stats = pc.Index(config.PINECONE_INDEX_NAME).describe_index_stats()
        if stats.total_vector_count > 0:
            print(f"[skip] Index already has {stats.total_vector_count} vectors. "
                  f"Use --recreate to rebuild.")
            return

    # 4. Embed + upsert
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=config.PINECONE_INDEX_NAME,
    )
    print("[done] Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and rebuild the Pinecone index")
    args = parser.parse_args()
    run_ingestion(recreate=args.recreate)

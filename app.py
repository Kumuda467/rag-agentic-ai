from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import build_rag_graph

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["graph"] = build_rag_graph()
    yield
    state.clear()


app = FastAPI(title="Agentic AI RAG API", version="1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1,
                       examples=["What is Agentic AI according to the eBook?"])


class Chunk(BaseModel):
    text: str
    page: Optional[int] = None
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[Chunk]
    confidence_score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    try:
        result = state["graph"].invoke({
            "question": request.query.strip(),
            "context": [], "sources": [], "answer": "", "score": 0.0,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    chunks = [
        Chunk(text=t, page=s["page"], similarity=s["similarity"])
        for t, s in zip(result["context"], result["sources"])
    ]
    return QueryResponse(
        answer=result["answer"],
        retrieved_chunks=chunks,
        confidence_score=result["score"],
    )

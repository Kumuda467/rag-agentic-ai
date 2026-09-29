from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config


class AgentState(TypedDict):
    question: str
    context: List[str]       # retrieved chunk texts
    sources: List[dict]      # page + per-chunk similarity score
    answer: str
    score: float             # overall confidence (0-1)


def build_rag_graph():
    config.validate()
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME, embedding=embeddings
    )
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)

    # ---------- Nodes ----------
    def retrieve_node(state: AgentState):
        results = vectorstore.similarity_search_with_score(
            state["question"], k=config.TOP_K
        )
        context, sources = [], []
        for doc, sim in results:
            context.append(doc.page_content)
            sources.append({
                "page": doc.metadata.get("page"),
                "similarity": round(float(sim), 4),
            })
        return {"context": context, "sources": sources}

    def generate_node(state: AgentState):
        context_str = "\n\n---\n\n".join(state["context"])
        prompt = f"""You are a strict question-answering assistant.
Answer the question using ONLY the context below. Do not use outside knowledge.
If the context does not contain enough information, reply exactly:
"{config.REFUSAL_MESSAGE}"

Context:
{context_str}

Question: {state['question']}

Answer:"""
        response = llm.invoke(prompt).content.strip()

        if config.REFUSAL_MESSAGE.lower() in response.lower():
            return {"answer": config.REFUSAL_MESSAGE, "score": 0.0}

        # Confidence = mean similarity of the top-3 retrieved chunks
        top = sorted((s["similarity"] for s in state["sources"]), reverse=True)[:3]
        confidence = round(sum(top) / len(top), 2) if top else 0.0
        return {"answer": response, "score": max(0.0, min(1.0, confidence))}

    def refuse_node(state: AgentState):
        return {"answer": config.REFUSAL_MESSAGE, "score": 0.0}

    # ---------- Routing ----------
    def route_after_retrieval(state: AgentState) -> str:
        best = max((s["similarity"] for s in state["sources"]), default=0.0)
        return "generate" if best >= config.MIN_RELEVANCE else "refuse"

    # ---------- Graph ----------
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("refuse", refuse_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_conditional_edges(
        "retrieve", route_after_retrieval,
        {"generate": "generate", "refuse": "refuse"},
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("refuse", END)

    return workflow.compile()

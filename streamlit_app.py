import streamlit as st
from src.graph import build_rag_graph

st.set_page_config(page_title="Agentic AI RAG Chatbot", layout="wide")


@st.cache_resource
def get_graph():
    return build_rag_graph()


graph = get_graph()
st.title("📘 Agentic AI eBook Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask a question about the Agentic AI eBook...")
if query:
    result = graph.invoke({"question": query, "context": [], "sources": [],
                           "answer": "", "score": 0.0})
    st.session_state.history.append((query, result))

for q, r in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(r["answer"])

if st.session_state.history:
    _, last = st.session_state.history[-1]
    with st.sidebar:
        st.metric("Confidence score", last["score"])
        st.subheader("Retrieved chunks")
        for t, s in zip(last["context"], last["sources"]):
            with st.expander(f"Page {s['page']} | similarity {s['similarity']}"):
                st.write(t)

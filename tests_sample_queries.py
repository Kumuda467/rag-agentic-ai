import json
import requests

URL = "http://127.0.0.1:8000/chat"

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",   # validation: should be refused
]

results = []
for q in QUERIES:
    print("=" * 80)
    print("Q:", q)
    r = requests.post(URL, json={"query": q}, timeout=60)
    r.raise_for_status()
    data = r.json()
    print("A:", data["answer"])
    print("Confidence:", data["confidence_score"])
    print("Chunks retrieved:", len(data["retrieved_chunks"]),
          "| pages:", [c["page"] for c in data["retrieved_chunks"]])
    results.append({"query": q, **data})

with open("sample_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nSaved full output to sample_results.json")

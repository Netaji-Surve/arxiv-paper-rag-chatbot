import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.retrieval.hybrid_retriever import build_hybrid_retriever

retriever = build_hybrid_retriever()

query = "How does RAG work with large language models?"
results = retriever.invoke(query)

print(f"\nQuery: {query}")
print(f"Results: {len(results)}\n")

for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Title  : {doc.metadata['title']}")
    print(f"Paper  : {doc.metadata['arxiv_id']}")
    print(f"Page   : {doc.metadata['page']}")
    print(f"Chunk  : {doc.page_content[:200]}...")
    print()

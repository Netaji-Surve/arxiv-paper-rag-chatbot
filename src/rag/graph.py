import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from langgraph.graph import StateGraph, START, END
from src.rag.state import RAGState
from src.rag.nodes import rewrite_query, retrieve, grade_documents, rerank, generate


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    #graph.add_node("grade_documents", grade_documents)
    graph.add_node("rerank", rerank)
    graph.add_node("generate", generate)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    #graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)

    return graph.compile()



def execute_workflow(query):
    workflow = build_rag_graph()
    result = workflow.invoke({
        "query": query,
        "rewritten_query": "",
        "documents": [],
        "answer": "",
        "citation": [],
    })
    #print("\nAnswer:", result["answer"])
    #print("\nCitations:")
    #for c in result["citation"]:
    #    print(f"  - [{c['arxiv_id']}] {c['title']}")
    print("Final State: ", result)
    return result
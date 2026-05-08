from typing import TypedDict
from langchain_core.documents import Document

class RAGState(TypedDict):
    query: str
    rewritten_query: str
    documents: list[Document]
    answer: str
    citation: list[dict]
    
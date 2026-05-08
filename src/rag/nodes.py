from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from src.rag.state import RAGState
from src.rag.llm import get_llm
from src.retrieval.hybrid_retriever import get_hybrid_retriever


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at reformulating search queries for scientific paper retrieval.

Rewrite the user query to improve retrieval from an AI/ML research paper vector database.
- Expand abbreviations (RAG → retrieval augmented generation, LLM → large language model)
- Add relevant technical synonyms
- Keep it specific and focused
- Return ONLY the rewritten query as a string, NO explanation or NO any other note """),
    ("human", "Original query: {query}"),
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are grading whether a retrieved document chunk is relevant to a user query.

Respond with 'yes' if the chunk contains information useful for answering the query.
Respond with 'no' if the chunk is unrelated, a bibliography entry, or generic filler.
Reply with only 'yes' or 'no'."""),
    ("human", "Query: {query}\n\nDocument chunk:\n{document}"),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI research assistant answering questions about AI/ML papers.

Use ONLY the provided context to answer. If the context does not contain enough information, say so clearly.
Do not hallucinate or use knowledge outside the provided context.
Be concise and precise.
When referencing information, cite the source using the exact paper title from the context in square brackets e.g. [Paper Title].
Only cite titles that appear in the context — do not invent or guess paper titles."""),
    ("human", """Context:
{context}

Question: {query}

Answer:"""),
])


# --------------------------------------------------------------------------- #
# Structured output schema for grading
# --------------------------------------------------------------------------- #

class GradeResult(BaseModel):
    relevant: str = Field(description="Is the document relevant? Answer 'yes' or 'no'")


# --------------------------------------------------------------------------- #
# Module-level singletons
# --------------------------------------------------------------------------- #

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def rewrite_query(state: RAGState) -> dict:
    chain = REWRITE_PROMPT | get_llm()
    response = chain.invoke({"query": state["query"]})
    return {"rewritten_query": response.content.strip()}


def retrieve(state: RAGState):
    rewritten_query = state['rewritten_query']
    retriever = get_hybrid_retriever()
    documents = retriever.invoke(rewritten_query)
    return {"documents": documents}


def grade_documents(state: RAGState) -> dict:
    grader = GRADE_PROMPT | get_llm().with_structured_output(GradeResult)
    relevant_docs = []
    for doc in state["documents"]:
        result = grader.invoke({
            "query": state["rewritten_query"],
            "document": doc.page_content,
        })
        if result.relevant.lower() == "yes":
            relevant_docs.append(doc)
    return {"documents": relevant_docs}


def rerank(state: RAGState) -> dict:
    if not state["documents"]:
        return {"documents": []}

    query = state["rewritten_query"]
    pairs = [(query, doc.page_content) for doc in state["documents"]]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(
        zip(state["documents"], scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return {"documents": [doc for doc, _ in ranked[:5]]}


def generate(state: RAGState) -> dict:
    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('title', 'Unknown')}]\n{doc.page_content}"
        for doc in state["documents"]
    )
    chain = GENERATE_PROMPT | get_llm()
    response = chain.invoke({"context": context, "query": state["query"]})

    citations = [
        {
            "arxiv_id": doc.metadata.get("arxiv_id"),
            "title": doc.metadata.get("title"),
            "authors": doc.metadata.get("authors"),
            "page": doc.metadata.get("page"),
        }
        for doc in state["documents"]
    ]
    seen = set()
    unique_citations = []
    for c in citations:
        if c["arxiv_id"] not in seen:
            seen.add(c["arxiv_id"])
            unique_citations.append(c)

    return {"answer": response.content.strip(), "citation": unique_citations}

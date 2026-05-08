"""
Generates a retrieval evaluation dataset from existing chunks.

Strategy:
  - 40 single-chunk questions: 1 chunk → LLM generates a question → that chunk is the relevant doc (grade 2)
  - 10 multi-chunk questions : 2-3 chunks from same paper → LLM generates a question needing all of them
                               primary chunk: grade 2, secondary chunks: grade 1

Output: tests/eval_dataset.json
"""

import sys
import os
import json
import random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import ChatPromptTemplate
from src.rag.llm import llm


CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "../data/chunks.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
MIN_CHUNK_LEN = 300
SINGLE_COUNT  = 40
MULTI_COUNT   = 10
RANDOM_SEED   = 42


SINGLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You generate evaluation questions for a RAG system.
Given a passage from an AI/ML research paper, generate ONE specific question that:
- Can be answered using ONLY this passage
- Is specific enough that a different passage would not answer it
- Asks about a concept, method, or finding in the text
Return ONLY the question, nothing else."""),
    ("human", "Passage:\n{chunk_text}\n\nQuestion:"),
])

MULTI_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You generate evaluation questions for a RAG system.
Given multiple passages from the same research paper, generate ONE question that:
- Requires information from ALL the passages to answer fully
- Is specific to the paper's content
- Cannot be answered from just one passage alone
Return ONLY the question, nothing else."""),
    ("human", "{passages}\n\nQuestion:"),
])


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)
    return [c for c in chunks if len(c["chunk_text"]) >= MIN_CHUNK_LEN]


def generate_single_questions(chunks: list[dict], n: int) -> list[dict]:
    random.seed(RANDOM_SEED)
    selected = random.sample(chunks, n)
    chain = SINGLE_PROMPT | llm
    results = []

    for i, chunk in enumerate(selected, 1):
        print(f"  [single {i}/{n}] {chunk['chunk_id']}")
        response = chain.invoke({"chunk_text": chunk["chunk_text"][:1500]})
        question = response.content.strip()

        results.append({
            "question": question,
            "relevant_chunks": [
                {"chunk_id": chunk["chunk_id"], "grade": 2}
            ],
            "source_title": chunk["title"],
        })

    return results


def generate_multi_questions(chunks: list[dict], n: int) -> list[dict]:
    random.seed(RANDOM_SEED + 1)

    # Group chunks by arxiv_id, keep papers with enough chunks
    by_paper: dict[str, list] = {}
    for c in chunks:
        by_paper.setdefault(c["arxiv_id"], []).append(c)
    eligible = [v for v in by_paper.values() if len(v) >= 5]

    selected_papers = random.sample(eligible, min(n, len(eligible)))
    chain = MULTI_PROMPT | llm
    results = []

    for i, paper_chunks in enumerate(selected_papers, 1):
        group = random.sample(paper_chunks, 3)
        passages = "\n\n---\n\n".join(
            f"Passage {j+1}:\n{c['chunk_text'][:800]}"
            for j, c in enumerate(group)
        )
        print(f"  [multi  {i}/{n}] paper={group[0]['arxiv_id']}")
        response = chain.invoke({"passages": passages})
        question = response.content.strip()

        results.append({
            "question": question,
            "relevant_chunks": [
                {"chunk_id": group[0]["chunk_id"], "grade": 2},
                {"chunk_id": group[1]["chunk_id"], "grade": 1},
                {"chunk_id": group[2]["chunk_id"], "grade": 1},
            ],
            "source_title": group[0]["title"],
        })

    return results


def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks with length >= {MIN_CHUNK_LEN} chars\n")

    print(f"Generating {SINGLE_COUNT} single-chunk questions...")
    single = generate_single_questions(chunks, SINGLE_COUNT)

    print(f"\nGenerating {MULTI_COUNT} multi-chunk questions...")
    multi = generate_multi_questions(chunks, MULTI_COUNT)

    dataset = single + multi
    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDone. Saved {len(dataset)} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

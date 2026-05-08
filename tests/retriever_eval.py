"""
Retriever evaluation using the synthetic dataset from generate_eval_dataset.py.

Metrics:
  - Hit Rate : 1 if any relevant chunk appears in top-k results
  - MRR      : 1 / rank of first relevant chunk
  - NDCG@k   : ranking quality weighted by relevance grades
"""

import sys
import os
import json
import math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../"))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.hybrid_retriever import get_hybrid_retriever


EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
TOP_K = 5


def hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg(retrieved_ids: list[str], grade_map: dict[str, int]) -> float:
    dcg  = sum(grade_map.get(rid, 0) / math.log2(rank + 1)
               for rank, rid in enumerate(retrieved_ids, start=1))
    ideal_grades = sorted(grade_map.values(), reverse=True)[:TOP_K]
    idcg = sum(g / math.log2(rank + 1)
               for rank, g in enumerate(ideal_grades, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    with open(EVAL_DATASET_PATH) as f:
        dataset = json.load(f)

    retriever = get_hybrid_retriever()
    all_hr, all_mrr, all_ndcg = [], [], []

    for i, item in enumerate(dataset, 1):
        question   = item["question"]
        grade_map  = {c["chunk_id"]: c["grade"] for c in item["relevant_chunks"]}
        relevant_ids = set(grade_map.keys())

        retrieved_docs = retriever.invoke(question)[:TOP_K]
        retrieved_ids  = [doc.metadata.get("chunk_id", "") for doc in retrieved_docs]

        hr = hit_rate(retrieved_ids, relevant_ids)
        rr = mrr(retrieved_ids, relevant_ids)
        ng = ndcg(retrieved_ids, grade_map)

        all_hr.append(hr)
        all_mrr.append(rr)
        all_ndcg.append(ng)

        print(f"[{i:02d}/{len(dataset)}] hr={hr:.0f}  mrr={rr:.2f}  ndcg={ng:.3f}  | {question[:60]}...")

    print("\n=== Retriever Evaluation Results ===")
    print(f"Hit Rate  : {np.mean(all_hr):.3f}")
    print(f"MRR       : {np.mean(all_mrr):.3f}")
    print(f"NDCG@{TOP_K}   : {np.mean(all_ndcg):.3f}")


if __name__ == "__main__":
    main()

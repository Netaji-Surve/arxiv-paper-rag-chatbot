# This is the backup file created for learning purpose
import json
import os
import pickle
from rank_bm25 import BM25Okapi


BM25_INDEX_PATH = "./data/bm25_index.pkl"


def build_bm25_index(
    chunks_path: str = "./data/chunks.json",
    index_path: str = BM25_INDEX_PATH,
) -> tuple[BM25Okapi, list[dict]]:
    chunks = _load_chunks(chunks_path)
    tokenized = [_tokenize(c["chunk_text"]) for c in chunks]

    index = BM25Okapi(tokenized)

    _save_index(index, chunks, index_path)
    print(f"BM25 index built: {len(chunks)} chunks → {index_path}")
    return index, chunks


def load_bm25_index(
    index_path: str = BM25_INDEX_PATH,
) -> tuple[BM25Okapi, list[dict]]:
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"BM25 index not found at {index_path}. Run build_bm25_index() first.")
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return data["index"], data["chunks"]


def search_bm25(
    query: str,
    index: BM25Okapi,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    tokens = _tokenize(query)
    scores = index.get_scores(tokens)

    # pair each chunk with its score and sort descending
    scored = sorted(
        enumerate(scores), key=lambda x: x[1], reverse=True
    )[:top_k]

    results = []
    for rank, (idx, score) in enumerate(scored):
        results.append({
            **chunks[idx],
            "bm25_score": float(score),
            "bm25_rank": rank + 1,
        })
    return results


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _load_chunks(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks file not found: {path}")
    with open(path) as f:
        return json.load(f)


def _save_index(index: BM25Okapi, chunks: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"index": index, "chunks": chunks}, f)

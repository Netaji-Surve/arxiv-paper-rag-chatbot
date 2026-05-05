import json
import os
import pickle
from rank_bm25 import BM25Okapi

BM_25_INDEX_PATH = "../../data/bm_25_index.pkl"

def build_bm25_index(chunks_path: str='../../data/chunks.json',
                     index_path: str= BM_25_INDEX_PATH) -> tuple[BM25Okapi, list[dict]]:
     
     # load chunks
    loaded_chunks = _load_chunks(chunk_path=chunks_path)
    # tokenize
    tokenized_chunk_text = [_tokenize(c['chunk_text']) for c in loaded_chunks]

    index = BM25Okapi(tokenized_chunk_text)
    print(len(index.get_scores('transformers')))
    _save_index(index=index, chunks=loaded_chunks, path=index_path)
    print(f"BM25 index built: {len(loaded_chunks)} chunks → {index_path}")
    return index, loaded_chunks


def _load_chunks(chunk_path) -> list[dict]:
    loaded_chunks = []
    try:
        with open(chunk_path, 'r') as f:
            loaded_chunks = json.load(f)
        return loaded_chunks
    except Exception as e:
        print(f"Exception while loading chunks: {e}")
        raise RuntimeError(f"Exception while loading chunks: {e}")

def _save_index(index: BM25Okapi, chunks: list[dict], path:str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'index': index, 'chunks': chunks}, f)
    

def _tokenize(text: str):
    return text.lower().split()

build_bm25_index()
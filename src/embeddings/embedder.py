import chromadb
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

COLLECTION_NAME = 'arxiv_papers'

def embed_and_index(chunks_path: str='./data/chunks.json',
                    chroma_dir: str='./data/vectorstore',
                    embedding_model_name: str='sentence-transformers/all-MiniLM-L6-v2',
                    batch_size: int = 100) -> chromadb.Collection:
    # load chunks
    loaded_chunks = load_chunks(chunk_path=chunks_path)

    # embedding model
    embedding_model = SentenceTransformerEmbeddingFunction(model_name = embedding_model_name)

    # init chromadb
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name=COLLECTION_NAME,
                                    embedding_function=embedding_model,
                                    metadata={"hnsw:space": "cosine"})

    existing_ids = set(collection.get()['ids'])

    new_chunks = [c for c in loaded_chunks if c['chunk_id'] not in existing_ids]

    if not new_chunks:
        print(f'all the chunks are already indexed. skipping indexing process.')
        return collection
    
    for i in range (0, len(new_chunks), batch_size):
        chunk_batch = new_chunks[i: i+batch_size]
        collection.add(

            ids=[c['chunk_id'] for c in chunk_batch],

            metadatas=[
                {
                    "arxiv_id": c["arxiv_id"],
                    "title": c["title"],
                    "authors": ", ".join(c["authors"]),
                    "published": c["published"],
                    "page": c["page"],
                    "chunk_index": c["chunk_index"]
                }
                for c in chunk_batch ],

            documents=[c['chunk_text'] for c in chunk_batch]
            )
        
    print(f"total indexed chunks: {collection.count()}")
    return collection

def load_chunks(chunk_path) -> list[dict]:
    loaded_chunks = []
    try:
        with open(chunk_path, 'r') as f:
            loaded_chunks = json.load(f)
        return loaded_chunks
    except Exception as e:
        print(f"Exception while loading chunks: {e}")
        raise RuntimeError(f"Exception while loading chunks: {e}")
    


         
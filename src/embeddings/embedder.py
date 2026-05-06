import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


COLLECTION_NAME = "arxiv_paper"


def embed_and_index(
    chunks_path: str = "./data/chunks.json",
    chroma_dir: str = "./data/vectorstore",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 100,
) -> Chroma:
    loaded_chunks = load_chunks(chunks_path)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

    existing_ids = set(vectorstore.get()["ids"])
    new_chunks = [c for c in loaded_chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("All chunks already indexed. Skipping.")
        return vectorstore

    print(f"Indexing {len(new_chunks)} new chunks (skipping {len(existing_ids)} existing)...")

    docs = [
        Document(
            page_content=c["chunk_text"],
            metadata={
                "arxiv_id": c["arxiv_id"],
                "title": c["title"],
                "authors": ", ".join(c["authors"]),
                "published": c["published"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
            },
        )
        for c in new_chunks
    ]
    ids = [c["chunk_id"] for c in new_chunks]

    for i in range(0, len(docs), batch_size):
        vectorstore.add_documents(docs[i : i + batch_size], ids=ids[i : i + batch_size])

    print(f"Total indexed: {vectorstore._collection.count()}")
    return vectorstore


def load_chunks(chunk_path: str) -> list[dict]:
    try:
        with open(chunk_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Exception while loading chunks: {e}")

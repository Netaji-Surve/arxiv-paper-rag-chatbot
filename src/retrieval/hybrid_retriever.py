import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

COLLECTION_NAME = "arxiv_paper"


def build_hybrid_retriever(
        chunk_path: str = "./data/chunks.json",
        chroma_dir: str = "./data/vectorstore",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
        weights: list[float] = [0.5, 0.5]
        ):
    chunks = _load_chunks(chunk_path)

    docs = _chunks_to_documents(chunks)
    dense_retriever = _build_dense_retriever(chroma_dir, embedding_model_name, top_k)
    sparse_retriever = _build_sparse_retriever(docs=docs, top_k=top_k)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=weights
    )
    return hybrid_retriever

def _load_chunks(chunk_path) -> list[dict]:
    try:
        with open(chunk_path, 'r') as f:
            chunks = json.load(f)
            return chunks
    except Exception as e:
        print(f'Error while loading chunks {e}')
        raise RuntimeError(f'Error while loading chunks {e}')

def _build_dense_retriever(chroma_dir: str, embedding_model_name: str, top_k = 5):
    embeddings = HuggingFaceEmbeddings(model_name = embedding_model_name)
    vectorstore = Chroma(collection_name=COLLECTION_NAME,
                          embedding_function=embeddings,
                          persist_directory=chroma_dir)
    return vectorstore.as_retriever(search_kwargs = {"k": top_k})

def _build_sparse_retriever(docs: list[Document], top_k: int) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = top_k
    return retriever

def _chunks_to_documents(chunks):
    docs = [ Document(page_content=c['chunk_text'], 
                      metadata = {
                "arxiv_id": c["arxiv_id"],
                "title": c["title"],
                "authors": ", ".join(c["authors"]),
                "published": c["published"],
                "page": c["page"],
                "chunk_id": c["chunk_id"],
            },) for c in chunks]
    return docs
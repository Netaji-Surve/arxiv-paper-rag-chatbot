"""Run ArXiv ingestion + PDF parsing. Usage: python ingest.py"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.arxiv_fetcher import fetch_papers, SEARCH_TOPICS
from src.ingestion.pdf_parser import parse_pdfs
from src.ingestion.chunker import chunk_papers
from src.embeddings.embedder import embed_and_index

if __name__ == "__main__":
    papers = fetch_papers(
        topics=SEARCH_TOPICS,
        max_paper_per_topic=5,
        pdf_dir="./data/pdfs",
        metadata_path="./data/metadata.json",
    )
    print(f"\nFetched {len(papers)} new papers.")

    parsed = parse_pdfs(
        metadata_path="./data/metadata.json",
        output_path="./data/parsed_papers.json",
    )
    print(f"Parsed {len(parsed)} papers total.")

    chunks = chunk_papers(parsed_path="./data/parsed_papers.json",
                 output_path="./data/chunks.json",
                 chunk_size=1000,
                 chunk_overlap=200)

    print(f"{len(chunks)} chunks created created total.")


    collection = embed_and_index(chunks_path="./data/chunks.json",
                    chroma_dir="./data/vectorstore",
                    batch_size=100,
                    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")
import arxiv
import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    categories: list[str]
    pdf_url: str
    pdf_path: Optional[str] = None


SEARCH_TOPICS = [
    "large language models",
    "transformer architecture",
    "diffusion models image generation",
    "instruction tuning fine-tuning LLM",
    "reinforcement learning from human feedback",
    "vision language models",
    "chain of thought reasoning",
]


def fetch_papers(
    topics: list[str] = SEARCH_TOPICS,
    max_paper_per_topic: int = 5,
    pdf_dir: str = "./data/pdfs",
    metadata_path: str = "./data/metadata.json",
) -> list[PaperMetadata]:
    Path(pdf_dir).mkdir(parents=True, exist_ok=True)

    existing = _load_existing_metadata(metadata_path)
    existing_ids = {p["arxiv_id"] for p in existing}

    client = arxiv.Client(page_size=max_paper_per_topic, delay_seconds=1.0)
    all_papers: list[PaperMetadata] = []

    for topic in topics:
        print(f"\nSearching: '{topic}'")
        search = arxiv.Search(
            query=topic,
            max_results=max_paper_per_topic,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = list(client.results(search))
        for result in tqdm(results, desc=f"  Downloading", unit="paper"):
            arxiv_id = result.entry_id.split("/")[-1]

            if arxiv_id in existing_ids:
                print(f"  Skipping {arxiv_id} (already downloaded)")
                continue

            pdf_path = _download_pdf(result, pdf_dir)

            paper = PaperMetadata(
                arxiv_id=arxiv_id,
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary.replace("\n", " "),
                published=result.published.strftime("%Y-%m-%d"),
                categories=result.categories,
                pdf_url=result.pdf_url,
                pdf_path=pdf_path,
            )
            all_papers.append(paper)
            existing_ids.add(arxiv_id)

        time.sleep(1)

    _save_metadata(existing + [asdict(p) for p in all_papers], metadata_path)
    print(f"\nFetched {len(all_papers)} new papers. Total: {len(existing) + len(all_papers)}")
    return all_papers


def _download_pdf(result: arxiv.Result, pdf_dir: str) -> Optional[str]:
    arxiv_id = result.entry_id.split("/")[-1]
    safe_id = arxiv_id.replace("/", "_")
    pdf_path = os.path.join(pdf_dir, f"{safe_id}.pdf")

    if os.path.exists(pdf_path):
        return pdf_path

    try:
        result.download_pdf(dirpath=pdf_dir, filename=f"{safe_id}.pdf")
        return pdf_path
    except Exception as e:
        print(f"  Failed to download {arxiv_id}: {e}")
        return None


def _load_existing_metadata(path: str) -> list[dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_metadata(papers: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(papers, f, indent=2)

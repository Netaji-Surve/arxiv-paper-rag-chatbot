import fitz
import json
import os
from pathlib import Path
from tqdm import tqdm


def parse_pdfs(
    metadata_path: str = "./data/metadata.json",
    output_path: str = "./data/parsed_papers.json",
) -> list[dict]:
    papers = _load_metadata(metadata_path)
    parsed = []

    for paper in tqdm(papers, desc="Parsing PDFs", unit="paper"):
        pdf_path = paper.get("pdf_path")
        if not pdf_path or not os.path.exists(pdf_path):
            print(f"  Skipping {paper['arxiv_id']} — PDF not found")
            continue

        pages = _extract_text(pdf_path)
        if not pages:
            print(f"  Skipping {paper['arxiv_id']} — no text extracted")
            continue

        parsed.append({
            "arxiv_id": paper["arxiv_id"],
            "title": paper["title"],
            "authors": paper["authors"],
            "published": paper["published"],
            "categories": paper["categories"],
            "abstract": paper["abstract"],
            "pages": pages,
            "total_pages": len(pages),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"\nParsed {len(parsed)}/{len(papers)} papers → {output_path}")
    return parsed


def _load_metadata(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _extract_text(pdf_path: str) -> list[dict]:
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = _clean(text)
            if len(text) > 100:  # skip near-blank pages
                pages.append({"page": page_num, "text": text})
        doc.close()
    except Exception as e:
        print(f"  Error reading {pdf_path}: {e}")
    return pages


def _clean(text: str) -> str:
    # fix hyphenated line breaks (e.g. "trans-\nformer" → "transformer")
    text = text.replace("-\n", "")
    # collapse multiple blank lines
    lines = [l for l in text.splitlines() if l.strip()]
    return " ".join(lines)

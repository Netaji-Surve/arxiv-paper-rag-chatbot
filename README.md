# ArXiv Paper RAG Chatbot

A production-style Retrieval Augmented Generation (RAG) chatbot that answers questions over **40 AI/ML research papers** fetched from ArXiv. Built with LangChain, LangGraph, hybrid search (BM25 + ChromaDB), CrossEncoder reranking, and a Chainlit UI.

---

## Architecture Diagram

> [View interactive diagram on Excalidraw](https://excalidraw.com/#json=V-U42YWhOuwuiX-hrGAAB,cAXRw5OmEwciuGXvZbE6mA)

The system is split into two phases: an **offline ingestion pipeline** that builds the search indexes, and a **runtime RAG pipeline** powered by LangGraph that answers user queries.

---

## Process & Workflow

### Phase 1 — Offline Data Ingestion

```
ArXiv API  →  40 PDFs  →  PyMuPDF Parser  →  Chunker  →  HuggingFace Embedder
                                                               ↙           ↘
                                                        ChromaDB       BM25 Index
                                                      (Dense Index)  (Sparse Index)
```

| Step | Tool | Detail |
|------|------|--------|
| **Fetch papers** | ArXiv API | 40 papers across 8 AI/ML topics (RAG, RLHF, diffusion models, LoRA, chain-of-thought, transformers, instruction tuning, evaluation) |
| **Parse PDFs** | PyMuPDF (LangChain wrapper) | Extracts text page-by-page with light cleaning |
| **Chunk text** | RecursiveCharacterTextSplitter | `chunk_size=1000`, `overlap=200` → **3893 chunks** |
| **Embed chunks** | `sentence-transformers/all-MiniLM-L6-v2` | Dense vectors stored in ChromaDB |
| **Build BM25 index** | `rank_bm25` | Sparse keyword index built at retriever load time |

---

### Phase 2 — LangGraph RAG Pipeline (Runtime)

```
User Query
    │
    ▼
[Node 1] Query Rewriter   ─── LLM expands abbreviations, adds synonyms
    │
    ▼
[Node 2] Hybrid Retriever ─── BM25 (0.5) + ChromaDB (0.5) via EnsembleRetriever (RRF fusion)
    │                          top-k = 5 from each, fused to top-10
    ▼
[Node 3] CrossEncoder Reranker ─── ms-marco-MiniLM-L-6-v2 scores all pairs → keeps top-5
    │
    ▼
[Node 4] LLM Generator    ─── Generates grounded answer with inline citations
    │
    ▼
Answer + Citations
```

Each node is a pure function operating on `RAGState` (a TypedDict). LangGraph manages the execution graph and state transitions.

#### Node Details

**Query Rewriter**
- Prompt instructs the LLM to expand abbreviations (RAG → retrieval augmented generation), add technical synonyms, and keep the query focused
- Improves recall for domain-specific queries

**Hybrid Retriever**
- `EnsembleRetriever` combines BM25 and ChromaDB with equal weights `[0.5, 0.5]`
- BM25 catches exact keyword matches (paper titles, author names, specific terms)
- ChromaDB catches semantic similarity (paraphrased questions, concept-level queries)
- Reciprocal Rank Fusion (RRF) merges the two ranked lists

**CrossEncoder Reranker**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` initialized once at module load (not per call)
- Scores `(query, chunk)` pairs together — more accurate than bi-encoder similarity
- Re-ranks top-10 retrieved chunks, keeps top-5 for generation

**LLM Generator**
- Formats top-5 chunks as structured context with paper titles
- Prompt instructs LLM to cite only titles present in context
- Returns answer + deduplicated citation list (arxiv_id, title, page)

---

### UI Layer — Chainlit

- Nested step display: `Processing > Rewriting query > Retrieving > Generating`
- HTML-rendered answer with styled citation cards
- LangSmith tracing enabled for every run

---

## Evaluation

### Concepts

RAG evaluation requires measuring two separate components:

| Component | What to measure | How |
|-----------|-----------------|-----|
| **Generator** | Does the answer faithfully reflect the retrieved context? | Faithfulness |
| **Generator** | Does the answer actually address the question? | Answer Relevancy |
| **Retriever** | Were the retrieved chunks relevant to the question? | Context Precision |
| **Retriever** | Did the retrieval cover enough of the needed information? | Context Recall |

Additionally, classic IR metrics measure **retrieval ranking quality**:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Hit Rate** | 1 if any relevant chunk in top-k | Did the retriever find anything useful? |
| **MRR** | `1 / rank` of first relevant chunk | How high up was the first useful result? |
| **NDCG@k** | `DCG / ideal DCG` with graded relevance | Overall ranking quality, penalises relevant docs buried low |

---

### Evaluation Setup

**RAGAS evaluation** (`tests/evaluate.py`)
- 15 hand-crafted questions with reference answers across all 8 paper topics
- Judge LLM: local `meta-llama-3-8b-instruct` via LM Studio
- Judge embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Sequential requests (`max_workers=1`) to avoid overwhelming the local LLM

**Retriever evaluation** (`tests/retriever_eval.py` + `tests/generate_eval_dataset.py`)
- **Synthetic dataset**: 40 single-chunk + 10 multi-chunk questions generated by LLM from corpus chunks
  - Single-chunk: 1 chunk → LLM generates a specific question → that `chunk_id` is the ground truth (grade 2)
  - Multi-chunk: 2–3 chunks from same paper → LLM generates a question needing all of them → graded relevance (primary=2, secondary=1)
- Retriever run at `top_k=5`
- NDCG uses graded relevance to reward finding higher-grade chunks at top ranks

---

### Results

#### RAGAS — Generator & Retriever Quality (15 questions, local LLM judge)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Faithfulness** | 0.6006 | ~60% of answer claims are directly supported by retrieved context. Indicates some hallucination or over-inference from the local 8B model |
| **Answer Relevancy** | 0.5082 | ~50% relevancy score. Some answers drift off-topic or are too generic — partially a limitation of the small judge LLM |
| **Context Precision** | 0.6552 | ~66% of retrieved chunks are relevant to the question. The hybrid retriever is pulling mostly on-topic content |
| **Context Recall** | 0.4840 | ~48% of the information needed to answer is covered by retrieved chunks. Retrieval coverage has room to improve |

> **Note:** All LLM-judge metrics are sensitive to the quality of the judge model. Scores measured with a local 8B model will be lower than with GPT-4 as judge — this is expected and consistent across all metrics.

#### IR Metrics — Retriever Ranking Quality (50 synthetic questions, exact chunk match)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Hit Rate** | 0.380 | 38% of queries return the exact source chunk in top-5 |
| **MRR** | 0.151 | First relevant chunk appears around rank 6–7 on average |
| **NDCG@5** | 0.199 | Overall ranking quality is low under exact-match criterion |

> **Important context:** These IR metrics use **exact chunk ID matching** — the retriever must return the specific chunk the question was generated from. This is a very strict criterion. The gap between Context Precision (0.66) and Hit Rate (0.38) reveals that the retriever *is* finding relevant content — just not always the *exact* source chunk. Two reasons: (1) overlapping chunks (overlap=200) mean adjacent chunks also contain the answer; (2) LLM-generated questions are sometimes abstract enough to match multiple chunks across papers. LLM-judged metrics (RAGAS) are more appropriate for RAG evaluation than exact-match IR metrics.

#### Key Observations

- **Context Precision (0.66) > Hit Rate (0.38)**: The retriever finds relevant content but not always the exact source chunk — typical with overlapping chunks and multi-chunk information spread.
- **Faithfulness (0.60)**: The generator mostly stays grounded but occasionally extends beyond what the context provides — common with smaller instruction-tuned models.
- **Answer Relevancy (0.51)**: Lower than expected. Investigation shows 3 questions with 0.0 scores (self-attention, chain-of-thought, RAG hallucinations) — likely caused by the judge LLM failing on those specific responses rather than actual bad answers.
- **Context Recall (0.48)**: Some questions require information spread across distant chunks that the top-5 window misses entirely.

#### Potential Improvements

| Issue | Fix |
|-------|-----|
| Low context recall | Increase `top_k`, tune chunk size, or add parent-document retrieval |
| Low faithfulness | Add a grading/filtering node before generation, or use a stronger generator |
| Retrieval coverage | Re-enable `grade_documents` node to filter irrelevant chunks before reranking |
| Low IR metrics | Use soft-match (embedding similarity) instead of exact chunk ID matching |
| Evaluation accuracy | Use GPT-4o-mini as judge LLM for more reliable RAGAS scores |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | `meta-llama-3-8b-instruct` via LM Studio (local) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense store | ChromaDB |
| Sparse search | BM25 (`rank_bm25`) |
| Retrieval fusion | LangChain `EnsembleRetriever` (RRF) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Pipeline orchestration | LangGraph |
| LLM framework | LangChain |
| UI | Chainlit |
| Monitoring | LangSmith |
| Evaluation | RAGAS |

---

## Project Structure

```
ArXiv-paper-rag-chatbot/
├── data/
│   ├── pdfs/               # Downloaded ArXiv PDFs
│   ├── parsed_papers.json  # Extracted text per page
│   ├── chunks.json         # 3893 text chunks with metadata
│   └── metadata.json       # Paper metadata (title, authors, arxiv_id)
├── src/
│   ├── ingestion/
│   │   ├── arxiv_fetcher.py    # Fetch & download papers
│   │   ├── pdf_parser.py       # PyMuPDF parsing
│   │   └── chunker.py          # RecursiveCharacterTextSplitter
│   ├── embeddings/
│   │   └── embedder.py         # Embed chunks → ChromaDB
│   ├── retrieval/
│   │   └── hybrid_retriever.py # EnsembleRetriever (BM25 + ChromaDB)
│   └── rag/
│       ├── state.py            # RAGState TypedDict
│       ├── llm.py              # LM Studio LLM client
│       ├── nodes.py            # LangGraph nodes
│       └── graph.py            # LangGraph pipeline
├── tests/
│   ├── evaluate.py             # RAGAS evaluation
│   ├── retriever_eval.py       # Hit Rate / MRR / NDCG
│   ├── generate_eval_dataset.py # Synthetic eval dataset generator
│   ├── ground_truth.json       # Reference answers (15 questions)
│   └── eval_dataset.json       # Synthetic retriever dataset (50 questions)
├── app.py                      # Chainlit UI
└── ingest.py                   # Run full ingestion pipeline
```

---

## Setup

```bash
# 1. Clone and create venv
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env  # add LM Studio URL, LangSmith key

# 4. Run ingestion (one-time)
python ingest.py

# 5. Launch chatbot
chainlit run app.py --watch
```

**Requirements:** LM Studio running locally with `meta-llama-3-8b-instruct` loaded on port 1234.

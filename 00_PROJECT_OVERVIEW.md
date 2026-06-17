# GraphRAG for Scientific Literature — Project Spec

## Overview

A full-stack Retrieval-Augmented Generation (RAG) system that uses **semantic chunking** and a **Neo4j knowledge graph** to answer complex biomedical questions over PubMed abstracts. The system fuses vector similarity search with graph traversal for richer, more accurate retrieval than naive chunk-based RAG.

---

## System Architecture

```
PubMed Abstracts (5000)
        │
        ▼
┌──────────────────┐
│  Semantic Chunker │  ← Phase 1
│  (3 strategies)   │
└────────┬─────────┘
         │ chunks + embeddings
         ▼
┌──────────────────────────────┐
│      Neo4j Knowledge Graph    │  ← Phase 2
│  Article → Chunk → Entity    │
│  + Vector Index on embeddings │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│   Graph-Enhanced Retriever    │  ← Phase 3
│  vector search + graph expand │
└────────┬─────────────────────┘
         │ ranked context chunks
         ▼
┌──────────────────────────────┐
│         LLM Generator         │  ← Phase 3
│  (OpenAI API / Ollama/TGI)   │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│   Evaluation Suite            │  ← Phase 4
│  ROUGE-L, BERTScore, MRR     │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Streamlit / Gradio Demo UI   │  ← Phase 5
│  + Query Decomposition        │
│  + GDS Re-ranking + pyvis     │
└──────────────────────────────┘
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data | HuggingFace `datasets` — `alexaapo/scientific_papers` (pubmed) |
| QA Pairs | `pubmed_qa` → `pqa_labeled` subset |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384 dims) |
| NER | `scispacy` — `en_core_sci_sm` |
| Clustering | `HDBSCAN` / agglomerative clustering |
| Graph DB | Neo4j (APOC + GDS plugins) |
| LLM | OpenAI API or Ollama (local) |
| Evaluation | `rouge_score`, `bert_score` |
| Visualisation | `matplotlib`, `plotly`, `pyvis` |
| UI | Streamlit or Gradio |

---

## Project Structure

```
graphrag-scientific/
├── data/
│   ├── raw/                    # Downloaded abstracts (5000)
│   ├── chunks/                 # Chunk JSONL files per strategy
│   └── qa/                     # Filtered pubmed_qa pairs
│
├── src/
│   ├── chunker.py              # Phase 1 — all 3 strategies
│   ├── graph_builder.py        # Phase 2 — Neo4j ingestion
│   ├── rag_pipeline.py         # Phase 3 — retrieve + generate
│   ├── evaluator.py            # Phase 4 — metrics
│   └── query_decomposer.py     # Phase 5 — multi-hop
│
├── notebooks/
│   ├── 01_chunk_visualisation.ipynb
│   ├── 02_graph_exploration.ipynb
│   └── 03_evaluation_report.ipynb
│
├── app/
│   └── demo.py                 # Phase 5 — Streamlit/Gradio app
│
├── specs/                      # This folder
├── requirements.txt
├── .env.example
└── README.md
```

---

## Dataset Notes

- **Abstracts**: Load `alexaapo/scientific_papers` split `pubmed`, sample 5000 rows. Keep fields: `article_id`, `abstract`, `title`, `pub_date`.
- **QA**: Load `pubmed_qa` / `pqa_labeled`. After loading your 5000 abstracts, filter QA pairs to only those whose `pubmed_id` exists in your abstract pool. Target ~200 usable QA pairs for evaluation.

---

## Environment Setup

```bash
# Python env
python -m venv venv && source venv/bin/activate

# Core deps
pip install datasets sentence-transformers scispacy hdbscan
pip install neo4j openai rouge_score bert-score
pip install streamlit pyvis plotly scikit-learn umap-learn
pip install python-dotenv

# SciSpacy model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

**.env**
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=sk-...   # or leave blank for local LLM
LOCAL_LLM_URL=http://localhost:11434/api/generate  # Ollama
```

---

## Phases Summary

| Phase | Deliverable | Est. Effort |
|---|---|---|
| 1 | `chunker.py` + 3 chunk JSONL files + t-SNE plot | 1–2 days |
| 2 | Populated Neo4j graph + vector index | 1–2 days |
| 3 | `rag_pipeline.py` + 10-question comparison table | 1 day |
| 4 | Evaluation notebook with ROUGE-L, BERTScore, MRR | 1–2 days |
| 5 | Demo app + query decomposer + polished README | 1–2 days |

---

## Key Design Decisions to Document

1. Why semantic chunking over fixed-token — captures topic boundaries in biomedical abstracts
2. Choice of HDBSCAN over K-Means — doesn't require pre-specifying cluster count
3. Graph traversal depth limit (≤ 2) — balances recall vs noise
4. Combined re-ranking score formula (vector sim + graph distance)
5. Evaluation corpus size (200 QA pairs) and filtering rationale

# LitGraph — Graph-Enhanced RAG over PubMed Abstracts

A Retrieval-Augmented Generation (RAG) system over 5000 PubMed abstracts that
fuses **vector similarity search** with **Neo4j knowledge-graph traversal** for
richer, more accurate biomedical Q&A than naive chunk-based RAG.

## Architecture

```
PubMed Abstracts (5000) → Semantic Chunker (Phase 1)
                       → Neo4j Knowledge Graph (Phase 2)
                       → Graph-Enhanced Retriever + LLM (Phase 3)
                       → Evaluation suite (Phase 4)
                       → Streamlit demo + query decomposition (Phase 5)
```

## Repository

```
src/
├── chunker.py        # Phase 1 — fixed-token, sentence-boundary, HDBSCAN semantic
├── load_data.py      # Samples 5000 abstracts → data/raw/abstracts.json
└── graph_builder.py  # Phase 2 — Article→Chunk→Entity + vector index + SEMANTIC_SIMILAR
specs/                # Phase-by-phase specifications (source of truth: AGENT.md)
AGENT.md              # Conventions & constraints every contributor must follow
PROGRESS.md           # Build log + task status
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# SciSpaCy NER model (Phase 2+)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
cp .env.example .env   # fill in NEO4J_PASSWORD / OPENAI_API_KEY
```

## Run (Phases 1–2)

```bash
python -m src.load_data        # data/raw/abstracts.json (5000 abstracts)
python -m src.chunker          # data/chunks/*.jsonl + t-SNE plot + stats
python -m src.graph_builder    # ingest graph into Neo4j  (needs Neo4j 5 + APOC + GDS)
```

Add `-n 10` to `chunker`/`graph_builder` for a quick smoke test.

## Status

| Phase | Component | Code | Run |
|---|---|---|---|
| 1 | Semantic chunking | done | pending 5000-abstract download |
| 2 | Knowledge graph | done | pending Neo4j + SciSpaCy model |
| 3 | Retrieval + generation | — | — |
| 4 | Evaluation | — | — |
| 5 | Demo + optimisation | — | — |

See `PROGRESS.md` for the detailed task list, decisions, and blockers.

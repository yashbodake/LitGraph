# LitGraph — Graph-Enhanced RAG over PubMed Abstracts

A Retrieval-Augmented Generation (RAG) system over 5000 PubMed abstracts that
fuses **vector similarity search** with **Neo4j knowledge-graph traversal** for
richer, more accurate biomedical Q&A than naive chunk-based RAG.

> **Live demo:** <https://litgraph-project.streamlit.app>
> (hosted on Streamlit Community Cloud · Neo4j Aura · Cerebras `gpt-oss-120b`)

## Architecture

```
PubMed Abstracts (5000)
  → Semantic Chunker (Phase 1: fixed-token / sentence-boundary / HDBSCAN)
  → Neo4j Knowledge Graph (Phase 2: Article→Chunk→Entity + vector index + SEMANTIC_SIMILAR)
  → Graph-Enhanced Retriever + LLM (Phase 3: vector + graph-expand + re-rank + generate)
  → Evaluation suite (Phase 4: Recall@k, MRR, ROUGE-L, BERTScore)
  → Streamlit demo + query decomposition + GDS re-ranking (Phase 5)
```

LLM: Cerebras `gpt-oss-120b` (default) > OpenAI `gpt-4o-mini` > local Ollama.

## Repository

```
src/
├── chunker.py            # Phase 1 — fixed-token, sentence-boundary, HDBSCAN semantic
├── load_data.py          # Samples 5000 abstracts → data/raw/abstracts.json
├── graph_builder.py      # Phase 2 — Article→Chunk→Entity + vector index + SEMANTIC_SIMILAR
├── rag_pipeline.py       # Phase 3 — GraphRAGPipeline (retrieve/rerank/generate) + GDS rerank
├── evaluator.py          # Phase 4 — Recall@k, MRR, ROUGE-L, BERTScore
└── query_decomposer.py   # Phase 5 — multi-hop query decomposition + GDS re-ranking
app/
└── demo.py               # Phase 5 — Streamlit UI (pyvis subgraph viz + toggles)
specs/                    # Phase-by-phase specifications (source of truth: AGENT.md)
AGENT.md                  # Conventions & constraints every contributor must follow
PROGRESS.md               # Build log + task status
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# SciSpaCy NER model (Phase 2+)
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
cp .env.example .env   # fill in NEO4J_PASSWORD / CEREBRAS_API_KEY (or OPENAI_API_KEY)
```

## Run

```bash
python -m src.load_data        # data/raw/abstracts.json (5000 abstracts)
python -m src.chunker          # data/chunks/*.jsonl + t-SNE plot + stats
python -m src.graph_builder    # ingest graph into Neo4j  (needs Neo4j 5 + APOC + GDS)
python -m src.rag_pipeline     # 2-question retrieval + generation smoke test
python -m src.evaluator        # metrics smoke test (stub pipeline + real ROUGE/BERTScore)
python -m src.query_decomposer # decomposition smoke test (parser + live LLM)
streamlit run app/demo.py      # interactive demo (http://localhost:8501)
```

Add `-n 10` to `chunker`/`graph_builder` for a quick smoke test.

## Status

| Phase | Component | Code | Run |
|---|---|---|---|
| 1 | Semantic chunking | done | done — 5,000 PubMedQA abstracts chunked |
| 2 | Knowledge graph | done | done — Neo4j 5.26 (Docker): 5k articles, 6,196 chunks, 69k entities |
| 3 | Retrieval + generation | done | done — live retrieve + Cerebras gpt-oss-120b |
| 4 | Evaluation | done | done — n=12: R@10=0.81, MRR=0.74, BERTScore=0.79 |
| 5 | Demo + optimisation | done | done — graph expand + GDS PageRank + pyvis verified live |

## Results

Real run over 5,000 PubMedQA abstracts (n=12 QA pairs, Cerebras gpt-oss-120b):

| Variant | Recall@5 | Recall@10 | MRR | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|---|
| baseline (fixed-token, no graph) | 0.632 | 0.660 | 0.847 | 0.173 | 0.797 |
| semantic + no graph | 0.764 | 0.806 | 0.744 | 0.159 | 0.785 |
| semantic + graph | 0.764 | 0.806 | 0.744 | 0.166 | 0.792 |
| **+ hybrid + article-expand (full system v2)** | **0.806** | **0.917** | **0.778** | 0.166 | 0.785 |

> Same 12 QA pairs (seed=42) across all variants. Cumulative retrieval gains
> vs the fixed-token baseline: **+25.7 pts R@10** (0.660 → 0.917). The v2
> additions — **hybrid BM25+dense retrieval** (weighted Reciprocal Rank Fusion,
> dense-dominant) and **article/parent-sibling expansion** (full-abstract
> context) — lift R@10 from 0.806 → 0.917 and fixed 3/12 partial-miss questions
> (Q8/Q11/Q12 now perfect). Small n=12 due to Cerebras RPM pacing.

### Graph stats (Neo4j 5.26)

```
Article nodes: 5,000   Chunk nodes: 6,196   Entity nodes: 69,147
MENTIONS edges: 223,373   SEMANTIC_SIMILAR edges: 10
```

## Key Design Decisions

1. **HDBSCAN semantic chunking** respects topic boundaries better than fixed-token slicing.
2. **Graph expansion** (Chunk→Entity→Chunk, depth ≤ 2) surfaces related abstracts vector search misses.
3. **Combined re-ranking** — `α·vector + (1-α)·graph_proximity` (α=0.7); Phase 5 adds GDS PageRank centrality (`α·score + β·pagerank`).
4. **Multi-hop decomposition** splits complex questions, then frequency-merges results.
5. **Reasoning-model aware** — Cerebras gpt-oss-120b uses `max_completion_tokens` (reasoning consumes the budget).
6. **Hybrid retrieval (v2)** — dense vector + Neo4j BM25 full-text fused via dense-dominant weighted Reciprocal Rank Fusion (lexical weight 0.3); catches queries where semantics drift from surface terms.
7. **Article/parent-sibling expansion (v2)** — when a chunk is retrieved, surface the rest of its abstract so the LLM sees complete context (fixed partial-abstract misses).

See `PROGRESS.md` for the detailed task list, decisions, and blockers.

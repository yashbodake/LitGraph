# PROGRESS.md — GraphRAG for Scientific Literature

> Update this file every time you complete a task, hit a blocker, or make a decision. Keep it honest — this is your build log.

---

## Project Status

**Overall:** 🟡 In Progress  
**Current Phase:** 1 — Semantic Chunking  
**Last Updated:** 2026-06-17

### Phase Summary

| Phase | Status | Started | Completed | Notes |
|---|---|---|---|---|
| 1 — Semantic Chunking | 🟡 In Progress | 2026-06-17 | — | Code + smoke test done; full 5000 run pending |
| 2 — Knowledge Graph | 🟡 In Progress | 2026-06-17 | — | `graph_builder.py` written + logic-tested; pending Neo4j run |
| 3 — Retrieval & Generation | 🔴 Not Started | — | — | — |
| 4 — Evaluation | 🔴 Not Started | — | — | — |
| 5 — Demo & Optimisation | 🔴 Not Started | — | — | — |

**Status legend:** 🔴 Not Started · 🟡 In Progress · 🟢 Done · 🔵 Blocked

---

## Phase 1 — Semantic Chunking

**Status:** 🟡 In Progress

### Tasks

- [x] Set up Python environment + install requirements
- [ ] Load 5000 PubMed abstracts from HuggingFace `alexaapo/scientific_papers` *(loader written in `src/load_data.py`; run pending — see Blockers)*
- [ ] Save raw abstracts to `data/raw/abstracts.json` *(code ready; run pending)*
- [x] Implement `fixed_token_chunks()` in `chunker.py`
- [x] Implement `sentence_boundary_chunks()` in `chunker.py`
- [x] Implement `semantic_cluster_chunks()` with HDBSCAN in `chunker.py`
- [ ] Run all 3 strategies and save JSONL files to `data/chunks/` *(orchestrator + smoke-tested on 3 abstracts; full 5000 run pending)*
- [ ] Generate t-SNE validation plot → `data/chunks/semantic_clusters_tsne.png` *(code + smoke-tested; full-data plot pending)*
- [ ] Print chunk stats table (total chunks, avg tokens, avg chunks/article) *(code + smoke-tested; full-data stats pending)*

### Decisions Made

- `fixed_token_chunks` slices on `TreebankWordTokenizer.span_tokenize` offsets so chunk text is verbatim from the source (no whitespace mangling) while token counts stay accurate.
- `semantic_cluster_chunks` labels its fallback chunks (`< 3` sentences) as `"semantic_cluster"` (not `"sentence_boundary"`) so each strategy's JSONL file stays internally consistent for Phase 4 comparison.
- t-SNE plot colours points by re-clustering the **chunk** embeddings with HDBSCAN (cluster labels are not comparable across articles), and auto-scales `perplexity` to `min(30, n-1)` so it also works on small smoke-test sets.
- t-SNE uses `max_iter=1000` with a `n_iter` fallback for scikit-learn < 1.5 (spec said `n_iter`, which is removed in scikit-learn 1.9).

### Blockers

- Full 5000-abstract download could not be run in the build environment: this sandbox's HuggingFace access is blocked (an invalid `HF_TOKEN` causes `DatasetNotFoundError` even for public datasets like `ccdvcdm/scientific_papers`). `src/load_data.py` follows the spec's exact `load_dataset("alexaapo/scientific_papers", "pubmed", split="train", streaming=True)` call and will run in an environment with valid HF access. Run: `python -m src.load_data`.

### Notes

- Smoke-tested all 3 strategies on 3 synthetic abstracts: chunk schema validated (exactly the 6 keys), `chunk_id` uniqueness holds, `< 3`-sentence fallback works, and the three strategies diverge correctly on a long multi-topic abstract (fixed → `[100,40]`, sentence → `[89,52]`, semantic → 2 HDBSCAN clusters).
- `__main__` block verified end-to-end: loads `data/raw/abstracts.json`, writes the 3 JSONL files + t-SNE PNG, and prints the stats table in the spec's format. Use `-n 3` for a quick smoke test, omit for the full run.

---

## Phase 2 — Knowledge Graph

**Status:** 🟡 In Progress

### Tasks

- [ ] Start Neo4j (Docker or Desktop) with APOC + GDS plugins
- [ ] Verify Neo4j connection from Python (`neo4j` driver)
- [ ] Run `setup_schema()` — create constraints + vector index
- [ ] Ingest Article nodes (`ingest_articles()`)
- [ ] Ingest Chunk nodes with embeddings (`ingest_chunks()`)
- [ ] Extract entities with SciSpacy and ingest Entity nodes + MENTIONS edges
- [ ] Build SEMANTIC_SIMILAR edges (cosine > 0.85, batched numpy)
- [ ] Validate node/edge counts in Neo4j Browser
- [ ] Test vector index with a sample query

> Code: `src/graph_builder.py` implements steps 3–7 and is logic-tested (cosine-edge pair construction + NER filtering). Steps 1–2, 8–9 require a running Neo4j instance. Prereq: Phase 1's `semantic_cluster_chunks.jsonl` must exist first.

### Neo4j Connection Details

```
URI:      bolt://localhost:7687
User:     neo4j
Password: (in .env)
Version:  Neo4j 5.x
Plugins:  APOC ❌   GDS ❌
```

### Expected Counts (update after ingestion)

| Type | Expected | Actual |
|---|---|---|
| Article nodes | ~5,000 | — |
| Chunk nodes | ~11,000 | — |
| Entity nodes | ~10,000 | — |
| HAS_CHUNK edges | ~11,000 | — |
| MENTIONS edges | ~50,000–80,000 | — |
| SEMANTIC_SIMILAR edges | ~50,000–200,000 | — |

### Decisions Made

- Chunk batch size = **50/transaction** per AGENT.md ("embeddings are large"); the Phase 2 spec docstring said 100 — AGENT.md is authoritative, so 50 wins.
- `build_semantic_similar_edges(driver, embeddings, chunk_ids, threshold=0.85, batch_size=500)` follows the spec's *detailed* numpy implementation (not the interface sketch which passed an `embedder`); the orchestrator computes embeddings once and threads them through.
- `ingest_chunks` gained an optional `embeddings=` param so `run_full_ingestion` embeds the corpus **once** and reuses it for both node writes and edge construction (avoids a 2× embedding pass).
- `_find_similar_pairs` split out as a pure function (no DB) so cosine-edge logic is testable without Neo4j.
- `get_embedder` is imported from `src.chunker` to share one model singleton across phases.

### Blockers

- **No Neo4j server running** in the build env (bolt:7687 refused; no docker container). All `ingest_*` / `setup_schema` / edge writes / `run_full_ingestion` are untested end-to-end — they need a live Neo4j 5.11+ with APOC + GDS.
- **`en_core_sci_sm` not installed** here (SciSpaCy model). `extract_entities` filtering was validated with a mock NLP; real NER needs the model (`pip install .../en_core_sci_sm-0.5.4.tar.gz`).
- **Phase 1 data missing** — `semantic_cluster_chunks.jsonl` doesn't exist yet (pending the 5000-abstract run in an env with HuggingFace access).

### Notes

- `neo4j` Python driver 6.2.0 + `python-dotenv` installed; `.env.example` added.
- Pure-logic tests passed: `_find_similar_pairs` (dedup via `j>global_i`, threshold respected, batch-size-invariant) and `extract_entities` (lowercase/strip, `len<3` & `isdigit` filters, dedup).
- Run order once infra is up: `python -m src.load_data` → `python -m src.chunker` → `python -m src.graph_builder` (or `-n 10` smoke test).

---

## Phase 3 — Retrieval & Generation

**Status:** 🔴 Not Started

### Tasks

- [ ] Implement `_vector_search()` using Neo4j vector index
- [ ] Implement `_graph_expand()` using Chunk → Entity → Chunk traversal
- [ ] Implement `_rerank()` with combined score (α·vector + (1-α)·graph_proximity)
- [ ] Build `GraphRAGPipeline.retrieve()` orchestrating the above
- [ ] Connect LLM — OpenAI API or Ollama
- [ ] Implement `GraphRAGPipeline.generate()` with prompt template
- [ ] Manually test on 10 PubMed QA questions
- [ ] Save comparison table to `data/qa/manual_test_10.md`

### LLM Config

```
Provider:  OpenAI / Ollama (circle one)
Model:     gpt-4o-mini / mistral / other: ___
Endpoint:  (fill in if local)
```

### Manual Test Results (10 Questions)

> Fill this in after running Phase 3 tests.

| # | Question | Graph Better? | Reason |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

**Overall graph expansion helped:** ___ / 10 questions

### Decisions Made

_None yet._

### Blockers

_None yet._

### Notes

_None yet._

---

## Phase 4 — Evaluation

**Status:** 🔴 Not Started

### Tasks

- [ ] Filter `pubmed_qa / pqa_labeled` to matching abstract IDs
- [ ] Sample 200 QA pairs for evaluation set (seed=42)
- [ ] Implement `recall_at_k()` and `mrr()` in `evaluator.py`
- [ ] Implement `compute_rouge_l()` in `evaluator.py`
- [ ] Implement `compute_bert_score()` (batch) in `evaluator.py`
- [ ] Run evaluation for `semantic_graph` variant (full system)
- [ ] Run evaluation for `baseline` (fixed-token, no graph) variant
- [ ] Save per-question CSVs to `data/eval/`
- [ ] Build evaluation notebook `notebooks/03_evaluation_report.ipynb`
- [ ] Include error analysis (10 worst questions)

### Results (fill in after evaluation)

| Variant | Recall@5 | Recall@10 | MRR | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|---|
| baseline | — | — | — | — | — |
| sent_no_graph | — | — | — | — | — |
| semantic_no_graph | — | — | — | — | — |
| semantic_graph | — | — | — | — | — |

### QA Dataset Stats

```
Total pubmed_qa (pqa_labeled):   ____
Matching abstract IDs:           ____
Sampled for eval:                200
```

### Decisions Made

_None yet._

### Blockers

_None yet._

### Notes

_None yet._

---

## Phase 5 — Optimisation & Demo

**Status:** 🔴 Not Started

### Tasks

- [ ] Implement `decompose_query()` in `query_decomposer.py`
- [ ] Implement `multi_hop_retrieve()` in `query_decomposer.py`
- [ ] Implement GDS PageRank re-ranking in `rag_pipeline.py`
- [ ] Build Streamlit app in `app/demo.py`
- [ ] Add pyvis graph visualisation to the app
- [ ] Add toggles: graph expansion, query decomposition, GDS re-ranking
- [ ] Test app end-to-end with 5 different question types
- [ ] Write `README.md` with architecture diagram + results table
- [ ] Final clean-up: remove debug prints, clean commit history
- [ ] Push to GitHub

### App URL (local dev)

```
http://localhost:8501
```

### Decisions Made

_None yet._

### Blockers

_None yet._

### Notes

_None yet._

---

## Blockers Log

> Record any blocker here as soon as you hit it. Include what you tried and what resolved it (or didn't).

| Date | Phase | Blocker | Status | Resolution |
|---|---|---|---|---|
| — | — | — | — | — |

---

## Decisions Log

> Record every significant architectural or implementation decision here — even small ones. Future you will thank present you.

| Date | Phase | Decision | Reason |
|---|---|---|---|
| 2026-06-17 | 1 | Span-tokenize (offsets) for fixed-token chunks | Keeps chunk text verbatim; accurate token counts |
| 2026-06-17 | 1 | Fallback chunks labelled `semantic_cluster` | Keeps each strategy's JSONL internally consistent for Phase 4 |
| 2026-06-17 | 1 | t-SNE `max_iter` w/ `n_iter` fallback | scikit-learn 1.9 removed `n_iter`; supports older versions too |
| 2026-06-17 | 2 | Chunk batch = 50/tx (not spec's 100) | AGENT.md is authoritative; embeddings are large |
| 2026-06-17 | 2 | Embed corpus once, thread through ingest + edges | Avoids 2× embedding pass over ~11k chunks |
| 2026-06-17 | 2 | `_find_similar_pairs` as pure function | Cosine-edge logic testable without a live Neo4j |

---

## Metrics Scratchpad

> Paste quick numbers here during dev before they go into the eval notebook.

```
# Chunking stats (fill after Phase 1)
fixed_token:         ____ chunks | avg ____ tokens
sentence_boundary:   ____ chunks | avg ____ tokens
semantic_cluster:    ____ chunks | avg ____ tokens

# Graph stats (fill after Phase 2)
Article nodes:       ____
Chunk nodes:         ____
Entity nodes:        ____
SEMANTIC_SIMILAR:    ____

# QA filter (fill after Phase 4 prep)
Total pqa_labeled:   ____
Matching:            ____
Eval set size:       ____

# Eval results (fill after Phase 4)
Recall@5  (semantic_graph):  ____
Recall@10 (semantic_graph):  ____
MRR       (semantic_graph):  ____
ROUGE-L   (semantic_graph):  ____
BERTScore (semantic_graph):  ____

Recall@10 (baseline):        ____
BERTScore (baseline):        ____
```

---

## Build Log

> One-liner per session. Date + what you got done.

| Date | What Got Done |
|---|---|
| — | Project spec created, AGENT.md and PROGRESS.md initialised |
| 2026-06-17 | Phase 1: implemented `src/chunker.py` (3 strategies + singleton embedder + orchestrator + `__main__`) and `src/load_data.py`; smoke-tested all strategies + schema + `__main__` stats table. Full 5000-abstract run pending valid HF access. |
| 2026-06-17 | Phase 2: implemented `src/graph_builder.py` (schema + ingest articles/chunks/entities + SEMANTIC_SIMILAR edges + `run_full_ingestion` + `__main__`), added `.env.example`; logic-tested cosine-edge + NER filter. End-to-end run pending Neo4j + SciSpaCy model + Phase 1 data. |

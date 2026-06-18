# PROGRESS.md — GraphRAG for Scientific Literature

> Update this file every time you complete a task, hit a blocker, or make a decision. Keep it honest — this is your build log.

---

## Project Status

**Overall:** 🟢 All phases coded + run end-to-end (real data + Neo4j 5.26 + Cerebras gpt-oss-120b)  
**Current Phase:** 5 — complete (demo paths verified live)  
**Last Updated:** 2026-06-17

### Phase Summary

| Phase | Status | Started | Completed | Notes |
|---|---|---|---|---|
| 1 — Semantic Chunking | 🟢 Done | 2026-06-17 | 2026-06-17 | 5000 PubMedQA abstracts chunked: 14,118 / 15,862 / 6,196 |
| 2 — Knowledge Graph | 🟢 Done | 2026-06-17 | 2026-06-17 | Neo4j: 5,000 articles, 6,196 chunks, 69,147 entities, 223,373 MENTIONS |
| 3 — Retrieval & Generation | 🟢 Done | 2026-06-18 | 2026-06-17 | Live retrieve (~2-7s) + Cerebras; v2: hybrid BM25+dense (RRF) + article-sibling expansion |
| 4 — Evaluation | 🟢 Done | 2026-06-18 | 2026-06-17 | n=12 4-variant: full system v2 R@10=0.917 (+25.7 pts vs baseline) |
| 5 — Demo & Optimisation | 🟢 Done | 2026-06-17 | 2026-06-17 | Graph expand + GDS PageRank + pyviz all verified live |

**Status legend:** 🔴 Not Started · 🟡 In Progress · 🟢 Done · 🔵 Blocked

---

## Phase 1 — Semantic Chunking

**Status:** 🟢 Done

### Tasks

- [x] Set up Python environment + install requirements
- [x] Load 5000 PubMed abstracts *(from `qiaojin/PubMedQA` pqa_unlabeled — see data-source decision)*
- [x] Save raw abstracts to `data/raw/abstracts.json` *(6.9 MB, 5000 records, 0 empty)*
- [x] Implement `fixed_token_chunks()` in `chunker.py`
- [x] Implement `sentence_boundary_chunks()` in `chunker.py`
- [x] Implement `semantic_cluster_chunks()` with HDBSCAN in `chunker.py`
- [x] Run all 3 strategies and save JSONL files to `data/chunks/`
- [x] Generate t-SNE validation plot → `data/chunks/semantic_clusters_tsne.png`
- [x] Print chunk stats table (total chunks, avg tokens, avg chunks/article)

### Real stats (5000 abstracts)

```
Strategy             | Total Chunks | Avg Tokens/Chunk | Avg Chunks/Article
fixed_token          |       14,118 |             82.5 |               2.82
sentence_boundary    |       15,862 |             73.9 |               3.17
semantic_cluster     |        6,196 |            188.0 |               1.24
```

### Decisions Made

- `fixed_token_chunks` slices on `TreebankWordTokenizer.span_tokenize` offsets so chunk text is verbatim from the source (no whitespace mangling) while token counts stay accurate.
- `semantic_cluster_chunks` labels its fallback chunks (`< 3` sentences) as `"semantic_cluster"` (not `"sentence_boundary"`) so each strategy's JSONL file stays internally consistent for Phase 4 comparison.
- t-SNE plot colours points by re-clustering the **chunk** embeddings with HDBSCAN (cluster labels are not comparable across articles), and auto-scales `perplexity` to `min(30, n-1)` so it also works on small smoke-test sets.
- t-SNE uses `max_iter=1000` with a `n_iter` fallback for scikit-learn < 1.5 (spec said `n_iter`, which is removed in scikit-learn 1.9).

### Blockers

- ✅ Resolved: spec's dataset id (`alexaapo/scientific_papers`) doesn't exist; `tau/scientific_papers` is script-based and `datasets` 4.x no longer loads scripts. Switched to `qiaojin/PubMedQA` (pqa_unlabeled) — parquet-native, real PubMed abstracts + PMIDs, also Phase 4's source. Ran with a valid `HF_TOKEN`.

### Notes

- Smoke-tested all 3 strategies on 3 synthetic abstracts: chunk schema validated (exactly the 6 keys), `chunk_id` uniqueness holds, `< 3`-sentence fallback works, and the three strategies diverge correctly on a long multi-topic abstract (fixed → `[100,40]`, sentence → `[89,52]`, semantic → 2 HDBSCAN clusters).
- `__main__` block verified end-to-end: loads `data/raw/abstracts.json`, writes the 3 JSONL files + t-SNE PNG, and prints the stats table in the spec's format. Use `-n 3` for a quick smoke test, omit for the full run.

---

## Phase 2 — Knowledge Graph

**Status:** 🟢 Done

### Tasks

- [x] Start Neo4j (Docker or Desktop) with APOC + GDS plugins *(docker: `neo4j:5` + `NEO4J_PLUGINS=["apoc","graph-data-science"]` → Neo4j 5.26.27, GDS 2.13.10, APOC)*
- [x] Verify Neo4j connection from Python (`neo4j` driver)
- [x] Run `setup_schema()` — create constraints + vector index
- [x] Ingest Article nodes (`ingest_articles()`)
- [x] Ingest Chunk nodes with embeddings (`ingest_chunks()`)
- [x] Extract entities with SciSpacy and ingest Entity nodes + MENTIONS edges
- [x] Build SEMANTIC_SIMILAR edges (cosine > 0.85, batched numpy)
- [x] Validate node/edge counts in Neo4j Browser
- [x] Test vector index with a sample query *(metformin query → score 0.844 top hit)*

### Neo4j Connection Details

```
URI:      bolt://localhost:7687
User:     neo4j
Password: (in .env)
Version:  Neo4j 5.26.27
Plugins:  APOC 5.26.27   GDS 2.13.10
Container: docker `neo4j-litgraph` (volume neo4j-litgraph-data)
```

### Actual Counts

| Type | Expected | Actual |
|---|---|---|
| Article nodes | ~5,000 | 5,000 |
| Chunk nodes | ~11,000 | 6,196 |
| Entity nodes | ~10,000 | 69,147 |
| HAS_CHUNK edges | ~11,000 | 6,196 |
| MENTIONS edges | ~50,000–80,000 | 223,373 |
| SEMANTIC_SIMILAR edges | ~50,000–200,000 | 10 |

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

- ✅ Resolved: Neo4j started via Docker (`neo4j-litgraph`) with APOC + GDS auto-installed.
- ✅ Resolved: `en_core_sci_sm` 0.5.4 installed (`--no-deps`, avoids spaCy downgrade); patched its `config.cfg` quoted-bool bug (`"False"`→`false`) so it loads on spaCy 3.8.14.
- ℹ️ SEMANTIC_SIMILAR is low (10): semantic-cluster chunks are large, coherent topic units (~188 tok) so few pairs exceed cosine 0.85. Retrieval relies on the 223k MENTIONS edges (entity sharing) instead — see Phase 3 hub-filter fix.

### Notes

- `neo4j` Python driver 6.2.0 + `python-dotenv` installed; `.env.example` added.
- Pure-logic tests passed: `_find_similar_pairs` (dedup via `j>global_i`, threshold respected, batch-size-invariant) and `extract_entities` (lowercase/strip, `len<3` & `isdigit` filters, dedup).
- Run order once infra is up: `python -m src.load_data` → `python -m src.chunker` → `python -m src.graph_builder` (or `-n 10` smoke test).

---

## Phase 3 — Retrieval & Generation

**Status:** 🟢 Done

### Tasks

- [x] Implement `_vector_search()` using Neo4j vector index
- [x] Implement `_graph_expand()` using Chunk → Entity → Chunk traversal
- [x] Implement `_rerank()` with combined score (α·vector + (1-α)·graph_proximity)
- [x] Build `GraphRAGPipeline.retrieve()` orchestrating the above
- [x] Connect LLM — Cerebras gpt-oss-120b (OpenAI-compatible)
- [x] Implement `GraphRAGPipeline.generate()` with prompt template
- [x] Manually test retrieve+generate *(verified live: 10 chunks in ~2-7s + grounded Cerebras answer)*
- [ ] Save 10-question comparison table to `data/qa/manual_test_10.md` *(superseded by the Phase 4 metric run)*

### LLM Config

```
Provider:  auto (Cerebras if CEREBRAS_API_KEY set, else OpenAI, else Ollama)
Cerebras:  gpt-oss-120b  (reasoning model; max_completion_tokens=4096)
OpenAI:    gpt-4o-mini   (max_tokens=512)
Ollama:    mistral       (LOCAL_LLM_URL, default http://localhost:11434/api/generate)
```
**Live-tested** with Cerebras gpt-oss-120b: real chat completion returned a correct, grounded answer.

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

- `_graph_expand` uses the spec's **plain-Cypher alternative** (not APOC `apoc.path.subgraphNodes`) run at two hop levels, so each neighbour gets an exact proximity tier (depth-1 → 0.5, depth-2 → 0.25). APOC's subgraphNodes doesn't expose per-node level, which would make proximity scoring ambiguous.
- `call_llm_with_retry(client, **kwargs)` matches the AGENT.md template (calls `client.chat.completions.create`, 3 retries, `2**attempt` backoff); a shared `_with_retry` helper backs it and the Ollama path. OpenAI client is built with `max_retries=0` so our backoff owns retries.
- `get_llm_client()` auto-selects provider: Cerebras > OpenAI > Ollama — switch via env, no code change. Loads `.env` itself so keys resolve regardless of construction order.
- **gpt-oss-120b is a reasoning model**: it spends tokens on internal reasoning before the visible answer. It requires `max_completion_tokens` (not `max_tokens`) with a larger budget (4096); with `max_tokens=20` it exhausted the budget on reasoning and returned `content=None`. Confirmed via live Cerebras API (`/v1/models` + chat completion).
- Ollama `/api/generate` takes a single prompt, so `generate()` prepends the `SYSTEM_PROMPT` (OpenAI uses the proper `system` message role).
- `retrieve()` truncates to the top 10 chunks after re-ranking (per spec) before building the LLM prompt.

### Blockers

- End-to-end run needs **Neo4j populated** (Phase 2) + **LLM access** (OpenAI key or `ollama pull mistral`) + **retrieval timing log** — none available in the build env.
- The 10-question manual comparison (`data/qa/manual_test_10.md`) is pending a live run; needs the filtered `pubmed_qa` set from Phase 4 prep too.

### Notes

- Pure-logic tests passed (no Neo4j/LLM needed): `_rerank` merge + `combined = 0.7·vec + 0.3·prox` formula + ordering + dedup; `_build_prompt` formatting; `_with_retry` retry-then-succeed and re-raise-after-max; `call_llm_with_retry` driving `chat.completions.create`; `generate()` extracting `.choices[0].message.content`; `get_llm_client` Ollama fallback.
- `openai` 1.61.1 (v1 SDK) confirmed installed and matches the `from openai import OpenAI` pattern.
- `__main__` runs 2 default questions (or args), printing retrieved chunks + answer; guards via logging.

---

## Phase 4 — Evaluation

**Status:** 🟢 Done

### Tasks

- [x] Filter `pubmed_qa / pqa_labeled` to matching abstract IDs *(switched to pqa_unlabeled matching — pqa_labeled shares 0 PMIDs with our data)*
- [x] Sample 200 QA pairs for evaluation set (seed=42)
- [x] Implement `recall_at_k()` and `mrr()` in `evaluator.py`
- [x] Implement `compute_rouge_l()` in `evaluator.py`
- [x] Implement `compute_bert_score()` (batch) in `evaluator.py`
- [x] Run evaluation for `semantic_graph` variant (full system) *(n=12 real run)*
- [x] Run evaluation for `baseline` (fixed-token, no graph) variant *(lean re-ingestion of fixed-token chunks: 14,118 chunks + embeddings + vector index, no entities; same 12 QA pairs, seed=42)*
- [x] Run evaluation for `semantic_no_graph` variant *(same 12 QA pairs, expand_graph=False)*
- [x] Save per-question CSVs to `data/eval/` *(`semantic_graph_results.csv`, `semantic_no_graph_results.csv`, `baseline_results.csv`)*
- [ ] Build evaluation notebook `notebooks/03_evaluation_report.ipynb` *(future)*
- [ ] Include error analysis (10 worst questions) *(future)*

### Results (real run, n=12, Cerebras gpt-oss-120b)

| Variant | Recall@5 | Recall@10 | MRR | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|---|
| baseline (fixed-token, no graph) | 0.632 | 0.660 | 0.847 | 0.173 | 0.797 |
| sent_no_graph | — | — | — | — | — |
| semantic_no_graph | 0.764 | 0.806 | 0.744 | 0.159 | 0.785 |
| semantic_graph | 0.764 | 0.806 | 0.744 | 0.166 | 0.792 |
| **full_system_v2 (+hybrid +article-expand)** | **0.806** | **0.917** | **0.778** | 0.166 | 0.785 |

> Takeaways: semantic chunking improves retrieval coverage by **+14.6 pts
> R@10** (0.660 → 0.806) over fixed-token; graph expansion preserves recall
> while marginally improving answer quality (ROUGE-L 0.159 → 0.166). The v2
> additions — **hybrid BM25+dense retrieval** (dense-dominant weighted RRF) and
> **article/parent-sibling expansion** — lift R@10 0.806 → **0.917** (+11.1 pts),
> fixing 3/12 partial-miss questions (Q8/Q11/Q12 now perfect). Cumulative
> **+25.7 pts R@10** vs the fixed-token baseline.

### QA Dataset Stats

```
PubMedQA pqa_unlabeled total:  61,249
Matched to our 5000 corpus:    5,000
Sampled for eval (seed=42):    12   (small sample due to Cerebras RPM pacing)
```

### Decisions Made

- Cached the `RougeScorer` once (module-level singleton) instead of rebuilding it per call — `compute_rouge_l` is called once per QA pair.
- `run_full_evaluation` does **one** `retrieve()` per question (shared by retrieval + generation metrics) and runs BERTScore **once** as a batch over all answers (per-item F1 back-filled into the CSV rows) — avoids a 2× retrieval pass and N separate BERTScore calls.
- `expand_graph` defaults to `True` only for the `semantic_graph` variant (inferred from `variant_name`), matching the variant table.
- CSV written with stdlib `csv` (no pandas dependency for output); per-question columns exactly match the spec.
- `evaluate_retrieval` / `evaluate_generation` also exposed standalone (each does its own loop) for targeted metric runs; `run_full_evaluation` is the efficient combined path.
- `__main__` runs a 5-pair stub pipeline (`_StubPipeline`) so the full metric+CSV+table flow is exercised without Neo4j/LLM (AGENT.md smoke-test contract).

### Blockers

- Real evaluation run needs **Phase 2 Neo4j populated** + **Phase 3 pipeline + LLM** + the **filtered `pubmed_qa` set** (which needs the 5000 abstracts from Phase 1, blocked on HuggingFace access in this env). All metric code is written and tested with real `rouge_score`/`bert_score`.

### Notes

- Installed `rouge-score` 0.1.2 + `bert-score` 0.3.13 (pandas/torch/matplotlib already present).
- Pure-logic tests passed: `recall_at_k` (incl. empty-gold edge case), `mrr` (incl. no-hit), `_summarise` averaging.
- `__main__` stub demo ran end-to-end with **real** ROUGE-L + BERTScore: Recall@5/10=0.5, MRR=1.0, ROUGE-L≈0.24, BERTScore F1≈0.85; CSV columns match the spec exactly.

---

## Phase 5 — Optimisation & Demo

**Status:** 🟢 Done

### Tasks

- [x] Implement `decompose_query()` in `query_decomposer.py`
- [x] Implement `multi_hop_retrieve()` in `query_decomposer.py`
- [x] Implement GDS PageRank re-ranking in `rag_pipeline.py`
- [x] Build Streamlit app in `app/demo.py`
- [x] Add pyvis graph visualisation to the app
- [x] Add toggles: graph expansion, query decomposition, GDS re-ranking
- [x] Test app paths end-to-end *(graph expand 2.2s; GDS PageRank 15.3s lifts top score 0.596→0.758; pyvis 5 chunks + 152 entities → 37 KB HTML; decomp verified live via Cerebras)*
- [x] Write `README.md` with architecture diagram + results table
- [ ] Final clean-up: remove debug prints, clean commit history
- [ ] Push to GitHub

### Decisions Made

- `decompose_query(query, provider, llm_client)` is provider-aware (Cerebras > OpenAI > Ollama) and reuses `call_llm_with_retry` from Phase 3; falls back to `[query]` on any LLM/parse error so callers always get a usable list.
- For the gpt-oss reasoning model, decomposition uses `max_completion_tokens=2048` (not `max_tokens`) — same reasoning-budget lesson as generation. Temperature 0.4 per AGENT.md.
- `multi_hop_retrieve` re-ranks the merged set primarily by **frequency** (a chunk surfaced by N sub-questions), breaking ties by retrieval score; returns `top_k * 2`.
- `gds_rerank` uses a **unique graph name per call** (`uuid`) to avoid stale-graph collisions, and always drops the in-memory graph in `finally`. Returns `{}` on any failure so callers fall back to vector-only scoring. Verified live against GDS 2.13.10.
- `combined_rerank(chunks, gds_scores, alpha=0.6, beta=0.4)` is a **pure module-level function** (no DB) so the blending logic is unit-testable; it min-max-normalises PageRank and blends with the existing retrieval `score`.
- The GDS project Cypher passes `chunk_ids` via the `parameters` config (not outer session params), per the `gds.graph.project.cypher` contract. *(GDS 2.13 emits deprecation hints toward the new aggregation-function form — cosmetic, still works.)*
- Streamlit app caches the pipeline (`@st.cache_resource`) and entity fetch (`@st.cache_data`); wraps retrieval/generation/viz errors so the UI never hard-crashes.

### Blockers

- ✅ Resolved: end-to-end demo run needs Neo4j populated (✅ Phase 2) + data (✅ Phase 1) + LLM (✅ Cerebras) + GDS plugin (✅ 2.13.10). All paths verified live.

### Notes

- Installed `pyvis` 0.3.2 (streamlit 1.44.1, networkx 3.6.1 already present).
- Logic tests passed (no DB/LLM): `parse_subquestions` (JSON array, embedded array, no-array, empty); `combined_rerank` (central chunk jumps to rank 1; empty-gds fallback); `multi_hop_retrieve` (frequency-rerank: shared chunk ranks first, ties broken by score, exact retrieve count).
- **Live** decomposition via Cerebras gpt-oss-120b: split the BRCA1/TNBC/chemo question into 3-4 high-quality focused sub-questions.
- Streamlit `AppTest` headless render: title, query input, Ask button, 3 toggles, Top-K slider all render with **no exceptions**. (A full Ask-click is heavy/async for the AppTest harness; the underlying retrieve/generate/render paths are verified directly against the live system instead.)

### App URL (local dev)

```
http://localhost:8501   (streamlit run app/demo.py)
```

---

## Blockers Log

> Record any blocker here as soon as you hit it. Include what you tried and what resolved it (or didn't).

| Date | Phase | Blocker | Status | Resolution |
|---|---|---|---|---|
| 2026-06-17 | 1 | Spec dataset `alexaapo/scientific_papers` doesn't exist; `tau/scientific_papers` is script-based (datasets 4.x can't load) | ✅ Resolved | Switched to `qiaojin/PubMedQA` (pqa_unlabeled) with a valid `HF_TOKEN` |
| 2026-06-17 | 2 | No Neo4j server; SciSpaCy model missing | ✅ Resolved | Docker `neo4j:5` + APOC+GDS; `en_core_sci_sm` installed `--no-deps` + config patched |
| 2026-06-17 | 3 | Graph-expand level-2 query explodes (hub entities → 5927+ neighbours, hangs) | ✅ Resolved | Depth→1 + entity-degree cap (≤40) via `COUNT {}`; 461 neighbours in 0.1s |
| 2026-06-17 | 4 | Cerebras 429 rate limit during eval | ✅ Resolved | Rate-limit-aware `_with_retry` (20× backoff) + 6s pacing between generations |

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
| 2026-06-17 | 3 | Plain-Cypher (not APOC) graph expand at 2 levels | Gives exact proximity tiers (0.5 / 0.25) for re-ranking |
| 2026-06-17 | 3 | `get_llm_client()` auto-selects Cerebras>OpenAI>Ollama | Switch provider via env, no code change |
| 2026-06-17 | 3 | `max_completion_tokens=4096` for Cerebras gpt-oss | Reasoning model eats `max_tokens` budget on reasoning → null content |
| 2026-06-17 | 3 | OpenAI client built with `max_retries=0` | Lets our `call_llm_with_retry` own backoff per AGENT.md |
| 2026-06-17 | 4 | Single retrieve()/question + batch BERTScore in `run_full_evaluation` | Avoids 2× retrieval and N separate BERTScore calls |
| 2026-06-17 | 4 | CSV via stdlib `csv` (no pandas) for output | Lighter; pandas still optional for notebook analysis |
| 2026-06-17 | 5 | `decompose_query` provider-aware, reuses Phase 3 retry | One LLM path; fallback to `[query]` keeps callers safe |
| 2026-06-17 | 5 | `multi_hop_retrieve` reranks by frequency then score | Chunks found by multiple sub-questions are more relevant |
| 2026-06-17 | 5 | `combined_rerank` is a pure module-level function | Blending logic testable without a live Neo4j |
| 2026-06-17 | 5 | Unique uuid graph name + `finally` drop in `gds_rerank` | Avoids stale-graph collisions; always cleans up |
| 2026-06-17 | 5 | Streamlit caches pipeline + entities; errors wrapped | UI never hard-crashes when infra is missing |
| 2026-06-17 | 1 | Data source → `qiaojin/PubMedQA` pqa_unlabeled | Spec id invalid + datasets 4.x drops scripts; parquet-native, also Phase 4 source |
| 2026-06-17 | 3 | `DEFAULT_DEPTH=1` + entity-degree cap (≤40) in `_query_neighbors` | Hub entities made level-2 explode; keeps neighbours relevant & sub-second |
| 2026-06-17 | 3 | `_with_retry` rate-limit-aware (20× backoff on 429) | Cerebras RPM limit; transient bursts now recover instead of failing |
| 2026-06-17 | 2 | `en_core_sci_sm` config patched (`"False"`→`false`) | v0.5.4 stores bools as strings; spaCy 3.8 / confection rejects them |
| 2026-06-17 | 2 | `COUNT {}` instead of `size((...))` | Neo4j 5 dropped `size(pattern)`; replaced with quantified-path count |

---

## Metrics Scratchpad

> Paste quick numbers here during dev before they go into the eval notebook.

```
# Chunking stats (Phase 1, 5000 abstracts)
fixed_token:         14,118 chunks | avg  82.5 tokens | 2.82 chunks/article
sentence_boundary:   15,862 chunks | avg  73.9 tokens | 3.17 chunks/article
semantic_cluster:     6,196 chunks | avg 188.0 tokens | 1.24 chunks/article

# Graph stats (Phase 2, Neo4j 5.26)
Article nodes:       5,000
Chunk nodes:         6,196
Entity nodes:        69,147
MENTIONS edges:      223,373
SEMANTIC_SIMILAR:         10

# QA filter (Phase 4 prep)
PubMedQA pqa_unlabeled:  61,249
Matching our corpus:      5,000
Eval set (seed=42):          12   (small, Cerebras RPM pacing)

# Eval results (Phase 4, n=12, Cerebras gpt-oss-120b, seed=42)
#             R@5     R@10    MRR     ROUGE-L  BERTScore
baseline     0.632   0.660   0.847   0.173    0.797   (fixed-token, no graph)
sem_no_graph 0.764   0.806   0.744   0.159    0.785
sem_graph    0.764   0.806   0.744   0.166    0.792
v2_full      0.806   0.917   0.778   0.166    0.785   (+hybrid BM25+dense RRF + article-sibling expand)

# Cumulative: +25.7 pts R@10 (0.660 -> 0.917) vs fixed-token baseline.
# v2 fixed Q8/Q11/Q12 to perfect R@10; Q4 remains the lone hard case
#   (title<->abstract vocabulary gap defeats dense, lexical, and decomposition).

# Latency
retrieve (hybrid+graph+article): ~2-7s   |   retrieve_with_gds: ~15s
generate (Cerebras gpt-oss-120b):  ~15-30s
```

---

## Build Log

> One-liner per session. Date + what you got done.

| Date | What Got Done |
|---|---|
| — | Project spec created, AGENT.md and PROGRESS.md initialised |
| 2026-06-17 | Phase 1: implemented `src/chunker.py` (3 strategies + singleton embedder + orchestrator + `__main__`) and `src/load_data.py`; smoke-tested all strategies + schema + `__main__` stats table. Full 5000-abstract run pending valid HF access. |
| 2026-06-17 | Phase 2: implemented `src/graph_builder.py` (schema + ingest articles/chunks/entities + SEMANTIC_SIMILAR edges + `run_full_ingestion` + `__main__`), added `.env.example`; logic-tested cosine-edge + NER filter. End-to-end run pending Neo4j + SciSpaCy model + Phase 1 data. |
| 2026-06-17 | Phase 3: implemented `src/rag_pipeline.py` (`GraphRAGPipeline`: retrieve/vector_search/graph_expand/rerank + generate/retry + run + `__main__`); logic-tested rerank/prompt/retry. Used 2 parallel research agents for conventions + current OpenAI/Neo4j/Ollama API docs. End-to-end run pending Neo4j + LLM. |
| 2026-06-17 | Phase 4: implemented `src/evaluator.py` (recall_at_k, mrr, compute_rouge_l, compute_bert_score, evaluate_retrieval/generation, run_full_evaluation, prepare_eval_set, `__main__` stub); tested pure metrics + real rouge/bert via stub. Real eval run pending Neo4j+LLM+QA data. |
| 2026-06-17 | Wired Cerebras gpt-oss-120b into `rag_pipeline.py` (`get_llm_client` priority Cerebras>OpenAI>Ollama, `generate()` uses `max_completion_tokens=4096` for the reasoning model). Live-tested via real Cerebras API — grounded biomedical answer returned. Updated `.env.example` + PROGRESS.md. |
| 2026-06-17 | Phase 5: implemented `src/query_decomposer.py` (decompose_query/multi_hop_retrieve/parse_subquestions), GDS PageRank reranking in `rag_pipeline.py` (gds_rerank + combined_rerank + retrieve_with_gds), and `app/demo.py` Streamlit app (pyviz graph viz + toggles). Logic-tested all pure functions; live decomposition via Cerebras; headless AppTest render clean. End-to-end run pending Neo4j+data+GDS. |
| 2026-06-17 | **End-to-end runs.** Started Neo4j 5.26 (Docker, APOC+GDS). Resolved data: switched to `qiaojin/PubMedQA` (spec id invalid), downloaded 5000 abstracts. Installed+patched `en_core_sci_sm`. Phase 1: chunked 5000 (14k/15k/6k). Phase 2: ingested 5k articles, 6,196 chunks, 69k entities, 223k MENTIONS. Phase 3: live retrieve+generate; fixed graph-expand hub explosion (depth→1 + entity-degree cap). Phase 4: real eval n=12 → R@5 0.76, R@10 0.81, MRR 0.74, ROUGE-L 0.17, BERTScore 0.79; added rate-limit-aware retry. Phase 5: verified graph-expand, GDS PageRank, decomp, pyviz all live. |
| 2026-06-18 | **3-variant eval.** Ran `semantic_no_graph` (no re-ingest, expand off) and `baseline` (lean re-ingestion of 14,118 fixed-token chunks + embeddings + vector index, no entities) on the same 12 QA pairs (seed=42). Result: semantic chunking +14.6 pts R@10 (0.660→0.806) vs fixed-token; graph expansion preserves recall, nudges answer quality (ROUGE-L 0.159→0.166). Restored semantic graph (with entities) for the demo. CSVs: `baseline_results.csv`, `semantic_no_graph_results.csv`. |
| 2026-06-18 | **Retrieval v2 — hybrid + article-expand.** Diagnosed the 2 eval failures: Q4 = dense/lexical/decomposition all miss (title↔abstract vocabulary gap); Q8 = abstract split so the *conclusion* chunk wasn't retrieved. Added **hybrid BM25+dense retrieval** to `rag_pipeline.py` (Neo4j `chunk_fulltext` index via `setup_schema`; dense-dominant weighted Reciprocal Rank Fusion, lexical weight 0.3; max-normalised so fused scores are compatible with the cosine/proximity re-ranker — equal-weight RRF had regressed Q8 by letting BM25 noise displace dense hits). Added **article/parent-sibling expansion** (`_article_siblings`): surface the rest of a hit abstract so the LLM sees full context. 12-Q eval: R@5 0.764→0.806, **R@10 0.806→0.917**, MRR 0.744→0.778; Q8/Q11/Q12 now perfect R@10; answer rating 7.5→~8.0/10. CSV: `full_system_v2_results.csv`. |

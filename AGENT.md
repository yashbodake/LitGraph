# AGENT.md — GraphRAG for Scientific Literature

> This file is the source of truth for any coding assistant (GitHub Copilot, Cursor, Claude, etc.) working on this project. Read this before generating any code.

---

## Project Identity

**Name:** GraphRAG for Scientific Literature  
**Stack:** Python 3.11+, Neo4j 5, sentence-transformers, SciSpacy, OpenAI/Ollama, Streamlit  
**Goal:** RAG pipeline over 5000 PubMed abstracts using semantic chunking + Neo4j graph traversal  
**Spec files:** `specs/` folder — read the relevant phase spec before implementing anything

---

## Folder Structure (Authoritative)

```
graphrag-scientific/
├── data/
│   ├── raw/                    # 5000 sampled PubMed abstracts (JSON)
│   ├── chunks/                 # Output JSONL files from chunker.py
│   └── qa/                     # Filtered pubmed_qa pairs + eval CSVs
│
├── src/
│   ├── chunker.py              # Phase 1
│   ├── graph_builder.py        # Phase 2
│   ├── rag_pipeline.py         # Phase 3
│   ├── evaluator.py            # Phase 4
│   └── query_decomposer.py     # Phase 5
│
├── notebooks/
│   ├── 01_chunk_visualisation.ipynb
│   ├── 02_graph_exploration.ipynb
│   └── 03_evaluation_report.ipynb
│
├── app/
│   └── demo.py                 # Phase 5 — Streamlit UI
│
├── specs/
│   ├── 00_PROJECT_OVERVIEW.md
│   ├── 01_PHASE1_CHUNKING.md
│   ├── 02_PHASE2_KNOWLEDGE_GRAPH.md
│   ├── 03_PHASE3_RETRIEVAL_GENERATION.md
│   ├── 04_PHASE4_EVALUATION.md
│   └── 05_PHASE5_DEMO_OPTIMISATION.md
│
├── AGENT.md                    # ← You are here
├── PROGRESS.md
├── requirements.txt
├── .env.example
└── README.md
```

---

## Environment Variables

All secrets live in `.env`. Never hardcode keys. Always use:

```python
from dotenv import load_dotenv
import os

load_dotenv()
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_LLM_URL  = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
```

**.env.example** (commit this, not `.env`):
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme
OPENAI_API_KEY=sk-...
LOCAL_LLM_URL=http://localhost:11434/api/generate
```

---

## Coding Conventions

### General

- Python 3.11+. Use type hints on all function signatures.
- `snake_case` for all functions and variables.
- Class names in `PascalCase`.
- Max line length: 100 characters.
- All public functions must have a docstring (one-liner minimum).

### Imports Order

```python
# 1. stdlib
import os, json, logging
from pathlib import Path

# 2. third-party
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# 3. local
from src.chunker import semantic_cluster_chunks
```

### Logging

Use `logging` instead of `print()` in all `src/` modules. Only `print()` in scripts and notebooks.

```python
import logging
logger = logging.getLogger(__name__)
logger.info("Ingesting %d chunks...", len(chunks))
```

### Error Handling

- Wrap Neo4j calls in try/except and log the error + query that failed.
- Never swallow exceptions silently.
- For LLM calls: catch rate limit errors and retry with exponential backoff (max 3 retries).

```python
import time

def call_llm_with_retry(client, **kwargs, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning("LLM call failed (attempt %d), retrying in %ds: %s", attempt+1, wait, e)
            time.sleep(wait)
```

---

## Neo4j Patterns

### Driver Initialisation (singleton)

```python
from neo4j import GraphDatabase

_driver = None

def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
    return _driver
```

### Session Pattern

Always use context manager. Never keep sessions open across functions.

```python
with get_driver().session() as session:
    result = session.run(cypher, **params)
    return [dict(r) for r in result]
```

### Batch Insert Pattern

Always use `UNWIND` for batch inserts, never loop single-row queries.

```python
# CORRECT
session.run("UNWIND $rows AS row MERGE (c:Chunk {chunk_id: row.chunk_id}) SET c.text = row.text", rows=batch)

# WRONG — never do this
for chunk in chunks:
    session.run("MERGE (c:Chunk {chunk_id: $id}) SET c.text = $text", id=chunk['chunk_id'], text=chunk['text'])
```

### Batch Size

- Chunk nodes (with embeddings): **50 per transaction** (embeddings are large)
- Other nodes / relationships: **500 per transaction**

---

## Embedding Conventions

- Model: `all-MiniLM-L6-v2` (384 dimensions) — **always this model, no substitutes**
- Singleton pattern — load once, reuse:

```python
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder
```

- Always normalise embeddings before cosine similarity comparisons:

```python
import numpy as np
embs = np.array(embeddings)
embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
```

- For batch embedding: use `embedder.encode(texts, batch_size=64, show_progress_bar=True)`

---

## Chunk Schema (JSONL)

Every chunk written to disk or passed between functions must conform to:

```python
{
    "article_id": str,       # PubMed article ID
    "chunk_id": str,         # f"{article_id}_{chunk_index}"
    "text": str,             # chunk text content
    "strategy": str,         # "fixed_token" | "sentence_boundary" | "semantic_cluster"
    "sentence_count": int,   # number of sentences in chunk
    "token_count": int       # approximate token count
}
```

---

## Cypher Query Guidelines

- Always use parameters (`$param`) — never f-strings in Cypher.
- Use `MERGE` (not `CREATE`) for nodes that may already exist.
- Use `MATCH` only when the node is guaranteed to exist.
- Always include `LIMIT` in exploratory queries.
- Name Cypher variables to match the node label (e.g. `(c:Chunk)`, `(e:Entity)`, `(a:Article)`).

---

## LLM Usage Guidelines

- Model for generation: `gpt-4o-mini` (OpenAI) or `mistral` (Ollama)
- Model for query decomposition: same — keep it consistent
- `temperature=0.2` for factual answer generation
- `temperature=0.4` for query decomposition (needs some creativity)
- `max_tokens=512` for answers, `max_tokens=256` for decomposition
- Always include a system prompt — never send bare user messages

---

## What the Coding Assistant MUST NOT Do

- Do not change the chunking model from `all-MiniLM-L6-v2` to anything else
- Do not use `CREATE` instead of `MERGE` in Neo4j inserts — duplicate nodes will break the pipeline
- Do not hardcode API keys, URIs, or passwords — always read from `.env`
- Do not use `print()` in `src/` modules — use `logging`
- Do not skip the `LIMIT` clause in exploratory Cypher queries
- Do not generate the SEMANTIC_SIMILAR edges by looping individual pairs — use batched matrix multiply
- Do not use async/await unless explicitly asked — keep all code synchronous for simplicity
- Do not change the chunk JSONL schema without updating this file

---

## When Adding a New Module

1. Create it under `src/`
2. Add its purpose and interface to the relevant spec file in `specs/`
3. Add any new dependencies to `requirements.txt`
4. Update `PROGRESS.md` with what was added

---

## Testing Guidance

- No formal test suite required, but each module should have an `if __name__ == "__main__":` block that runs a small smoke test.
- For `chunker.py`: test on 3 abstracts, print chunk counts per strategy.
- For `graph_builder.py`: test on 10 chunks, verify node counts in Neo4j.
- For `rag_pipeline.py`: test on 2 questions, print retrieved chunks and answer.
- For `evaluator.py`: test on 5 QA pairs, print metric values.

---

## Key Architectural Decisions (Do Not Reverse Without Good Reason)

| Decision | Rationale |
|---|---|
| HDBSCAN for semantic clustering | No need to pre-specify cluster count; handles noise gracefully |
| Depth ≤ 2 for graph traversal | Depth 3+ adds too much noise from weakly related entities |
| α=0.7 for combined re-ranking | Empirical default; tunable in Phase 4 evaluation |
| Semantic chunks as primary strategy | Best Recall@10 and BERTScore vs fixed/sentence strategies |
| UNWIND batch inserts | Orders of magnitude faster than single-row transactions |
| Singleton embedder | Avoids reloading 90MB model on every function call |

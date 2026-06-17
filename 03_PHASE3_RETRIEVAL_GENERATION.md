# Phase 3 — Graph-Enhanced Retrieval & Generation

## Goal

Build a `retrieve()` function that combines Neo4j vector search with graph traversal to fetch richer context than pure vector search alone. Then connect an LLM to generate final answers. Manually test on 10 PubMed QA pairs to validate the pipeline end-to-end.

---

## Prerequisites

- Phase 2 complete: Neo4j populated with chunks, entities, embeddings, and SEMANTIC_SIMILAR edges
- LLM access: OpenAI API key set in `.env`, OR Ollama running locally (`ollama pull mistral`)

---

## Module: `src/rag_pipeline.py`

### Full Interface

```python
class GraphRAGPipeline:
    def __init__(self, driver, embedder, llm_client):
        self.driver = driver
        self.embedder = embedder
        self.llm = llm_client

    def retrieve(self, query: str, top_k: int = 5, expand_graph: bool = True) -> list[dict]:
        """
        Step 1: Embed query
        Step 2: Vector search → initial top_k chunks
        Step 3 (if expand_graph): Graph traversal for additional context
        Step 4: Deduplicate + re-rank
        Returns list of chunk dicts with 'chunk_id', 'text', 'score'
        """

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """
        Formats retrieved chunks into a prompt.
        Calls LLM and returns the answer string.
        """

    def run(self, query: str, top_k: int = 5, expand_graph: bool = True) -> dict:
        """
        Orchestrates retrieve → generate.
        Returns {query, answer, retrieved_chunks, expand_graph}
        """
```

---

## Step 1 — Vector Search

Embed the query with the same `all-MiniLM-L6-v2` model used during ingestion, then call the Neo4j vector index:

```python
def _vector_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
    cypher = """
    CALL db.index.vector.queryNodes('chunk_embedding', $top_k, $embedding)
    YIELD node AS chunk, score
    RETURN chunk.chunk_id AS chunk_id,
           chunk.text     AS text,
           chunk.article_id AS article_id,
           score
    """
    with self.driver.session() as session:
        result = session.run(cypher, top_k=top_k, embedding=query_embedding)
        return [dict(r) for r in result]
```

---

## Step 2 — Graph Expansion

Given the initial top-k chunk IDs, traverse the graph to find additional contextually related chunks:

```python
def _graph_expand(self, seed_chunk_ids: list[str], depth: int = 2) -> list[dict]:
    cypher = """
    UNWIND $seed_ids AS seed_id
    MATCH (c:Chunk {chunk_id: seed_id})
    CALL apoc.path.subgraphNodes(c, {
        relationshipFilter: 'MENTIONS>|<MENTIONS',
        maxLevel: $depth,
        labelFilter: '+Chunk'
    }) YIELD node AS neighbor
    WHERE NOT neighbor.chunk_id IN $seed_ids
    RETURN DISTINCT neighbor.chunk_id AS chunk_id,
                     neighbor.text     AS text,
                     neighbor.article_id AS article_id
    """
    with self.driver.session() as session:
        result = session.run(cypher, seed_ids=seed_chunk_ids, depth=depth)
        return [dict(r) for r in result]
```

> **Alternative without APOC** (plain Cypher):
```cypher
UNWIND $seed_ids AS seed_id
MATCH (seed:Chunk {chunk_id: seed_id})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(neighbor:Chunk)
WHERE NOT neighbor.chunk_id IN $seed_ids
RETURN DISTINCT neighbor.chunk_id, neighbor.text, neighbor.article_id
```

---

## Step 3 — Re-ranking

After combining vector results + graph-expanded neighbors, re-rank using a combined score:

```
combined_score = α * vector_similarity + (1 - α) * graph_proximity_score
```

Where:
- `α = 0.7` (tunable)
- **Vector similarity** = cosine score from vector index (0–1), or 0.0 for graph-expanded chunks
- **Graph proximity score**: inverse of hop distance from seed chunks (depth-1 neighbor → 0.5, depth-2 → 0.25)

```python
def _rerank(self, vector_results: list[dict], graph_results: list[dict], alpha: float = 0.7) -> list[dict]:
    """
    Merges vector_results (with 'score') and graph_results (score=0).
    Assigns graph proximity scores based on which seed chunk each neighbor came from.
    Returns sorted list by combined_score descending.
    """
```

After re-ranking, **keep top 10 chunks** for the LLM prompt.

---

## Step 4 — LLM Generation

### Prompt Template

```python
SYSTEM_PROMPT = """You are a biomedical research assistant. 
Answer the question using ONLY the provided context passages. 
If the context doesn't contain enough information to answer, say so clearly.
Be concise and factual. Do not hallucinate."""

def _build_prompt(self, query: str, chunks: list[dict]) -> str:
    context = "\n\n".join([
        f"[Chunk {i+1} | Article {c['article_id']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ])
    return f"{context}\n\nQuestion: {query}\nAnswer:"
```

### OpenAI Client

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate(self, query, context_chunks):
    prompt = self._build_prompt(query, context_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=512
    )
    return response.choices[0].message.content
```

### Local LLM (Ollama fallback)

```python
import requests

def generate_local(query, context_chunks, model="mistral"):
    prompt = build_prompt(query, context_chunks)
    response = requests.post(
        os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate"),
        json={"model": model, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]
```

---

## Manual Test: 10 Questions

Select 10 questions from your filtered `pubmed_qa` QA pairs. For each, run:
1. `retrieve(query, top_k=5, expand_graph=False)` → `answer_no_graph`
2. `retrieve(query, top_k=5, expand_graph=True)` → `answer_with_graph`

Produce a comparison table (save as `data/qa/manual_test_10.md`):

| # | Question | Answer (No Graph) | Answer (With Graph) | Gold Answer | Graph Better? |
|---|---|---|---|---|---|
| 1 | ... | ... | ... | ... | ✅ / ❌ |

**Manual judgment criteria:**
- Does the answer address the question?
- Is it factually consistent with the gold answer?
- Does graph expansion add relevant biomedical detail?

---

## Deliverables Checklist

- [ ] `src/rag_pipeline.py` with `GraphRAGPipeline` class
- [ ] `retrieve()` working with both `expand_graph=True` and `False`
- [ ] `generate()` connected to OpenAI or local LLM
- [ ] `data/qa/manual_test_10.md` comparison table
- [ ] Console log showing retrieval time per query

---

## Dependencies for This Phase

```
neo4j
sentence-transformers
openai          # if using OpenAI
requests        # if using Ollama
python-dotenv
```

# Phase 2 — Build the Knowledge Graph in Neo4j

## Goal

Ingest all chunks (semantic strategy), extract biomedical entities, and load everything into a Neo4j graph with a vector index on chunk embeddings. The graph enables both vector similarity search and graph traversal in Phase 3.

---

## Prerequisites

- Phase 1 complete: `data/chunks/semantic_cluster_chunks.jsonl` exists
- Neo4j Desktop or Docker running locally with **APOC** and **GDS** plugins installed
- `.env` populated with `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

### Neo4j via Docker (quickstart)

```bash
docker run \
  --name neo4j-graphrag \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -e NEO4J_PLUGINS='["apoc","graph-data-science"]' \
  neo4j:5
```

---

## Graph Schema

### Nodes

```
(:Article {
    article_id: STRING,       # unique
    title: STRING,
    pub_date: STRING
})

(:Chunk {
    chunk_id: STRING,         # unique — e.g. "12345678_0"
    article_id: STRING,       # FK reference
    text: STRING,
    embedding: LIST<FLOAT>,   # 384 dims — all-MiniLM-L6-v2
    strategy: STRING,
    token_count: INTEGER
})

(:Entity {
    name: STRING,             # lowercased
    type: STRING              # e.g. CHEMICAL, DISEASE, GENE, ORG, etc.
})
```

### Relationships

```
(:Article)-[:HAS_CHUNK]->(:Chunk)
(:Chunk)-[:MENTIONS]->(:Entity)
(:Chunk)-[:SEMANTIC_SIMILAR {score: FLOAT}]->(:Chunk)   # cosine > 0.85
```

---

## Constraints & Indexes

Run these once at setup (idempotent):

```cypher
CREATE CONSTRAINT article_id_unique IF NOT EXISTS
  FOR (a:Article) REQUIRE a.article_id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
  FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;

CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS
  FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE;

-- Vector index (requires Neo4j 5.11+)
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
  FOR (c:Chunk) ON (c.embedding)
  OPTIONS {
    indexConfig: {
      `vector.dimensions`: 384,
      `vector.similarity_function`: 'cosine'
    }
  };
```

---

## Ingestion Pipeline (`src/graph_builder.py`)

### Module Interface

```python
def setup_schema(driver) -> None:
    """Creates constraints and vector index."""

def ingest_articles(driver, abstracts: list[dict]) -> None:
    """Batch-inserts Article nodes."""

def ingest_chunks(driver, chunks: list[dict], embedder) -> None:
    """
    Batch-inserts Chunk nodes with embeddings.
    Uses UNWIND for performance.
    Batch size: 100 chunks per transaction.
    """

def extract_entities(text: str, nlp) -> list[dict]:
    """
    Runs SciSpacy NER on a chunk text.
    Returns list of {name, type} dicts.
    """

def ingest_entities(driver, chunk_id: str, entities: list[dict]) -> None:
    """MERGE Entity nodes and create MENTIONS relationships."""

def build_semantic_similar_edges(driver, embedder, similarity_threshold: float = 0.85) -> None:
    """
    Computes cosine similarity between all chunk embeddings.
    Creates SEMANTIC_SIMILAR edges where score > threshold.
    Use batch matrix multiply to avoid O(n²) single-pair loops.
    """

def run_full_ingestion(chunks_path: str, abstracts: list[dict]) -> None:
    """Master function: calls all steps above in order."""
```

---

## Batch Insert Pattern (UNWIND)

**Article nodes:**
```cypher
UNWIND $rows AS row
MERGE (a:Article {article_id: row.article_id})
SET a.title = row.title, a.pub_date = row.pub_date
```

**Chunk nodes + HAS_CHUNK relationship:**
```cypher
UNWIND $rows AS row
MATCH (a:Article {article_id: row.article_id})
MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.text = row.text,
    c.embedding = row.embedding,
    c.strategy = row.strategy,
    c.token_count = row.token_count
MERGE (a)-[:HAS_CHUNK]->(c)
```

**Entity nodes + MENTIONS relationship:**
```cypher
UNWIND $entities AS ent
MERGE (e:Entity {name: ent.name, type: ent.type})
WITH e, ent
MATCH (c:Chunk {chunk_id: ent.chunk_id})
MERGE (c)-[:MENTIONS]->(e)
```

**SEMANTIC_SIMILAR edges:**
```cypher
UNWIND $pairs AS pair
MATCH (c1:Chunk {chunk_id: pair.chunk_id_1})
MATCH (c2:Chunk {chunk_id: pair.chunk_id_2})
MERGE (c1)-[:SEMANTIC_SIMILAR {score: pair.score}]->(c2)
```

---

## Entity Extraction (SciSpacy)

```python
import scispacy
import spacy

nlp = spacy.load("en_core_sci_sm")

def extract_entities(text: str, nlp) -> list[dict]:
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "name": ent.text.lower().strip(),
            "type": ent.label_
        })
    return entities
```

> **Filter out**: entities with `len(name) < 3` or purely numeric names to reduce noise.

---

## Semantic Similar Edge Construction

To avoid O(n²) loops across 10k+ chunks, compute similarity in **batches using numpy**:

```python
import numpy as np
from itertools import combinations

def build_semantic_similar_edges(driver, all_embeddings: list, chunk_ids: list, threshold=0.85, batch_size=500):
    """
    Splits chunk embeddings into batches.
    For each batch × batch pair, computes cosine similarity matrix.
    Collects pairs above threshold and bulk-inserts into Neo4j.
    """
    # Normalise once
    embs = np.array(all_embeddings)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    
    pairs = []
    for i in range(0, len(embs), batch_size):
        batch = embs[i:i+batch_size]
        sims = batch @ embs.T  # (batch, total)
        for bi, row in enumerate(sims):
            global_i = i + bi
            for j, score in enumerate(row):
                if j > global_i and score > threshold:
                    pairs.append({
                        "chunk_id_1": chunk_ids[global_i],
                        "chunk_id_2": chunk_ids[j],
                        "score": float(score)
                    })
    # Bulk insert pairs
    ...
```

> **Warning:** With 10k chunks, full pairwise is 10k² = 100M comparisons. Use batched matrix multiply and only store above-threshold pairs. Expect ~50k–200k SEMANTIC_SIMILAR edges.

---

## Expected Graph Size

| Node / Edge Type | Approximate Count |
|---|---|
| Article nodes | ~5,000 |
| Chunk nodes | ~11,000 |
| Entity nodes | ~10,000 |
| HAS_CHUNK edges | ~11,000 |
| MENTIONS edges | ~50,000–80,000 |
| SEMANTIC_SIMILAR edges | ~50,000–200,000 |

---

## Validation Queries

Run these in Neo4j Browser to verify ingestion:

```cypher
-- Node counts
MATCH (a:Article) RETURN count(a) AS articles;
MATCH (c:Chunk)   RETURN count(c) AS chunks;
MATCH (e:Entity)  RETURN count(e) AS entities;

-- Sample subgraph
MATCH (a:Article)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
RETURN a, c, e LIMIT 25;

-- Test vector index
CALL db.index.vector.queryNodes('chunk_embedding', 5, $embedding)
YIELD node, score
RETURN node.chunk_id, node.text, score;
```

---

## Deliverables Checklist

- [ ] Neo4j running with APOC + GDS plugins confirmed
- [ ] `src/graph_builder.py` with all functions above
- [ ] All Article, Chunk, Entity nodes ingested
- [ ] Vector index `chunk_embedding` created and queryable
- [ ] SEMANTIC_SIMILAR edges built
- [ ] Validation queries pass in Neo4j Browser

---

## Dependencies for This Phase

```
neo4j          # Python driver
scispacy
en_core_sci_sm
numpy
sentence-transformers
python-dotenv
```

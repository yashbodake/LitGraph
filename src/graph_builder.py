"""Phase 2 — build the knowledge graph in Neo4j.

Ingests semantic chunks (from Phase 1), extracts biomedical entities with
SciSpacy, and loads Article -> Chunk -> Entity nodes plus a vector index on
chunk embeddings and ``SEMANTIC_SIMILAR`` edges (cosine > threshold).

Driver/embedder are singletons (see AGENT.md). Batch inserts use ``UNWIND``;
chunk nodes go 50 per transaction (embeddings are large), everything else 500.
"""

# 1. stdlib
import json
import logging
import os
from pathlib import Path

# 2. third-party
import numpy as np

# 3. local
from src.chunker import get_embedder

logger = logging.getLogger(__name__)

# Batch sizes pinned by AGENT.md (authoritative).
CHUNK_BATCH_SIZE = 50
EDGE_BATCH_SIZE = 500
DEFAULT_SIMILARITY_THRESHOLD = 0.85

# --------------------------------------------------------------------------- #
# Singletons
# --------------------------------------------------------------------------- #
_driver = None


def get_driver():
    """Return the shared Neo4j driver, created once from env/Streamlit secrets."""
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase

        from src.config import get_secret

        uri = get_secret("NEO4J_URI", "bolt://localhost:7687")
        user = get_secret("NEO4J_USER", "neo4j")
        password = get_secret("NEO4J_PASSWORD")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4j driver created for %s", uri)
    return _driver


_nlp = None


def get_nlp():
    """Return a cached SciSpacy ``en_core_sci_sm`` pipeline (loaded once)."""
    global _nlp
    if _nlp is None:
        import scispacy  # noqa: F401  (registers the biomedical pipeline)
        import spacy

        logger.info("Loading SciSpacy model 'en_core_sci_sm'...")
        _nlp = spacy.load("en_core_sci_sm")
    return _nlp


def _chunks_from_jsonl(chunks_path: str) -> list[dict]:
    """Read a Phase-1 JSONL chunk file into a list of chunk dicts."""
    chunks: list[dict] = []
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    return chunks


def _run_cypher(session, cypher: str, **params):
    """Execute one Cypher statement, logging the failing query on error."""
    try:
        return list(session.run(cypher, **params))
    except Exception:
        logger.exception("Cypher failed: %s | params=%s", cypher, list(params))
        raise


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def setup_schema(driver) -> None:
    """Create uniqueness constraints and the chunk-embedding vector index."""
    statements = [
        """
        CREATE CONSTRAINT article_id_unique IF NOT EXISTS
          FOR (a:Article) REQUIRE a.article_id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
          FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
        """,
        """
        CREATE CONSTRAINT entity_name_type_unique IF NOT EXISTS
          FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE
        """,
        """
        CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
          FOR (c:Chunk) ON (c.embedding)
          OPTIONS {
            indexConfig: {
              `vector.dimensions`: 384,
              `vector.similarity_function`: 'cosine'
            }
          }
        """,
        # BM25 full-text index for hybrid retrieval (Reciprocal Rank Fusion).
        """
        CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
          FOR (c:Chunk) ON EACH [c.text]
        """,
    ]
    with driver.session() as session:
        for cypher in statements:
            _run_cypher(session, cypher)
    logger.info("Schema (constraints + vector + fulltext indexes) ready.")


# --------------------------------------------------------------------------- #
# Articles
# --------------------------------------------------------------------------- #
def ingest_articles(driver, abstracts: list[dict]) -> None:
    """Batch-insert Article nodes (500 per transaction) via UNWIND."""
    rows = [
        {
            "article_id": str(a.get("article_id", "")),
            "title": a.get("title", ""),
            "pub_date": a.get("pub_date", ""),
        }
        for a in abstracts
    ]
    cypher = """
    UNWIND $rows AS row
    MERGE (a:Article {article_id: row.article_id})
    SET a.title = row.title, a.pub_date = row.pub_date
    """
    with driver.session() as session:
        for i in range(0, len(rows), EDGE_BATCH_SIZE):
            _run_cypher(session, cypher, rows=rows[i:i + EDGE_BATCH_SIZE])
    logger.info("Ingested %d Article nodes.", len(rows))


# --------------------------------------------------------------------------- #
# Chunks (with embeddings)
# --------------------------------------------------------------------------- #
def ingest_chunks(
    driver, chunks: list[dict], embedder, embeddings: list = None
) -> None:
    """Batch-insert Chunk nodes with embeddings + HAS_CHUNK edges (50/tx).

    ``embeddings`` may be supplied to avoid re-embedding; otherwise the texts
    are encoded here with ``embedder``.
    """
    texts = [c["text"] for c in chunks]
    if embeddings is None:
        logger.info("Embedding %d chunks...", len(texts))
        embeddings = embedder.encode(
            texts, batch_size=64, show_progress_bar=True
        )

    cypher = """
    UNWIND $rows AS row
    MATCH (a:Article {article_id: row.article_id})
    MERGE (c:Chunk {chunk_id: row.chunk_id})
    SET c.text = row.text,
        c.embedding = row.embedding,
        c.strategy = row.strategy,
        c.token_count = row.token_count,
        c.article_id = row.article_id
    MERGE (a)-[:HAS_CHUNK]->(c)
    """
    with driver.session() as session:
        for start in range(0, len(chunks), CHUNK_BATCH_SIZE):
            window = chunks[start:start + CHUNK_BATCH_SIZE]
            emb_window = embeddings[start:start + CHUNK_BATCH_SIZE]
            rows = [
                {
                    "article_id": c["article_id"],
                    "chunk_id": c["chunk_id"],
                    "text": c["text"],
                    "embedding": list(map(float, emb)),
                    "strategy": c["strategy"],
                    "token_count": c["token_count"],
                }
                for c, emb in zip(window, emb_window)
            ]
            _run_cypher(session, cypher, rows=rows)
    logger.info("Ingested %d Chunk nodes (with embeddings).", len(chunks))


# --------------------------------------------------------------------------- #
# Entities (SciSpacy NER)
# --------------------------------------------------------------------------- #
def extract_entities(text: str, nlp) -> list[dict]:
    """Run SciSpacy NER on ``text``, returning deduplicated {name, type} dicts.

    Filters out noise: names shorter than 3 chars or purely numeric.
    """
    doc = nlp(text)
    seen: set[tuple[str, str]] = set()
    entities: list[dict] = []
    for ent in doc.ents:
        name = ent.text.lower().strip()
        if len(name) < 3 or name.isdigit():
            continue
        key = (name, ent.label_)
        if key in seen:
            continue
        seen.add(key)
        entities.append({"name": name, "type": ent.label_})
    return entities


def ingest_entities(driver, chunk_id: str, entities: list[dict]) -> None:
    """MERGE Entity nodes and create MENTIONS relationships for one chunk."""
    if not entities:
        return
    rows = [{"name": e["name"], "type": e["type"]} for e in entities]
    cypher = """
    UNWIND $entities AS ent
    MERGE (e:Entity {name: ent.name, type: ent.type})
    WITH e, ent
    MATCH (c:Chunk {chunk_id: $chunk_id})
    MERGE (c)-[:MENTIONS]->(e)
    """
    with driver.session() as session:
        _run_cypher(session, cypher, chunk_id=chunk_id, entities=rows)


# --------------------------------------------------------------------------- #
# SEMANTIC_SIMILAR edges (batched matrix multiply)
# --------------------------------------------------------------------------- #
def _find_similar_pairs(
    embeddings, chunk_ids: list[str], threshold: float, batch_size: int
) -> list[dict]:
    """Return chunk-id pairs whose cosine similarity exceeds ``threshold``.

    Uses batched matrix multiplication to avoid an O(n^2) Python loop.
    Pure function (no DB) so it can be unit-tested without Neo4j.
    """
    embs = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    pairs: list[dict] = []
    n = len(embs)
    for i in range(0, n, batch_size):
        sims = embs[i:i + batch_size] @ embs.T  # (batch, total)
        for offset, row in enumerate(sims):
            global_i = i + offset
            for j in range(global_i + 1, n):
                score = float(row[j])
                if score > threshold:
                    pairs.append(
                        {
                            "chunk_id_1": chunk_ids[global_i],
                            "chunk_id_2": chunk_ids[j],
                            "score": score,
                        }
                    )
    return pairs


def build_semantic_similar_edges(
    driver, embeddings, chunk_ids: list[str], threshold: float = 0.85,
    batch_size: int = EDGE_BATCH_SIZE,
) -> None:
    """Create SEMANTIC_SIMILAR edges for chunk pairs with cosine > threshold."""
    pairs = _find_similar_pairs(embeddings, chunk_ids, threshold, batch_size)
    logger.info(
        "Found %d SEMANTIC_SIMILAR pairs (threshold=%.2f).", len(pairs), threshold
    )

    cypher = """
    UNWIND $pairs AS pair
    MATCH (c1:Chunk {chunk_id: pair.chunk_id_1})
    MATCH (c2:Chunk {chunk_id: pair.chunk_id_2})
    MERGE (c1)-[:SEMANTIC_SIMILAR {score: pair.score}]->(c2)
    """
    with driver.session() as session:
        for i in range(0, len(pairs), EDGE_BATCH_SIZE):
            _run_cypher(session, cypher, pairs=pairs[i:i + EDGE_BATCH_SIZE])


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def log_graph_counts(driver) -> dict:
    """Query and log node/edge counts; return them as a dict."""
    queries = {
        "articles": "MATCH (a:Article) RETURN count(a) AS n",
        "chunks": "MATCH (c:Chunk) RETURN count(c) AS n",
        "entities": "MATCH (e:Entity) RETURN count(e) AS n",
        "has_chunk": "MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) AS n",
        "mentions": "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS n",
        "semantic_similar": "MATCH ()-[r:SEMANTIC_SIMILAR]->() RETURN count(r) AS n",
    }
    counts: dict[str, int] = {}
    with driver.session() as session:
        for key, cypher in queries.items():
            result = _run_cypher(session, cypher)
            counts[key] = result[0]["n"] if result else 0
    logger.info("Graph counts: %s", counts)
    return counts


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_full_ingestion(chunks_path: str, abstracts: list[dict]) -> None:
    """Master pipeline: schema -> articles -> chunks -> entities -> edges."""
    driver = get_driver()
    setup_schema(driver)
    ingest_articles(driver, abstracts)

    chunks = _chunks_from_jsonl(chunks_path)
    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks (once, reused for edges)...", len(texts))
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True)

    ingest_chunks(driver, chunks, embedder, embeddings=embeddings)

    nlp = get_nlp()
    for i, chunk in enumerate(chunks, start=1):
        entities = extract_entities(chunk["text"], nlp)
        ingest_entities(driver, chunk["chunk_id"], entities)
        if i % 500 == 0:
            logger.info("Extracted entities for %d/%d chunks...", i, len(chunks))

    build_semantic_similar_edges(
        driver, embeddings, [c["chunk_id"] for c in chunks]
    )

    log_graph_counts(driver)


# --------------------------------------------------------------------------- #
# Smoke / full-run entry point (AGENT.md: test on 10 chunks)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    parser = argparse.ArgumentParser(description="Ingest the knowledge graph.")
    parser.add_argument(
        "-c", "--chunks", default="data/chunks/semantic_cluster_chunks.jsonl",
        help="path to semantic chunk JSONL (default: data/chunks/...)",
    )
    parser.add_argument(
        "-a", "--abstracts", default="data/raw/abstracts.json",
        help="path to abstracts JSON (default: data/raw/abstracts.json)",
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=None,
        help="limit #chunks for a quick smoke test (e.g. 10)",
    )
    args = parser.parse_args()

    if not Path(args.chunks).exists():
        print(f"Chunks file not found: {args.chunks}")
        print("Run `python -m src.chunker` first to generate Phase-1 chunks.")
        sys.exit(1)

    abstracts = []
    if Path(args.abstracts).exists():
        with open(args.abstracts, encoding="utf-8") as fh:
            abstracts = json.load(fh)
    else:
        logger.warning("Abstracts file missing; Article nodes will be skipped.")

    chunks = _chunks_from_jsonl(args.chunks)
    if args.limit:
        keep_ids = {c["chunk_id"] for c in chunks[: args.limit]}
        chunks = chunks[: args.limit]
        # Restrict articles to those referenced by the sampled chunks.
        keep_arts = {c["article_id"] for c in chunks}
        abstracts = [a for a in abstracts if str(a.get("article_id")) in keep_arts]
        # Write the trimmed chunk set to a temp file for run_full_ingestion.
        tmp = Path(args.chunks).with_suffix(".smoke.jsonl")
        with open(tmp, "w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        args.chunks = str(tmp)
        logger.info("Smoke test: %d chunks, %d articles.", len(chunks), len(abstracts))

    run_full_ingestion(args.chunks, abstracts)

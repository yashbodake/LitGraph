"""Phase 3 — graph-enhanced retrieval & generation.

Combines Neo4j vector search with knowledge-graph traversal to fetch richer
context than pure vector search, then generates an answer with an LLM
(OpenAI primary, local Ollama fallback).

Driver/embedder are the singletons defined in the earlier phases; heavy LLM
imports are lazy so the module imports cleanly without them installed.
"""

# 1. stdlib
import logging
import os
import re
import time
import uuid

# 2. third-party — (none at import time; openai/requests loaded lazily)

# 3. local
from src.chunker import get_embedder
from src.graph_builder import get_driver

logger = logging.getLogger(__name__)

# Defaults pinned by AGENT.md / the Phase 3 spec.
DEFAULT_TOP_K = 5
DEFAULT_DEPTH = 1
DEFAULT_ALPHA = 0.7
FINAL_CONTEXT_K = 10
# Hub entities (e.g. "patients", "cancer") can be mentioned by thousands of
# chunks, which makes Chunk->Entity->Chunk explode. Only traverse through
# reasonably specific entities to keep neighbours genuinely related.
MAX_ENTITY_DEGREE = 40
ANSWER_MODEL_OPENAI = "gpt-4o-mini"
ANSWER_MODEL_OLLAMA = "mistral"
ANSWER_MODEL_CEREBRAS = "gpt-oss-120b"
ANSWER_TEMPERATURE = 0.2
ANSWER_MAX_TOKENS = 512
# gpt-oss is a reasoning model: it spends tokens on internal reasoning before
# the visible answer, so it needs a larger completion budget than gpt-4o-mini.
CEREBRAS_MAX_COMPLETION_TOKENS = 4096
MAX_LLM_RETRIES = 3
VECTOR_INDEX = "chunk_embedding"
# Hybrid retrieval: Neo4j full-text (BM25) index fused with dense vectors.
FULLTEXT_INDEX = "chunk_fulltext"
RRF_K = 60  # Reciprocal Rank Fusion constant (standard value).
# Dense stays primary; lexical acts as a gentle booster so noisy BM25 hits
# cannot displace strong vector matches (e.g. short, generic queries).
RRF_DENSE_WEIGHT = 1.0
RRF_LEXICAL_WEIGHT = 0.3

# GDS PageRank re-ranking (Phase 5B).
GDS_RERANK_ALPHA = 0.6
GDS_RERANK_BETA = 0.4

SYSTEM_PROMPT = """You are a biomedical research assistant.
Answer the question using ONLY the provided context passages.
If the context doesn't contain enough information to answer, say so clearly.
Be concise and factual. Do not hallucinate."""


# --------------------------------------------------------------------------- #
# LLM client factory + retry
# --------------------------------------------------------------------------- #
def get_llm_client():
    """Return ``(provider, client)``. Priority: Cerebras > OpenAI > Ollama.

    Cerebras is OpenAI-compatible, so it uses the OpenAI client pointed at the
    Cerebras base URL with the gpt-oss model.
    """
    from dotenv import load_dotenv

    from src.config import get_secret

    load_dotenv()
    if get_secret("CEREBRAS_API_KEY"):
        from openai import OpenAI

        model = get_secret("CEREBRAS_MODEL", ANSWER_MODEL_CEREBRAS)
        logger.info("Using Cerebras LLM (%s).", model)
        return "cerebras", OpenAI(
            api_key=get_secret("CEREBRAS_API_KEY"),
            base_url=get_secret(
                "CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"
            ),
            max_retries=0,
        )
    if get_secret("OPENAI_API_KEY"):
        from openai import OpenAI

        logger.info("Using OpenAI LLM (%s).", ANSWER_MODEL_OPENAI)
        # max_retries=0 so our own call_llm_with_retry owns backoff.
        return "openai", OpenAI(
            api_key=get_secret("OPENAI_API_KEY"), max_retries=0
        )
    logger.info("No cloud key set; using local Ollama LLM (%s).", ANSWER_MODEL_OLLAMA)
    return "ollama", None


def _is_rate_limited(exc: Exception) -> bool:
    """True if ``exc`` looks like an API rate-limit / quota error (HTTP 429)."""
    if getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return any(tok in msg for tok in ("429", "rate limit", "too many requests", "quota"))


def _with_retry(action, what: str = "LLM call", max_retries: int = MAX_LLM_RETRIES):
    """Run ``action`` with backoff; rate-limit (429) errors get a longer cooldown."""
    for attempt in range(max_retries):
        try:
            return action()
        except Exception as exc:  # retry any LLM error, per AGENT.md error handling
            if attempt == max_retries - 1:
                raise
            # Rate limits need a longer cooldown than a transient blip.
            base = 2 ** attempt
            wait = base * 20 if _is_rate_limited(exc) else base
            logger.warning(
                "%s failed (attempt %d), retrying in %ds: %s",
                what, attempt + 1, wait, exc,
            )
            time.sleep(wait)


def call_llm_with_retry(client, max_retries: int = MAX_LLM_RETRIES, **kwargs):
    """OpenAI chat completion with retry (AGENT.md error-handling template)."""
    return _with_retry(
        lambda: client.chat.completions.create(**kwargs), "OpenAI call", max_retries
    )


def _ollama_generate(prompt: str, model: str) -> str:
    """POST to the local Ollama ``/api/generate`` endpoint; return the text."""
    import requests

    url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
    resp = requests.post(
        url, json={"model": model, "prompt": prompt, "stream": False}, timeout=120
    )
    resp.raise_for_status()
    return resp.json()["response"]


# --------------------------------------------------------------------------- #
# GDS PageRank re-ranking (Phase 5B)
# --------------------------------------------------------------------------- #
# Project a subgraph of the retrieved chunks + their entities. The inner Cypher
# queries use ``$chunk_ids`` resolved from the GDS ``parameters`` config (NOT the
# outer session params), per the gds.graph.project.cypher contract.
_GDS_PROJECT_CYPHER = """
CALL gds.graph.project.cypher(
  $graph_name,
  'MATCH (n)
     WHERE (n:Chunk AND n.chunk_id IN $chunk_ids)
        OR (n:Entity AND EXISTS { (c:Chunk)-[:MENTIONS]->(n)
                                  WHERE c.chunk_id IN $chunk_ids })
   RETURN id(n) AS id',
  'MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
    WHERE c.chunk_id IN $chunk_ids
   RETURN id(c) AS source, id(e) AS target',
  { parameters: { chunk_ids: $chunk_ids } }
)
YIELD graphName
RETURN graphName
"""

_GDS_PAGERANK_CYPHER = """
CALL gds.pageRank.stream($graph_name)
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
WHERE node:Chunk
RETURN node.chunk_id AS chunk_id, score
ORDER BY score DESC
"""


def combined_rerank(
    chunks: list[dict],
    gds_scores: dict[str, float],
    alpha: float = GDS_RERANK_ALPHA,
    beta: float = GDS_RERANK_BETA,
) -> list[dict]:
    """Blend retrieval score with normalized GDS PageRank; return re-sorted list.

    ``final = alpha * base_score + beta * normalized_pagerank`` where the base
    score is each chunk's existing ``score`` (the vector+graph combined score
    from ``retrieve``) and the PageRank scores are min-max normalised to [0, 1].

    Pure function (no DB) so it is unit-testable without a live Neo4j.
    """
    max_gds = max(gds_scores.values()) if gds_scores else 0.0
    for chunk in chunks:
        raw = gds_scores.get(chunk["chunk_id"], 0.0)
        norm = (raw / max_gds) if max_gds > 0 else 0.0
        chunk["score"] = alpha * chunk.get("score", 0.0) + beta * norm
    return sorted(chunks, key=lambda c: c["score"], reverse=True)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class GraphRAGPipeline:
    """Graph-enhanced RAG: vector search + graph expansion + LLM generation."""

    def __init__(self, driver=None, embedder=None, llm_client=None, provider=None):
        """Wire up driver, embedder and LLM client (defaults from singletons/env)."""
        self.driver = driver or get_driver()
        self.embedder = embedder or get_embedder()
        if llm_client is None or provider is None:
            resolved_provider, resolved_client = get_llm_client()
            llm_client = llm_client or resolved_client
            provider = provider or resolved_provider
        self.llm = llm_client
        self.provider = provider

    # -- retrieval -----------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        expand_graph: bool = True,
        hybrid: bool = True,
        expand_article: bool = True,
    ) -> list[dict]:
        """Embed the query, retrieve (hybrid BM25+dense), graph-expand, article-expand, re-rank."""
        start = time.time()
        query_embedding = self.embedder.encode([query])[0].tolist()
        if hybrid:
            base_results = self._hybrid_search(query, query_embedding, top_k)
        else:
            base_results = self._vector_search(query_embedding, top_k)

        graph_results: list[dict] = []
        if expand_graph and base_results:
            seed_ids = [r["chunk_id"] for r in base_results]
            graph_results = self._graph_expand(seed_ids, depth=DEFAULT_DEPTH)

        ranked = self._rerank(base_results, graph_results, alpha=DEFAULT_ALPHA)
        # Parent/article-level context: prioritise full abstracts of the most
        # relevant articles over marginally-related tail chunks.
        if expand_article and ranked:
            head = ranked[:5]
            tail = ranked[5:]
            head_ids = {c["chunk_id"] for c in head}
            siblings = self._article_siblings(head, head=5)
            merged: list[dict] = []
            seen_ids: set[str] = set()
            for chunk in head + siblings + tail:
                cid = chunk["chunk_id"]
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    merged.append(chunk)
            ranked = merged[:FINAL_CONTEXT_K]
        else:
            ranked = ranked[:FINAL_CONTEXT_K]
        logger.info(
            "retrieve('%s') -> %d chunks in %.2fs (hybrid=%s, graph=%s, article=%s)",
            query[:60], len(ranked), time.time() - start,
            hybrid, expand_graph, expand_article,
        )
        return ranked

    def _vector_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Query the ``chunk_embedding`` vector index for the top-k chunks."""
        cypher = """
        CALL db.index.vector.queryNodes($index, $top_k, $embedding)
        YIELD node AS chunk, score
        RETURN chunk.chunk_id   AS chunk_id,
               chunk.text       AS text,
               chunk.article_id AS article_id,
               score
        """
        with self.driver.session() as session:
            result = session.run(
                cypher, index=VECTOR_INDEX, top_k=top_k, embedding=query_embedding
            )
            return [dict(r) for r in result]

    @staticmethod
    def _sanitize_fulltext_query(query: str) -> str:
        """Strip Lucene-special chars so free text is queried as bag-of-words."""
        terms = re.sub(r"[^\w\s]", " ", query).strip()
        return terms or query

    def _lexical_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 full-text search over Chunk.text via the Neo4j full-text index.

        Falls back to an empty list (dense-only) if the index is missing or the
        query yields no terms.
        """
        terms = self._sanitize_fulltext_query(query)
        if not terms:
            return []
        cypher = """
        CALL db.index.fulltext.queryNodes($index, $terms) YIELD node AS chunk, score
        RETURN chunk.chunk_id   AS chunk_id,
               chunk.text       AS text,
               chunk.article_id AS article_id,
               score
        ORDER BY score DESC LIMIT $top_k
        """
        try:
            with self.driver.session() as session:
                return [
                    dict(r)
                    for r in session.run(
                        cypher,
                        index=FULLTEXT_INDEX,
                        terms=terms,
                        top_k=top_k,
                    )
                ]
        except Exception as exc:
            logger.warning("Lexical search failed (%s); using dense-only.", exc)
            return []

    def _hybrid_search(
        self, query: str, query_embedding: list[float], top_k: int
    ) -> list[dict]:
        """Fuse dense vector + lexical (BM25) results via Reciprocal Rank Fusion.

        Each result's ``score`` becomes its fused RRF score so downstream
        re-ranking (graph proximity) blends with retrieval quality.
        """
        dense = self._vector_search(query_embedding, top_k)
        lexical = self._lexical_search(query, top_k)
        scores: dict[str, float] = {}
        meta: dict[str, dict] = {}
        for rank, r in enumerate(dense):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + RRF_DENSE_WEIGHT / (
                RRF_K + rank + 1
            )
            meta[cid] = r
        for rank, r in enumerate(lexical):
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + RRF_LEXICAL_WEIGHT / (
                RRF_K + rank + 1
            )
            meta.setdefault(cid, r)
        ordered = sorted(scores, key=scores.get, reverse=True)
        # Max-normalise so fused scores live in [0, 1], compatible with cosine
        # similarities and graph-proximity tiers in the downstream re-ranker.
        top = scores[ordered[0]] if ordered else 1.0
        return [
            {**meta[cid], "score": scores[cid] / top if top else 0.0}
            for cid in ordered
        ]

    def _article_siblings(
        self, chunks: list[dict], head: int = 5
    ) -> list[dict]:
        """Return sibling chunks sharing an article with the ``head`` top chunks.

        Parent/article-level context: when an abstract is split into several
        chunks and only part of it is retrieved, surface the rest so the LLM
        sees the complete abstract. Siblings are sourced only from the top
        ``head`` chunks (the most relevant articles) to avoid context bloat.
        Only the supplied ``chunks`` are excluded; low-ranked graph neighbours
        are NOT excluded so a sibling can be promoted into the context.
        """
        aids = list(
            dict.fromkeys(
                c["article_id"]
                for c in chunks[:head]
                if c.get("article_id") is not None
            )
        )
        if not aids:
            return []
        seen = [c["chunk_id"] for c in chunks]
        cypher = """
        UNWIND $aids AS aid
        MATCH (:Article {article_id: aid})-[:HAS_CHUNK]->(ch:Chunk)
        WHERE NOT ch.chunk_id IN $seen
        RETURN ch.chunk_id AS chunk_id, ch.text AS text,
               ch.article_id AS article_id
        """
        try:
            with self.driver.session() as session:
                rows = [
                    {
                        **dict(r),
                        "vector_score": 0.0,
                        "graph_proximity": 0.0,
                    }
                    for r in session.run(cypher, aids=aids, seen=seen)
                ]
            return rows
        except Exception as exc:
            logger.warning("Article-sibling expansion failed (%s).", exc)
            return []

    def _graph_expand(
        self, seed_chunk_ids: list[str], depth: int = DEFAULT_DEPTH
    ) -> list[dict]:
        """Find chunk neighbours sharing entities with seeds, tagged by proximity.

        Uses the spec's plain-Cypher alternative (no APOC) so each neighbour can
        be tagged with an exact hop tier: depth-1 -> 0.5, depth-2 -> 0.25.
        """
        neighbors: dict[str, dict] = {}
        for row in self._query_neighbors(seed_chunk_ids, level=1):
            neighbors[row["chunk_id"]] = {**row, "graph_proximity": 0.5}
        if depth >= 2:
            for row in self._query_neighbors(seed_chunk_ids, level=2):
                # Keep the best (closest) proximity; level-1 already present wins.
                if row["chunk_id"] not in neighbors:
                    neighbors[row["chunk_id"]] = {**row, "graph_proximity": 0.25}
        return list(neighbors.values())

    def _query_neighbors(self, seed_ids: list[str], level: int) -> list[dict]:
        """Return chunk dicts related to ``seed_ids`` at the given hop level.

        Hub entities (mentioned by > ``MAX_ENTITY_DEGREE`` chunks) are skipped so
        the Chunk->Entity->Chunk traversal stays focused and fast.
        """
        if level == 1:
            cypher = """
            UNWIND $seed_ids AS seed_id
            MATCH (:Chunk {chunk_id: seed_id})-[:MENTIONS]->(e:Entity)
                  <-[:MENTIONS]-(neighbor:Chunk)
            WHERE NOT neighbor.chunk_id IN $seed_ids
              AND COUNT { (:Chunk)-[:MENTIONS]->(e) } <= $max_degree
            RETURN DISTINCT neighbor.chunk_id   AS chunk_id,
                            neighbor.text       AS text,
                            neighbor.article_id AS article_id
            """
        else:
            cypher = """
            UNWIND $seed_ids AS seed_id
            MATCH (:Chunk {chunk_id: seed_id})-[:MENTIONS]->(e1:Entity)
                  <-[:MENTIONS]-(mid:Chunk)-[:MENTIONS]->(e2:Entity)
                  <-[:MENTIONS]-(neighbor:Chunk)
            WHERE NOT neighbor.chunk_id IN $seed_ids
              AND neighbor.chunk_id <> mid.chunk_id
              AND COUNT { (:Chunk)-[:MENTIONS]->(e1) } <= $max_degree
              AND COUNT { (:Chunk)-[:MENTIONS]->(e2) } <= $max_degree
            RETURN DISTINCT neighbor.chunk_id   AS chunk_id,
                            neighbor.text       AS text,
                            neighbor.article_id AS article_id
            """
        with self.driver.session() as session:
            result = session.run(
                cypher, seed_ids=seed_ids, max_degree=MAX_ENTITY_DEGREE
            )
            return [dict(r) for r in result]

    def _rerank(
        self,
        vector_results: list[dict],
        graph_results: list[dict],
        alpha: float = DEFAULT_ALPHA,
    ) -> list[dict]:
        """Merge vector + graph hits and rank by combined score (descending).

        combined = alpha * vector_similarity + (1 - alpha) * graph_proximity
        Vector hits carry their cosine score; graph neighbours carry 0.0.
        Graph neighbours carry their proximity tier; vector hits carry 0.0.
        Pure function of its inputs (no DB) — unit-testable.
        """
        merged: dict[str, dict] = {}
        for r in vector_results:
            merged[r["chunk_id"]] = {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "article_id": r.get("article_id"),
                "vector_score": float(r.get("score", 0.0)),
                "graph_proximity": 0.0,
            }
        for r in graph_results:
            cid = r["chunk_id"]
            prox = float(r.get("graph_proximity", 0.0))
            if cid in merged:
                merged[cid]["graph_proximity"] = max(
                    merged[cid]["graph_proximity"], prox
                )
            else:
                merged[cid] = {
                    "chunk_id": cid,
                    "text": r["text"],
                    "article_id": r.get("article_id"),
                    "vector_score": 0.0,
                    "graph_proximity": prox,
                }
        for c in merged.values():
            c["score"] = alpha * c["vector_score"] + (1 - alpha) * c["graph_proximity"]
        return sorted(merged.values(), key=lambda c: c["score"], reverse=True)

    # -- generation ----------------------------------------------------------
    def _build_prompt(self, query: str, chunks: list[dict]) -> str:
        """Format retrieved chunks + the question into the user prompt."""
        context = "\n\n".join(
            f"[Chunk {i + 1} | Article {c['article_id']}]\n{c['text']}"
            for i, c in enumerate(chunks)
        )
        return f"{context}\n\nQuestion: {query}\nAnswer:"

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """Build the prompt and call the LLM (Cerebras > OpenAI > Ollama)."""
        prompt = self._build_prompt(query, context_chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if self.provider == "cerebras":
            # gpt-oss is a reasoning model: use max_completion_tokens (not
            # max_tokens) with a budget large enough for reasoning + answer.
            response = call_llm_with_retry(
                self.llm,
                model=os.getenv("CEREBRAS_MODEL", ANSWER_MODEL_CEREBRAS),
                messages=messages,
                temperature=ANSWER_TEMPERATURE,
                max_completion_tokens=CEREBRAS_MAX_COMPLETION_TOKENS,
            )
            return response.choices[0].message.content
        if self.provider == "openai":
            response = call_llm_with_retry(
                self.llm,
                model=ANSWER_MODEL_OPENAI,
                messages=messages,
                temperature=ANSWER_TEMPERATURE,
                max_tokens=ANSWER_MAX_TOKENS,
            )
            return response.choices[0].message.content
        # Ollama /api/generate takes a single prompt; prepend the system instructions.
        return _with_retry(
            lambda: _ollama_generate(f"{SYSTEM_PROMPT}\n\n{prompt}", ANSWER_MODEL_OLLAMA),
            "Ollama call",
        )

    # -- GDS re-ranking (Phase 5B) ------------------------------------------ #
    def gds_rerank(self, chunk_ids: list[str]) -> dict[str, float]:
        """Project the retrieved subgraph, run PageRank, return ``{chunk_id: score}``.

        Uses a unique graph name per call (avoids stale-graph collisions) and
        always drops the in-memory graph afterwards. Returns ``{}`` if GDS is
        unavailable or the projection fails, so callers can fall back to
        vector-only scoring.
        """
        if not chunk_ids:
            return {}
        graph_name = f"retrieved_subgraph_{uuid.uuid4().hex[:8]}"
        try:
            with self.driver.session() as session:
                session.run(
                    _GDS_PROJECT_CYPHER,
                    graph_name=graph_name,
                    chunk_ids=chunk_ids,
                )
                result = session.run(_GDS_PAGERANK_CYPHER, graph_name=graph_name)
                scores = {row["chunk_id"]: row["score"] for row in result}
            logger.info("GDS PageRank scored %d chunks.", len(scores))
            return scores
        except Exception as exc:
            logger.warning("GDS rerank failed (%s); returning empty scores.", exc)
            return {}
        finally:
            try:
                with self.driver.session() as session:
                    session.run("CALL gds.graph.drop($name, false)", name=graph_name)
            except Exception:
                pass  # graph may never have been created; nothing to drop

    def retrieve_with_gds(
        self, query: str, top_k: int = DEFAULT_TOP_K, expand_graph: bool = True,
    ) -> list[dict]:
        """Retrieve, then re-rank with GDS PageRank centrality (alpha=0.6)."""
        chunks = self.retrieve(query, top_k=top_k, expand_graph=expand_graph)
        if not chunks:
            return chunks
        gds_scores = self.gds_rerank([c["chunk_id"] for c in chunks])
        return combined_rerank(chunks, gds_scores) if gds_scores else chunks

    # -- orchestration -------------------------------------------------------
    def run(
        self, query: str, top_k: int = DEFAULT_TOP_K, expand_graph: bool = True
    ) -> dict:
        """Retrieve context then generate an answer; return the full result dict."""
        chunks = self.retrieve(query, top_k=top_k, expand_graph=expand_graph)
        answer = self.generate(query, chunks)
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": chunks,
            "expand_graph": expand_graph,
        }


# --------------------------------------------------------------------------- #
# Smoke / full-run entry point (AGENT.md: test on 2 questions)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    default_questions = [
        "Does metformin reduce the risk of cancer in diabetic patients?",
        "What is the role of CRISPR-Cas9 off-target effects in gene therapy safety?",
    ]
    questions = sys.argv[1:] or default_questions

    pipeline = GraphRAGPipeline()
    for question in questions:
        print(f"\n=== Question: {question} ===")
        result = pipeline.run(question, expand_graph=True)
        print("--- Retrieved chunks ---")
        for chunk in result["retrieved_chunks"]:
            print(f"  [{chunk['chunk_id']}] score={chunk['score']:.3f}")
        print("--- Answer ---")
        print(result["answer"])

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
import time

# 2. third-party — (none at import time; openai/requests loaded lazily)

# 3. local
from src.chunker import get_embedder
from src.graph_builder import get_driver

logger = logging.getLogger(__name__)

# Defaults pinned by AGENT.md / the Phase 3 spec.
DEFAULT_TOP_K = 5
DEFAULT_DEPTH = 2
DEFAULT_ALPHA = 0.7
FINAL_CONTEXT_K = 10
ANSWER_MODEL_OPENAI = "gpt-4o-mini"
ANSWER_MODEL_OLLAMA = "mistral"
ANSWER_TEMPERATURE = 0.2
ANSWER_MAX_TOKENS = 512
MAX_LLM_RETRIES = 3
VECTOR_INDEX = "chunk_embedding"

SYSTEM_PROMPT = """You are a biomedical research assistant.
Answer the question using ONLY the provided context passages.
If the context doesn't contain enough information to answer, say so clearly.
Be concise and factual. Do not hallucinate."""


# --------------------------------------------------------------------------- #
# LLM client factory + retry
# --------------------------------------------------------------------------- #
def get_llm_client():
    """Return ``(provider, client)``: OpenAI if a key is set, else local Ollama."""
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        logger.info("Using OpenAI LLM (%s).", ANSWER_MODEL_OPENAI)
        # max_retries=0 so our own call_llm_with_retry owns backoff.
        return "openai", OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=0)
    logger.info("OPENAI_API_KEY unset; using local Ollama LLM (%s).", ANSWER_MODEL_OLLAMA)
    return "ollama", None


def _with_retry(action, what: str = "LLM call", max_retries: int = MAX_LLM_RETRIES):
    """Run ``action`` with exponential backoff (1s, 2s, 4s); re-raise after the last try."""
    for attempt in range(max_retries):
        try:
            return action()
        except Exception as exc:  # retry any LLM error, per AGENT.md error handling
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
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
        self, query: str, top_k: int = DEFAULT_TOP_K, expand_graph: bool = True
    ) -> list[dict]:
        """Embed the query, vector-search, optionally graph-expand, then re-rank."""
        start = time.time()
        query_embedding = self.embedder.encode([query])[0].tolist()
        vector_results = self._vector_search(query_embedding, top_k)

        graph_results: list[dict] = []
        if expand_graph and vector_results:
            seed_ids = [r["chunk_id"] for r in vector_results]
            graph_results = self._graph_expand(seed_ids, depth=DEFAULT_DEPTH)

        ranked = self._rerank(vector_results, graph_results, alpha=DEFAULT_ALPHA)
        ranked = ranked[:FINAL_CONTEXT_K]
        logger.info(
            "retrieve('%s') -> %d chunks in %.2fs (expand_graph=%s)",
            query[:60], len(ranked), time.time() - start, expand_graph,
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
        """Return chunk dicts related to ``seed_ids`` at the given hop level."""
        if level == 1:
            cypher = """
            UNWIND $seed_ids AS seed_id
            MATCH (:Chunk {chunk_id: seed_id})-[:MENTIONS]->(e:Entity)
                  <-[:MENTIONS]-(neighbor:Chunk)
            WHERE NOT neighbor.chunk_id IN $seed_ids
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
            RETURN DISTINCT neighbor.chunk_id   AS chunk_id,
                            neighbor.text       AS text,
                            neighbor.article_id AS article_id
            """
        with self.driver.session() as session:
            result = session.run(cypher, seed_ids=seed_ids)
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
        """Build the prompt and call the LLM (OpenAI primary, Ollama fallback)."""
        prompt = self._build_prompt(query, context_chunks)
        if self.provider == "openai":
            response = call_llm_with_retry(
                self.llm,
                model=ANSWER_MODEL_OPENAI,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=ANSWER_TEMPERATURE,
                max_tokens=ANSWER_MAX_TOKENS,
            )
            return response.choices[0].message.content
        # Ollama /api/generate takes a single prompt; prepend the system instructions.
        return _with_retry(
            lambda: _ollama_generate(f"{SYSTEM_PROMPT}\n\n{prompt}", ANSWER_MODEL_OLLAMA),
            "Ollama call",
        )

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

"""Phase 5A — multi-hop query decomposition.

Complex biomedical questions span several concepts. This module decomposes a
question into 2-4 focused sub-questions via the LLM, retrieves chunks for each,
then merges + de-duplicates + re-ranks the union by retrieval frequency (chunks
surfaced by multiple sub-questions rank higher).

Provider-aware (Cerebras > OpenAI > Ollama) so it reuses the pipeline's client.
"""

# 1. stdlib
import json
import logging
import os
import re

# 2. third-party — (none at import time)

# 3. local
from src.rag_pipeline import (
    ANSWER_MODEL_CEREBRAS,
    ANSWER_MODEL_OPENAI,
    ANSWER_MODEL_OLLAMA,
    _ollama_generate,
    _with_retry,
    call_llm_with_retry,
)

logger = logging.getLogger(__name__)

# AGENT.md: temperature=0.4 (some creativity) for decomposition, max 256 tokens.
DECOMPOSE_TEMPERATURE = 0.4
DECOMPOSE_MAX_TOKENS = 256
# gpt-oss reasons before emitting JSON; budget covers reasoning + the array.
CEREBRAS_DECOMP_TOKENS = 2048
MAX_SUB_QUESTIONS = 4

DECOMPOSE_SYSTEM = "You are a biomedical research assistant."

DECOMPOSE_PROMPT = """Break the following complex question into 2-4 simpler, focused sub-questions
that together cover the original question.
Return ONLY a JSON array of strings. No explanation.

Question: {query}

Example output: ["sub-question 1", "sub-question 2", "sub-question 3"]"""


def parse_subquestions(response: str) -> list[str]:
    """Extract a JSON string array from an LLM response; ``[]`` on failure.

    Pure function (no LLM/DB) so it is trivially unit-testable.
    """
    if not response:
        return []
    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if not match:
            return []
        parsed = json.loads(match.group())
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        return []


def decompose_query(
    query: str, provider: str, llm_client
) -> list[str]:
    """Decompose ``query`` into 2-4 sub-questions via the LLM.

    Falls back to ``[query]`` on any LLM/parse error so callers always get a
    usable list. ``provider``/``llm_client`` come from ``GraphRAGPipeline``.
    """
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM},
        {"role": "user", "content": DECOMPOSE_PROMPT.format(query=query)},
    ]
    try:
        if provider in ("cerebras", "openai"):
            kwargs: dict = {
                "messages": messages,
                "temperature": DECOMPOSE_TEMPERATURE,
            }
            if provider == "cerebras":
                kwargs["model"] = os.getenv("CEREBRAS_MODEL", ANSWER_MODEL_CEREBRAS)
                kwargs["max_completion_tokens"] = CEREBRAS_DECOMP_TOKENS
            else:
                kwargs["model"] = ANSWER_MODEL_OPENAI
                kwargs["max_tokens"] = DECOMPOSE_MAX_TOKENS
            response = call_llm_with_retry(llm_client, **kwargs)
            text = response.choices[0].message.content
        else:
            prompt = f"{DECOMPOSE_SYSTEM}\n\n{DECOMPOSE_PROMPT.format(query=query)}"
            text = _with_retry(
                lambda: _ollama_generate(prompt, ANSWER_MODEL_OLLAMA),
                "Ollama decompose",
            )
    except Exception as exc:
        logger.warning(
            "Query decomposition failed (%s); using original query only.", exc
        )
        return [query]

    sub_questions = parse_subquestions(text or "")
    if not sub_questions:
        logger.info("Decomposition yielded no sub-questions; using original query.")
        return [query]
    logger.info("Decomposed '%s' into %d sub-questions.", query[:50], len(sub_questions))
    return sub_questions[:MAX_SUB_QUESTIONS]


def multi_hop_retrieve(
    query: str, pipeline, top_k: int = 5
) -> list[dict]:
    """Decompose, retrieve per sub-question, merge + de-dupe, frequency re-rank.

    Steps (per Phase 5 spec):
      1. Decompose the query into sub-questions.
      2. ``retrieve()`` for each sub-question.
      3. Merge all retrieved chunks.
      4. De-duplicate by ``chunk_id``.
      5. Re-rank by frequency (a chunk found by N sub-retrievals ranks higher),
         breaking ties by retrieval score.
      6. Return ``top_k * 2`` chunks.
    """
    sub_questions = decompose_query(query, pipeline.provider, pipeline.llm)
    seen: dict[str, dict] = {}
    freq: dict[str, int] = {}

    for sub_q in sub_questions:
        chunks = pipeline.retrieve(sub_q, top_k=top_k, expand_graph=True)
        for chunk in chunks:
            cid = chunk["chunk_id"]
            freq[cid] = freq.get(cid, 0) + 1
            if cid not in seen:
                seen[cid] = chunk

    merged = list(seen.values())
    for chunk in merged:
        chunk["frequency"] = freq[chunk["chunk_id"]]
    # Primary: how many sub-questions surfaced it; secondary: retrieval score.
    merged.sort(
        key=lambda c: (c["frequency"], c.get("score", 0.0)), reverse=True
    )

    limit = max(top_k * 2, 1)
    result = merged[:limit]
    logger.info(
        "multi_hop_retrieve('%s') -> %d unique chunks from %d sub-questions.",
        query[:50], len(result), len(sub_questions),
    )
    return result


# --------------------------------------------------------------------------- #
# Smoke / full-run entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    sample = (
        "What is the relationship between BRCA1 mutations and chemotherapy "
        "response in triple-negative breast cancer?"
    )

    # 1) Pure parser tests (no LLM/DB needed).
    print("=== parse_subquestions tests ===")
    cases = {
        '["a", "b", "c"]': ["a", "b", "c"],
        "noise [\"x\",\"y\"] trailing": ["x", "y"],
        "no array here": [],
        "[]": [],
    }
    for raw, expected in cases.items():
        got = parse_subquestions(raw)
        status = "OK" if got == expected else "FAIL"
        print(f"  [{status}] {raw!r} -> {got}")

    # 2) Live decomposition via the configured LLM provider (Cerebras/OpenAI/Ollama).
    print("\n=== decompose_query (live LLM) ===")
    from src.rag_pipeline import GraphRAGPipeline

    # dummy driver/embedder isolate the LLM path; provider picked up from .env
    pipe = GraphRAGPipeline(driver=object(), embedder=object())
    print(f"provider: {pipe.provider}")
    sub_qs = decompose_query(sample, pipe.provider, pipe.llm)
    for i, sq in enumerate(sub_qs, 1):
        print(f"  Sub-Q{i}: {sq}")

"""Phase 1 — semantic chunking for PubMed abstracts.

Implements three chunking strategies:

* ``fixed_token``        — naive non-overlapping token windows (baseline)
* ``sentence_boundary``  — greedily merge whole sentences up to a token budget
* ``semantic_cluster``   — HDBSCAN over sentence embeddings, one chunk per topic

The orchestrator :func:`run_all_strategies` writes one JSONL file per strategy
into ``data/chunks/`` plus a t-SNE validation plot of the semantic chunks.

Every chunk conforms to the schema defined in AGENT.md.
"""

# 1. stdlib
import json
import logging
import os

# 2. third-party
import numpy as np

logger = logging.getLogger(__name__)

# Defaults pinned by the spec.
DEFAULT_MAX_TOKENS = 100
DEFAULT_MIN_CLUSTER_SIZE = 2

# --------------------------------------------------------------------------- #
# Singleton embedder — model fixed by AGENT.md, never substitute.
# --------------------------------------------------------------------------- #
_embedder = None


def get_embedder():
    """Return the shared ``all-MiniLM-L6-v2`` sentence embedder (loaded once)."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer 'all-MiniLM-L6-v2'...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# --------------------------------------------------------------------------- #
# NLTK bootstrap + tokenisation helpers
# --------------------------------------------------------------------------- #
_nltk_ready = False
_tokenizer = None


def _ensure_nltk() -> None:
    """Ensure the NLTK punkt tokenisers are present, downloading if missing."""
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk

    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            logger.info("Downloading NLTK '%s'...", pkg)
            nltk.download(pkg, quiet=True)
    _nltk_ready = True


def _word_tokenizer():
    """Return a cached Treebank word tokeniser instance."""
    global _tokenizer
    if _tokenizer is None:
        _ensure_nltk()
        from nltk.tokenize import TreebankWordTokenizer

        _tokenizer = TreebankWordTokenizer()
    return _tokenizer


def _word_spans(text: str) -> list[tuple[int, int]]:
    """Return the ``(start, end)`` character span of each token in ``text``."""
    try:
        return list(_word_tokenizer().span_tokenize(text))
    except Exception:
        return []


def _sentences(text: str) -> list[str]:
    """Split ``text`` into non-empty sentences via the NLTK punkt tokeniser."""
    _ensure_nltk()
    from nltk.tokenize import sent_tokenize

    return [s for s in sent_tokenize(text) if s.strip()]


def _token_count(text: str) -> int:
    """Approximate token count via the Treebank word tokeniser."""
    return len(_word_spans(text))


def _build_chunk(article_id: str, index: int, text: str, strategy: str) -> dict:
    """Assemble a single schema-compliant chunk dict."""
    return {
        "article_id": article_id,
        "chunk_id": f"{article_id}_{index}",
        "text": text,
        "strategy": strategy,
        "sentence_count": len(_sentences(text)),
        "token_count": _token_count(text),
    }


# --------------------------------------------------------------------------- #
# Strategy 1 — fixed token baseline
# --------------------------------------------------------------------------- #
def fixed_token_chunks(
    text: str, article_id: str, max_tokens: int = DEFAULT_MAX_TOKENS
) -> list[dict]:
    """Split ``text`` into non-overlapping windows of ``max_tokens`` tokens."""
    spans = _word_spans(text)
    if not spans:
        return []
    chunks: list[dict] = []
    for index, start in enumerate(range(0, len(spans), max_tokens)):
        window = spans[start:start + max_tokens]
        chunk_text = text[window[0][0]:window[-1][1]]
        chunks.append(_build_chunk(article_id, index, chunk_text, "fixed_token"))
    return chunks


# --------------------------------------------------------------------------- #
# Strategy 2 — sentence boundary
# --------------------------------------------------------------------------- #
def _greedy_sentence_chunks(
    sentences: list[str], article_id: str, strategy: str, max_tokens: int
) -> list[dict]:
    """Merge sentences greedily, flushing a chunk before ``max_tokens`` is exceeded."""
    chunks: list[dict] = []
    current: list[str] = []
    current_tokens = 0
    index = 0
    for sent in sentences:
        n = _token_count(sent)
        if current and current_tokens + n > max_tokens:
            chunks.append(_build_chunk(article_id, index, " ".join(current), strategy))
            index += 1
            current = [sent]
            current_tokens = n
        else:
            current.append(sent)
            current_tokens += n
    if current:
        chunks.append(_build_chunk(article_id, index, " ".join(current), strategy))
    return chunks


def sentence_boundary_chunks(
    text: str, article_id: str, max_tokens: int = DEFAULT_MAX_TOKENS
) -> list[dict]:
    """Chunk ``text`` at sentence boundaries, never exceeding ``max_tokens`` per chunk."""
    sentences = _sentences(text)
    if not sentences:
        return []
    return _greedy_sentence_chunks(sentences, article_id, "sentence_boundary", max_tokens)


# --------------------------------------------------------------------------- #
# Strategy 3 — semantic clustering (primary)
# --------------------------------------------------------------------------- #
def _assign_noise_to_nearest(embeddings: np.ndarray, labels) -> list[int]:
    """Reassign every noise label (-1) to its nearest true cluster centroid."""
    labels = [int(l) for l in labels]
    real = sorted({l for l in labels if l != -1})
    if -1 not in labels or not real:
        return labels
    centroids = {
        l: embeddings[[i for i, x in enumerate(labels) if x == l]].mean(axis=0)
        for l in real
    }
    for i, l in enumerate(labels):
        if l == -1:
            labels[i] = min(
                real, key=lambda c: float(np.linalg.norm(embeddings[i] - centroids[c]))
            )
    return labels


def semantic_cluster_chunks(
    text: str,
    article_id: str,
    embedder,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> list[dict]:
    """Cluster sentences with HDBSCAN; each cluster becomes one chunk.

    Falls back to sentence-boundary grouping when an abstract has fewer than 3
    sentences (too few for HDBSCAN to form clusters).
    """
    sentences = _sentences(text)
    if len(sentences) < 3:
        return _greedy_sentence_chunks(
            sentences, article_id, "semantic_cluster", DEFAULT_MAX_TOKENS
        )

    import hdbscan

    embeddings = np.asarray(embedder.encode(sentences))
    raw_labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean"
    ).fit_predict(embeddings)
    labels = _assign_noise_to_nearest(embeddings, raw_labels)

    # Group sentence indices by cluster, preserving first-appearance order.
    grouped: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        grouped.setdefault(lbl, []).append(i)
    ordered_labels = sorted(grouped, key=lambda c: grouped[c][0])

    chunks: list[dict] = []
    for index, lbl in enumerate(ordered_labels):
        chunk_text = " ".join(sentences[i] for i in grouped[lbl])
        chunks.append(_build_chunk(article_id, index, chunk_text, "semantic_cluster"))
    return chunks


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _write_jsonl(chunks: list[dict], path: str) -> None:
    """Write ``chunks`` as JSON Lines to ``path``."""
    with open(path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks -> %s", len(chunks), path)


def _strategy_stats(chunks: list[dict], n_articles: int, name: str) -> dict:
    """Compute summary statistics for one strategy's chunk set."""
    total = len(chunks)
    avg_tokens = (sum(c["token_count"] for c in chunks) / total) if total else 0.0
    avg_per_article = (total / n_articles) if n_articles else 0.0
    return {
        "name": name,
        "total": total,
        "avg_tokens": avg_tokens,
        "avg_per_article": avg_per_article,
    }


def _format_stats_table(rows: list[dict]) -> str:
    """Render the strategy comparison table as a fixed-width string."""
    lines = [
        f"{'Strategy':<20} | {'Total Chunks':>12} | "
        f"{'Avg Tokens/Chunk':>16} | {'Avg Chunks/Article':>18}",
        "-" * 76,
    ]
    for r in rows:
        lines.append(
            f"{r['name']:<20} | {r['total']:>12,} | "
            f"{r['avg_tokens']:>16.1f} | {r['avg_per_article']:>18.2f}"
        )
    return "\n".join(lines)


def _make_tsne_plot(chunks: list[dict], output_path: str, embedder) -> None:
    """Save a t-SNE scatter plot of chunk embeddings coloured by cluster."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import hdbscan
    from sklearn.manifold import TSNE

    if not chunks:
        logger.warning("No semantic chunks to plot; skipping t-SNE.")
        return

    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks for t-SNE plot...", len(texts))
    embs = np.asarray(embedder.encode(texts, batch_size=64, show_progress_bar=True))
    cluster_labels = np.asarray(
        hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean").fit_predict(embs)
    )

    if len(embs) < 2:
        logger.warning("Too few chunks (%d) for t-SNE; skipping plot.", len(embs))
        return
    perplexity = max(1, min(30, len(embs) - 1))
    logger.info("Running t-SNE (perplexity=%d, max_iter=1000)...", perplexity)
    common = dict(n_components=2, perplexity=perplexity, random_state=42)
    try:
        coords = TSNE(max_iter=1000, **common).fit_transform(embs)
    except TypeError:
        # Older sklearn (<1.5) used n_iter instead of max_iter.
        coords = TSNE(n_iter=1000, **common).fit_transform(embs)

    noise = cluster_labels == -1
    plt.figure(figsize=(10, 8))
    if (~noise).any():
        scatter = plt.scatter(
            coords[~noise, 0], coords[~noise, 1], s=8, c=cluster_labels[~noise],
            cmap="tab20",
        )
        plt.colorbar(scatter, label="cluster")
    if noise.any():
        plt.scatter(
            coords[noise, 0], coords[noise, 1], s=6, c="lightgrey", label="noise"
        )
        plt.legend(loc="best")
    plt.title("Semantic Chunk Clusters — PubMed Abstracts (5000)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved t-SNE plot -> %s", output_path)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_all_strategies(abstracts: list[dict], output_dir: str) -> None:
    """Run all 3 strategies, write per-strategy JSONL + t-SNE plot, log stats."""
    os.makedirs(output_dir, exist_ok=True)
    embedder = get_embedder()
    n_articles = len(abstracts)

    fixed_all: list[dict] = []
    sent_all: list[dict] = []
    sem_all: list[dict] = []
    for i, ab in enumerate(abstracts, start=1):
        text = ab.get("abstract", "") or ""
        aid = str(ab.get("article_id", ""))
        fixed_all.extend(fixed_token_chunks(text, aid))
        sent_all.extend(sentence_boundary_chunks(text, aid))
        sem_all.extend(semantic_cluster_chunks(text, aid, embedder))
        if i % 500 == 0:
            logger.info("Processed %d/%d articles...", i, n_articles)

    _write_jsonl(fixed_all, os.path.join(output_dir, "fixed_token_chunks.jsonl"))
    _write_jsonl(sent_all, os.path.join(output_dir, "sentence_boundary_chunks.jsonl"))
    _write_jsonl(sem_all, os.path.join(output_dir, "semantic_cluster_chunks.jsonl"))
    _make_tsne_plot(
        sem_all, os.path.join(output_dir, "semantic_clusters_tsne.png"), embedder
    )

    rows = [
        _strategy_stats(fixed_all, n_articles, "fixed_token"),
        _strategy_stats(sent_all, n_articles, "sentence_boundary"),
        _strategy_stats(sem_all, n_articles, "semantic_cluster"),
    ]
    logger.info("Chunk stats:\n%s", _format_stats_table(rows))


# --------------------------------------------------------------------------- #
# Smoke / full-run entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, format="%(message)s"
    )
    parser = argparse.ArgumentParser(description="Run all 3 chunking strategies.")
    parser.add_argument(
        "abstracts_path", nargs="?", default="data/raw/abstracts.json",
        help="path to abstracts JSON (default: data/raw/abstracts.json)",
    )
    parser.add_argument(
        "-o", "--output-dir", default="data/chunks",
        help="directory for JSONL + plot output (default: data/chunks)",
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=None,
        help="limit number of abstracts (e.g. 3 for a quick smoke test)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.abstracts_path):
        print(f"Abstracts file not found: {args.abstracts_path}")
        print("Run `python -m src.load_data` first to download samples.")
        sys.exit(1)

    with open(args.abstracts_path, encoding="utf-8") as fh:
        abstracts = json.load(fh)
    if args.limit:
        abstracts = abstracts[: args.limit]
    print(f"Loaded {len(abstracts)} abstracts from {args.abstracts_path}")
    run_all_strategies(abstracts, args.output_dir)

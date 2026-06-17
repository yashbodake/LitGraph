# Phase 1 — Semantic Chunking

## Goal

Produce three different chunk sets from 5000 PubMed abstracts. The semantic strategy should produce chunks that respect topic boundaries rather than arbitrary token windows. All chunks are saved as JSON Lines for downstream ingestion.

---

## Input

- 5000 PubMed abstracts sampled from `alexaapo/scientific_papers` (HuggingFace)
- Each record: `article_id`, `abstract`, `title`, `pub_date`

---

## Output

Three JSONL files in `data/chunks/`:

```
data/chunks/
├── fixed_token_chunks.jsonl
├── sentence_boundary_chunks.jsonl
└── semantic_cluster_chunks.jsonl
```

Each line in every file:
```json
{
  "article_id": "12345678",
  "chunk_id": "12345678_0",
  "text": "...",
  "strategy": "semantic_cluster",
  "sentence_count": 3,
  "token_count": 87
}
```

---

## Strategies

### Strategy 1 — Fixed Token Baseline

- Split each abstract into non-overlapping windows of **100 tokens**
- No overlap between consecutive chunks
- Tokenise using `nltk.word_tokenize` or HuggingFace tokenizer
- Purpose: establishes a naive baseline for evaluation in Phase 4

```python
def fixed_token_chunks(text: str, article_id: str, max_tokens: int = 100) -> list[dict]:
    ...
```

---

### Strategy 2 — Sentence Boundary

- Split abstract into individual sentences using `nltk.sent_tokenize`
- Greedily merge sentences until token count would exceed 100 tokens
- Start a new chunk when limit is hit
- Preserves sentence integrity (no mid-sentence cuts)

```python
def sentence_boundary_chunks(text: str, article_id: str, max_tokens: int = 100) -> list[dict]:
    ...
```

---

### Strategy 3 — Semantic Clustering *(primary strategy)*

**Step-by-step:**

1. Sentence-tokenise the abstract (same as strategy 2)
2. Embed every sentence using `all-MiniLM-L6-v2` (384-dim vectors)
3. Cluster the sentence embeddings using **HDBSCAN** with:
   - `min_cluster_size=2`
   - `metric='euclidean'`
   - Noise label (`-1`) → assign each noise sentence to its nearest cluster centroid
4. Group sentences by cluster label → each group is one chunk
5. If abstract has < 3 sentences (HDBSCAN can't cluster), fall back to sentence-boundary strategy

```python
def semantic_cluster_chunks(text: str, article_id: str, embedder, min_cluster_size: int = 2) -> list[dict]:
    ...
```

**Why HDBSCAN:** Doesn't require specifying number of clusters in advance. Well-suited for variable-length abstracts where number of topics is unknown.

---

## Validation Plot

Produce a **t-SNE or UMAP scatter plot** of chunk embeddings (semantic strategy only):
- Each point = one chunk embedding (mean-pool its sentence embeddings)
- Colour by cluster label
- Title: "Semantic Chunk Clusters — PubMed Abstracts (5000)"
- Save as `data/chunks/semantic_clusters_tsne.png`

For t-SNE: use `perplexity=30`, `n_iter=1000`, `random_state=42`.

---

## Chunker Module Interface

`src/chunker.py` should expose:

```python
# Main entry point
def run_all_strategies(abstracts: list[dict], output_dir: str) -> None:
    """
    Runs all 3 chunking strategies on the abstract list.
    Saves a JSONL file per strategy to output_dir.
    Also saves the t-SNE validation plot.
    """

# Individual strategy functions (importable separately)
def fixed_token_chunks(text, article_id, max_tokens=100) -> list[dict]
def sentence_boundary_chunks(text, article_id, max_tokens=100) -> list[dict]
def semantic_cluster_chunks(text, article_id, embedder, min_cluster_size=2) -> list[dict]
```

---

## Data Loading Snippet

```python
from datasets import load_dataset
import random

ds = load_dataset("alexaapo/scientific_papers", "pubmed", split="train", streaming=True)
abstracts = []
for i, row in enumerate(ds):
    if i >= 5000:
        break
    abstracts.append({
        "article_id": row["article_id"],
        "abstract": row["abstract"],
        "title": row["article"][:200],   # title field name — verify on load
        "pub_date": row.get("pub_date", "")
    })
```

> **Note:** Verify exact field names after first load with `print(ds.features)`.

---

## Chunk Stats to Log (stdout)

After generating all three chunk sets, print a summary:

```
Strategy            | Total Chunks | Avg Tokens/Chunk | Avg Chunks/Article
--------------------|--------------|------------------|-------------------
fixed_token         | 14,321       | 98.2             | 2.86
sentence_boundary   | 12,809       | 94.7             | 2.56
semantic_cluster    | 11,204       | 108.3            | 2.24
```

---

## Deliverables Checklist

- [ ] `src/chunker.py` with all 3 strategy functions
- [ ] `data/chunks/fixed_token_chunks.jsonl`
- [ ] `data/chunks/sentence_boundary_chunks.jsonl`
- [ ] `data/chunks/semantic_cluster_chunks.jsonl`
- [ ] `data/chunks/semantic_clusters_tsne.png`
- [ ] Chunk stats printed to console (or logged)

---

## Dependencies for This Phase

```
datasets
sentence-transformers
hdbscan
nltk
scikit-learn          # for t-SNE
matplotlib
umap-learn            # optional, alternative to t-SNE
```

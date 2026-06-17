# Phase 4 — Full Evaluation

## Goal

Systematically measure both **retrieval quality** (Recall@K, MRR) and **generation quality** (ROUGE-L, BERTScore) across ~200 QA pairs. Produce a final report comparing all system variants: fixed-chunk baseline, sentence-boundary + no graph, and semantic-cluster + graph expansion.

---

## Prerequisites

- Phase 3 complete: `GraphRAGPipeline` working end-to-end
- `data/qa/` contains filtered `pubmed_qa` pairs (those whose `pubmed_id` is in your 5000 abstracts)
- Both fixed-token and semantic-cluster chunk sets are in Neo4j (or evaluate fixed-token retrieval using a separate index / in-memory approach)

---

## QA Dataset Preparation

```python
from datasets import load_dataset

qa_ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")

# Your 5000 article IDs (loaded from data/raw or Phase 1 output)
valid_ids = set(your_article_ids)

# Filter to matching QA pairs
filtered_qa = [
    {
        "pubmed_id": str(row["pubmed_id"]),
        "question": row["question"],
        "long_answer": row["long_answer"],           # gold generation target
        "final_decision": row["final_decision"]      # yes/no/maybe
    }
    for row in qa_ds
    if str(row["pubmed_id"]) in valid_ids
]

# Sample 200 for evaluation (or use all if < 200)
import random
random.seed(42)
eval_set = random.sample(filtered_qa, min(200, len(filtered_qa)))
```

---

## Retrieval Metrics

For each question in `eval_set`:

1. Run `retrieve(question, top_k=10, expand_graph=True)` → list of returned chunk IDs
2. Get **gold chunk IDs**: all chunks whose `article_id == pubmed_id` for this question
3. Compute metrics below

### Recall@K

```
Recall@K = |retrieved_top_k ∩ gold_chunks| / |gold_chunks|
```

```python
def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in gold_ids)
    return hits / len(gold_ids) if gold_ids else 0.0
```

Report **Recall@5** and **Recall@10** averaged over all 200 questions.

### Mean Reciprocal Rank (MRR)

```
MRR = (1/|Q|) * Σ (1 / rank_of_first_relevant_chunk)
```

```python
def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0
```

Report MRR@10 averaged over all questions.

---

## Generation Metrics

For each question, generate an answer using retrieved chunks, then compare to the `long_answer` (gold).

### ROUGE-L

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def compute_rouge_l(prediction: str, reference: str) -> float:
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure
```

### BERTScore

```python
from bert_score import score as bert_score

def compute_bert_score(predictions: list[str], references: list[str]) -> dict:
    P, R, F1 = bert_score(predictions, references, lang="en", model_type="distilbert-base-uncased")
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }
```

> Compute BERTScore in **batch** across all 200 questions at once for efficiency.

---

## System Variants to Compare

Run the full evaluation pipeline for each variant:

| Variant ID | Chunking | Graph Expansion | Description |
|---|---|---|---|
| `baseline` | fixed_token | No | Fixed 100-token chunks, vector-only retrieval |
| `sent_no_graph` | sentence_boundary | No | Sentence chunks, vector-only |
| `semantic_no_graph` | semantic_cluster | No | Semantic chunks, vector-only |
| `semantic_graph` | semantic_cluster | Yes | **Full system** — semantic + graph expansion |

> If running all variants is time-consuming, prioritise `baseline` vs `semantic_graph` for the contrast.

---

## Evaluation Script Interface

`src/evaluator.py`:

```python
def evaluate_retrieval(pipeline, eval_set: list[dict], ks: list[int] = [5, 10]) -> dict:
    """
    Runs retrieval for each QA pair, returns avg Recall@k and MRR.
    """

def evaluate_generation(pipeline, eval_set: list[dict]) -> dict:
    """
    Runs full pipeline (retrieve + generate) for each QA pair.
    Returns avg ROUGE-L and BERTScore F1.
    """

def run_full_evaluation(pipeline, eval_set: list[dict], variant_name: str, output_dir: str) -> None:
    """
    Runs both retrieval and generation eval.
    Saves per-question results as CSV and prints summary table.
    """
```

---

## Results Format

Save per-question results to `data/eval/{variant_name}_results.csv`:

```
pubmed_id,question,recall_at_5,recall_at_10,mrr,rouge_l,bert_score_f1
```

Print aggregate summary table to stdout:

```
Variant              | Recall@5 | Recall@10 | MRR  | ROUGE-L | BERTScore F1
---------------------|----------|-----------|------|---------|-------------
baseline             | 0.31     | 0.44      | 0.29 | 0.18    | 0.72
sent_no_graph        | 0.38     | 0.51      | 0.34 | 0.21    | 0.74
semantic_no_graph    | 0.43     | 0.57      | 0.39 | 0.23    | 0.76
semantic_graph       | 0.52     | 0.65      | 0.47 | 0.27    | 0.79
```

---

## Evaluation Notebook

Create `notebooks/03_evaluation_report.ipynb` with:

1. QA dataset preparation code
2. Retrieval metric results (bar charts per variant)
3. Generation metric results (grouped bar chart: ROUGE-L vs BERTScore)
4. Scatter plot: MRR vs BERTScore per question (coloured by variant)
5. Error analysis: 10 worst-performing questions and why they failed
6. Key takeaways in markdown cells

---

## Deliverables Checklist

- [ ] `src/evaluator.py` with all metric functions
- [ ] `data/eval/` folder with CSV results per variant
- [ ] `notebooks/03_evaluation_report.ipynb` with charts and analysis
- [ ] Summary table printed to console
- [ ] At least 2 variants compared (baseline vs semantic_graph minimum)

---

## Dependencies for This Phase

```
rouge_score
bert-score
torch         # needed by bert-score
pandas        # for CSV handling and analysis
matplotlib
plotly        # for notebook charts
```

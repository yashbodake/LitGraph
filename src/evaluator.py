"""Phase 4 — evaluation: retrieval and generation metrics.

Measures retrieval quality (Recall@K, MRR) and generation quality
(ROUGE-L, BERTScore) over a QA eval set, for each system variant, and writes
per-question CSVs plus an aggregate summary table.

The pure metric helpers (``recall_at_k``, ``mrr``) need no external services;
``compute_rouge_l`` / ``compute_bert_score`` lazy-import their libraries.
"""

# 1. stdlib
import csv
import logging
import os
import random

# 2. third-party — rouge_score / bert_score imported lazily inside functions

logger = logging.getLogger(__name__)

DEFAULT_KS = [5, 10]
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_SEED = 42
BERTSCORE_MODEL = "distilbert-base-uncased"

_CSV_COLUMNS = [
    "pubmed_id", "question", "recall_at_5", "recall_at_10", "mrr",
    "rouge_l", "bert_score_f1",
]


# --------------------------------------------------------------------------- #
# Retrieval metrics (pure)
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Fraction of ``gold_ids`` present in the top-``k`` retrieved ids."""
    if not gold_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids[:k] if cid in gold_ids)
    return hits / len(gold_ids)


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant chunk (0.0 if none in the list)."""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


# --------------------------------------------------------------------------- #
# Generation metrics
# --------------------------------------------------------------------------- #
_rouge_scorer = None


def _get_rouge_scorer():
    """Return a cached ROUGE-L scorer (loaded once)."""
    global _rouge_scorer
    if _rouge_scorer is None:
        from rouge_score import rouge_scorer

        _rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _rouge_scorer


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure between ``prediction`` and ``reference`` (stemmed)."""
    return _get_rouge_scorer().score(reference, prediction)["rougeL"].fmeasure


def _bert_score_tensors(predictions: list[str], references: list[str]):
    """Run BERTScore once over the batch; return the (P, R, F1) tensors."""
    from bert_score import score as bert_score

    return bert_score(
        predictions, references, lang="en", model_type=BERTSCORE_MODEL
    )


def compute_bert_score(predictions: list[str], references: list[str]) -> dict:
    """Batch BERTScore over predictions/references → {precision, recall, f1}."""
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p, r, f1 = _bert_score_tensors(predictions, references)
    return {
        "precision": p.mean().item(),
        "recall": r.mean().item(),
        "f1": f1.mean().item(),
    }


# --------------------------------------------------------------------------- #
# QA dataset preparation + gold chunks
# --------------------------------------------------------------------------- #
def prepare_eval_set(
    article_ids, sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED
) -> list[dict]:
    """Filter pubmed_qa ``pqa_labeled`` to ``article_ids`` and sample up to N."""
    from datasets import load_dataset

    valid = {str(a) for a in article_ids}
    qa_ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    filtered = [
        {
            "pubmed_id": str(row["pubmed_id"]),
            "question": row["question"],
            "long_answer": row["long_answer"],
            "final_decision": row["final_decision"],
        }
        for row in qa_ds
        if str(row["pubmed_id"]) in valid
    ]
    logger.info("pubmed_qa matched %d / %d pairs.", len(filtered), len(qa_ds))
    random.seed(seed)
    return random.sample(filtered, min(sample_size, len(filtered)))


def load_gold_chunks(driver) -> dict[str, set[str]]:
    """Return ``{article_id: {chunk_id, ...}}`` for every chunk in the graph."""
    cypher = "MATCH (c:Chunk) RETURN c.article_id AS aid, c.chunk_id AS cid"
    gold: dict[str, set[str]] = {}
    with driver.session() as session:
        for rec in session.run(cypher):
            gold.setdefault(rec["aid"], set()).add(rec["cid"])
    logger.info("Loaded gold chunks for %d articles.", len(gold))
    return gold


# --------------------------------------------------------------------------- #
# Evaluation drivers
# --------------------------------------------------------------------------- #
def evaluate_retrieval(
    pipeline, eval_set: list[dict], ks: list[int] = DEFAULT_KS,
    expand_graph: bool = True, gold_chunks_by_article: dict = None,
) -> dict:
    """Run retrieval per QA pair; return mean Recall@k and MRR@max(ks)."""
    if gold_chunks_by_article is None:
        gold_chunks_by_article = load_gold_chunks(pipeline.driver)
    top_k = max(ks)
    totals = {f"recall@{k}": 0.0 for k in ks}
    totals["mrr"] = 0.0
    for qa in eval_set:
        retrieved = [
            c["chunk_id"]
            for c in pipeline.retrieve(
                qa["question"], top_k=top_k, expand_graph=expand_graph
            )
        ]
        gold = gold_chunks_by_article.get(qa["pubmed_id"], set())
        for k in ks:
            totals[f"recall@{k}"] += recall_at_k(retrieved, gold, k)
        totals["mrr"] += mrr(retrieved, gold)
    n = len(eval_set) or 1
    return {key: val / n for key, val in totals.items()}


def evaluate_generation(
    pipeline, eval_set: list[dict], expand_graph: bool = True
) -> dict:
    """Run retrieve + generate per QA pair; return mean ROUGE-L and BERTScore F1."""
    preds, refs, rouge_total = [], [], 0.0
    for qa in eval_set:
        chunks = pipeline.retrieve(
            qa["question"], expand_graph=expand_graph
        )
        answer = pipeline.generate(qa["question"], chunks)
        rouge_total += compute_rouge_l(answer, qa["long_answer"])
        preds.append(answer)
        refs.append(qa["long_answer"])
    bert = compute_bert_score(preds, refs) if preds else {"f1": 0.0}
    n = len(eval_set) or 1
    return {"rouge_l": rouge_total / n, "bert_score_f1": bert["f1"]}


def run_full_evaluation(
    pipeline, eval_set: list[dict], variant_name: str, output_dir: str,
    expand_graph: bool = None, ks: list[int] = DEFAULT_KS,
    gold_chunks_by_article: dict = None,
) -> dict:
    """Run retrieval + generation eval, write per-question CSV, return summary.

    ``expand_graph`` defaults to True only for the ``semantic_graph`` variant.
    Retrieval and generation share one ``retrieve()`` call per question, and
    BERTScore runs once as a batch over all answers.
    """
    if expand_graph is None:
        expand_graph = variant_name == "semantic_graph"
    os.makedirs(output_dir, exist_ok=True)
    if gold_chunks_by_article is None:
        gold_chunks_by_article = load_gold_chunks(pipeline.driver)

    top_k = max(ks)
    rows: list[dict] = []
    preds: list[str] = []
    refs: list[str] = []
    for i, qa in enumerate(eval_set, start=1):
        retrieved = pipeline.retrieve(
            qa["question"], top_k=top_k, expand_graph=expand_graph
        )
        retrieved_ids = [c["chunk_id"] for c in retrieved]
        gold = gold_chunks_by_article.get(qa["pubmed_id"], set())
        answer = pipeline.generate(qa["question"], retrieved)
        rouge_l = compute_rouge_l(answer, qa["long_answer"])
        preds.append(answer)
        refs.append(qa["long_answer"])
        rows.append(
            {
                "pubmed_id": qa["pubmed_id"],
                "question": qa["question"],
                "recall_at_5": recall_at_k(retrieved_ids, gold, 5),
                "recall_at_10": recall_at_k(retrieved_ids, gold, 10),
                "mrr": mrr(retrieved_ids, gold),
                "rouge_l": rouge_l,
                "bert_score_f1": None,  # filled after the batch BERTScore pass
            }
        )
        if i % 50 == 0:
            logger.info("Scored %d/%d questions for '%s'...", i, len(eval_set), variant_name)

    if preds:
        _, _, f1_tensor = _bert_score_tensors(preds, refs)
        for row, val in zip(rows, f1_tensor.tolist()):
            row["bert_score_f1"] = val

    _write_results_csv(rows, os.path.join(output_dir, f"{variant_name}_results.csv"))
    summary = _summarise(rows)
    summary["variant"] = variant_name
    logger.info(
        "Summary for '%s':\n%s", variant_name, _format_summary_row(summary)
    )
    return summary


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _write_results_csv(rows: list[dict], path: str) -> None:
    """Write per-question metric rows to ``path`` as CSV."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _CSV_COLUMNS})
    logger.info("Wrote %d rows -> %s", len(rows), path)


def _summarise(rows: list[dict]) -> dict:
    """Average the per-question metrics into a summary dict."""
    n = len(rows) or 1
    return {
        "recall@5": sum(r["recall_at_5"] for r in rows) / n,
        "recall@10": sum(r["recall_at_10"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        "rouge_l": sum(r["rouge_l"] for r in rows) / n,
        "bert_score_f1": sum(r["bert_score_f1"] or 0.0 for r in rows) / n,
    }


_SUMMARY_HEADER = (
    f"{'Variant':<20} | {'Recall@5':>8} | {'Recall@10':>10} | "
    f"{'MRR':>6} | {'ROUGE-L':>8} | {'BERTScore F1':>13}"
)


def _format_summary_row(summary: dict) -> str:
    """Format one summary dict as a fixed-width table row."""
    return (
        f"{summary.get('variant', ''):<20} | {summary['recall@5']:>8.3f} | "
        f"{summary['recall@10']:>10.3f} | {summary['mrr']:>6.3f} | "
        f"{summary['rouge_l']:>8.3f} | {summary['bert_score_f1']:>13.3f}"
    )


def print_summary_table(summaries: list[dict]) -> None:
    """Print the variant comparison table (header + rows) to stdout."""
    print(_SUMMARY_HEADER)
    print("-" * len(_SUMMARY_HEADER))
    for s in summaries:
        print(_format_summary_row(s))


# --------------------------------------------------------------------------- #
# Smoke entry point (AGENT.md: test on 5 QA pairs, print metric values)
# --------------------------------------------------------------------------- #
class _StubPipeline:
    """Minimal pipeline stand-in so the evaluator is testable without Neo4j/LLM."""

    def __init__(self, catalog):
        self._catalog = catalog  # {question: ([{"chunk_id":...}], answer)}
        self.driver = None

    def retrieve(self, query, top_k=10, expand_graph=True):
        return self._catalog[query][0]

    def generate(self, query, chunks):
        return self._catalog[query][1]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    # 5 synthetic QA pairs: (retrieved chunks, generated answer, gold answer).
    catalog = {
        "Does metformin cut cancer risk?": (
            [{"chunk_id": "1_0"}, {"chunk_id": "1_1"}, {"chunk_id": "9_0"}],
            "Metformin may reduce colorectal cancer risk in diabetic patients.",
            "Metformin reduces cancer incidence in patients with type 2 diabetes.",
        ),
        "Is CRISPR safe for gene therapy?": (
            [{"chunk_id": "3_0"}, {"chunk_id": "5_0"}],
            "High-fidelity CRISPR variants reduce off-target effects substantially.",
            "CRISPR-Cas9 off-target activity remains a safety concern for therapy.",
        ),
        "Do statins prevent heart attacks?": (
            [{"chunk_id": "7_0"}, {"chunk_id": "8_0"}, {"chunk_id": "7_1"}],
            "Statins lower cholesterol and reduce myocardial infarction risk.",
            "Statins are effective for the secondary prevention of heart attacks.",
        ),
        "Can deep learning diagnose skin cancer?": (
            [{"chunk_id": "2_0"}, {"chunk_id": "6_0"}],
            "CNNs can classify skin lesions with dermatologist-level accuracy.",
            "Deep learning models match dermatologists in skin cancer classification.",
        ),
        "Does sleep loss impair cognition?": (
            [{"chunk_id": "4_0"}],
            "Sleep deprivation impairs cognitive performance.",
            "Sleep loss negatively affects attention and memory.",
        ),
    }
    eval_set = [
        {
            "pubmed_id": str(pid),
            "question": question,
            "long_answer": gold_answer,
        }
        for pid, (question, (_, _generated, gold_answer)) in enumerate(
            catalog.items(), start=1
        )
    ]
    # Gold = the question's first retrieved chunk (a hit) + an unseen id (a miss),
    # so the demo exercises partial-overlap retrieval metrics rather than 0 or 1.
    gold = {
        str(pid): {chunks[0]["chunk_id"], "unrelated_chunk"}
        for pid, (chunks, _, _) in enumerate(catalog.values(), start=1)
    }

    pipe = _StubPipeline({q: (chunks, answer) for q, (chunks, answer, _) in catalog.items()})
    summary = run_full_evaluation(
        pipe, eval_set, variant_name="smoke", output_dir="data/eval",
        expand_graph=False, gold_chunks_by_article=gold,
    )
    print_summary_table([summary])

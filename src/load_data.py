"""Download a sample of PubMed abstracts from HuggingFace to ``data/raw/``.

Samples ``n`` (default 5000) abstracts from the ``alexaapo/scientific_papers``
dataset (``pubmed`` config) in streaming mode and persists them as JSON.
"""

# 1. stdlib
import json
import logging
import os

logger = logging.getLogger(__name__)

DATASET_ID = "alexaapo/scientific_papers"
DATASET_CONFIG = "pubmed"
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_OUTPUT_PATH = "data/raw/abstracts.json"


def _extract_record(row: dict) -> dict:
    """Normalise one dataset row to the project's abstract schema."""
    raw_id = row.get("article_id") or row.get("id") or row.get("pmid") or ""
    raw_abstract = row.get("abstract") or row.get("article") or ""
    raw_title = row.get("title") or row.get("article") or raw_abstract
    raw_date = row.get("pub_date") or row.get("year") or row.get("date") or ""
    return {
        "article_id": str(raw_id),
        "abstract": (str(raw_abstract)).strip(),
        "title": (str(raw_title)[:200]).strip(),
        "pub_date": str(raw_date),
    }


def load_abstracts(n: int = DEFAULT_SAMPLE_SIZE) -> list[dict]:
    """Stream ``n`` PubMed abstracts from HuggingFace as normalised dicts."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split="train", streaming=True)
    try:
        logger.info("Dataset features: %s", ds.features)
    except Exception:
        pass

    abstracts: list[dict] = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        if i == 0:
            logger.info("First row keys: %s", list(row.keys()))
        abstracts.append(_extract_record(row))
    logger.info("Loaded %d abstracts from %s.", len(abstracts), DATASET_ID)
    return abstracts


def save_abstracts(
    abstracts: list[dict], output_path: str = DEFAULT_OUTPUT_PATH
) -> None:
    """Persist ``abstracts`` to JSON, creating parent directories as needed."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(abstracts, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %d abstracts -> %s", len(abstracts), output_path)


def main(
    n: int = DEFAULT_SAMPLE_SIZE, output_path: str = DEFAULT_OUTPUT_PATH
) -> None:
    """Load ``n`` abstracts and save them to ``output_path``."""
    abstracts = load_abstracts(n)
    save_abstracts(abstracts, output_path)
    print(f"Saved {len(abstracts)} abstracts to {output_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Sample PubMed abstracts to JSON.")
    parser.add_argument(
        "-n", "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help="number of abstracts to sample (default: 5000)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help="output JSON path (default: data/raw/abstracts.json)",
    )
    args = parser.parse_args()
    main(args.sample_size, args.output)

#!/usr/bin/env python3
"""Add embedding-based semantic agreement from repeated answer samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_state import load_json, save_json_atomic


def mean_pairwise_cosine(embeddings) -> float | None:
    n = len(embeddings)
    if n < 2:
        return None
    total, pairs = 0.0, 0
    for left in range(n):
        for right in range(left + 1, n):
            total += float((embeddings[left] * embeddings[right]).sum())
            pairs += 1
    # Cosine similarity [-1, 1] is mapped to a confidence-like [0, 1] value.
    return max(0.0, min(1.0, (total / pairs + 1) / 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit("Install research extras first: pip install -r requirements-research.txt") from error

    records = load_json(args.input)
    model = SentenceTransformer(args.model)
    for record in records:
        answers = record.get("sample_answers") or []
        if len(answers) < 2:
            record["semantic_agreement"] = None
            continue
        embeddings = model.encode(answers, normalize_embeddings=True, convert_to_tensor=True)
        record["semantic_agreement"] = mean_pairwise_cosine(embeddings)
        record["semantic_model_id"] = args.model
    save_json_atomic(args.output, records)
    print(f"Saved semantic-agreement scores: {args.output}")


if __name__ == "__main__":
    main()

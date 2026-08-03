"""
Phase 3 — Build scaled question set from Hugging Face datasets.

Run in Colab:
  !pip install -q datasets
  !python phase_3/build_question_set.py

Outputs: phase_3/artifacts/questions_700.json (and copies schema compatible with sample.py)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

PHASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = PHASE_DIR / "artifacts"
OUT_PATH = ARTIFACTS / "questions_700.json"

N_TRIVIA = 400
N_MATH = 400


def _sample_indices(n_available: int, n_want: int, label: str) -> list[int]:
    if n_available < n_want:
        raise ValueError(f"{label}: wanted {n_want} rows but only {n_available} available")
    return random.sample(range(n_available), n_want)


def _normalize_trivia_row(row, idx: int) -> dict | None:
    question = (row.get("question") or "").strip()
    answer = row.get("answer")
    if isinstance(answer, dict):
        ground_truth = (answer.get("value") or "").strip()
        aliases = answer.get("aliases") or []
    else:
        ground_truth = str(answer or "").strip()
        aliases = []

    aliases = [str(a).strip() for a in aliases if str(a).strip()]
    if not question or not ground_truth:
        return None

    return {
        "id": f"trivia_{idx:04d}",
        "domain": "trivia",
        "question": question,
        "ground_truth": ground_truth,
        "answer_aliases": aliases,
    }


def _normalize_math_row(row, idx: int) -> dict | None:
    question = (row.get("prompt") or row.get("question") or "").strip()
    ground_truth = str(row.get("solution") or row.get("ground_truth") or "").strip()
    if not question or not ground_truth:
        return None
    # DAPO solution should be short final answer (e.g. "34", "131")
    if len(ground_truth) > 64:
        return None
    return {
        "id": f"math_{idx:04d}",
        "domain": "math",
        "question": question,
        "ground_truth": ground_truth,
    }


def build_trivia(n: int) -> list[dict]:
    from datasets import load_dataset

    print("Loading trivia_qa (validation split)...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    indices = _sample_indices(len(ds), n, "trivia_qa")

    out = []
    for i, idx in enumerate(indices):
        rec = _normalize_trivia_row(ds[idx], i)
        if rec is None:
            continue
        out.append(rec)
    print(f"  trivia kept: {len(out)}/{n} requested")
    return out


def build_math(n: int) -> list[dict]:
    from datasets import load_dataset

    print("Loading DAPO-Math-17k-Processed (en)...")
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    indices = _sample_indices(len(ds), n, "DAPO-Math")

    out = []
    for i, idx in enumerate(indices):
        rec = _normalize_math_row(ds[idx], i)
        if rec is None:
            continue
        out.append(rec)
    print(f"  math kept: {len(out)}/{n} requested")
    return out


def validate_questions(questions: list[dict]) -> None:
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids detected")

    for q in questions:
        for key in ("id", "domain", "question", "ground_truth"):
            if key not in q or not str(q[key]).strip():
                raise ValueError(f"Missing/empty {key} in {q.get('id')}")

    domains = {}
    for q in questions:
        domains[q["domain"]] = domains.get(q["domain"], 0) + 1
    print(f"  domains: {domains}")


def preview(questions: list[dict], k: int = 3) -> None:
    print("\nSample rows (this is normal output — not an error):")
    for q in questions[:k]:
        gt = q["ground_truth"]
        preview_gt = gt if len(gt) <= 40 else gt[:37] + "..."
        qpreview = q["question"][:80].replace("\n", " ")
        print(f"  [{q['domain']}] id={q['id']} gt={preview_gt!r}")
        print(f"    Q: {qpreview}...")


def main():
    trivia = build_trivia(N_TRIVIA)
    math = build_math(N_MATH)
    all_questions = trivia + math
    random.shuffle(all_questions)

    validate_questions(all_questions)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\nSaved {len(all_questions)} questions -> {OUT_PATH}")
    print(f"  trivia: {len(trivia)}, math: {len(math)}")
    preview(all_questions)

    print("\nLimitations (state in paper):")
    print("  - Trivia: held-out validation split.")
    print("  - Math: sampled from DAPO en train — NOT a guaranteed clean hold-out")
    print("    (Gnosis checkpoint may have seen similar math during training).")
    print("\nTo run sample.py on this set in Colab:")
    print("  !cp phase_3/artifacts/questions_700.json questions.json")
    print("  !PYTHONPATH=/content/GATE/Gnosis python sample.py")


if __name__ == "__main__":
    main()

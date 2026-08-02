#!/usr/bin/env python3
"""
Phase 2 — AUROC for Gnosis score vs wrong answers (no GPU).

Reads baseline_results.json only. Checks whether gnosis_score (or flipped
risk = 1 - score) ranks wrong answers above correct ones.

Usage:
  python phase_2/auroc_analysis.py

Optional env:
  ARTIFACTS_DIR=phase_2/artifacts
  BASELINE_PATH=/path/to/baseline_results.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = PHASE_DIR / "artifacts"


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {path}")


def resolve_baseline_path() -> Path:
    explicit = os.environ.get("BASELINE_PATH", "").strip()
    if explicit:
        return Path(explicit)
    artifacts = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS))
    for p in (
        artifacts / "baseline_results.json",
        PHASE_DIR.parent / "baseline_results.json",
    ):
        if p.exists():
            return p
    raise SystemExit(
        "Missing baseline_results.json. Set BASELINE_PATH or put file in phase_2/artifacts/."
    )


def roc_auc_score(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score as sklearn_auc
        return float(sklearn_auc(y_true, y_score))
    except ImportError:
        return _roc_auc_manual(y_true, y_score)


def _roc_auc_manual(y_true, y_score):
    """Rank-based AUROC without sklearn."""
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative labels for AUROC")
    rank_sum = 0.0
    for i, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += i
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def compute_auroc(records):
    scores = [float(r["gnosis_score"]) for r in records]
    # 1 = wrong (positive class we want to detect)
    labels = [0 if r["baseline_correct"] else 1 for r in records]

    n_wrong = sum(labels)
    n_correct = len(labels) - n_wrong

    correct_scores = [s for s, y in zip(scores, labels) if y == 0]
    wrong_scores = [s for s, y in zip(scores, labels) if y == 1]

    # High gnosis_score = model thinks correct → flip for "risk" of being wrong
    risk_scores = [1.0 - s for s in scores]

    auroc_raw = roc_auc_score(labels, scores)
    auroc_risk = roc_auc_score(labels, risk_scores)

    return {
        "n": len(records),
        "n_wrong": n_wrong,
        "n_correct": n_correct,
        "auroc_gnosis_score": round(auroc_raw, 4),
        "auroc_risk_1_minus_score": round(auroc_risk, 4),
        "correct_score_mean": round(sum(correct_scores) / len(correct_scores), 4) if correct_scores else None,
        "correct_score_min": round(min(correct_scores), 4) if correct_scores else None,
        "wrong_score_mean": round(sum(wrong_scores) / len(wrong_scores), 4) if wrong_scores else None,
        "wrong_score_max": round(max(wrong_scores), 4) if wrong_scores else None,
        "interpretation_note": (
            "AUROC uses risk = 1 - gnosis_score (higher risk = more likely wrong). "
            f"Only {n_wrong} wrong answers — unstable on small samples."
        ),
    }


def interpret_auroc(auroc_risk: float) -> str:
    if auroc_risk >= 0.7:
        return "Signal has some discriminative power; threshold tuning may help."
    if auroc_risk <= 0.55:
        return "Near random (0.5) — score barely separates wrong from correct."
    return "Weak separation — better than chance but not strong."


def print_auroc_report(metrics: dict):
    print("\n" + "=" * 72)
    print("AUROC — Can Gnosis score rank wrong answers?")
    print("=" * 72)
    print(f"N = {metrics['n']}  |  wrong = {metrics['n_wrong']}  |  correct = {metrics['n_correct']}")
    print(f"AUROC (raw gnosis_score):     {metrics['auroc_gnosis_score']:.4f}")
    print(f"AUROC (risk = 1 - score):     {metrics['auroc_risk_1_minus_score']:.4f}  ← use this")
    print(
        f"Correct scores — mean: {metrics['correct_score_mean']}, "
        f"min: {metrics['correct_score_min']}"
    )
    print(
        f"Wrong scores   — mean: {metrics['wrong_score_mean']}, "
        f"max: {metrics['wrong_score_max']}"
    )
    print(f"\n{interpret_auroc(metrics['auroc_risk_1_minus_score'])}")
    print(f"Note: {metrics['interpretation_note']}")


def main():
    baseline_path = resolve_baseline_path()
    records = load_json(baseline_path)
    print(f"Loaded: {baseline_path} ({len(records)} records)")

    metrics = compute_auroc(records)
    print_auroc_report(metrics)

    artifacts = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS))
    out = artifacts / "auroc.json"
    save_json(out, metrics)
    return metrics


if __name__ == "__main__":
    main()

"""Threshold sweep metrics for Gnosis score gating (no GPU)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DEFAULT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def threshold_metrics(
    records: list[dict],
    threshold: float,
    score_key: str = "gnosis_score",
    correct_key: str = "baseline_correct",
) -> dict:
    """Detection counts at a single threshold (intervene if score < threshold)."""
    flagged = [r for r in records if r[score_key] < threshold]
    caught = sum(1 for r in flagged if not r[correct_key])
    unnecessary = sum(1 for r in flagged if r[correct_key])
    total_wrong = sum(1 for r in records if not r[correct_key])
    n_flagged = len(flagged)
    missed = sum(
        1 for r in records if r[score_key] >= threshold and not r[correct_key]
    )
    return {
        "threshold": threshold,
        "flagged": n_flagged,
        "caught": caught,
        "unnecessary": unnecessary,
        "missed": missed,
        "precision": round(caught / n_flagged, 4) if n_flagged else 0.0,
        "recall": round(caught / total_wrong, 4) if total_wrong else 0.0,
        "total_wrong": total_wrong,
        "n": len(records),
    }


def sweep_thresholds(
    records: list[dict],
    thresholds: list[float] | None = None,
    **kwargs,
) -> list[dict]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    return [threshold_metrics(records, t, **kwargs) for t in thresholds]


def _main():
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    baseline = os.environ.get("BASELINE_PATH", "").strip()
    path = Path(baseline) if baseline else Path(__file__).parent / "artifacts" / "baseline_results.json"
    if not path.exists():
        path = repo / "baseline_results.json"
    with open(path) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} from {path}\n")
    print(f"{'τ':>5} {'flag':>5} {'caught':>6} {'prec':>6} {'rec':>6} {'unnec':>6}")
    for row in sweep_thresholds(records):
        print(
            f"{row['threshold']:5.2f} {row['flagged']:5d} {row['caught']:6d} "
            f"{row['precision']:6.3f} {row['recall']:6.3f} {row['unnecessary']:6d}"
        )


if __name__ == "__main__":
    _main()

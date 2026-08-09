#!/usr/bin/env python3
"""
Phase 3 — AUROC for Gnosis score vs wrong answers (no GPU).

Reads Phase 3 baseline_results.json (800 questions). Reports overall AUROC
and per-domain breakdown (trivia vs math). Compares to Phase 2 if available.

Usage:
  python phase_3/auroc_analysis.py

Colab (baseline on Drive):
  BASELINE_PATH=/content/drive/MyDrive/gate_phase3_baseline.json python phase_3/auroc_analysis.py

Optional env:
  BASELINE_PATH=path/to/baseline_results.json
  ARTIFACTS_DIR=phase_3/artifacts
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE_DIR.parent
DEFAULT_ARTIFACTS = PHASE_DIR / "artifacts"
PHASE2_ARTIFACTS = REPO_ROOT / "phase_2" / "artifacts"

# Reuse core metrics from Phase 2
sys.path.insert(0, str(REPO_ROOT / "phase_2"))
from auroc_analysis import compute_auroc, interpret_auroc, load_json, print_auroc_report, save_json


def resolve_baseline_path() -> Path:
    explicit = os.environ.get("BASELINE_PATH", "").strip()
    if explicit:
        return Path(explicit)
    artifacts = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS))
    for p in (
        artifacts / "baseline_results.json",
        REPO_ROOT / "baseline_results.json",
    ):
        if p.exists():
            return p
    raise SystemExit(
        "Missing baseline file. Set BASELINE_PATH or place baseline_results.json in "
        "phase_3/artifacts/.\n"
        "Colab: BASELINE_PATH=/content/drive/MyDrive/gate_phase3_baseline.json "
        "python phase_3/auroc_analysis.py"
    )


def subset_metrics(records: list[dict], label: str) -> dict | None:
    if not records:
        return None
    m = compute_auroc(records)
    m["label"] = label
    m["interpretation_note"] = (
        f"AUROC uses risk = 1 - gnosis_score. Subset: {label} (n={m['n']})."
    )
    return m


def load_phase2_auroc() -> dict | None:
    path = PHASE2_ARTIFACTS / "auroc.json"
    if path.exists():
        return load_json(path)
    return None


def print_domain_report(metrics: dict):
    print("\n" + "-" * 72)
    print(f"  {metrics['label']}: n={metrics['n']}  wrong={metrics['n_wrong']}  correct={metrics['n_correct']}")
    print(f"  AUROC (risk = 1 - score): {metrics['auroc_risk_1_minus_score']:.4f}")
    print(f"  Correct mean score: {metrics['correct_score_mean']}  |  Wrong mean score: {metrics['wrong_score_mean']}")
    print(f"  {interpret_auroc(metrics['auroc_risk_1_minus_score'])}")


def print_phase2_comparison(phase3_risk: float, phase2: dict | None):
    if phase2 is None:
        print("\n(No phase_2/artifacts/auroc.json found for comparison.)")
        return
    p2 = phase2["auroc_risk_1_minus_score"]
    delta = round(phase3_risk - p2, 4)
    print("\n" + "=" * 72)
    print("Comparison — Phase 2 (n=100) vs Phase 3 (n=800)")
    print("=" * 72)
    print(f"  Phase 2 AUROC (risk): {p2:.4f}  (wrong={phase2['n_wrong']}/{phase2['n']})")
    print(f"  Phase 3 AUROC (risk): {phase3_risk:.4f}")
    print(f"  Delta: {delta:+.4f}")


def main():
    baseline_path = resolve_baseline_path()
    records = load_json(baseline_path)
    print(f"Loaded: {baseline_path} ({len(records)} records)")

    overall = compute_auroc(records)
    overall["label"] = "overall"
    overall["interpretation_note"] = (
        "AUROC uses risk = 1 - gnosis_score (higher risk = more likely wrong). "
        f"Phase 3 scaled set: n={overall['n']}."
    )

    print_auroc_report(overall)

    by_domain: dict[str, list] = {}
    for r in records:
        by_domain.setdefault(r.get("domain", "unknown"), []).append(r)

    domain_metrics = []
    print("\n" + "=" * 72)
    print("Per-domain AUROC")
    print("=" * 72)
    for domain in sorted(by_domain):
        m = subset_metrics(by_domain[domain], label=domain)
        if m:
            domain_metrics.append(m)
            print_domain_report(m)

    phase2 = load_phase2_auroc()
    print_phase2_comparison(overall["auroc_risk_1_minus_score"], phase2)

    artifacts = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS))
    out = {
        "baseline_path": str(baseline_path),
        "overall": overall,
        "by_domain": {m["label"]: m for m in domain_metrics},
        "phase2_comparison": phase2,
    }
    out_path = artifacts / "auroc.json"
    save_json(out_path, out)
    print(f"\nFull report saved: {out_path}")
    return out


if __name__ == "__main__":
    main()

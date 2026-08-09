#!/usr/bin/env python3
"""
Phase 3 — Bootstrap confidence intervals (no GPU).

Computes bootstrap 95% CIs for:
  - AUROC (risk = 1 - gnosis_score), overall and by domain
  - Threshold sweep rows (precision, recall, flagged count)

Usage:
  BASELINE_PATH=/content/drive/MyDrive/gate_phase3_baseline.json python phase_3/bootstrap_analysis.py

Env:
  BASELINE_PATH, ARTIFACTS_DIR=phase_3/artifacts
  N_BOOT=2000
  SEED=42
  CI=0.95
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE_DIR.parent
DEFAULT_ARTIFACTS = PHASE_DIR / "artifacts"

sys.path.insert(0, str(PHASE_DIR))
sys.path.insert(0, str(REPO_ROOT / "phase_2"))
from auroc_analysis import _roc_auc_manual, load_json, save_json

from threshold_sweep import DEFAULT_THRESHOLDS, sweep_thresholds, threshold_metrics


def resolve_baseline_path() -> Path:
    explicit = os.environ.get("BASELINE_PATH", "").strip()
    if explicit:
        return Path(explicit)
    artifacts = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS))
    for p in (artifacts / "baseline_results.json", REPO_ROOT / "baseline_results.json"):
        if p.exists():
            return p
    raise SystemExit("Set BASELINE_PATH or place baseline_results.json in phase_3/artifacts/.")


def bootstrap_auroc(records: list[dict], n_boot: int, seed: int, ci: float) -> dict:
    rng = random.Random(seed)
    n = len(records)
    scores = [float(r["gnosis_score"]) for r in records]
    labels = [0 if r["baseline_correct"] else 1 for r in records]
    risk = [1.0 - s for s in scores]

    if sum(labels) == 0 or sum(labels) == n:
        return {"auroc_risk": None, "ci_low": None, "ci_high": None, "n_boot": n_boot}

    samples = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        y = [labels[i] for i in idx]
        s = [risk[i] for i in idx]
        if sum(y) == 0 or sum(y) == len(y):
            continue
        try:
            samples.append(_roc_auc_manual(y, s))
        except ValueError:
            continue

    if not samples:
        return {"auroc_risk": None, "ci_low": None, "ci_high": None, "n_boot": 0}

    samples.sort()
    point = _roc_auc_manual(labels, risk)
    alpha = (1 - ci) / 2
    lo = samples[int(alpha * len(samples))]
    hi = samples[int((1 - alpha) * len(samples)) - 1]
    return {
        "auroc_risk_point": round(point, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n_boot_effective": len(samples),
        "ci_level": ci,
    }


def bootstrap_threshold_row(
    records: list[dict],
    threshold: float,
    n_boot: int,
    seed: int,
    ci: float,
) -> dict:
    rng = random.Random(seed + int(threshold * 1000))
    n = len(records)
    prec_samples, rec_samples, flag_samples = [], [], []

    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sample = [records[i] for i in idx]
        m = threshold_metrics(sample, threshold)
        prec_samples.append(m["precision"])
        rec_samples.append(m["recall"])
        flag_samples.append(m["flagged"])

    point = threshold_metrics(records, threshold)
    alpha = (1 - ci) / 2

    def ci_band(arr):
        arr = sorted(arr)
        return round(arr[int(alpha * len(arr))], 4), round(arr[int((1 - alpha) * len(arr)) - 1], 4)

    p_lo, p_hi = ci_band(prec_samples)
    r_lo, r_hi = ci_band(rec_samples)
    f_lo, f_hi = ci_band(flag_samples)

    return {
        **point,
        "precision_ci": [p_lo, p_hi],
        "recall_ci": [r_lo, r_hi],
        "flagged_ci": [int(f_lo), int(f_hi)],
    }


def print_auroc_ci(label: str, stats: dict):
    if stats.get("auroc_risk_point") is None:
        print(f"  {label}: insufficient labels for AUROC")
        return
    print(
        f"  {label}: AUROC={stats['auroc_risk_point']:.4f}  "
        f"95% CI [{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]  "
        f"(boot={stats['n_boot_effective']})"
    )


def main():
    n_boot = int(os.environ.get("N_BOOT", "2000"))
    seed = int(os.environ.get("SEED", "42"))
    ci = float(os.environ.get("CI", "0.95"))
    thresholds = DEFAULT_THRESHOLDS
    env_t = os.environ.get("THRESHOLDS", "").strip()
    if env_t:
        thresholds = [float(x) for x in env_t.split(",")]

    baseline_path = resolve_baseline_path()
    records = load_json(baseline_path)
    print(f"Loaded: {baseline_path} ({len(records)} records)")
    print(f"Bootstrap: n={n_boot}, seed={seed}, CI={ci}")

    by_domain: dict[str, list] = {}
    for r in records:
        by_domain.setdefault(r.get("domain", "unknown"), []).append(r)

    auroc_ci = {"overall": bootstrap_auroc(records, n_boot, seed, ci)}
    for domain, subset in sorted(by_domain.items()):
        auroc_ci[domain] = bootstrap_auroc(subset, n_boot, seed + 1, ci)

    print("\n" + "=" * 72)
    print("Bootstrap AUROC (risk = 1 - score)")
    print("=" * 72)
    for label, stats in auroc_ci.items():
        print_auroc_ci(label, stats)

    sweep_ci = [
        bootstrap_threshold_row(records, t, n_boot, seed + 2, ci) for t in thresholds
    ]

    print("\n" + "=" * 72)
    print("Threshold sweep with bootstrap CIs (overall)")
    print("=" * 72)
    print(f"{'τ':>5} {'flag':>5} {'flag CI':>12} {'prec':>6} {'prec CI':>16} {'rec':>6} {'rec CI':>16}")
    for row in sweep_ci:
        print(
            f"{row['threshold']:5.2f} {row['flagged']:5d} "
            f"[{row['flagged_ci'][0]:3d},{row['flagged_ci'][1]:3d}] "
            f"{row['precision']:6.3f} [{row['precision_ci'][0]:.3f},{row['precision_ci'][1]:.3f}] "
            f"{row['recall']:6.3f} [{row['recall_ci'][0]:.3f},{row['recall_ci'][1]:.3f}]"
        )

    artifacts = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS))
    out = {
        "baseline_path": str(baseline_path),
        "n_boot": n_boot,
        "seed": seed,
        "ci_level": ci,
        "auroc_ci": auroc_ci,
        "threshold_sweep_ci": sweep_ci,
    }
    out_path = artifacts / "bootstrap.json"
    save_json(out_path, out)
    print(f"\nSaved: {out_path}")
    return out


if __name__ == "__main__":
    main()

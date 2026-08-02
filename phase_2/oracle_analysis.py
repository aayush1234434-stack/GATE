#!/usr/bin/env python3
"""
Phase 2 — Oracle & condition comparison (no GPU).

Reads saved artifacts and prints:
  - Detection: Gnosis vs Random vs Oracle (perfect wrong-only gate)
  - Outcome: Baseline vs Random vs Gnosis (if gnosis_results.json exists)
  - Theoretical oracle outcome ceiling (if regen always fixes wrong answers)
  - AUROC: can gnosis_score rank wrong vs correct? (via auroc_analysis)

Usage:
  python phase_2/oracle_analysis.py

Optional env:
  ARTIFACTS_DIR=phase_2/artifacts
  THRESHOLD=0.85
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PHASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE_DIR.parent
DEFAULT_ARTIFACTS = PHASE_DIR / "artifacts"
THRESHOLD = float(os.environ.get("THRESHOLD", "0.85"))


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {path}")


def baseline_wrong_set(records):
    return {i for i, r in enumerate(records) if not r["baseline_correct"]}


def detection_metrics(records, flagged_indices: set[int], label: str):
    wrong = baseline_wrong_set(records)
    flagged = flagged_indices
    caught = len(wrong & flagged)
    missed = len(wrong - flagged)
    unnecessary = len(flagged - wrong)
    intervened = len(flagged)
    precision = caught / intervened if intervened else 0.0
    recall = caught / len(wrong) if wrong else 0.0

    return {
        "label": label,
        "threshold_or_policy": label,
        "intervened": intervened,
        "caught": caught,
        "missed": missed,
        "unnecessary": unnecessary,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "total_wrong": len(wrong),
    }


def outcome_summary(records, label: str):
    n = len(records)
    baseline_wrong = sum(1 for r in records if not r["baseline_correct"])
    final_wrong = sum(1 for r in records if not r["final_correct"])
    intervened = [r for r in records if r.get("intervened")]
    fixed = sum(1 for r in intervened if not r["baseline_correct"] and r["final_correct"])
    broke = sum(1 for r in intervened if r["baseline_correct"] and not r["final_correct"])
    still_wrong = sum(1 for r in intervened if not r["final_correct"])
    wrong_among = sum(1 for r in intervened if not r["baseline_correct"])

    return {
        "label": label,
        "n": n,
        "baseline_wrong": baseline_wrong,
        "final_wrong": final_wrong,
        "intervened": len(intervened),
        "wrong_among_intervened": wrong_among,
        "fixed": fixed,
        "broke": broke,
        "still_wrong_after_regen": still_wrong,
        "hallucination_reduction": baseline_wrong - final_wrong,
    }


def gnosis_flagged_indices(records, threshold=THRESHOLD):
    return {i for i, r in enumerate(records) if r["gnosis_score"] < threshold}


def random_flagged_indices(picked_path: Path, n_records: int):
    if not picked_path.exists():
        return set()
    meta = load_json(picked_path)
    return set(meta["indices"])


def find_gnosis_results(artifacts: Path) -> Path | None:
    for name in ("gnosis_results.json", "results.json"):
        p = artifacts / name
        if p.exists():
            return p
    p = REPO_ROOT / "results.json"
    return p if p.exists() else None


def print_detection_table(rows):
    print("\n" + "=" * 72)
    print("DETECTION COMPARISON (who gets flagged for intervention?)")
    print("=" * 72)
    print(
        f"{'Condition':<22} {'Flagged':>8} {'Caught':>7} {'Missed':>7} "
        f"{'Unnec.':>7} {'Prec.':>7} {'Recall':>7}"
    )
    print("-" * 72)
    for r in rows:
        print(
            f"{r['label']:<22} {r['intervened']:>8} {r['caught']:>7} {r['missed']:>7} "
            f"{r['unnecessary']:>7} {r['precision']:>7.2%} {r['recall']:>7.2%}"
        )


def print_outcome_table(rows):
    print("\n" + "=" * 72)
    print("OUTCOME COMPARISON (after regenerate intervention)")
    print("=" * 72)
    print(
        f"{'Condition':<22} {'Base wrong':>11} {'Final wrong':>12} "
        f"{'Fixed':>6} {'Broke':>6} {'Net Δ':>6}"
    )
    print("-" * 72)
    for r in rows:
        print(
            f"{r['label']:<22} {r['baseline_wrong']:>11} {r['final_wrong']:>12} "
            f"{r.get('fixed', '-'):>6} {r.get('broke', '-'):>6} "
            f"{r['hallucination_reduction']:>6}"
        )


def print_wrong_questions(records):
    wrong = [r for r in records if not r["baseline_correct"]]
    print("\n" + "=" * 72)
    print(f"BASELINE WRONG ANSWERS ({len(wrong)} total)")
    print("=" * 72)
    for r in wrong:
        print(
            f"  id={r.get('id')} score={r['gnosis_score']:.4f} "
            f"flagged@0.85={'yes' if r['gnosis_score'] < THRESHOLD else 'no'}  "
            f"domain={r.get('domain')}  Q: {r['question'][:70]}..."
        )


def main():
    artifacts = Path(os.environ.get("ARTIFACTS_DIR", DEFAULT_ARTIFACTS))
    baseline_path = artifacts / "baseline_results.json"
    random_path = artifacts / "random_results.json"
    picked_path = artifacts / "random_picked_ids.json"
    out_path = artifacts / "oracle_analysis.json"

    if not baseline_path.exists():
        raise SystemExit(f"Missing {baseline_path}")

    baseline = load_json(baseline_path)
    n = len(baseline)
    baseline_summary = outcome_summary(baseline, "No intervention (baseline)")

    # --- Detection ---
    gnosis_det = detection_metrics(
        baseline,
        gnosis_flagged_indices(baseline),
        f"Gnosis (score < {THRESHOLD})",
    )
    random_det = detection_metrics(
        baseline,
        random_flagged_indices(picked_path, n),
        "Random (matched budget)",
    )
    oracle_det = detection_metrics(
        baseline,
        baseline_wrong_set(baseline),
        "Oracle (perfect: flag all wrong)",
    )

    print_detection_table([gnosis_det, random_det, oracle_det])

    # --- Outcome ---
    outcome_rows = [baseline_summary]

    if random_path.exists():
        random_results = load_json(random_path)
        outcome_rows.append(outcome_summary(random_results, "Random-gated regenerate"))
    else:
        print(f"\nNote: missing {random_path.name} — skip random outcome row")

    gnosis_path = find_gnosis_results(artifacts)
    if gnosis_path:
        gnosis_results = load_json(gnosis_path)
        outcome_rows.append(outcome_summary(gnosis_results, "Gnosis-gated regenerate"))
    else:
        print(
            f"\nNote: missing gnosis_results.json — add to {artifacts} for Gnosis outcome row. "
            "Detection for Gnosis is still computed from baseline scores."
        )

    # Theoretical oracle outcome ceiling (no GPU): assume regen fixes every wrong answer
    bw = baseline_summary["baseline_wrong"]
    oracle_outcome = {
        "label": "Oracle outcome (theoretical ceiling)",
        "n": n,
        "baseline_wrong": bw,
        "final_wrong": 0,
        "intervened": bw,
        "fixed": bw,
        "broke": 0,
        "hallucination_reduction": bw,
        "note": "Assumes regenerate fixes all known-wrong answers; not measured unless you run oracle regen.",
    }
    outcome_rows.append(oracle_outcome)

    print_outcome_table(outcome_rows)

    print_wrong_questions(baseline)

    # --- AUROC ---
    import sys
    sys.path.insert(0, str(PHASE_DIR))
    from auroc_analysis import compute_auroc, interpret_auroc, print_auroc_report

    auroc_metrics = compute_auroc(baseline)
    print_auroc_report(auroc_metrics)
    save_json(artifacts / "auroc.json", auroc_metrics)

    # --- Interpretation ---
    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    if random_path.exists() and gnosis_path:
        g = outcome_rows[-2] if len(outcome_rows) >= 3 else None
        r = next((x for x in outcome_rows if x["label"] == "Random-gated regenerate"), None)
        if g and r:
            if r["hallucination_reduction"] >= g["hallucination_reduction"]:
                print("  Random ≥ Gnosis on net reduction → Gnosis score may not help vs chance.")
            else:
                print("  Gnosis > Random on net reduction → Gnosis may help select who to fix.")
    print(f"  Oracle detection recall: {oracle_det['recall']:.0%} (upper bound for any signal)")
    print(f"  Gnosis detection recall: {gnosis_det['recall']:.0%} at threshold {THRESHOLD}")
    if gnosis_det["recall"] < oracle_det["recall"]:
        print("  Gnosis catches fewer wrong answers than a perfect oracle would.")
    print(f"  AUROC (risk = 1 - score): {auroc_metrics['auroc_risk_1_minus_score']:.4f} — "
          f"{interpret_auroc(auroc_metrics['auroc_risk_1_minus_score'])}")

    report = {
        "threshold": THRESHOLD,
        "n_questions": n,
        "auroc": auroc_metrics,
        "detection": {
            "gnosis": gnosis_det,
            "random": random_det,
            "oracle": oracle_det,
        },
        "outcome": outcome_rows,
        "baseline_wrong_ids": [
            r.get("id") for r in baseline if not r["baseline_correct"]
        ],
    }
    save_json(out_path, report)


if __name__ == "__main__":
    main()

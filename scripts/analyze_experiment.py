#!/usr/bin/env python3
"""Produce paper-facing statistics from baseline and intervention artifacts.

This script never chooses a threshold and never edits input artifacts.  Use
``evaluate_baselines.py`` for validation-only threshold selection, then use
this script once on the locked test output and its intervention arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from confidence_baselines import BASELINES, baseline_risks
from experiment_state import load_json, save_json_atomic
from statistical_analysis import (
    compute_cost_summary,
    paired_outcome_test,
    paired_revision_test,
    per_group_counts,
    ranking_statistics,
)


def _statistics_for_records(records: list[dict], n_boot: int, seed: int, ci: float) -> dict:
    report = {
        "n": len(records),
        "n_wrong": sum(not bool(record.get("baseline_correct")) for record in records),
        "domains": per_group_counts(records, "domain"),
        "models": per_group_counts(records, "model_name"),
        "baselines": {},
        "cost": compute_cost_summary(records),
    }
    for offset, name in enumerate(BASELINES):
        labels, risks = baseline_risks(records, name)
        if labels:
            report["baselines"][name] = ranking_statistics(
                labels, risks, n_boot=n_boot, seed=seed + 10 * offset, ci=ci
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline result JSON.")
    parser.add_argument("--gnosis", type=Path, help="Gnosis intervention result JSON.")
    parser.add_argument(
        "--random",
        type=Path,
        action="append",
        default=[],
        help="One or more matched random-arm result JSON files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", default="all", help="Restrict every input to a split label (for example: test)."
    )
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args()
    if args.n_boot < 1:
        raise SystemExit("--n-boot must be positive")
    if not 0 < args.ci < 1:
        raise SystemExit("--ci must be in (0, 1)")

    def selected(path: Path) -> list[dict]:
        rows = load_json(path)
        if args.split.lower() == "all":
            return rows
        return [row for row in rows if row.get("split") == args.split]

    baseline = selected(args.baseline)
    if not baseline:
        raise SystemExit(f"No records in split={args.split!r}; use --split all for legacy artifacts.")
    report = {
        "version": "statistical_analysis_v1",
        "inputs": {
            "baseline": str(args.baseline),
            "gnosis": str(args.gnosis) if args.gnosis else None,
            "random": [str(path) for path in args.random],
        },
        "bootstrap": {"n_boot": args.n_boot, "seed": args.seed, "ci_level": args.ci},
        "split": args.split,
        "baseline": _statistics_for_records(baseline, args.n_boot, args.seed, args.ci),
        "intervention_arms": {},
        "paired_arm_comparisons": {},
    }
    if args.gnosis:
        gnosis = selected(args.gnosis)
        report["intervention_arms"]["gnosis"] = {
            "revision": paired_revision_test(gnosis, n_boot=args.n_boot, seed=args.seed, ci=args.ci),
            "cost": compute_cost_summary(gnosis),
        }
    else:
        gnosis = None
    for offset, path in enumerate(args.random, start=1):
        arm_name = f"random_{offset}"
        random_arm = selected(path)
        report["intervention_arms"][arm_name] = {
            "revision": paired_revision_test(
                random_arm, n_boot=args.n_boot, seed=args.seed + offset, ci=args.ci
            ),
            "cost": compute_cost_summary(random_arm),
        }
        if gnosis is not None:
            report["paired_arm_comparisons"][f"gnosis_vs_{arm_name}"] = paired_outcome_test(
                gnosis, random_arm, n_boot=args.n_boot, seed=args.seed + 100 + offset, ci=args.ci
            )

    save_json_atomic(args.output, report)
    print(
        f"Saved {args.output}; baseline n={report['baseline']['n']} "
        f"wrong={report['baseline']['n_wrong']}"
    )


if __name__ == "__main__":
    main()

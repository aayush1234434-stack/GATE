#!/usr/bin/env python3
"""Evaluate confidence baselines with calibration, validation, and test isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from confidence_baselines import (
    BASELINES,
    available_feature_fields,
    baseline_risks,
    budget_metrics,
    calibrated_risk,
    evaluate_ranking,
    fit_logistic_calibrator,
    quantile_thresholds,
    select_threshold,
    threshold_metrics,
)
from experiment_splits import attach_splits, records_for_split
from experiment_state import load_json, save_json_atomic


def report_for_baseline(
    records: list[dict], name: str, budgets: list[float], thresholds: list[float] | None = None
) -> dict:
    labels, risks = baseline_risks(records, name)
    return {
        "ranking": evaluate_ranking(labels, risks),
        "budget_metrics": [budget_metrics(labels, risks, budget) for budget in budgets] if labels else [],
        "threshold_sweep": [threshold_metrics(labels, risks, threshold) for threshold in thresholds or []],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase3_research.json")
    args = parser.parse_args()

    config = load_json(args.config)
    records = attach_splits(load_json(args.records), load_json(args.splits))
    calibration = records_for_split(records, "calibration")
    validation = records_for_split(records, "validation")
    test = records_for_split(records, "test")
    budgets = config["evaluation"]["intervention_budgets"]

    report = {
        "version": "baseline_evaluation_v1",
        "records": str(args.records),
        "splits": str(args.splits),
        "config": str(args.config),
        "counts": {"calibration": len(calibration), "validation": len(validation), "test": len(test)},
        "baselines": {},
    }
    for name in BASELINES:
        if baseline_risks(calibration, name)[0] and baseline_risks(test, name)[0]:
            validation_labels, validation_risks = baseline_risks(validation, name)
            thresholds = quantile_thresholds(validation_risks)
            selected_threshold = select_threshold(validation_labels, validation_risks, thresholds)
            test_labels, test_risks = baseline_risks(test, name)
            report["baselines"][name] = {
                "calibration": report_for_baseline(calibration, name, budgets),
                "validation": report_for_baseline(validation, name, budgets, thresholds),
                "test": report_for_baseline(test, name, budgets, thresholds),
                "selected_threshold_from_validation": selected_threshold,
                "test_at_selected_threshold": (
                    threshold_metrics(test_labels, test_risks, selected_threshold["threshold"])
                    if selected_threshold else None
                ),
            }

    preferred = config["evaluation"].get("calibrator_features", [])
    available = [field for field in preferred if field in available_feature_fields(calibration)]
    split_health = {
        name: {
            "n": len(rows),
            "n_wrong": sum(not row["baseline_correct"] for row in rows),
            "has_both_classes": 0 < sum(not row["baseline_correct"] for row in rows) < len(rows),
        }
        for name, rows in {"calibration": calibration, "validation": validation, "test": test}.items()
    }
    report["split_health"] = split_health
    if available and all(values["has_both_classes"] for values in split_health.values()):
        model = fit_logistic_calibrator(calibration, available)
        for record in records:
            record["learned_calibrator_risk"] = calibrated_risk(record, model)
        report["learned_calibrator"] = {
            "model": model,
            "validation": _report_learned(validation, budgets),
            "test": _report_learned(test, budgets),
        }
    else:
        report["learned_calibrator"] = {
            "skipped": "Calibration, validation, and test each need both classes and complete configured features."
        }

    save_json_atomic(args.output, report)
    print(json.dumps(report["counts"], indent=2))
    print(f"Saved: {args.output}")


def _report_learned(records: list[dict], budgets: list[float]) -> dict:
    pairs = [
        (0 if record["baseline_correct"] else 1, record["learned_calibrator_risk"])
        for record in records
        if record.get("learned_calibrator_risk") is not None
    ]
    labels = [label for label, _ in pairs]
    risks = [risk for _, risk in pairs]
    return {
        "ranking": evaluate_ranking(labels, risks),
        "budget_metrics": [budget_metrics(labels, risks, budget) for budget in budgets] if labels else [],
    }


if __name__ == "__main__":
    main()

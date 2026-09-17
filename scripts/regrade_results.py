#!/usr/bin/env python3
"""Regrade saved baseline or intervention records with the current evaluator.

This writes a new file rather than modifying an existing result artifact. It is
intended for migrating legacy pilot outputs that used the former substring
grader, while preserving each changed legacy label for auditability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval_utils import build_question_lookup, enrich_records, grade_record
from experiment_state import load_json, save_json_atomic

GRADING_PROTOCOL = "conservative_match_v1"


def regrade_record(record: dict, lookup: dict | None) -> tuple[dict, bool]:
    updated = dict(record)
    changed = False

    baseline_correct = grade_record(updated, updated["baseline_answer"], lookup)
    if updated.get("baseline_correct") != baseline_correct:
        updated["legacy_baseline_correct"] = updated.get("baseline_correct")
        changed = True
    updated["baseline_correct"] = baseline_correct

    if updated.get("intervened") and updated.get("regen_answer") is not None:
        regen_correct = grade_record(updated, updated["regen_answer"], lookup)
        if updated.get("regen_correct") != regen_correct:
            updated["legacy_regen_correct"] = updated.get("regen_correct")
            changed = True
        updated["regen_correct"] = regen_correct
        updated["final_answer"] = updated["regen_answer"]
        updated["final_correct"] = regen_correct
    else:
        updated["final_answer"] = updated["baseline_answer"]
        updated["final_correct"] = baseline_correct

    updated["grading_protocol"] = GRADING_PROTOCOL
    return updated, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Optional question-set JSON used to attach official answer aliases.",
    )
    args = parser.parse_args()

    records = load_json(args.input)
    questions = load_json(args.questions) if args.questions else None
    records = enrich_records(records, questions)
    lookup = build_question_lookup(questions) if questions else None

    regraded, changed = [], 0
    for record in records:
        updated, did_change = regrade_record(record, lookup)
        regraded.append(updated)
        changed += did_change

    save_json_atomic(args.output, regraded)
    correct = sum(record["baseline_correct"] for record in regraded)
    print(
        f"Saved {len(regraded)} records to {args.output}; "
        f"changed labels: {changed}; baseline correct: {correct}/{len(regraded)}"
    )


if __name__ == "__main__":
    main()

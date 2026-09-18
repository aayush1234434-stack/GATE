#!/usr/bin/env python3
"""Audit whether an evaluation run contains enough observed model errors.

The target is a design requirement, not a stopping rule chosen after looking at
detector quality. If the run misses it, enlarge the *predeclared* benchmark
configuration or evaluate a harder held-out suite, then rebuild splits before
any threshold tuning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from experiment_state import load_json, save_json_atomic
from statistical_analysis import per_group_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-wrong", type=int, default=150)
    parser.add_argument("--minimum-wrong-per-domain", type=int, default=20)
    args = parser.parse_args()
    records = load_json(args.records)
    by_domain = per_group_counts(records, "domain")
    total_wrong = sum(group["n_wrong"] for group in by_domain.values())
    failing_domains = {
        domain: group["n_wrong"] for domain, group in by_domain.items()
        if group["n_wrong"] < args.minimum_wrong_per_domain
    }
    report = {
        "records": str(args.records), "n": len(records), "n_wrong": total_wrong,
        "minimum_wrong": args.minimum_wrong,
        "minimum_wrong_per_domain": args.minimum_wrong_per_domain,
        "by_domain": by_domain,
        "meets_total_target": total_wrong >= args.minimum_wrong,
        "domains_below_target": failing_domains,
        "ready_for_primary_analysis": total_wrong >= args.minimum_wrong and not failing_domains,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        save_json_atomic(args.output, report)
    if not report["ready_for_primary_analysis"]:
        raise SystemExit("Error target not met; do not present this as the primary powered analysis.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a deterministic calibration/validation/test split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_splits import make_split_manifest
from experiment_state import load_json, save_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Question-set or baseline JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase3_research.json")
    args = parser.parse_args()

    config = load_json(args.config)
    split_config = config["splits"]
    manifest = make_split_manifest(
        load_json(args.input),
        seed=int(split_config["seed"]),
        ratios=split_config["ratios"],
        stratum_key=split_config.get("stratum_key", "domain"),
    )
    manifest["source"] = str(args.input)
    manifest["config"] = str(args.config)
    save_json_atomic(args.output, manifest)
    print(json.dumps({"output": str(args.output), "n_records": manifest["n_records"], "strata": manifest["stratum_counts"]}, indent=2))


if __name__ == "__main__":
    main()

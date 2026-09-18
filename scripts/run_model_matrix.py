#!/usr/bin/env python3
"""Run each declared, Gnosis-compatible model into isolated artifact paths.

By default this prints commands only. Pass ``--run`` on a CUDA machine to run
them sequentially. Do not add a checkpoint to the config until its correctness
head has been verified; arbitrary base models cannot produce a Gnosis score.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def command_for(model: dict, config: Path, questions: Path, splits: Path, output_dir: Path) -> tuple[dict, str]:
    env = os.environ.copy()
    env.update({
        "CONFIG_PATH": str(config), "MODEL_NAME": model["name"],
        "QUESTIONS_PATH": str(questions), "SPLITS_PATH": str(splits),
        "BASELINE_PATH": str(output_dir / f"baseline_{model['name']}.json"),
    })
    command = [sys.executable, "phase_3/run_baseline.py"]
    rendered = " ".join(f"{key}={shlex.quote(env[key])}" for key in ("CONFIG_PATH", "MODEL_NAME", "QUESTIONS_PATH", "SPLITS_PATH", "BASELINE_PATH")) + " " + " ".join(command)
    return env, rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase3_research.json")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Execute commands sequentially instead of printing them.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    models = [model for model in config.get("models", []) if model.get("supports_gnosis_correctness_head")]
    if not models:
        raise SystemExit("No Gnosis-compatible models declared in the config.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        env, rendered = command_for(model, args.config, args.questions, args.splits, args.output_dir)
        print(rendered)
        if args.run:
            subprocess.run([sys.executable, "phase_3/run_baseline.py"], cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

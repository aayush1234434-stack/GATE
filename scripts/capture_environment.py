#!/usr/bin/env python3
"""Capture package, platform, and Git provenance for an experiment artifact."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = ("torch", "transformers", "datasets", "pytest")
EXPERIMENT_ENV = (
    "DEVICE",
    "SEED",
    "INTERVENTION_PROTOCOL",
    "GNOSIS_MODEL_ID",
    "THRESHOLD",
    "TEMPERATURE",
    "TOP_P",
    "MAX_NEW_TOKENS",
)


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(args, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "experiment_environment": {key: os.environ.get(key) for key in EXPERIMENT_ENV},
        "git_commit": git_value(["git", "rev-parse", "HEAD"]),
        "git_submodules": git_value(["git", "submodule", "status", "--recursive"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved environment metadata: {args.output}")


if __name__ == "__main__":
    main()

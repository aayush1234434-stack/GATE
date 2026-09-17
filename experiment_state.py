"""Safe, resumable JSON state for intervention experiments."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path


def load_json(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json_atomic(path: str | Path, payload) -> None:
    """Atomically save JSON so interrupted Colab sessions preserve checkpoints."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def reset_intervention(record: dict) -> None:
    """Restore a record to its baseline answer without changing baseline fields."""
    record["intervened"] = False
    record["final_answer"] = record["baseline_answer"]
    record["final_correct"] = record["baseline_correct"]
    record["final_gnosis_score"] = record["gnosis_score"]
    for key in (
        "regen_answer",
        "regen_correct",
        "regen_gnosis_score",
        "selection",
        "intervention_protocol",
        "intervention_seed",
    ):
        record.pop(key, None)


def _same_baseline(left: dict, right: dict) -> bool:
    """Use stable identity fields to ensure a checkpoint fits the current run."""
    return (
        left.get("id") == right.get("id")
        and left.get("question") == right.get("question")
        and left.get("baseline_answer") == right.get("baseline_answer")
        and left.get("baseline_correct") == right.get("baseline_correct")
        and left.get("gnosis_score") == right.get("gnosis_score")
    )


def prepare_intervention_results(
    baseline: list[dict],
    pick_indices: list[int],
    output_path: str | Path,
    selection: str,
    protocol: str,
    seed: int,
) -> list[dict]:
    """Load a compatible checkpoint or initialize one, preserving completed picks."""
    target = Path(output_path)
    picks = set(pick_indices)
    if target.exists():
        results = load_json(target)
        if len(results) != len(baseline) or any(
            not _same_baseline(saved, source)
            for saved, source in zip(results, baseline)
        ):
            raise ValueError(
                f"Checkpoint {target} does not match the current baseline. "
                "Use a new output path rather than mixing experiments."
            )
    else:
        results = copy.deepcopy(baseline)

    for index, record in enumerate(results):
        complete = (
            index in picks
            and record.get("intervened")
            and record.get("selection") == selection
            and record.get("intervention_protocol") == protocol
            and record.get("intervention_seed") == seed
            and "regen_answer" in record
        )
        if not complete:
            reset_intervention(record)
    return results

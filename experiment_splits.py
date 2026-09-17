"""Deterministic, stratified calibration/validation/test splits."""

from __future__ import annotations

import random
from collections import defaultdict


DEFAULT_RATIOS = {"calibration": 0.20, "validation": 0.20, "test": 0.60}


def validate_ratios(ratios: dict[str, float]) -> None:
    if set(ratios) != set(DEFAULT_RATIOS):
        raise ValueError(f"Split names must be {sorted(DEFAULT_RATIOS)}")
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("Every split ratio must be positive")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")


def _allocation(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation that exactly preserves a stratum size."""
    raw = {name: n * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = n - sum(counts.values())
    order = sorted(ratios, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def make_split_manifest(
    records: list[dict],
    *,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    stratum_key: str = "domain",
) -> dict:
    """Assign every unique record id to one split, stratified by ``stratum_key``."""
    ratios = ratios or DEFAULT_RATIOS
    validate_ratios(ratios)
    groups: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for record in records:
        identifier = record.get("id")
        if identifier is None:
            raise ValueError("Every record needs a stable id before split assignment")
        if identifier in seen:
            raise ValueError(f"Duplicate id in split input: {identifier!r}")
        seen.add(identifier)
        groups[str(record.get(stratum_key, "unknown"))].append(record)

    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    stratum_counts = {}
    for stratum, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: str(row["id"]))
        rng.shuffle(rows)
        counts = _allocation(len(rows), ratios)
        stratum_counts[stratum] = counts
        cursor = 0
        for split in ("calibration", "validation", "test"):
            for row in rows[cursor : cursor + counts[split]]:
                assignments[str(row["id"])] = split
            cursor += counts[split]

    return {
        "version": "split_manifest_v1",
        "seed": seed,
        "stratum_key": stratum_key,
        "ratios": ratios,
        "n_records": len(records),
        "stratum_counts": stratum_counts,
        "assignments": assignments,
    }


def attach_splits(records: list[dict], manifest: dict) -> list[dict]:
    """Return copied records annotated with their manifest-defined split."""
    assignments = manifest["assignments"]
    out = []
    for record in records:
        split = assignments.get(str(record.get("id")))
        if split is None:
            raise ValueError(f"Record id {record.get('id')!r} is absent from split manifest")
        item = dict(record)
        item["split"] = split
        out.append(item)
    return out


def records_for_split(records: list[dict], split: str) -> list[dict]:
    return [record for record in records if record.get("split") == split]

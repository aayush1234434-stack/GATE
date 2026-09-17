from experiment_splits import attach_splits, make_split_manifest


def records():
    return [
        {"id": f"trivia-{index}", "domain": "trivia"} for index in range(10)
    ] + [
        {"id": f"math-{index}", "domain": "math"} for index in range(10)
    ]


def test_splits_are_deterministic_disjoint_and_stratified():
    first = make_split_manifest(records(), seed=7)
    second = make_split_manifest(records(), seed=7)
    assert first["assignments"] == second["assignments"]
    assert set(first["assignments"].values()) == {"calibration", "validation", "test"}
    assert first["stratum_counts"]["math"] == {"calibration": 2, "validation": 2, "test": 6}
    assert len(first["assignments"]) == 20


def test_attach_splits_preserves_records_and_adds_assignment():
    manifest = make_split_manifest(records())
    assigned = attach_splits(records(), manifest)
    assert all(row["split"] in {"calibration", "validation", "test"} for row in assigned)

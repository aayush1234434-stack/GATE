from experiment_state import prepare_intervention_results, save_json_atomic


def baseline_record(identifier: int) -> dict:
    return {
        "id": identifier,
        "question": f"Question {identifier}",
        "baseline_answer": f"Answer {identifier}",
        "baseline_correct": True,
        "gnosis_score": 0.9,
        "intervened": False,
        "final_answer": f"Answer {identifier}",
        "final_correct": True,
        "final_gnosis_score": 0.9,
    }


def test_resume_preserves_only_compatible_completed_picks(tmp_path):
    baseline = [baseline_record(1), baseline_record(2)]
    output = tmp_path / "random.json"
    first = prepare_intervention_results(
        baseline, [0], output, selection="random", protocol="critique_revision_v1", seed=42
    )
    first[0].update(
        {
            "intervened": True,
            "selection": "random",
            "intervention_protocol": "critique_revision_v1",
            "intervention_seed": 42,
            "regen_answer": "Revised",
            "final_answer": "Revised",
        }
    )
    save_json_atomic(output, first)

    resumed = prepare_intervention_results(
        baseline, [0], output, selection="random", protocol="critique_revision_v1", seed=42
    )
    assert resumed[0]["regen_answer"] == "Revised"

    changed_protocol = prepare_intervention_results(
        baseline, [0], output, selection="random", protocol="independent_resample_v1", seed=42
    )
    assert not changed_protocol[0]["intervened"]
    assert "regen_answer" not in changed_protocol[0]

    changed_seed = prepare_intervention_results(
        baseline, [0], output, selection="random", protocol="critique_revision_v1", seed=43
    )
    assert not changed_seed[0]["intervened"]
    assert "intervention_seed" not in changed_seed[0]


def test_resume_rejects_a_checkpoint_from_a_different_baseline(tmp_path):
    output = tmp_path / "random.json"
    baseline = [baseline_record(1)]
    save_json_atomic(output, baseline)
    changed = [baseline_record(1)]
    changed[0]["baseline_answer"] = "Different answer"

    import pytest

    with pytest.raises(ValueError, match="does not match"):
        prepare_intervention_results(
            changed, [0], output, selection="random", protocol="critique_revision_v1", seed=42
        )

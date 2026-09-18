from statistical_analysis import (
    calibration_summary,
    compute_cost_summary,
    paired_outcome_test,
    paired_revision_test,
    risk_coverage_curve,
)


def test_risk_coverage_trusts_lowest_risk_answers_first():
    labels = [0, 1, 0, 1]
    risks = [0.1, 0.9, 0.2, 0.8]
    curve = risk_coverage_curve(labels, risks, points=3)
    assert curve[0]["residual_risk"] is None
    assert curve[1]["trusted"] == 2
    assert curve[1]["residual_risk"] == 0.0
    assert curve[-1]["residual_risk"] == 0.5


def test_calibration_has_brier_ece_and_nonempty_reliability_bins():
    summary = calibration_summary([0, 1], [0.1, 0.9], bins=2)
    assert abs(summary["brier"] - 0.01) < 1e-12
    assert abs(summary["ece"] - 0.1) < 1e-12
    assert len(summary["bins"]) == 2


def test_paired_tests_keep_question_identity_and_detect_improvement():
    reference = [
        {"id": "a", "baseline_correct": False, "final_correct": True, "intervened": True},
        {"id": "b", "baseline_correct": True, "final_correct": True, "intervened": False},
    ]
    comparison = [
        {"id": "a", "baseline_correct": False, "final_correct": False, "intervened": True},
        {"id": "b", "baseline_correct": True, "final_correct": True, "intervened": False},
    ]
    paired = paired_outcome_test(reference, comparison, n_boot=20, seed=1, ci=0.95)
    revision = paired_revision_test(reference, n_boot=20, seed=1, ci=0.95)
    assert paired["paired_n"] == 2
    assert paired["delta_final_accuracy_reference_minus_comparison"]["point"] == 0.5
    assert revision["improved_wrong_to_correct"] == 1
    assert revision["harmed_correct_to_wrong"] == 0


def test_compute_cost_uses_observed_intervention_metrics():
    records = [
        {
            "generation_samples": 2,
            "sample_generation_metrics": [
                {"generated_tokens": 10, "generation_seconds": 1.0},
                {"generated_tokens": 12, "generation_seconds": 1.2},
            ],
            "intervened": True,
            "regen_generation_metrics": {"generated_tokens": 8, "generation_seconds": 0.8},
        },
        {"generation_samples": 1, "generated_tokens": 11, "generation_seconds": 1.1, "intervened": False},
    ]
    cost = compute_cost_summary(records)
    assert cost["baseline_generation_calls"] == 3
    assert cost["baseline_generated_tokens"] == 33
    assert cost["counterfactual_all_regenerate_generated_tokens"] == 16
    assert cost["intervention_token_savings_vs_all_regenerate"] == 0.5

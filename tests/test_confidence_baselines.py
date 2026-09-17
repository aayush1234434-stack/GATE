from confidence_baselines import (
    baseline_risks,
    budget_metrics,
    fit_logistic_calibrator,
    quantile_thresholds,
    roc_auc,
    select_threshold,
    threshold_metrics,
)


RECORDS = [
    {"baseline_correct": True, "gnosis_score": 0.9, "mean_token_entropy": 0.1},
    {"baseline_correct": True, "gnosis_score": 0.8, "mean_token_entropy": 0.2},
    {"baseline_correct": False, "gnosis_score": 0.2, "mean_token_entropy": 0.8},
    {"baseline_correct": False, "gnosis_score": 0.1, "mean_token_entropy": 0.9},
]


def test_gnosis_risk_ranks_wrong_records_above_correct_records():
    labels, risks = baseline_risks(RECORDS, "gnosis")
    assert roc_auc(labels, risks) == 1.0


def test_budget_and_threshold_metrics_are_explicit():
    labels, risks = baseline_risks(RECORDS, "gnosis")
    assert budget_metrics(labels, risks, 0.5)["recall"] == 1.0
    thresholds = quantile_thresholds(risks, count=3)
    assert len(thresholds) == 3
    assert threshold_metrics(labels, risks, thresholds[-1])["selected"] == 1
    selected = select_threshold(labels, risks, thresholds)
    assert selected is not None
    assert selected["f1"] == 1.0


def test_logistic_calibrator_trains_on_declared_features_only():
    model = fit_logistic_calibrator(RECORDS, ["gnosis_score", "mean_token_entropy"], steps=100)
    assert model["features"] == ["gnosis_score", "mean_token_entropy"]
    assert len(model["weights"]) == 2

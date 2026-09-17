import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase_2"))

from auroc_analysis import _roc_auc_manual, roc_auc_score


def test_auroc_uses_average_ranks_for_ties():
    assert _roc_auc_manual([1, 0], [0.5, 0.5]) == 0.5


def test_auroc_orders_positive_examples_above_negatives():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc_score(labels, scores) == 1.0


def test_auroc_is_independent_of_optional_sklearn_installation():
    labels = [0, 1, 0, 1]
    scores = [0.1, 0.5, 0.5, 0.9]
    assert roc_auc_score(labels, scores) == _roc_auc_manual(labels, scores)

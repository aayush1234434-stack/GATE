"""Feature baselines, calibration, and held-out evaluation for GATE."""

from __future__ import annotations

import math


BASELINES = {
    "gnosis": {"field": "gnosis_score", "risk": lambda value: 1.0 - value},
    "token_logprob": {"field": "mean_token_logprob", "risk": lambda value: -value},
    "token_entropy": {"field": "mean_token_entropy", "risk": lambda value: value},
    "self_consistency": {"field": "self_consistency", "risk": lambda value: 1.0 - value},
    "semantic_agreement": {"field": "semantic_agreement", "risk": lambda value: 1.0 - value},
    "external_verifier": {"field": "external_verifier_score", "risk": lambda value: 1.0 - value},
}


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def baseline_risks(records: list[dict], baseline: str) -> tuple[list[int], list[float]]:
    spec = BASELINES[baseline]
    labels, risks = [], []
    for record in records:
        value = record.get(spec["field"])
        if _finite(value) and "baseline_correct" in record:
            labels.append(0 if record["baseline_correct"] else 1)
            risks.append(float(spec["risk"](float(value))))
    return labels, risks


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum, index = 0.0, 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        mean_rank = ((index + 1) + end) / 2
        rank_sum += mean_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    seen, hits, total = 0, 0, 0.0
    for _, label in pairs:
        seen += 1
        if label:
            hits += 1
            total += hits / seen
    return total / positives


def expected_calibration_error(labels: list[int], risks: list[float], bins: int = 10) -> float | None:
    if not labels or any(score < 0 or score > 1 for score in risks):
        return None
    total, n = 0.0, len(labels)
    for bucket in range(bins):
        lower, upper = bucket / bins, (bucket + 1) / bins
        indices = [
            i for i, score in enumerate(risks)
            if lower <= score < upper or (bucket == bins - 1 and score == upper)
        ]
        if indices:
            observed = sum(labels[i] for i in indices) / len(indices)
            confidence = sum(risks[i] for i in indices) / len(indices)
            total += len(indices) / n * abs(observed - confidence)
    return total


def evaluate_ranking(labels: list[int], risks: list[float]) -> dict:
    return {
        "n": len(labels),
        "n_wrong": sum(labels),
        "auroc": roc_auc(labels, risks),
        "average_precision": average_precision(labels, risks),
        "ece_risk": expected_calibration_error(labels, risks),
    }


def budget_metrics(labels: list[int], risks: list[float], budget: float) -> dict:
    if not 0 < budget <= 1:
        raise ValueError("budget must be in (0, 1]")
    n = len(labels)
    count = max(1, math.ceil(n * budget)) if n else 0
    selected = sorted(range(n), key=lambda index: risks[index], reverse=True)[:count]
    caught = sum(labels[index] for index in selected)
    total_wrong = sum(labels)
    return {
        "budget": budget,
        "selected": count,
        "caught": caught,
        "precision": caught / count if count else None,
        "recall": caught / total_wrong if total_wrong else None,
        "threshold": risks[selected[-1]] if selected else None,
    }


def quantile_thresholds(risks: list[float], count: int = 10) -> list[float]:
    """Return a stable multi-threshold grid derived from validation risks only."""
    if not risks:
        return []
    ordered = sorted(risks)
    indices = {round(index * (len(ordered) - 1) / max(count - 1, 1)) for index in range(count)}
    return sorted({ordered[index] for index in indices})


def threshold_metrics(labels: list[int], risks: list[float], threshold: float) -> dict:
    selected = [index for index, risk in enumerate(risks) if risk >= threshold]
    caught = sum(labels[index] for index in selected)
    total_wrong = sum(labels)
    return {
        "threshold": threshold,
        "selected": len(selected),
        "caught": caught,
        "precision": caught / len(selected) if selected else None,
        "recall": caught / total_wrong if total_wrong else None,
    }


def select_threshold(labels: list[int], risks: list[float], thresholds: list[float]) -> dict | None:
    """Select the validation threshold with the best F1; ties prefer lower coverage."""
    if not thresholds or not (0 < sum(labels) < len(labels)):
        return None
    candidates = []
    for threshold in thresholds:
        metrics = threshold_metrics(labels, risks, threshold)
        precision, recall = metrics["precision"], metrics["recall"]
        metrics["f1"] = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
        candidates.append(metrics)
    return max(candidates, key=lambda metrics: (metrics["f1"], -metrics["selected"]))


def available_feature_fields(records: list[dict]) -> list[str]:
    return [
        spec["field"]
        for spec in BASELINES.values()
        if all(_finite(record.get(spec["field"])) for record in records)
    ]


def fit_logistic_calibrator(records: list[dict], feature_fields: list[str], steps: int = 800, lr: float = 0.1) -> dict:
    """Fit a small dependency-free logistic calibrator on calibration data only."""
    rows = [
        record for record in records
        if "baseline_correct" in record and all(_finite(record.get(field)) for field in feature_fields)
    ]
    if not rows or not feature_fields:
        raise ValueError("Calibration requires labels and at least one complete feature")
    x = [[float(row[field]) for field in feature_fields] for row in rows]
    y = [0.0 if row["baseline_correct"] else 1.0 for row in rows]
    means = [sum(row[i] for row in x) / len(x) for i in range(len(feature_fields))]
    scales = [
        max(math.sqrt(sum((row[i] - means[i]) ** 2 for row in x) / len(x)), 1e-8)
        for i in range(len(feature_fields))
    ]
    x = [[(value - means[i]) / scales[i] for i, value in enumerate(row)] for row in x]
    weights = [0.0] * len(feature_fields)
    prevalence = min(max(sum(y) / len(y), 1e-6), 1 - 1e-6)
    bias = math.log(prevalence / (1 - prevalence))
    for _ in range(steps):
        gradients = [0.0] * len(weights)
        bias_gradient = 0.0
        for row, label in zip(x, y):
            prediction = 1 / (1 + math.exp(-max(min(bias + sum(w * v for w, v in zip(weights, row)), 30), -30)))
            error = prediction - label
            bias_gradient += error
            for index, value in enumerate(row):
                gradients[index] += error * value
        for index in range(len(weights)):
            weights[index] -= lr * gradients[index] / len(x)
        bias -= lr * bias_gradient / len(x)
    return {"features": feature_fields, "means": means, "scales": scales, "weights": weights, "bias": bias}


def calibrated_risk(record: dict, model: dict) -> float | None:
    try:
        values = [float(record[field]) for field in model["features"]]
    except (KeyError, TypeError, ValueError):
        return None
    normalized = [(value - mean) / scale for value, mean, scale in zip(values, model["means"], model["scales"])]
    logit = max(min(model["bias"] + sum(w * x for w, x in zip(model["weights"], normalized)), 30), -30)
    return 1 / (1 + math.exp(-logit))

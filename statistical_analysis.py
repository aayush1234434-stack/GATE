"""Statistical reporting primitives for selective-intervention experiments.

All functions use only the standard library so the paper-facing analysis can be
re-run without a particular SciPy or scikit-learn version.  Metrics treat
``1`` as an incorrect answer (risk event) and higher scores as higher risk.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from confidence_baselines import average_precision, expected_calibration_error, roc_auc


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def bootstrap_interval(
    values: list,
    statistic,
    *,
    n_boot: int = 2_000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Percentile bootstrap interval for a statistic over independently sampled rows."""
    if not values:
        return {"point": None, "ci_low": None, "ci_high": None, "n_boot_effective": 0}
    point = statistic(values)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            samples.append(float(value))
    alpha = (1.0 - ci) / 2.0
    return {
        "point": point,
        "ci_low": _quantile(samples, alpha),
        "ci_high": _quantile(samples, 1.0 - alpha),
        "ci_level": ci,
        "n_boot_effective": len(samples),
    }


def brier_score(labels: list[int], probabilities: list[float]) -> float | None:
    if not labels or len(labels) != len(probabilities):
        return None
    if any(probability < 0 or probability > 1 for probability in probabilities):
        return None
    return sum((label - probability) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)


def calibration_summary(labels: list[int], risks: list[float], bins: int = 10) -> dict:
    """Reliability table and scalar calibration metrics for probabilities of error."""
    table = []
    if not labels or len(labels) != len(risks) or any(risk < 0 or risk > 1 for risk in risks):
        return {"brier": None, "ece": None, "bins": table}
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        rows = [i for i, risk in enumerate(risks) if lo <= risk < hi or (index == bins - 1 and risk == hi)]
        if rows:
            table.append(
                {
                    "lower": lo,
                    "upper": hi,
                    "n": len(rows),
                    "mean_predicted_risk": sum(risks[i] for i in rows) / len(rows),
                    "observed_error_rate": sum(labels[i] for i in rows) / len(rows),
                }
            )
    return {"brier": brier_score(labels, risks), "ece": expected_calibration_error(labels, risks, bins), "bins": table}


def risk_coverage_curve(labels: list[int], risks: list[float], points: int = 21) -> list[dict]:
    """Residual error rate after trusting the lowest-risk fraction of answers.

    Coverage is the fraction answered without intervention.  At coverage 1,
    residual risk is ordinary baseline error; at low coverage the gate has
    deferred most answers.
    """
    if not labels or len(labels) != len(risks):
        return []
    order = sorted(range(len(labels)), key=lambda index: risks[index])
    curve = []
    for point in range(points):
        coverage = point / max(points - 1, 1)
        retained = int(round(coverage * len(order)))
        trusted = order[:retained]
        errors = sum(labels[index] for index in trusted)
        curve.append(
            {
                "coverage": coverage,
                "trusted": retained,
                "deferred": len(order) - retained,
                "trusted_errors": errors,
                "residual_risk": errors / retained if retained else None,
            }
        )
    return curve


def ranking_statistics(labels: list[int], risks: list[float], *, n_boot: int, seed: int, ci: float) -> dict:
    pairs = list(zip(labels, risks))

    def metric(metric_name):
        def calculate(sample):
            sample_labels = [label for label, _ in sample]
            sample_risks = [risk for _, risk in sample]
            if metric_name == "auroc":
                return roc_auc(sample_labels, sample_risks)
            if metric_name == "auprc":
                return average_precision(sample_labels, sample_risks)
            raise ValueError(metric_name)

        return calculate

    return {
        "n": len(labels),
        "n_wrong": sum(labels),
        "auroc": bootstrap_interval(pairs, metric("auroc"), n_boot=n_boot, seed=seed, ci=ci),
        "auprc": bootstrap_interval(pairs, metric("auprc"), n_boot=n_boot, seed=seed + 1, ci=ci),
        "calibration": calibration_summary(labels, risks),
        "risk_coverage_curve": risk_coverage_curve(labels, risks),
    }


def _binomial_two_sided_pvalue(discordant_positive: int, discordant_negative: int) -> float:
    """Exact two-sided sign/McNemar p-value under a fair Bernoulli null."""
    n = discordant_positive + discordant_negative
    if n == 0:
        return 1.0
    # Use log-gamma rather than ``comb / 2**n`` so large benchmark runs do not
    # overflow while converting an integer denominator to float.
    lower_tail = sum(
        math.exp(
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            - n * math.log(2)
        )
        for k in range(min(discordant_positive, discordant_negative) + 1)
    )
    return min(1.0, 2.0 * lower_tail)


def paired_outcome_test(reference: list[dict], comparison: list[dict], *, n_boot: int, seed: int, ci: float) -> dict:
    """Paired final-accuracy analysis for two arms sharing the same question IDs.

    The exact McNemar test compares final correctness.  A paired bootstrap
    interval estimates the difference in final accuracy (reference minus
    comparison).  Rows absent from either output are deliberately excluded.
    """
    right = {str(row.get("id")): row for row in reference if row.get("id") is not None}
    left = {str(row.get("id")): row for row in comparison if row.get("id") is not None}
    shared = sorted(set(right).intersection(left))
    pairs = [
        (int(bool(right[qid].get("final_correct"))), int(bool(left[qid].get("final_correct"))))
        for qid in shared
    ]
    reference_only = sum(1 for first, second in pairs if first == 1 and second == 0)
    comparison_only = sum(1 for first, second in pairs if first == 0 and second == 1)
    interval = bootstrap_interval(
        pairs,
        lambda sample: sum(first - second for first, second in sample) / len(sample) if sample else None,
        n_boot=n_boot,
        seed=seed,
        ci=ci,
    )
    return {
        "paired_n": len(pairs),
        "reference_final_accuracy": sum(first for first, _ in pairs) / len(pairs) if pairs else None,
        "comparison_final_accuracy": sum(second for _, second in pairs) / len(pairs) if pairs else None,
        "delta_final_accuracy_reference_minus_comparison": interval,
        "discordant_reference_only_correct": reference_only,
        "discordant_comparison_only_correct": comparison_only,
        "mcnemar_exact_two_sided_pvalue": _binomial_two_sided_pvalue(reference_only, comparison_only),
    }


def paired_revision_test(records: list[dict], *, n_boot: int, seed: int, ci: float) -> dict:
    """Paired before/after test restricted to examples actually intervened on."""
    selected = [record for record in records if record.get("intervened")]
    before_after = [
        (int(bool(record.get("baseline_correct"))), int(bool(record.get("final_correct"))))
        for record in selected
        if "baseline_correct" in record and "final_correct" in record
    ]
    improved = sum(1 for before, after in before_after if before == 0 and after == 1)
    harmed = sum(1 for before, after in before_after if before == 1 and after == 0)
    interval = bootstrap_interval(
        before_after,
        lambda sample: sum(after - before for before, after in sample) / len(sample) if sample else None,
        n_boot=n_boot,
        seed=seed,
        ci=ci,
    )
    return {
        "n_intervened_with_outcomes": len(before_after),
        "improved_wrong_to_correct": improved,
        "harmed_correct_to_wrong": harmed,
        "delta_accuracy_after_minus_before": interval,
        "mcnemar_exact_two_sided_pvalue": _binomial_two_sided_pvalue(improved, harmed),
    }


def compute_cost_summary(records: list[dict]) -> dict:
    """Summarize observed generation tokens/calls and the counterfactual all-revise cost."""
    baseline_calls = sum(max(1, int(record.get("generation_samples", 1))) for record in records)
    baseline_tokens = sum(
        sum(metric.get("generated_tokens", 0) for metric in record.get("sample_generation_metrics", []))
        if record.get("sample_generation_metrics")
        else int(record.get("generated_tokens", 0)) * max(1, int(record.get("generation_samples", 1)))
        for record in records
    )
    baseline_seconds = sum(
        sum(metric.get("generation_seconds", 0.0) for metric in record.get("sample_generation_metrics", []))
        if record.get("sample_generation_metrics")
        else float(record.get("generation_seconds", 0.0)) * max(1, int(record.get("generation_samples", 1)))
        for record in records
    )
    selected = [record for record in records if record.get("intervened")]
    intervention_tokens = sum(int(record.get("regen_generation_metrics", {}).get("generated_tokens", 0)) for record in selected)
    intervention_seconds = sum(float(record.get("regen_generation_metrics", {}).get("generation_seconds", 0.0)) for record in selected)
    mean_regen_tokens = intervention_tokens / len(selected) if selected else None
    mean_regen_seconds = intervention_seconds / len(selected) if selected else None
    all_regenerate_tokens = mean_regen_tokens * len(records) if mean_regen_tokens is not None else None
    all_regenerate_seconds = mean_regen_seconds * len(records) if mean_regen_seconds is not None else None
    return {
        "n_questions": len(records),
        "baseline_generation_calls": baseline_calls,
        "baseline_generated_tokens": baseline_tokens,
        "baseline_generation_seconds": baseline_seconds,
        "intervention_calls": len(selected),
        "intervention_rate": len(selected) / len(records) if records else None,
        "intervention_generated_tokens": intervention_tokens,
        "intervention_generation_seconds": intervention_seconds,
        "counterfactual_all_regenerate_generated_tokens": all_regenerate_tokens,
        "counterfactual_all_regenerate_generation_seconds": all_regenerate_seconds,
        "intervention_token_savings_vs_all_regenerate": (
            1 - intervention_tokens / all_regenerate_tokens if all_regenerate_tokens else None
        ),
        "intervention_time_savings_vs_all_regenerate": (
            1 - intervention_seconds / all_regenerate_seconds if all_regenerate_seconds else None
        ),
        "note": "Token and time savings exclude score-forward-pass cost; report hardware and model separately.",
    }


def per_group_counts(records: list[dict], key: str) -> dict:
    groups = defaultdict(lambda: {"n": 0, "n_wrong": 0})
    for record in records:
        name = str(record.get(key, "unknown"))
        groups[name]["n"] += 1
        groups[name]["n_wrong"] += int(not bool(record.get("baseline_correct")))
    return dict(sorted(groups.items()))

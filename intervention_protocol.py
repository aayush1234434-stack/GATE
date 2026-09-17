"""Explicit intervention protocols for selective-generation experiments."""

from __future__ import annotations


CRITIQUE_REVISION_V1 = "critique_revision_v1"
INDEPENDENT_RESAMPLE_V1 = "independent_resample_v1"
SUPPORTED_PROTOCOLS = {CRITIQUE_REVISION_V1, INDEPENDENT_RESAMPLE_V1}


def resolve_protocol(value: str | None) -> str:
    """Validate an intervention protocol, defaulting to genuine revision."""
    protocol = (value or CRITIQUE_REVISION_V1).strip().lower()
    if protocol not in SUPPORTED_PROTOCOLS:
        valid = ", ".join(sorted(SUPPORTED_PROTOCOLS))
        raise ValueError(f"Unknown intervention protocol {protocol!r}. Choose one of: {valid}")
    return protocol


def intervention_instruction(domain: str) -> str:
    """Return a domain-specific instruction for revising a prior answer."""
    if domain == "math":
        return (
            "Audit the previous answer, re-solve the problem, and correct it if needed. "
            "Put only the final answer within \\boxed{}."
        )
    if domain == "mmlu_pro":
        return (
            "Audit the previous answer against the choices and correct it if needed. "
            "Put only the choice letter within \\boxed{}."
        )
    return (
        "Audit the previous answer carefully and correct it if needed. "
        "Put only the final answer within \\boxed{}."
    )


def build_intervention_question(
    question: str,
    baseline_answer: str,
    protocol: str = CRITIQUE_REVISION_V1,
) -> str:
    """Build the user content used for a regeneration intervention.

    ``critique_revision_v1`` exposes the original answer so the model can
    actually review it. ``independent_resample_v1`` is retained only for
    reproducing legacy pilot experiments and should be reported as resampling.
    """
    protocol = resolve_protocol(protocol)
    if protocol == INDEPENDENT_RESAMPLE_V1:
        return question
    return (
        f"Original question:\n{question}\n\n"
        f"Previous answer to audit:\n{baseline_answer}\n\n"
        "Produce a revised answer."
    )

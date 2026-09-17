import pytest

from intervention_protocol import (
    CRITIQUE_REVISION_V1,
    INDEPENDENT_RESAMPLE_V1,
    build_intervention_question,
    resolve_protocol,
)


def test_revision_protocol_includes_the_baseline_answer():
    prompt = build_intervention_question(
        "What is 2 + 2?", "The answer is 5.", CRITIQUE_REVISION_V1
    )
    assert "What is 2 + 2?" in prompt
    assert "The answer is 5." in prompt
    assert "Previous answer to audit" in prompt


def test_legacy_resample_protocol_contains_only_the_question():
    assert build_intervention_question(
        "What is 2 + 2?", "The answer is 5.", INDEPENDENT_RESAMPLE_V1
    ) == "What is 2 + 2?"


def test_unknown_protocol_is_rejected():
    with pytest.raises(ValueError):
        resolve_protocol("rewrite_until_correct")

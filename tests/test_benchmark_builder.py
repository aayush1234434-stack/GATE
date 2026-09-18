import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "phase_3" / "build_question_set.py"
SPEC = importlib.util.spec_from_file_location("benchmark_builder", MODULE)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)

SOURCE = {"dataset": "unit", "split": "test", "license": "test"}


def test_math_normalizer_uses_final_boxed_answer():
    record = BUILDER.normalize_math(
        {"id": "m1", "problem": "What is 2 + 2?", "solution": "Work. \\boxed{4}"}, 0, SOURCE
    )
    assert record["ground_truth"] == "4"
    assert record["source_split"] == "test"


def test_multiple_choice_normalizers_preserve_letter_and_text_alias():
    arc = BUILDER.normalize_arc(
        {"id": "a1", "question": "Which?", "choices": {"label": ["A", "B"], "text": ["no", "yes"]}, "answerKey": "B"},
        0,
        SOURCE,
    )
    mmlu = BUILDER.normalize_mmlu(
        {"id": "k1", "question": "Which?", "choices": ["no", "yes"], "answer": 1}, 0, SOURCE
    )
    assert arc["ground_truth"] == "B" and arc["answer_aliases"] == ["yes"]
    assert mmlu["ground_truth"] == "B" and mmlu["answer_aliases"] == ["yes"]

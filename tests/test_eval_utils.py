from eval_utils import grade_record, is_correct


def test_boxed_final_answer_is_graded_exactly():
    assert is_correct("Reasoning... therefore \\boxed{42}.", "42")
    assert not is_correct("Reasoning... therefore \\boxed{41}.", "42")


def test_textual_answers_use_whole_token_matching_not_substrings():
    assert is_correct("The answer is Mercury.", "Mercury")
    assert not is_correct("The answer is a waterfall.", "water")
    assert not is_correct("I found 14 examples.", "4")


def test_aliases_are_supported():
    record = {
        "domain": "trivia",
        "ground_truth": "United States",
        "answer_aliases": ["USA", "US"],
    }
    assert grade_record(record, "The answer is USA.")


def test_nested_boxed_latex_is_extracted():
    assert is_correct("Final: \\boxed{\\frac{1}{2}}", "\\frac{1}{2}")


def test_math_normalization_accepts_common_latex_equivalents():
    assert is_correct("Final: \\boxed{\\sqrt{\\pi}/2}", "sqrt(pi)/2", domain="math")

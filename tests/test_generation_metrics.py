from generation_metrics import answer_signature, self_consistency


def test_self_consistency_uses_normalized_final_answers():
    answers = ["Result: \\boxed{Paris}", "\\boxed{Paris}", "\\boxed{London}"]
    assert answer_signature(answers[0]) == "paris"
    assert self_consistency(answers) == 2 / 3


def test_single_sample_has_no_self_consistency_signal():
    assert self_consistency(["\\boxed{Paris}"]) is None

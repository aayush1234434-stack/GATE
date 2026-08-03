"""Answer correctness checks shared across experiment scripts."""


def is_correct(answer, ground_truth, answer_aliases=None) -> bool:
    """True if ground_truth or any alias appears in the model answer (case-insensitive)."""
    answer_l = answer.strip().lower()
    candidates = [str(ground_truth).strip()]
    if answer_aliases:
        candidates.extend(str(a).strip() for a in answer_aliases)

    seen = set()
    for candidate in candidates:
        key = candidate.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        if key in answer_l:
            return True
    return False

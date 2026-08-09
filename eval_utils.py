"""Answer correctness checks shared across experiment scripts."""


def build_question_lookup(questions: list[dict]) -> dict:
    """Index ground_truth and answer_aliases by question id and question text."""
    by_id: dict = {}
    by_question: dict = {}
    for q in questions:
        meta = {
            "ground_truth": q.get("ground_truth"),
            "answer_aliases": q.get("answer_aliases"),
        }
        qid = q.get("id")
        if qid is not None:
            by_id[qid] = meta
        text = q.get("question")
        if text:
            by_question[text] = meta
    return {"by_id": by_id, "by_question": by_question}


def enrich_record(record: dict, lookup: dict | None) -> dict:
    """Attach answer_aliases from lookup when missing on a result record."""
    if lookup is None or record.get("answer_aliases") is not None:
        return record
    meta = lookup["by_id"].get(record.get("id")) or lookup["by_question"].get(
        record.get("question")
    )
    if meta and meta.get("answer_aliases"):
        record = dict(record)
        record["answer_aliases"] = meta["answer_aliases"]
    return record


def enrich_records(records: list[dict], questions: list[dict] | None = None) -> list[dict]:
    """Fill missing answer_aliases on baseline/regen records from the question set."""
    if not questions:
        return records
    lookup = build_question_lookup(questions)
    return [enrich_record(r, lookup) for r in records]


def grading_fields(record: dict, lookup: dict | None = None) -> tuple[str, list | None]:
    """Return (ground_truth, answer_aliases) for grading a record."""
    aliases = record.get("answer_aliases")
    if aliases is not None:
        return record["ground_truth"], aliases
    if lookup:
        meta = lookup["by_id"].get(record.get("id")) or lookup["by_question"].get(
            record.get("question")
        )
        if meta:
            return meta["ground_truth"], meta.get("answer_aliases")
    return record["ground_truth"], None


def grade_record(record: dict, answer: str, lookup: dict | None = None) -> bool:
    """Grade an answer against a result record, using aliases when available."""
    ground_truth, aliases = grading_fields(record, lookup)
    return is_correct(answer, ground_truth, aliases)


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

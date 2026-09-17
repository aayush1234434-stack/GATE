"""Conservative answer correctness checks shared across experiment scripts."""

from __future__ import annotations

import re
import unicodedata


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
    return is_correct(answer, ground_truth, aliases, record.get("domain"))


def _boxed_contents(text: str) -> list[str]:
    """Extract balanced LaTex ``\\boxed{...}`` contents, preserving the final answer."""
    values = []
    start = 0
    marker = "\\boxed{"
    while True:
        index = text.find(marker, start)
        if index < 0:
            return values
        depth = 1
        cursor = index + len(marker)
        content_start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            values.append(text[content_start : cursor - 1])
        start = cursor


def _normalize(value: object) -> str:
    """Normalize harmless formatting while retaining mathematical operators."""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("−", "-")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" $.,;:!?")


def _normalize_math(value: object) -> str:
    """Normalize common equivalent LaTex forms used by the bundled math set."""
    text = _normalize(value)
    # Apply repeatedly so simple nested fractions are handled from the inside out.
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)
        text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("\\pi", "pi").replace("{", "").replace("}", "")
    return re.sub(r"\s+", "", text)


def _whole_answer_match(answer: str, candidate: str) -> bool:
    """Match a candidate as a complete token sequence, never as a raw substring."""
    if answer == candidate:
        return True
    if not candidate:
        return False
    # Letter/digit boundaries prevent e.g. "art" matching "partial" or "4" matching "14".
    pattern = rf"(?<![0-9a-z]){re.escape(candidate)}(?![0-9a-z])"
    return re.search(pattern, answer) is not None


def is_correct(answer, ground_truth, answer_aliases=None, domain: str | None = None) -> bool:
    """Conservatively grade an answer against ground truth and official aliases.

    Prefer the model's final ``\\boxed{...}`` answer when present. Otherwise,
    accept an exact normalized answer or a whole-token textual mention. This
    avoids the old raw-substring rule while still supporting concise trivia
    answers embedded in a sentence. ``domain`` is reserved for future
    task-specific normalizers and keeps all callers on a stable API.
    """
    normalizer = _normalize_math if domain == "math" else _normalize
    candidates = [str(ground_truth).strip()]
    if answer_aliases:
        candidates.extend(str(a).strip() for a in answer_aliases)

    seen = set()
    for candidate in candidates:
        key = normalizer(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        boxed = [normalizer(value) for value in _boxed_contents(str(answer))]
        if any(key == value for value in boxed):
            return True
        if _whole_answer_match(normalizer(answer), key):
            return True
    return False

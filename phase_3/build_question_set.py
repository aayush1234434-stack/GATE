"""Build a fixed, diverse, held-out evaluation set from public benchmarks.

The builder samples deterministically from documented source splits and writes
both question-level provenance and a separate source manifest. It does *not*
claim to create errors: after generation, run ``scripts/check_error_target.py``
to verify that the evaluated model produced the planned 100--200+ errors.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _boxed_final(text: object) -> str:
    r"""Return the final balanced \boxed{} payload, if the source provides one."""
    value, marker, start, found = str(text or ""), "\\boxed{", 0, []
    while True:
        begin = value.find(marker, start)
        if begin < 0:
            break
        cursor, depth, content = begin + len(marker), 1, begin + len(marker)
        while cursor < len(value) and depth:
            depth += (value[cursor] == "{") - (value[cursor] == "}")
            cursor += 1
        if depth == 0:
            found.append(value[content : cursor - 1].strip())
        start = cursor
    return found[-1] if found else ""


def _sample(rows, count: int, rng: random.Random, label: str):
    if len(rows) < count:
        raise ValueError(f"{label}: requested {count} examples but source has only {len(rows)}")
    return [rows[index] for index in rng.sample(range(len(rows)), count)]


def _provenance(source: dict, source_id: object) -> dict:
    return {
        "source_dataset": source["dataset"], "source_config": source.get("config"),
        "source_split": source["split"], "source_id": str(source_id),
        "source_license": source.get("license"),
    }


def _with_choices(question: str, choices: list[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return f"{question.strip()}\n\nChoices:\n" + "\n".join(
        f"{labels[index]}. {choice}" for index, choice in enumerate(choices)
    )


def normalize_trivia(row: dict, index: int, source: dict) -> dict | None:
    answer = row.get("answer") or {}
    value = (answer.get("value") if isinstance(answer, dict) else answer) or ""
    question = str(row.get("question") or "").strip()
    aliases = answer.get("aliases", []) if isinstance(answer, dict) else []
    if not question or not str(value).strip():
        return None
    return {
        "id": f"trivia_{index:05d}", "domain": "trivia", "question": question,
        "ground_truth": str(value).strip(),
        "answer_aliases": [str(alias).strip() for alias in aliases if str(alias).strip()],
        **_provenance(source, row.get("question_id", row.get("id", index))),
    }


def normalize_math(row: dict, index: int, source: dict) -> dict | None:
    question = str(row.get("problem") or row.get("prompt") or row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip() or _boxed_final(row.get("solution"))
    if not question or not answer or len(answer) > 160:
        return None
    return {
        "id": f"math_{index:05d}", "domain": "math", "question": question,
        "ground_truth": answer, **_provenance(source, row.get("id", index)),
    }


def normalize_arc(row: dict, index: int, source: dict) -> dict | None:
    choices = row.get("choices") or {}
    labels, texts = choices.get("label", []), choices.get("text", [])
    answer, question = str(row.get("answerKey") or "").strip(), str(row.get("question") or "").strip()
    if answer not in labels or not texts or not question:
        return None
    answer_text = str(texts[labels.index(answer)]).strip()
    if not answer_text:
        return None
    return {
        "id": f"science_{index:05d}", "domain": "science", "question": _with_choices(question, list(texts)),
        "ground_truth": answer, "answer_aliases": [answer_text],
        **_provenance(source, row.get("id", index)),
    }


def normalize_mmlu(row: dict, index: int, source: dict) -> dict | None:
    choices = [str(choice).strip() for choice in row.get("choices", [])]
    answer = row.get("answer")
    if isinstance(answer, int) or str(answer).isdigit():
        answer_index = int(answer)
    else:
        answer_index = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(str(answer).strip().upper())
    question = str(row.get("question") or "").strip()
    if not question or not (0 <= answer_index < len(choices)):
        return None
    return {
        "id": f"knowledge_{index:05d}", "domain": "knowledge", "question": _with_choices(question, choices),
        "ground_truth": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[answer_index], "answer_aliases": [choices[answer_index]],
        "subject": row.get("subject"), **_provenance(source, row.get("id", index)),
    }


NORMALIZERS = {"trivia": normalize_trivia, "math": normalize_math, "science": normalize_arc, "knowledge": normalize_mmlu}


def load_source(source: dict):
    from datasets import load_dataset
    kwargs = {"split": source["split"]}
    if source.get("configs"):
        # The public MATH mirror stores each mathematical subject as a config.
        # Materialize the small benchmark rows with their category recorded so
        # sampling remains deterministic across the aggregate source.
        rows = []
        for config in source["configs"]:
            rows.extend(
                {**dict(row), "_gate_source_config": config}
                for row in load_dataset(source["dataset"], config, **kwargs)
            )
        return rows
    return load_dataset(source["dataset"], source["config"], **kwargs) if source.get("config") else load_dataset(source["dataset"], **kwargs)


def build_questions(config: dict) -> tuple[list[dict], list[dict]]:
    rng = random.Random(config["seed"])
    questions, manifest = [], []
    for source in config["sources"]:
        selected = _sample(load_source(source), source["count"], rng, source["name"])
        normalize = NORMALIZERS[source["domain"]]
        built = [
            normalize(
                dict(row),
                len(questions) + offset,
                {**source, "config": dict(row).get("_gate_source_config", source.get("config"))},
            )
            for offset, row in enumerate(selected)
        ]
        built = [row for row in built if row]
        if len(built) != source["count"]:
            raise ValueError(f"{source['name']}: sampled records failed normalization; inspect this dataset schema before running.")
        questions.extend(built)
        manifest.append({key: value for key, value in source.items() if key != "count"} | {"n": len(built)})
    rng.shuffle(questions)
    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")
    return questions, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "benchmark_v1.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts" / "questions_v1.json")
    parser.add_argument("--manifest-out", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    questions, manifest = build_questions(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(questions, indent=2) + "\n")
    manifest_path = args.manifest_out or args.output.with_name(args.output.stem + "_sources.json")
    manifest_path.write_text(json.dumps({"benchmark_version": config["version"], "seed": config["seed"], "sources": manifest}, indent=2) + "\n")
    by_domain = {}
    for question in questions:
        by_domain[question["domain"]] = by_domain.get(question["domain"], 0) + 1
    print(f"Saved {len(questions)} questions to {args.output}: {by_domain}")
    print(f"Saved source manifest to {manifest_path}")
    print("Next: generate baseline results, then run scripts/check_error_target.py.")


if __name__ == "__main__":
    main()

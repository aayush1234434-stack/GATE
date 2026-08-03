"""
Phase 3 — Baseline only (Pass 1): answer + Gnosis score for every question.

Designed for long Colab runs (~800 questions):
  - Saves after every question (atomic write)
  - Resumes automatically from partial checkpoint
  - Output format matches baseline_results.json from Phase 2

Usage (Colab):
  !pip uninstall -y transformers && pip install -e /content/GATE/Gnosis/transformers
  !PYTHONPATH=/content/GATE/Gnosis python phase_3/run_baseline.py

Env overrides:
  QUESTIONS_PATH=phase_3/artifacts/questions_700.json
  BASELINE_PATH=phase_3/artifacts/baseline_results.json
  RERUN_BASELINE=1          # ignore checkpoint, start fresh
  CHECKPOINT_EVERY=1          # save every N completed questions (default 1)
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
GNOSIS_DIR = REPO_ROOT / "Gnosis"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GNOSIS_DIR))

from eval_utils import is_correct
from src.demo import build_chat_prompt, correctness_prob, generate_with_hf, has_correctness_head

PHASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = PHASE_DIR / "artifacts"

GNOSIS_MODEL_ID = os.environ.get("GNOSIS_MODEL_ID", "AmirhoseinGH/Gnosis-Qwen3-1.7B-Hybrid")
QUESTIONS_PATH = Path(os.environ.get("QUESTIONS_PATH", ARTIFACTS / "questions_700.json"))
BASELINE_PATH = Path(os.environ.get("BASELINE_PATH", ARTIFACTS / "baseline_results.json"))
CHECKPOINT_EVERY = max(1, int(os.environ.get("CHECKPOINT_EVERY", "1")))
THRESHOLD = float(os.environ.get("THRESHOLD", "0.85"))

SYSTEM_PROMPTS = {
    "math": "Please reason step by step, and put your final answer within \\boxed{}.",
    "trivia": "This is a trivia question. Put your final answer within \\boxed{}.",
    "mmlu_pro": (
        "You are solving multiple-choice questions. Please reason step by step, "
        "and put your final answer with only the choice letter within \\boxed{}."
    ),
}


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json_atomic(path: Path, obj) -> None:
    """Write JSON atomically so a crash mid-write never corrupts the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"Saved: {path} ({len(obj)} records)")


def load_model():
    print("Loading tokenizer and model (this may take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(GNOSIS_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GNOSIS_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    ).cuda().eval()
    if not has_correctness_head(model):
        raise RuntimeError(
            "Loaded model is missing the Gnosis correctness head. "
            "Install the custom Transformers fork: "
            "`pip uninstall -y transformers && pip install -e Gnosis/transformers`."
        )
    print("Model loaded.")
    return model, tokenizer


def ask_gnosis(model, tokenizer, question, system_prompt, max_new_tokens=1536):
    prompt = build_chat_prompt(tokenizer, question=question, system_prompt=system_prompt)
    answer = generate_with_hf(
        model,
        tokenizer,
        prompt,
        torch.device("cuda"),
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.95,
    )
    score = float(
        correctness_prob(
            model,
            tokenizer,
            prompt + answer,
            torch.device("cuda"),
            max_len_for_scoring=None,
        )
    )
    return answer, score


def make_record(q: dict, answer: str, score: float, correct: bool) -> dict:
    """Same schema as Phase 2 baseline_results.json."""
    return {
        "id": q.get("id"),
        "domain": q.get("domain", "trivia"),
        "question": q["question"],
        "ground_truth": q["ground_truth"],
        "baseline_answer": answer,
        "baseline_correct": correct,
        "gnosis_score": score,
        "intervened": False,
        "final_answer": answer,
        "final_correct": correct,
        "final_gnosis_score": score,
    }


def record_is_complete(rec: dict) -> bool:
    required = (
        "baseline_answer",
        "baseline_correct",
        "gnosis_score",
        "final_answer",
        "final_correct",
        "final_gnosis_score",
    )
    return all(k in rec for k in required)


def index_checkpoint(records: list[dict]) -> dict:
    by_id = {}
    for rec in records:
        qid = rec.get("id")
        if qid is not None and record_is_complete(rec):
            by_id[qid] = rec
    return by_id


def baseline_is_complete(records: list[dict], questions: list[dict]) -> bool:
    if len(records) != len(questions):
        return False
    by_id = index_checkpoint(records)
    return all(q.get("id") in by_id for q in questions)


def print_score_stats(results):
    correct_scores = [r["gnosis_score"] for r in results if r["baseline_correct"]]
    wrong_scores = [r["gnosis_score"] for r in results if not r["baseline_correct"]]

    if correct_scores:
        print(
            f"Correct — mean: {statistics.mean(correct_scores):.4f}, "
            f"min: {min(correct_scores):.4f}"
        )
    else:
        print("Correct — no correct answers")

    if wrong_scores:
        print(
            f"Wrong   — mean: {statistics.mean(wrong_scores):.4f}, "
            f"max: {max(wrong_scores):.4f}"
        )
    else:
        print("Wrong   — no wrong answers")


def print_threshold_metrics(results, threshold=THRESHOLD):
    caught = sum(
        1 for r in results if r["gnosis_score"] < threshold and not r["baseline_correct"]
    )
    unnecessary = sum(
        1 for r in results if r["gnosis_score"] < threshold and r["baseline_correct"]
    )
    missed = sum(
        1 for r in results if r["gnosis_score"] >= threshold and not r["baseline_correct"]
    )
    trusted = sum(
        1 for r in results if r["gnosis_score"] >= threshold and r["baseline_correct"]
    )

    print("\n" + "=" * 60)
    print(f"Gnosis detection metrics @ threshold={threshold}")
    print("=" * 60)
    print(f"score <  {threshold} and wrong   (caught hallucinations):     {caught}")
    print(f"score <  {threshold} and correct (unnecessary interventions): {unnecessary}")
    print(f"score >= {threshold} and wrong   (missed hallucinations):     {missed}")
    print(f"score >= {threshold} and correct (trusted correct answers):   {trusted}")
    print(f"Total flagged for intervention: {caught + unnecessary}")


def run_baseline(model, tokenizer, questions: list[dict], checkpoint: dict | None):
    checkpoint = checkpoint or {}
    results: list[dict] = []
    completed_since_save = 0
    total = len(questions)
    already_done = sum(1 for q in questions if q.get("id") in checkpoint)

    print("\n" + "=" * 60)
    print("Phase 3 baseline (Pass 1 only)")
    if already_done:
        print(f"Resuming: {already_done}/{total} already in checkpoint")
    print(f"Questions: {QUESTIONS_PATH}")
    print(f"Output:    {BASELINE_PATH}")
    print("=" * 60)

    for i, q in enumerate(questions, start=1):
        qid = q.get("id")
        if qid in checkpoint:
            results.append(checkpoint[qid])
            continue

        domain = q.get("domain", "trivia")
        system_prompt = SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["trivia"])
        answer, score = ask_gnosis(model, tokenizer, q["question"], system_prompt)
        correct = is_correct(answer, q["ground_truth"], q.get("answer_aliases"))
        record = make_record(q, answer, score, correct)
        results.append(record)
        completed_since_save += 1

        preview_q = q["question"][:120].replace("\n", " ")
        print(f"[{i}/{total}] id={qid} domain={domain} correct={correct} score={score:.4f}")
        print(f"  Q: {preview_q}...")
        print("-" * 80)

        if completed_since_save >= CHECKPOINT_EVERY:
            save_json_atomic(BASELINE_PATH, results)
            completed_since_save = 0

    save_json_atomic(BASELINE_PATH, results)
    return results


def main():
    force_rerun = os.environ.get("RERUN_BASELINE", "").strip().lower() in {"1", "true", "yes"}

    if not QUESTIONS_PATH.exists():
        raise SystemExit(
            f"Questions file not found: {QUESTIONS_PATH}\n"
            "Run `python phase_3/build_question_set.py` first."
        )

    questions = load_json(QUESTIONS_PATH)
    print(f"Loaded {len(questions)} questions from {QUESTIONS_PATH}")

    checkpoint: dict | None = None
    if BASELINE_PATH.exists() and not force_rerun:
        existing = load_json(BASELINE_PATH)
        checkpoint = index_checkpoint(existing)
        print(f"Found checkpoint: {BASELINE_PATH} ({len(checkpoint)} complete records)")
        if baseline_is_complete(existing, questions):
            print("Baseline already complete — nothing to do.")
            results = [checkpoint[q["id"]] for q in questions]
        else:
            model, tokenizer = load_model()
            results = run_baseline(model, tokenizer, questions, checkpoint)
    elif force_rerun:
        print("RERUN_BASELINE=1 → ignoring existing checkpoint")
        model, tokenizer = load_model()
        results = run_baseline(model, tokenizer, questions, None)
    else:
        model, tokenizer = load_model()
        results = run_baseline(model, tokenizer, questions, None)

    correct = sum(r["baseline_correct"] for r in results)
    wrong = len(results) - correct
    print(f"\nBaseline — Total: {len(results)} | Correct: {correct} | Wrong: {wrong}")
    print_score_stats(results)
    print_threshold_metrics(results)
    print(f"\nDone. Results: {BASELINE_PATH}")


if __name__ == "__main__":
    main()

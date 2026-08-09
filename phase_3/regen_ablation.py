#!/usr/bin/env python3
"""
Phase 3 — Scaled regeneration ablation (trivia, no RAG).

Compares Gnosis-gated regenerate vs random-matched control on Phase 3 baseline.
Same regenerate prompts as sample.py / Phase 2 (stricter prompt, no RAG).

Usage (Colab):
  export BASELINE_PATH=/content/drive/MyDrive/gate_phase3_baseline.json
  export QUESTIONS_PATH=/content/GATE/phase_3/artifacts/questions_700.json
  !PYTHONPATH=/content/GATE/Gnosis python phase_3/regen_ablation.py

Env:
  THRESHOLD=0.50          # intervene if gnosis_score < THRESHOLD (default 0.50)
  DOMAIN=trivia           # default trivia only
  SEED=42                 # random arm seed
  ARM=both                # gnosis | random | both (default both)
  SKIP_GNOSIS=1           # skip gnosis arm (e.g. if already done)
  SKIP_RANDOM=1           # skip random arm
  GNOSIS_OUT=...          # default phase_3/artifacts/regen_gnosis_results.json
  RANDOM_OUT=...          # default phase_3/artifacts/regen_random_results.json
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
GNOSIS_DIR = REPO_ROOT / "Gnosis"
PHASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = PHASE_DIR / "artifacts"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GNOSIS_DIR))

from eval_utils import build_question_lookup, enrich_records, grade_record
from src.demo import build_chat_prompt, correctness_prob, generate_with_hf, has_correctness_head

# Reuse Phase 2 summarize / comparison helpers
sys.path.insert(0, str(REPO_ROOT / "phase_2"))
from random_baseline import print_comparison, print_summary, summarize

GNOSIS_MODEL_ID = os.environ.get("GNOSIS_MODEL_ID", "AmirhoseinGH/Gnosis-Qwen3-1.7B-Hybrid")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.50"))
DOMAIN = os.environ.get("DOMAIN", "trivia").strip()
SEED = int(os.environ.get("SEED", "42"))
ARM = os.environ.get("ARM", "both").strip().lower()

REGENERATE_PROMPTS = {
    "math": (
        "Your previous answer may be wrong. Carefully re-solve the problem step by step. "
        "Check each step. Put only the final answer within \\boxed{}."
    ),
    "trivia": (
        "Your previous answer may be wrong. Think carefully and answer again. "
        "Put only the final answer within \\boxed{}."
    ),
}


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json_atomic(path: Path, obj) -> None:
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
    print(f"Saved: {path}")


def resolve_baseline_path() -> Path:
    explicit = os.environ.get("BASELINE_PATH", "").strip()
    if explicit:
        return Path(explicit)
    for p in (ARTIFACTS / "baseline_results.json", REPO_ROOT / "baseline_results.json"):
        if p.exists():
            return p
    raise SystemExit("Set BASELINE_PATH to Phase 3 baseline JSON.")


def resolve_questions_path() -> Path:
    explicit = os.environ.get("QUESTIONS_PATH", "").strip()
    if explicit:
        return Path(explicit)
    return ARTIFACTS / "questions_700.json"


def load_model():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(GNOSIS_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GNOSIS_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    ).cuda().eval()
    if not has_correctness_head(model):
        raise RuntimeError("Install Gnosis Transformers fork before running.")
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
            model, tokenizer, prompt + answer, torch.device("cuda"), max_len_for_scoring=None
        )
    )
    return answer, score


def filter_subset(records: list[dict], domain: str) -> list[dict]:
    if not domain or domain.lower() == "all":
        return records
    return [r for r in records if r.get("domain") == domain]


def gnosis_pick_indices(subset: list[dict], threshold: float) -> list[int]:
    return [i for i, r in enumerate(subset) if r["gnosis_score"] < threshold]


def random_pick_indices(n_total: int, n_pick: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    return sorted(indices[:n_pick])


def reset_intervention_fields(results: list[dict]) -> None:
    for r in results:
        r["intervened"] = False
        r["final_answer"] = r["baseline_answer"]
        r["final_correct"] = r["baseline_correct"]
        r["final_gnosis_score"] = r["gnosis_score"]
        for k in ("regen_answer", "regen_correct", "regen_gnosis_score", "selection"):
            r.pop(k, None)


def run_regen_arm(
    model,
    tokenizer,
    subset: list[dict],
    pick_indices: list[int],
    question_lookup: dict,
    selection: str,
    out_path: Path,
) -> list[dict]:
    pick_set = set(pick_indices)
    if out_path.exists():
        results = load_json(out_path)
        if len(results) != len(subset):
            results = copy.deepcopy(subset)
            reset_intervention_fields(results)
    else:
        results = copy.deepcopy(subset)
        reset_intervention_fields(results)

    done = {
        i
        for i in pick_set
        if results[i].get("intervened") and results[i].get("regen_answer") is not None
    }
    remaining = [i for i in pick_indices if i not in done]

    print("\n" + "=" * 60)
    print(f"{selection.upper()} ARM: {len(pick_indices)} total, {len(done)} done, {len(remaining)} left")
    print("=" * 60)

    for i in pick_indices:
        if i in done:
            continue
        r = results[i]
        domain = r.get("domain", "trivia")
        system_prompt = REGENERATE_PROMPTS.get(domain, REGENERATE_PROMPTS["trivia"])
        answer, score = ask_gnosis(model, tokenizer, r["question"], system_prompt)
        correct = grade_record(r, answer, question_lookup)

        r["selection"] = selection
        r["intervened"] = True
        r["regen_answer"] = answer
        r["regen_correct"] = correct
        r["regen_gnosis_score"] = score
        r["final_answer"] = answer
        r["final_correct"] = correct
        r["final_gnosis_score"] = score

        print(
            f"[{selection}] id={r.get('id')} baseline_ok={r['baseline_correct']} "
            f"regen_ok={correct} score={score:.4f}"
        )
        save_json_atomic(out_path, results)

    return results


def main():
    baseline_path = resolve_baseline_path()
    questions_path = resolve_questions_path()
    gnosis_out = Path(
        os.environ.get("GNOSIS_OUT", ARTIFACTS / "regen_gnosis_results.json")
    )
    random_out = Path(
        os.environ.get("RANDOM_OUT", ARTIFACTS / "regen_random_results.json")
    )
    comparison_out = Path(
        os.environ.get("COMPARISON_OUT", ARTIFACTS / "regen_comparison.json")
    )

    skip_gnosis = os.environ.get("SKIP_GNOSIS", "").strip().lower() in {"1", "true", "yes"}
    skip_random = os.environ.get("SKIP_RANDOM", "").strip().lower() in {"1", "true", "yes"}
    run_gnosis = ARM in {"gnosis", "both"} and not skip_gnosis
    run_random = ARM in {"random", "both"} and not skip_random

    records = load_json(baseline_path)
    questions = load_json(questions_path) if questions_path.exists() else None
    if questions:
        records = enrich_records(records, questions)
        lookup = build_question_lookup(questions)
    else:
        lookup = build_question_lookup(
            [
                {
                    "id": r.get("id"),
                    "question": r.get("question"),
                    "ground_truth": r.get("ground_truth"),
                    "answer_aliases": r.get("answer_aliases"),
                }
                for r in records
            ]
        )

    subset = filter_subset(records, DOMAIN)
    print(f"Loaded {len(records)} baseline records; subset domain={DOMAIN!r} n={len(subset)}")
    print(f"Threshold τ={THRESHOLD}")

    gnosis_indices = gnosis_pick_indices(subset, THRESHOLD)
    print(f"Gnosis would flag {len(gnosis_indices)} / {len(subset)} in subset")

    if not gnosis_indices:
        raise SystemExit("No questions flagged at this threshold — lower THRESHOLD or change DOMAIN.")

    random_indices = random_pick_indices(len(subset), len(gnosis_indices), SEED)

    model, tokenizer = load_model()

    gnosis_results = None
    random_results = None

    if run_gnosis:
        gnosis_results = run_regen_arm(
            model, tokenizer, subset, gnosis_indices, lookup, "gnosis", gnosis_out
        )
    elif gnosis_out.exists():
        gnosis_results = load_json(gnosis_out)

    if run_random:
        random_results = run_regen_arm(
            model, tokenizer, subset, random_indices, lookup, "random", random_out
        )
    elif random_out.exists():
        random_results = load_json(random_out)

    comparison = {
        "baseline_path": str(baseline_path),
        "domain": DOMAIN,
        "threshold": THRESHOLD,
        "seed": SEED,
        "n_subset": len(subset),
        "n_intervene": len(gnosis_indices),
        "gnosis_indices": gnosis_indices,
        "random_indices": random_indices,
    }

    if gnosis_results:
        gnosis_summary = summarize(gnosis_results, f"GNOSIS-GATED REGEN (τ={THRESHOLD}, {DOMAIN})")
        print_summary(gnosis_summary)
        comparison["gnosis"] = gnosis_summary

    if random_results:
        random_summary = summarize(random_results, f"RANDOM REGEN (matched N, {DOMAIN})")
        print_summary(random_summary)
        comparison["random"] = random_summary

    if gnosis_results and random_results:
        print_comparison(comparison["gnosis"], comparison["random"])
        g_red = comparison["gnosis"]["hallucination_reduction"]
        r_red = comparison["random"]["hallucination_reduction"]
        if g_red > r_red:
            comparison["verdict"] = "gnosis_better"
        elif r_red > g_red:
            comparison["verdict"] = "random_better"
        else:
            comparison["verdict"] = "tie"

    save_json_atomic(comparison_out, comparison)
    print(f"\nDone. Comparison: {comparison_out}")


if __name__ == "__main__":
    main()

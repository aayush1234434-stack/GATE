"""
Phase 2 — Random-gated regenerate control.

Key test: if intervening on N random questions works as well as intervening
on Gnosis's N low-score questions, then the Gnosis score is not helping.

Uses the SAME regenerate prompts as sample.py Pass 2.
Does NOT re-run baseline — loads baseline_results.json.

Usage (Colab):
  %cd /content/GATE
  !PYTHONPATH=/content/GATE/Gnosis python phase_2/random_baseline.py

Optional env:
  BASELINE_PATH=/path/to/baseline_results.json
  GNOSIS_RESULTS=/path/to/results.json   # used to match intervention count
  N_INTERVENE=22                         # override count (default: match Gnosis or 22)
  SEED=42
  SKIP_MODEL=1                           # only print comparison if both result files exist
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PHASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PHASE_DIR)
GNOSIS_DIR = os.path.join(REPO_ROOT, "Gnosis")
sys.path.insert(0, GNOSIS_DIR)

from src.demo import build_chat_prompt, generate_with_hf, correctness_prob, has_correctness_head

GNOSIS_MODEL_ID = "AmirhoseinGH/Gnosis-Qwen3-1.7B-Hybrid"
THRESHOLD = 0.85

DEFAULT_BASELINE = os.path.join(REPO_ROOT, "baseline_results.json")
DEFAULT_GNOSIS_RESULTS = os.path.join(REPO_ROOT, "results.json")
RANDOM_RESULTS_PATH = os.path.join(PHASE_DIR, "random_results.json")
COMPARISON_PATH = os.path.join(PHASE_DIR, "comparison.json")
PICKED_IDS_PATH = os.path.join(PHASE_DIR, "random_picked_ids.json")

REGENERATE_PROMPTS = {
    "math": (
        "Your previous answer may be wrong. Carefully re-solve the problem step by step. "
        "Check each step. Put only the final answer within \\boxed{}."
    ),
    "trivia": (
        "Your previous answer may be wrong. Think carefully and answer again. "
        "Put only the final answer within \\boxed{}."
    ),
    "mmlu_pro": (
        "Your previous answer may be wrong. Re-evaluate the choices carefully. "
        "Put only the choice letter within \\boxed{}."
    ),
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {path}")


def is_correct(answer, ground_truth):
    return str(ground_truth).strip().lower() in answer.strip().lower()


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
            "Install the custom Transformers fork from Gnosis before running."
        )
    print("Model loaded.")
    return model, tokenizer


def ask_gnosis(model, tokenizer, question, system_prompt, max_new_tokens=1536):
    prompt = build_chat_prompt(tokenizer, question=question, system_prompt=system_prompt)
    answer = generate_with_hf(
        model, tokenizer, prompt, torch.device("cuda"),
        max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.95,
    )
    score = float(
        correctness_prob(
            model, tokenizer, prompt + answer, torch.device("cuda"),
            max_len_for_scoring=None,
        )
    )
    return answer, score


def summarize(results, label):
    baseline_wrong = sum(1 for r in results if not r["baseline_correct"])
    final_wrong = sum(1 for r in results if not r["final_correct"])
    intervened = [r for r in results if r.get("intervened")]
    fixed = sum(1 for r in intervened if not r["baseline_correct"] and r["final_correct"])
    broke = sum(1 for r in intervened if r["baseline_correct"] and not r["final_correct"])
    still_wrong = sum(1 for r in intervened if not r["final_correct"])
    caught_wrong_intervened = sum(1 for r in intervened if not r["baseline_correct"])

    summary = {
        "label": label,
        "n": len(results),
        "baseline_wrong": baseline_wrong,
        "final_wrong": final_wrong,
        "intervened": len(intervened),
        "wrong_among_intervened": caught_wrong_intervened,
        "fixed": fixed,
        "broke": broke,
        "still_wrong_after_regen": still_wrong,
        "hallucination_reduction": baseline_wrong - final_wrong,
        "baseline_wrong_rate": baseline_wrong / len(results) if results else 0.0,
        "final_wrong_rate": final_wrong / len(results) if results else 0.0,
    }
    return summary


def print_summary(summary):
    print("\n" + "=" * 60)
    print(summary["label"])
    print("=" * 60)
    n = summary["n"]
    print(
        f"Baseline wrong: {summary['baseline_wrong']}/{n} "
        f"({100 * summary['baseline_wrong_rate']:.1f}%)"
    )
    print(
        f"Final wrong:    {summary['final_wrong']}/{n} "
        f"({100 * summary['final_wrong_rate']:.1f}%)"
    )
    print(f"Intervened:     {summary['intervened']}")
    print(f"  Wrong among intervened: {summary['wrong_among_intervened']}")
    print(f"  Fixed (wrong -> correct): {summary['fixed']}")
    print(f"  Broke (correct -> wrong): {summary['broke']}")
    print(f"  Still wrong after regen:  {summary['still_wrong_after_regen']}")
    print(f"Hallucination reduction:    {summary['hallucination_reduction']} fewer wrong answers")


def print_comparison(gnosis_summary, random_summary):
    print("\n" + "=" * 60)
    print("GNOSIS vs RANDOM (same intervention budget)")
    print("=" * 60)
    rows = [
        ("Intervened", "intervened"),
        ("Wrong among intervened", "wrong_among_intervened"),
        ("Fixed", "fixed"),
        ("Broke", "broke"),
        ("Still wrong after regen", "still_wrong_after_regen"),
        ("Final wrong", "final_wrong"),
        ("Hallucination reduction", "hallucination_reduction"),
    ]
    print(f"{'Metric':<28} {'Gnosis':>10} {'Random':>10} {'Delta(G-R)':>12}")
    print("-" * 62)
    for name, key in rows:
        g = gnosis_summary[key]
        r = random_summary[key]
        print(f"{name:<28} {g:>10} {r:>10} {g - r:>12}")

    print("\nInterpretation:")
    if random_summary["hallucination_reduction"] >= gnosis_summary["hallucination_reduction"]:
        print(
            "  Random did as well or better than Gnosis on net reduction → "
            "score may not be helping for this intervention."
        )
    else:
        print(
            "  Gnosis beat random on net reduction → score may be useful "
            "for selecting who to fix."
        )


def count_gnosis_intervened(gnosis_results):
    return sum(1 for r in gnosis_results if r.get("intervened"))


def resolve_n_intervene(baseline, gnosis_path):
    env_n = os.environ.get("N_INTERVENE", "").strip()
    if env_n:
        return int(env_n)
    if os.path.exists(gnosis_path):
        g = load_json(gnosis_path)
        n = count_gnosis_intervened(g)
        if n > 0:
            print(f"Matching Gnosis intervention count from {gnosis_path}: N={n}")
            return n
    # Fallback: count how many would be flagged at threshold on baseline
    flagged = sum(1 for r in baseline if r["gnosis_score"] < THRESHOLD)
    if flagged > 0:
        print(f"No Gnosis results found; using baseline score<{THRESHOLD} count: N={flagged}")
        return flagged
    print("Defaulting N_INTERVENE=22")
    return 22


def pick_random_indices(n_total, n_pick, seed):
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    return sorted(indices[:n_pick])


def run_random_intervention(model, tokenizer, baseline, pick_indices):
    results = copy.deepcopy(baseline)
    pick_set = set(pick_indices)

    # Reset intervention fields from any previous Gnosis run if baseline was a full results dump
    for r in results:
        r["intervened"] = False
        r["final_answer"] = r["baseline_answer"]
        r["final_correct"] = r["baseline_correct"]
        r["final_gnosis_score"] = r["gnosis_score"]
        for k in ("regen_answer", "regen_correct", "regen_gnosis_score", "selection"):
            r.pop(k, None)

    print("\n" + "=" * 60)
    print(f"RANDOM PASS: regenerate on {len(pick_indices)} randomly selected questions")
    print("=" * 60)

    for i, r in enumerate(results):
        if i not in pick_set:
            continue
        if r.get("intervened") and "regen_answer" in r:
            continue

        domain = r.get("domain", "trivia")
        system_prompt = REGENERATE_PROMPTS.get(domain, REGENERATE_PROMPTS["trivia"])
        answer, score = ask_gnosis(model, tokenizer, r["question"], system_prompt)
        correct = is_correct(answer, r["ground_truth"])

        r["selection"] = "random"
        r["intervened"] = True
        r["regen_answer"] = answer
        r["regen_correct"] = correct
        r["regen_gnosis_score"] = score
        r["final_answer"] = answer
        r["final_correct"] = correct
        r["final_gnosis_score"] = score

        print(f"RANDOM INTERVENE Q: {r['question']}")
        print(f"Baseline score: {r['gnosis_score']:.4f} | Baseline correct: {r['baseline_correct']}")
        print(f"Regen A: {answer}")
        print(f"Regen Correct: {correct} | Regen Score: {score:.4f}")
        print("-" * 100)
        save_json(RANDOM_RESULTS_PATH, results)

    return results


def main():
    baseline_path = os.environ.get("BASELINE_PATH", DEFAULT_BASELINE)
    gnosis_path = os.environ.get("GNOSIS_RESULTS", DEFAULT_GNOSIS_RESULTS)
    seed = int(os.environ.get("SEED", "42"))
    skip_model = os.environ.get("SKIP_MODEL", "").strip() in {"1", "true", "True", "yes"}

    if not os.path.exists(baseline_path):
        raise SystemExit(
            f"Missing baseline file: {baseline_path}\n"
            "Copy baseline_results.json from Colab into the repo root, or set BASELINE_PATH."
        )

    baseline = load_json(baseline_path)
    print(f"Loaded baseline: {baseline_path} ({len(baseline)} records)")

    n_intervene = resolve_n_intervene(baseline, gnosis_path)
    if n_intervene > len(baseline):
        raise SystemExit(f"N_INTERVENE={n_intervene} > baseline size {len(baseline)}")

    pick_indices = pick_random_indices(len(baseline), n_intervene, seed)
    picked_meta = {
        "seed": seed,
        "n_intervene": n_intervene,
        "indices": pick_indices,
        "ids": [baseline[i].get("id") for i in pick_indices],
        "questions": [baseline[i]["question"] for i in pick_indices],
    }
    save_json(PICKED_IDS_PATH, picked_meta)
    print(f"Seed={seed} | Picked {n_intervene} indices: {pick_indices}")

    if skip_model:
        if not os.path.exists(RANDOM_RESULTS_PATH):
            raise SystemExit("SKIP_MODEL=1 but phase_2/random_results.json is missing.")
        random_results = load_json(RANDOM_RESULTS_PATH)
    else:
        model, tokenizer = load_model()
        random_results = run_random_intervention(model, tokenizer, baseline, pick_indices)
        save_json(RANDOM_RESULTS_PATH, random_results)

    random_summary = summarize(random_results, "RANDOM-GATED REGENERATE")
    print_summary(random_summary)

    comparison = {"random": random_summary, "seed": seed, "n_intervene": n_intervene}

    if os.path.exists(gnosis_path):
        gnosis_results = load_json(gnosis_path)
        gnosis_summary = summarize(gnosis_results, "GNOSIS-GATED REGENERATE (t=0.85)")
        print_summary(gnosis_summary)
        print_comparison(gnosis_summary, random_summary)
        comparison["gnosis"] = gnosis_summary
        g_red = gnosis_summary["hallucination_reduction"]
        r_red = random_summary["hallucination_reduction"]
        if g_red > r_red:
            comparison["verdict"] = "gnosis_better"
        elif r_red > g_red:
            comparison["verdict"] = "random_better"
        else:
            comparison["verdict"] = "tie"
    else:
        print(
            f"\nNo Gnosis results at {gnosis_path}. "
            "Copy results.json there to auto-compare, or compare manually."
        )
        # Record known pilot numbers if file missing (optional reference)
        comparison["gnosis_reference_from_pilot"] = {
            "note": "From Colab run (manual); replace when results.json is available",
            "baseline_wrong": 5,
            "final_wrong": 5,
            "intervened": 22,
            "fixed": 1,
            "broke": 1,
            "still_wrong_after_regen": 3,
            "hallucination_reduction": 0,
        }

    save_json(COMPARISON_PATH, comparison)
    print(f"\nDone. Outputs in {PHASE_DIR}")


if __name__ == "__main__":
    main()

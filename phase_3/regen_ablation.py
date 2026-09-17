#!/usr/bin/env python3
"""
Phase 3 — Scaled critique-and-revise ablation (trivia, no RAG).

Compares Gnosis-gated critique-and-revise vs a random-matched control on the
Phase 3 baseline. The default protocol exposes the baseline answer for audit.

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
  RANDOM_SEEDS=11,23,37   # repeated random matched controls (default: SEED)
  INTERVENTION_PROTOCOL=critique_revision_v1  # default; legacy resample is opt-in
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
GNOSIS_DIR = REPO_ROOT / "Gnosis"
PHASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = PHASE_DIR / "artifacts"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GNOSIS_DIR))

from eval_utils import build_question_lookup, enrich_records, grade_record
from experiment_state import load_json, prepare_intervention_results, save_json_atomic
from intervention_protocol import (
    CRITIQUE_REVISION_V1,
    build_intervention_question,
    intervention_instruction,
    resolve_protocol,
)
from runtime import describe_runtime, model_dtype, resolve_device, set_experiment_seed
from src.demo import build_chat_prompt, correctness_prob, generate_with_hf, has_correctness_head

# Reuse Phase 2 summarize / comparison helpers
sys.path.insert(0, str(REPO_ROOT / "phase_2"))
from random_baseline import print_comparison, print_summary, summarize

GNOSIS_MODEL_ID = os.environ.get("GNOSIS_MODEL_ID", "AmirhoseinGH/Gnosis-Qwen3-1.7B-Hybrid")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.50"))
DOMAIN = os.environ.get("DOMAIN", "trivia").strip()
SEED = int(os.environ.get("SEED", "42"))
ARM = os.environ.get("ARM", "both").strip().lower()
CONFIG_PATH = os.environ.get("CONFIG_PATH", "").strip()
if CONFIG_PATH:
    with open(CONFIG_PATH, "r") as config_file:
        EXPERIMENT_CONFIG = json.load(config_file)
else:
    EXPERIMENT_CONFIG = {}
INTERVENTION_PROTOCOL = resolve_protocol(
    os.environ.get("INTERVENTION_PROTOCOL", CRITIQUE_REVISION_V1)
)


def parse_seeds(value: str, fallback: int) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    return seeds or [fallback]


_configured_random_seeds = EXPERIMENT_CONFIG.get("random_control_seeds", [SEED])
RANDOM_SEEDS = parse_seeds(
    os.environ.get("RANDOM_SEEDS", ",".join(str(seed) for seed in _configured_random_seeds)), SEED
)


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
    device = resolve_device()
    print(f"Runtime: {describe_runtime(device)}")
    tokenizer = AutoTokenizer.from_pretrained(GNOSIS_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GNOSIS_MODEL_ID,
        torch_dtype=model_dtype(device),
        trust_remote_code=True,
        use_cache=False,
    ).to(device).eval()
    if not has_correctness_head(model):
        raise RuntimeError("Install Gnosis Transformers fork before running.")
    print("Model loaded.")
    return model, tokenizer, device


def ask_gnosis(model, tokenizer, device, question, system_prompt, max_new_tokens=1536):
    prompt = build_chat_prompt(tokenizer, question=question, system_prompt=system_prompt)
    answer = generate_with_hf(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.95,
    )
    score = float(
        correctness_prob(
            model, tokenizer, prompt + answer, device, max_len_for_scoring=None
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


def run_regen_arm(
    model,
    tokenizer,
    device,
    subset: list[dict],
    pick_indices: list[int],
    question_lookup: dict,
    selection: str,
    out_path: Path,
    generation_seed: int,
) -> list[dict]:
    results = prepare_intervention_results(
        subset,
        pick_indices,
        out_path,
        selection=selection,
        protocol=INTERVENTION_PROTOCOL,
        seed=generation_seed,
    )

    done = {
        i
        for i in pick_indices
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
        system_prompt = intervention_instruction(domain)
        intervention_question = build_intervention_question(
            r["question"], r["baseline_answer"], INTERVENTION_PROTOCOL
        )
        answer, score = ask_gnosis(
            model, tokenizer, device, intervention_question, system_prompt
        )
        correct = grade_record(r, answer, question_lookup)

        r["selection"] = selection
        r["intervention_protocol"] = INTERVENTION_PROTOCOL
        r["intervention_seed"] = generation_seed
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
        print(f"Saved: {out_path}")

    return results


def main():
    set_experiment_seed(SEED)
    print(f"Experiment seed: {SEED}")
    baseline_path = resolve_baseline_path()
    questions_path = resolve_questions_path()
    gnosis_out = Path(
        os.environ.get("GNOSIS_OUT", ARTIFACTS / "regen_gnosis_results.json")
    )
    random_out_setting = os.environ.get("RANDOM_OUT", "").strip()
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

    random_indices_by_seed = {
        seed: random_pick_indices(len(subset), len(gnosis_indices), seed)
        for seed in RANDOM_SEEDS
    }

    model, tokenizer, device = load_model()

    gnosis_results = None
    random_results_by_seed: dict[int, list[dict]] = {}

    if run_gnosis:
        gnosis_results = run_regen_arm(
            model, tokenizer, device, subset, gnosis_indices, lookup, "gnosis", gnosis_out, SEED
        )
    elif gnosis_out.exists():
        gnosis_results = load_json(gnosis_out)

    for random_seed, random_indices in random_indices_by_seed.items():
        if random_out_setting:
            if len(RANDOM_SEEDS) > 1 and "{seed}" not in random_out_setting:
                raise SystemExit("Use RANDOM_OUT with a {seed} placeholder when RANDOM_SEEDS has multiple values.")
            random_out = Path(random_out_setting.format(seed=random_seed))
        elif len(RANDOM_SEEDS) == 1:
            random_out = ARTIFACTS / "regen_random_results.json"
        else:
            random_out = ARTIFACTS / f"regen_random_seed_{random_seed}.json"

        if run_random:
            set_experiment_seed(random_seed)
            random_results_by_seed[random_seed] = run_regen_arm(
                model,
                tokenizer,
                device,
                subset,
                random_indices,
                lookup,
                "random",
                random_out,
                random_seed,
            )
        elif random_out.exists():
            random_results_by_seed[random_seed] = load_json(random_out)

    comparison = {
        "baseline_path": str(baseline_path),
        "domain": DOMAIN,
        "threshold": THRESHOLD,
        "seed": SEED,
        "intervention_protocol": INTERVENTION_PROTOCOL,
        "n_subset": len(subset),
        "n_intervene": len(gnosis_indices),
        "gnosis_indices": gnosis_indices,
        "random_seeds": RANDOM_SEEDS,
        "random_indices_by_seed": random_indices_by_seed,
    }

    if gnosis_results:
        gnosis_summary = summarize(gnosis_results, f"GNOSIS-GATED REGEN (τ={THRESHOLD}, {DOMAIN})")
        print_summary(gnosis_summary)
        comparison["gnosis"] = gnosis_summary

    random_summaries = {}
    for random_seed, random_results in random_results_by_seed.items():
        random_summary = summarize(
            random_results, f"RANDOM REGEN (matched N, {DOMAIN}, seed={random_seed})"
        )
        print_summary(random_summary)
        random_summaries[str(random_seed)] = random_summary
    if random_summaries:
        comparison["random_by_seed"] = random_summaries

    if gnosis_results and random_summaries:
        verdict_by_seed = {}
        for random_seed, random_summary in random_summaries.items():
            print_comparison(comparison["gnosis"], random_summary)
            g_red = comparison["gnosis"]["hallucination_reduction"]
            r_red = random_summary["hallucination_reduction"]
            verdict_by_seed[random_seed] = (
                "gnosis_better" if g_red > r_red else "random_better" if r_red > g_red else "tie"
            )
        comparison["verdict_by_seed"] = verdict_by_seed

    save_json_atomic(comparison_out, comparison)
    print(f"Saved: {comparison_out}")
    print(f"\nDone. Comparison: {comparison_out}")


if __name__ == "__main__":
    main()

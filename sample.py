"""
Baseline + Gnosis-gated regenerate experiment.

1. Answer every question (baseline).
2. If gnosis_score < THRESHOLD, regenerate with a stricter prompt (no RAG).
3. Report detection metrics and before/after hallucination rates.

Resume / skip Pass 1:
  SKIP_BASELINE=1 python sample.py
  # or if baseline_results.json already exists with all questions, Pass 1 is skipped
"""

import json
import os
import statistics
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GNOSIS_DIR = os.path.join(SCRIPT_DIR, "Gnosis")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, GNOSIS_DIR)

from src.demo import build_chat_prompt, generate_with_hf, correctness_prob, has_correctness_head

from eval_utils import is_correct

GNOSIS_MODEL_ID = "AmirhoseinGH/Gnosis-Qwen3-1.7B-Hybrid"
THRESHOLD = 0.85
BASELINE_PATH = os.path.join(SCRIPT_DIR, "baseline_results.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.json")
CHECKPOINT_EVERY = 5  # save baseline progress every N questions

SYSTEM_PROMPTS = {
    "math": "Please reason step by step, and put your final answer within \\boxed{}.",
    "trivia": "This is a trivia question. Put your final answer within \\boxed{}.",
    "mmlu_pro": (
        "You are solving multiple-choice questions. Please reason step by step, "
        "and put your final answer with only the choice letter within \\boxed{}."
    ),
}

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


with open(os.path.join(SCRIPT_DIR, "questions.json"), "r") as file:
    data = json.load(file)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {path}")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


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
            "Install the custom Transformers fork from the Gnosis repo before "
            "running this script. In Colab: clone the Gnosis repo and run "
            "`pip uninstall -y transformers && pip install -e /content/GATE/Gnosis/transformers`."
        )
    print("Model loaded.")
    return model, tokenizer


def ask_gnosis(model, tokenizer, question, system_prompt, max_new_tokens=1536):
    prompt = build_chat_prompt(tokenizer, question=question, system_prompt=system_prompt)

    answer = generate_with_hf(
        model, tokenizer, prompt, torch.device("cuda"),
        max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.95,
    )

    score = correctness_prob(
        model, tokenizer, prompt + answer, torch.device("cuda"),
        max_len_for_scoring=None,
    )
    score = float(score)

    return answer, score


def print_score_stats(results, score_key="gnosis_score", correct_key="correct"):
    correct_scores = [r[score_key] for r in results if r[correct_key]]
    wrong_scores = [r[score_key] for r in results if not r[correct_key]]

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


def print_threshold_metrics(results, threshold=THRESHOLD, correct_key="baseline_correct"):
    caught = sum(1 for r in results if r["gnosis_score"] < threshold and not r[correct_key])
    unnecessary = sum(1 for r in results if r["gnosis_score"] < threshold and r[correct_key])
    missed = sum(1 for r in results if r["gnosis_score"] >= threshold and not r[correct_key])
    trusted_correct = sum(1 for r in results if r["gnosis_score"] >= threshold and r[correct_key])
    flagged = caught + unnecessary

    print("\n" + "=" * 60)
    print(f"Gnosis detection metrics @ threshold={threshold}")
    print("=" * 60)
    print(f"score <  {threshold} and wrong   (caught hallucinations):     {caught}")
    print(f"score <  {threshold} and correct (unnecessary interventions): {unnecessary}")
    print(f"score >= {threshold} and wrong   (missed hallucinations):     {missed}")
    print(f"score >= {threshold} and correct (trusted correct answers):   {trusted_correct}")
    print(f"Total flagged for intervention: {flagged}")
    return flagged


def baseline_is_complete(results):
    if len(results) != len(data):
        return False
    return all(
        "gnosis_score" in r and "baseline_answer" in r and "baseline_correct" in r
        for r in results
    )


def run_pass1(model, tokenizer, existing=None):
    """Run baseline; resume from partial checkpoint if provided."""
    results = list(existing) if existing else []
    done_ids = {r.get("id") for r in results if r.get("id") is not None}
    done_questions = {r["question"] for r in results}

    print("\n" + "=" * 60)
    print("PASS 1: Baseline (no intervention)")
    if results:
        print(f"Resuming from checkpoint: {len(results)}/{len(data)} already done")
    print("=" * 60)

    for q in data:
        qid = q.get("id")
        if (qid is not None and qid in done_ids) or q["question"] in done_questions:
            continue

        domain = q.get("domain", "trivia")
        system_prompt = SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["trivia"])
        answer, score = ask_gnosis(model, tokenizer, q["question"], system_prompt)
        correct = is_correct(answer, q["ground_truth"], q.get("answer_aliases"))

        record = {
            "id": qid,
            "domain": domain,
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
        results.append(record)

        print(f"Q: {q['question']}")
        print(f"A: {answer}")
        print(f"Ground Truth: {q['ground_truth']} | Correct: {correct} | Score: {score:.4f}")
        print("-" * 100)

        if len(results) % CHECKPOINT_EVERY == 0:
            save_json(BASELINE_PATH, results)

    save_json(BASELINE_PATH, results)
    return results


def run_pass2(model, tokenizer, results):
    print("\n" + "=" * 60)
    print(f"PASS 2: Regenerate intervention (score < {THRESHOLD})")
    print("=" * 60)

    for r in results:
        if r["gnosis_score"] >= THRESHOLD:
            continue
        if r.get("intervened") and "regen_answer" in r:
            print(f"Skip already intervened: {r['question'][:80]}...")
            continue

        domain = r["domain"]
        system_prompt = REGENERATE_PROMPTS.get(domain, REGENERATE_PROMPTS["trivia"])
        answer, score = ask_gnosis(model, tokenizer, r["question"], system_prompt)
        correct = is_correct(answer, r["ground_truth"], r.get("answer_aliases"))

        r["intervened"] = True
        r["regen_answer"] = answer
        r["regen_correct"] = correct
        r["regen_gnosis_score"] = score
        r["final_answer"] = answer
        r["final_correct"] = correct
        r["final_gnosis_score"] = score

        print(f"INTERVENE Q: {r['question']}")
        print(f"Baseline score: {r['gnosis_score']:.4f} | Baseline correct: {r['baseline_correct']}")
        print(f"Regen A: {answer}")
        print(f"Regen Correct: {correct} | Regen Score: {score:.4f}")
        print("-" * 100)

        save_json(RESULTS_PATH, results)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
force_skip = os.environ.get("SKIP_BASELINE", "").strip() in {"1", "true", "True", "yes"}
force_rerun = os.environ.get("RERUN_BASELINE", "").strip() in {"1", "true", "True", "yes"}

partial = None
if os.path.exists(BASELINE_PATH) and not force_rerun:
    partial = load_json(BASELINE_PATH)
    print(f"Found {BASELINE_PATH} with {len(partial)} records")

model, tokenizer = load_model()

if force_skip:
    if not partial or not baseline_is_complete(partial):
        raise SystemExit(
            "SKIP_BASELINE=1 but baseline_results.json is missing or incomplete. "
            "Upload/save a complete baseline_results.json first."
        )
    results = partial
    print("SKIP_BASELINE=1 → skipping Pass 1")
elif partial and baseline_is_complete(partial) and not force_rerun:
    results = partial
    print("Complete baseline checkpoint found → skipping Pass 1")
else:
    results = run_pass1(model, tokenizer, existing=partial if not force_rerun else None)

baseline_correct = sum(r["baseline_correct"] for r in results)
baseline_wrong = len(results) - baseline_correct
print(f"\nBaseline — Total: {len(results)} | Correct: {baseline_correct} | Wrong: {baseline_wrong}")
print_score_stats(
    [{"gnosis_score": r["gnosis_score"], "correct": r["baseline_correct"]} for r in results]
)
print_threshold_metrics(results, THRESHOLD)

# Always keep a baseline checkpoint before Pass 2
save_json(BASELINE_PATH, results)

results = run_pass2(model, tokenizer, results)

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------
final_correct = sum(r["final_correct"] for r in results)
final_wrong = len(results) - final_correct
intervened = [r for r in results if r["intervened"]]
fixed = sum(1 for r in intervened if not r["baseline_correct"] and r["final_correct"])
still_wrong = sum(1 for r in intervened if not r["final_correct"])
broke = sum(1 for r in intervened if r["baseline_correct"] and not r["final_correct"])

print("\n" + "=" * 60)
print("BEFORE vs AFTER")
print("=" * 60)
print(f"Baseline wrong: {baseline_wrong}/{len(results)} ({100 * baseline_wrong / len(results):.1f}%)")
print(f"Final wrong:    {final_wrong}/{len(results)} ({100 * final_wrong / len(results):.1f}%)")
print(f"Intervened:     {len(intervened)}")
print(f"  Fixed (wrong -> correct): {fixed}")
print(f"  Broke (correct -> wrong): {broke}")
print(f"  Still wrong after regen:  {still_wrong}")
print(f"Hallucination reduction:    {baseline_wrong - final_wrong} fewer wrong answers")

save_json(RESULTS_PATH, results)
print(f"\nDone. Baseline: {BASELINE_PATH} | Full: {RESULTS_PATH}")

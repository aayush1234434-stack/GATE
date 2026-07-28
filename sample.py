"""
Baseline + Gnosis-gated regenerate experiment.

1. Answer every question (baseline).
2. If gnosis_score < THRESHOLD, regenerate with a stricter prompt (no RAG).
3. Report detection metrics and before/after hallucination rates.
"""

import json
import os
import statistics
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GNOSIS_DIR = os.path.join(SCRIPT_DIR, "Gnosis")
sys.path.insert(0, GNOSIS_DIR)

from src.demo import build_chat_prompt, generate_with_hf, correctness_prob, has_correctness_head

GNOSIS_MODEL_ID = "AmirhoseinGH/Gnosis-Qwen3-1.7B-Hybrid"
THRESHOLD = 0.85

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


def is_correct(answer, ground_truth):
    return ground_truth.strip().lower() in answer.strip().lower()


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


model, tokenizer = load_model()

# ---------------------------------------------------------------------------
# Pass 1: baseline (no intervention)
# ---------------------------------------------------------------------------
results = []

print("\n" + "=" * 60)
print("PASS 1: Baseline (no intervention)")
print("=" * 60)

for q in data:
    domain = q.get("domain", "trivia")
    system_prompt = SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["trivia"])
    answer, score = ask_gnosis(model, tokenizer, q["question"], system_prompt)
    correct = is_correct(answer, q["ground_truth"])

    record = {
        "id": q.get("id"),
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

baseline_correct = sum(r["baseline_correct"] for r in results)
baseline_wrong = len(results) - baseline_correct
print(f"\nBaseline — Total: {len(results)} | Correct: {baseline_correct} | Wrong: {baseline_wrong}")
print_score_stats(
    [{"gnosis_score": r["gnosis_score"], "correct": r["baseline_correct"]} for r in results]
)
print_threshold_metrics(results, THRESHOLD)

# ---------------------------------------------------------------------------
# Pass 2: regenerate only when gnosis_score < THRESHOLD
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"PASS 2: Regenerate intervention (score < {THRESHOLD})")
print("=" * 60)

for r in results:
    if r["gnosis_score"] >= THRESHOLD:
        continue

    domain = r["domain"]
    system_prompt = REGENERATE_PROMPTS.get(domain, REGENERATE_PROMPTS["trivia"])
    answer, score = ask_gnosis(model, tokenizer, r["question"], system_prompt)
    correct = is_correct(answer, r["ground_truth"])

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

out_path = os.path.join(SCRIPT_DIR, "results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved detailed results to {out_path}")

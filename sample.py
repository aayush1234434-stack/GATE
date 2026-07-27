import json
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.demo import build_chat_prompt, generate_with_hf, correctness_prob

GNOSIS_MODEL_ID = "AmirhoseinGH/Gnosis-Qwen3-1.7B-Hybrid"

threshold = [0.5,0.6,0.7,0.8,0.9]

with open("questions.json","r") as file:
    data = json.load(file)


def load_model():
    print("Loading tokenizer and model (this may take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(GNOSIS_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GNOSIS_MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()
    print("Model loaded.")
    return model, tokenizer

def ask_gnosis(model, tokenizer, question, max_new_tokens=512):
    
    prompt = build_chat_prompt(tokenizer, question=question)

    answer = generate_with_hf(
        model, tokenizer, prompt, torch.device("cuda"),
        max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.95
    )

    score = correctness_prob(
        model, tokenizer, prompt + answer, torch.device("cuda"),
        max_len_for_scoring=None
    )

    return answer, score


model, tokenizer = load_model()

results = []

for q in data:
    answer, score = ask_gnosis(model, tokenizer, q["question"])
    correct = q["ground_truth"].strip().lower() in answer.strip().lower()
    record = {
        "question": q["question"],
        "ground_truth": q["ground_truth"],
        "correct": correct,
        "gnosis_score": score,
    }
    results.append(record)
    print(f"Q: {q['question']}")
    print(f"A: {answer}")
    print(f"Ground Truth: {q['ground_truth']} | Correct: {correct} | Score: {score}")
    print("-" * 100)

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nTotal: {len(results)} | Correct: {sum(r['correct'] for r in results)}")

correct_scores = [r["gnosis_score"] for r in results if r["correct"]]
wrong_scores = [r["gnosis_score"] for r in results if not r["correct"]]

if correct_scores:
    print(f"Correct — mean: {statistics.mean(correct_scores):.4f}, min: {min(correct_scores):.4f}")
else:
    print("Correct — no correct answers")

if wrong_scores:
    print(f"Wrong   — mean: {statistics.mean(wrong_scores):.4f}, max: {max(wrong_scores):.4f}")
else:
    print("Wrong   — no wrong answers")

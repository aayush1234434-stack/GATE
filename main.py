
"""
Standalone script: ask a question, get Qwen's answer + Gnosis's correctness score.
Usage: python main.py
"""

import os
import sys
import torch

# Resolve paths relative to this script's own location — works no matter
# what folder the repo gets cloned into, as long as Gnosis/ sits next to main.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GNOSIS_DIR = os.path.join(SCRIPT_DIR, "Gnosis")

sys.path.insert(0, GNOSIS_DIR)
os.chdir(GNOSIS_DIR)

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.demo import build_chat_prompt, generate_with_hf, correctness_prob

GNOSIS_MODEL_ID = "AmirhoseinGH/Gnosis-Qwen3-1.7B-Hybrid"

SYSTEM_PROMPTS = {
    "math": "Please reason step by step, and put your final answer within \\boxed{}.",
    "trivia": "This is a trivia question. Put your final answer within \\boxed{}.",
    "mmlu_pro": "You are solving multiple-choice questions. Please reason step by step, and put your final answer with only the choice letter within \\boxed{}."
}


def load_model():
    print("Loading tokenizer and model (this may take a minute)...")
    tokenizer = AutoTokenizer.from_pretrained(GNOSIS_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        GNOSIS_MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()
    print("Model loaded.")
    return model, tokenizer


def ask_gnosis(model, tokenizer, question, task_type="trivia", max_new_tokens=512):
    system_prompt = SYSTEM_PROMPTS[task_type]
    prompt = build_chat_prompt(tokenizer, question=question, system_prompt=system_prompt)

    answer = generate_with_hf(
        model, tokenizer, prompt, torch.device("cuda"),
        max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.95
    )

    score = correctness_prob(
        model, tokenizer, prompt + answer, torch.device("cuda"),
        max_len_for_scoring=None
    )

    return answer, score


if __name__ == "__main__":
    model, tokenizer = load_model()

    question = "How many r's are in strawberry?"
    task_type = "trivia"

    answer, score = ask_gnosis(model, tokenizer, question, task_type=task_type)

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print(f"Answer:\n{answer}")
    print(f"\nGnosis correctness score: {score:.4f}")
    print("=" * 60)

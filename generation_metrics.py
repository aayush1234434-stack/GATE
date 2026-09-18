"""Generation-time confidence features used by the baseline study."""

from __future__ import annotations

from collections import Counter
import math
import time

import torch

from eval_utils import _boxed_contents, _normalize


@torch.no_grad()
def generate_with_hf_metrics(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, dict]:
    """Generate an answer and capture per-token log-probability and entropy."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = enc["input_ids"].shape[1]
    started = time.perf_counter()
    generated = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        output_scores=True,
    )
    elapsed = time.perf_counter() - started
    token_ids = generated.sequences[0, input_len:]
    answer = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

    log_probs, entropies = [], []
    for step_logits, token_id in zip(generated.scores, token_ids):
        log_distribution = torch.log_softmax(step_logits[0].float(), dim=-1)
        distribution = log_distribution.exp()
        log_probs.append(float(log_distribution[int(token_id)]))
        entropies.append(float(-(distribution * log_distribution).sum()))
    return answer, {
        "generated_tokens": len(log_probs),
        "mean_token_logprob": sum(log_probs) / len(log_probs) if log_probs else None,
        "min_token_logprob": min(log_probs) if log_probs else None,
        "mean_token_entropy": sum(entropies) / len(entropies) if entropies else None,
        "max_token_entropy": max(entropies) if entropies else None,
        "generation_seconds": elapsed,
    }


def answer_signature(answer: str) -> str:
    """A conservative answer signature used for exact self-consistency."""
    boxed = _boxed_contents(answer)
    return _normalize(boxed[-1] if boxed else answer)


def self_consistency(answers: list[str]) -> float | None:
    """Fraction of samples agreeing with the modal normalized final answer."""
    if len(answers) < 2:
        return None
    counts = Counter(answer_signature(answer) for answer in answers)
    return max(counts.values()) / len(answers)


def stable_mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None and math.isfinite(value)]
    return sum(present) / len(present) if present else None

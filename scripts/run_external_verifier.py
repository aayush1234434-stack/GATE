#!/usr/bin/env python3
"""Add an independent OpenAI-compatible verifier score with safe checkpoints.

This script is not run automatically. It makes paid/network requests only when
the user explicitly invokes it with an endpoint, model, and API-key variable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_state import load_json, save_json_atomic


def verifier_prompt(question: str, answer: str) -> str:
    return (
        "Judge whether the proposed answer correctly answers the question. "
        "Reply with only YES or NO.\n\n"
        f"Question:\n{question}\n\nProposed answer:\n{answer}"
    )


def parse_verdict(text: str) -> float:
    verdict = text.strip().upper()
    if verdict.startswith("YES"):
        return 1.0
    if verdict.startswith("NO"):
        return 0.0
    raise ValueError(f"Verifier did not return YES or NO: {text!r}")


def verify(endpoint: str, api_key: str, model: str, question: str, answer: str) -> tuple[float, str]:
    payload = json.dumps({
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": verifier_prompt(question, answer)}],
    }).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        body = json.load(response)
    text = body["choices"][0]["message"]["content"]
    return parse_verdict(text), text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. https://host/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before running the external verifier.")

    if args.output.exists():
        records = load_json(args.output)
    else:
        records = load_json(args.input)
    for record in records:
        if record.get("external_verifier_model") == args.model and "external_verifier_score" in record:
            continue
        score, verdict = verify(args.endpoint, api_key, args.model, record["question"], record["baseline_answer"])
        record["external_verifier_score"] = score
        record["external_verifier_verdict"] = verdict
        record["external_verifier_model"] = args.model
        save_json_atomic(args.output, records)
        print(f"Verified id={record.get('id')} score={score:.0f}")


if __name__ == "__main__":
    main()

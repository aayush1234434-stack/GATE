import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_external_verifier.py"
SPEC = importlib.util.spec_from_file_location("external_verifier", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_verifier_prompt_is_answer_blind_and_parseable():
    prompt = MODULE.verifier_prompt("What is 2 + 2?", "The answer is 4.")
    assert "ground truth" not in prompt.lower()
    assert "What is 2 + 2?" in prompt
    assert MODULE.parse_verdict("YES") == 1.0
    assert MODULE.parse_verdict("no") == 0.0

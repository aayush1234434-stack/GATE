# Gnosis arm — Phase 2 pilot notes

Copied from Colab run (gated regenerate @ threshold 0.85).

## Detection (from baseline)
- Caught (score < 0.85 & wrong): 3
- Unnecessary (score < 0.85 & correct): 19
- Missed (score ≥ 0.85 & wrong): 2
- Trusted correct: 76
- Flagged: 22

## Before vs after (Gnosis regenerate)
- Baseline wrong: 5/100 (5.0%)
- Final wrong: 5/100 (5.0%)
- Intervened: 22
- Fixed (wrong → correct): 1
- Broke (correct → wrong): 1
- Still wrong after regen: 3
- Hallucination reduction: 0

## Intervention type
Prompt-only regenerate (no RAG), same model, stricter system prompt when score < 0.85.

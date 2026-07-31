# Phase 2 — Gnosis-gated regenerate vs Random control

## Question
Does Gnosis’s score help pick *who* to regenerate, or would regenerating the same number of **random** questions work just as well?

## What already ran (Gnosis arm)
- Policy: intervene if `gnosis_score < 0.85`
- Intervention: same stricter regenerate prompt (no RAG)
- Result from pilot:
  - Flagged / intervened: **22**
  - Fixed: **1**, Broke: **1**, Still wrong after regen: **3**
  - Net hallucination reduction: **0** (5/100 → 5/100)

Put your Colab artifacts in the repo root (or point env vars):
- `baseline_results.json` — required (skip redoing Pass 1)
- `results.json` — Gnosis Pass 2 output (for auto comparison)

## Random arm (this folder)
Script: `random_baseline.py`

1. Load baseline answers/scores
2. Pick **N** random questions (`N` = Gnosis intervene count from `results.json`, else 22)
3. Apply the **same** regenerate prompts
4. Compare fixed / broke / final wrong rate vs Gnosis

### Colab
```python
%cd /content/GATE
!git pull

# Ensure baseline_results.json (and ideally results.json) are in /content/GATE
!PYTHONPATH=/content/GATE/Gnosis python phase_2/random_baseline.py
```

Optional:
```bash
SEED=42 N_INTERVENE=22 PYTHONPATH=/content/GATE/Gnosis python phase_2/random_baseline.py
```

### Outputs
| File | Meaning |
|---|---|
| `random_picked_ids.json` | Which questions were randomly selected (seeded) |
| `random_results.json` | Full results after random regenerate |
| `comparison.json` | Side-by-side Gnosis vs Random metrics |

## How to interpret
| Outcome | Meaning |
|---|---|
| Gnosis net reduction ≫ Random | Score helps select who to fix |
| Random ≈ Gnosis | Score isn’t helping for this intervention |
| Random > Gnosis | Score may be selecting the wrong cases |

## Why this phase ends here
This is the critical control for the paper. Only after this comparison should you decide whether to scale data, try RAG, or write up a negative/positive result.

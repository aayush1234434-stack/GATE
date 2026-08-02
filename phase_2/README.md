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

## What you need to run Phase 2 (random control)

| You have | What to run | Time |
|----------|-------------|------|
| **`results.json`** (full Gnosis run) | Only `random_baseline.py` | ~20–40 min (18 regenerations) |
| **`baseline_results.json` only** | `random_baseline.py` + set `N_INTERVENE=18` if no results.json | same |
| **Neither file** | Full `sample.py` first, then random | ~1.5h + ~30 min |

`results.json` **includes** baseline answers/scores — the script can use it instead of `baseline_results.json`.

Put files in repo root **or** `phase_2/artifacts/` (see `artifacts/README.md` for committed snapshots).

Required inputs:
- `baseline_results.json` **or** `results.json` — baseline answers + Gnosis scores (no Pass 1 redo)
- `results.json` — optional but needed for automatic Gnosis vs Random table

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
| `oracle_analysis.json` | Full detection + outcome table (`oracle_analysis.py`) |

Run analysis (no GPU): `python phase_2/oracle_analysis.py`

## How to interpret
| Outcome | Meaning |
|---|---|
| Gnosis net reduction ≫ Random | Score helps select who to fix |
| Random ≈ Gnosis | Score isn’t helping for this intervention |
| Random > Gnosis | Score may be selecting the wrong cases |

## Why this phase ends here
This is the critical control for the paper. Only after this comparison should you decide whether to scale data, try RAG, or write up a negative/positive result.

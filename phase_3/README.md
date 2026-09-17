# Phase 3 — Scale up evaluation

**Colab step-by-step:** see [COLAB.md](./COLAB.md) (start here).

## Build question set (Colab, no GPU)

Only needed once if `questions_700.json` is not already in the repo.

```python
%cd /content/GATE
!git pull
!pip install -q datasets
!python phase_3/build_question_set.py
```

Output: `phase_3/artifacts/questions_700.json` (800 questions: 400 trivia + 400 math)

## Run baseline + Gnosis scores (Pass 1, resumable)

Use `run_baseline.py`. **Save checkpoints to Google Drive** so Colab disconnects don't wipe progress:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["BASELINE_PATH"] = "/content/drive/MyDrive/gate_phase3_baseline.json"

%cd /content/GATE
!pip uninstall -y transformers -q && pip install -e /content/GATE/Gnosis/transformers -q
!PYTHONPATH=/content/GATE/Gnosis python phase_3/run_baseline.py
```

Full copy-paste cells: [COLAB.md](./COLAB.md)

### Paper-facing configuration

Build a stratified calibration/validation/test manifest **before** generation,
then pass it with the versioned configuration. This run records token
log-probability, entropy, and five-sample self-consistency by default:

```bash
python scripts/build_splits.py \
  --input phase_3/artifacts/questions_700.json \
  --output phase_3/artifacts/splits.json

CONFIG_PATH=configs/phase3_research.json \
SPLITS_PATH=phase_3/artifacts/splits.json \
PYTHONPATH=Gnosis python phase_3/run_baseline.py
```

## If you see long JSON with LaTeX / `[asy]` blocks

That is **normal**, not an error. Competition-math prompts include diagrams and long problem text.

## AUROC (no GPU, after baseline completes)

```python
%cd /content/GATE
!BASELINE_PATH=/content/drive/MyDrive/gate_phase3_baseline.json python phase_3/auroc_analysis.py
```

Or if the file is in `phase_3/artifacts/`:

```bash
python phase_3/auroc_analysis.py
```

Output: `phase_3/artifacts/auroc.json` (overall + trivia/math breakdown, Phase 2 comparison)

## Threshold sweep (no GPU)

```bash
BASELINE_PATH=/content/drive/MyDrive/gate_phase3_baseline.json python phase_3/threshold_sweep.py
```

## Bootstrap CIs (no GPU)

```bash
BASELINE_PATH=/content/drive/MyDrive/gate_phase3_baseline.json python phase_3/bootstrap_analysis.py
```

Output: `phase_3/artifacts/bootstrap.json` (AUROC + threshold sweep 95% CIs)

## Critique-and-revise ablation — trivia, no RAG (GPU)

The default protocol supplies the baseline answer for critique and revision.
Gnosis-gated vs random-matched control. Resumes only compatible checkpoints.
Set `INTERVENTION_PROTOCOL=independent_resample_v1` only to reproduce the
legacy independent-resampling condition.

With `CONFIG_PATH=configs/phase3_research.json`, five seeded random arms are
run and saved separately. This estimates the random-control variance instead
of relying on one favorable seed.

```python
import os
os.environ["BASELINE_PATH"] = "/content/drive/MyDrive/gate_phase3_baseline.json"
os.environ["QUESTIONS_PATH"] = "/content/GATE/phase_3/artifacts/questions_700.json"
os.environ["THRESHOLD"] = "0.50"   # or 0.60

%cd /content/GATE
!PYTHONPATH=/content/GATE/Gnosis python phase_3/regen_ablation.py
```

Outputs:
- `phase_3/artifacts/regen_gnosis_results.json`
- `phase_3/artifacts/regen_random_results.json`
- `phase_3/artifacts/regen_comparison.json`

Resume partial runs: re-run the same command. Skip a finished arm with `SKIP_GNOSIS=1` or `SKIP_RANDOM=1`.

## Grading / aliases

`eval_utils.enrich_records()` and `grade_record()` attach TriviaQA `answer_aliases` when scoring regen answers. Baseline records now store `answer_aliases` when present in the question set.

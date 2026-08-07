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

## If you see long JSON with LaTeX / `[asy]` blocks

That is **normal**, not an error. Competition-math prompts include diagrams and long problem text.

## AUROC (no GPU, after baseline completes)

```python
BASELINE_PATH=/content/drive/MyDrive/gate_phase3_baseline.json python phase_2/auroc_analysis.py
```

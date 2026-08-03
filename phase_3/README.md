# Phase 3 — Scale up evaluation

## Build 700-question set (Colab, no GPU)

```python
%cd /content/GATE
!git pull
!pip install -q datasets
!python phase_3/build_question_set.py
```

Output: `phase_3/artifacts/questions_700.json`

Trivia rows include `answer_aliases`; `sample.py` uses them in `eval_utils.is_correct()`.

## If you see long JSON with LaTeX / `[asy]` blocks

That is **normal**, not an error. Competition-math prompts include diagrams and long problem text. The script prints a short preview at the end.

## Run baseline + Gnosis scores (Pass 1 only, resumable)

Use `run_baseline.py` instead of `sample.py` for Phase 3. It saves after **every** question and resumes if Colab dies.

```python
%cd /content/GATE
!pip uninstall -y transformers && pip install -e /content/GATE/Gnosis/transformers
!PYTHONPATH=/content/GATE/Gnosis python phase_3/run_baseline.py
```

Output: `phase_3/artifacts/baseline_results.json` (same format as Phase 2 `baseline_results.json`).

**Resume:** re-run the same command — it picks up from the checkpoint automatically.

**Start over:** `RERUN_BASELINE=1 python phase_3/run_baseline.py`

**Copy checkpoint to Drive** after each session so you can restore if the runtime is recycled:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp phase_3/artifacts/baseline_results.json /content/drive/MyDrive/gate_phase3_baseline.json
```

## AUROC (no GPU, after baseline completes)

```python
BASELINE_PATH=phase_3/artifacts/baseline_results.json python phase_2/auroc_analysis.py
```

## Commit artifacts

```bash
git add phase_3/
git commit -m "Add Phase 3 question builder and scaled dataset"
git push
```

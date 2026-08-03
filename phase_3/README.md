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

## Run evaluation on the new set

```python
!cp phase_3/artifacts/questions_700.json questions.json
!PYTHONPATH=/content/GATE/Gnosis python sample.py
```

## Commit artifacts

```bash
git add phase_3/
git commit -m "Add Phase 3 question builder and scaled dataset"
git push
```

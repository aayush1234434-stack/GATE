# Phase 3 on Google Colab (copy-paste guide)

Use this when Colab disconnects mid-run. Checkpoints go to **Google Drive**, so progress survives runtime resets.

**Repo:** https://github.com/aayush1234434-stack/GATE

---

## One-time: open Colab

1. Go to https://colab.research.google.com
2. **File → New notebook**
3. **Runtime → Change runtime type → GPU** (T4 is fine)
4. Run the cells below **in order**

---

## Cell 1 — Mount Google Drive (every new session)

```python
from google.colab import drive
drive.mount('/content/drive')
```

Approve the Google permission prompt.

Your checkpoint will live at:
`/content/drive/MyDrive/gate_phase3_baseline.json`

---

## Cell 2 — Clone or update the repo (every new session)

```python
import os

if not os.path.isdir("/content/GATE"):
    !git clone --recurse-submodules https://github.com/aayush1234434-stack/GATE.git /content/GATE
else:
    %cd /content/GATE
    !git pull
```

If `Gnosis` submodule is empty after clone:

```python
%cd /content/GATE
!git submodule update --init --recursive
```

---

## Cell 3 — Build questions (only if you don't have them yet)

Skip this if `questions_700.json` already exists on GitHub or in the repo.

```python
%cd /content/GATE
!pip install -q datasets
!python phase_3/build_question_set.py
```

Output: `phase_3/artifacts/questions_700.json` (800 questions)

---

## Cell 4 — Install Gnosis Transformers fork (every new session)

```python
%cd /content/GATE
!pip uninstall -y transformers -q
!pip install -e /content/GATE/Gnosis/transformers -q
```

---

## Cell 5 — Check checkpoint progress (optional)

```python
import json, os

CHECKPOINT = "/content/drive/MyDrive/gate_phase3_baseline.json"

if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        data = json.load(f)
    correct = sum(r["baseline_correct"] for r in data)
    print(f"Checkpoint found: {len(data)}/800 done ({correct} correct)")
else:
    print("No checkpoint yet — starting from question 1")
```

---

## Cell 6 — Run baseline (main job)

**Important:** `BASELINE_PATH` must point to Drive, not `/content/`.

```python
import os

os.environ["BASELINE_PATH"] = "/content/drive/MyDrive/gate_phase3_baseline.json"

%cd /content/GATE
!PYTHONPATH=/content/GATE/Gnosis python phase_3/run_baseline.py
```

What happens:
- Saves after **every** question to Drive
- If Colab disconnects, re-run **Cells 1, 2, 4, 6** — it resumes automatically
- Expect **many sessions** (1–4 hrs each on free Colab); that's normal

---

## Cell 7 — Backup checkpoint (run after each session)

```python
!cp /content/drive/MyDrive/gate_phase3_baseline.json \
     /content/drive/MyDrive/gate_phase3_baseline_backup.json

import json
with open("/content/drive/MyDrive/gate_phase3_baseline.json") as f:
    print(f"Backup OK — {len(json.load(f))} records")
```

---

## After all 800 questions finish

### AUROC (no GPU)

```python
%cd /content/GATE
import os
os.environ["BASELINE_PATH"] = "/content/drive/MyDrive/gate_phase3_baseline.json"
!BASELINE_PATH=/content/drive/MyDrive/gate_phase3_baseline.json python phase_2/auroc_analysis.py
```

### Download results to your laptop

```python
from google.colab import files
files.download("/content/drive/MyDrive/gate_phase3_baseline.json")
```

---

## Start over from scratch

```python
import os
os.remove("/content/drive/MyDrive/gate_phase3_baseline.json")  # only if you mean it!
os.environ["RERUN_BASELINE"] = "1"
os.environ["BASELINE_PATH"] = "/content/drive/MyDrive/gate_phase3_baseline.json"
%cd /content/GATE
!PYTHONPATH=/content/GATE/Gnosis python phase_3/run_baseline.py
```

---

## Quick reference — what to re-run when Colab dies

| Cell | Re-run after disconnect? |
|------|--------------------------|
| 1 Mount Drive | Yes |
| 2 Clone/pull repo | Yes |
| 3 Build questions | No (unless file missing) |
| 4 Install transformers | Yes |
| 5 Check progress | Optional |
| 6 Run baseline | Yes (resumes) |
| 7 Backup | After session ends |

---

## Troubleshooting

**`Questions file not found`**
→ Run Cell 3, or `!ls /content/GATE/phase_3/artifacts/`

**`missing the Gnosis correctness head`**
→ Re-run Cell 4 (custom Transformers fork)

**`No GPU`**
→ Runtime → Change runtime type → GPU, then re-run from Cell 4

**Progress stuck at same number**
→ Check Drive path: `!ls -lh /content/drive/MyDrive/gate_phase3*`

**Session keeps dying**
→ Keep tab open; run in shorter bursts — resume is safe as long as Drive checkpoint exists

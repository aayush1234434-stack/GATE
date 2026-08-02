# Committed experiment snapshots

Put frozen JSON outputs here so they survive Colab restarts and can be pushed to GitHub.

## What to copy after a run

| File | Source | Required for |
|------|--------|----------------|
| `baseline_results.json` | `/content/GATE/baseline_results.json` | Re-run random arm without Pass 1 |
| `gnosis_results.json` | `/content/GATE/results.json` | Gnosis vs Random comparison |
| `random_results.json` | `/content/GATE/phase_2/random_results.json` | Random arm record |
| `comparison.json` | `/content/GATE/phase_2/comparison.json` | Side-by-side metrics |
| `random_picked_ids.json` | `/content/GATE/phase_2/random_picked_ids.json` | Which questions were picked (seed 42) |

## Colab one-liner (after experiments finish)

```python
!mkdir -p /content/GATE/phase_2/artifacts
!cp /content/GATE/baseline_results.json /content/GATE/phase_2/artifacts/ 2>/dev/null || true
!cp /content/GATE/results.json /content/GATE/phase_2/artifacts/gnosis_results.json
!cp /content/GATE/phase_2/random_results.json /content/GATE/phase_2/artifacts/ 2>/dev/null || true
!cp /content/GATE/phase_2/comparison.json /content/GATE/phase_2/artifacts/ 2>/dev/null || true
!cp /content/GATE/phase_2/random_picked_ids.json /content/GATE/phase_2/artifacts/ 2>/dev/null || true
```

Then `git add phase_2/artifacts/` and commit.

## Re-run Phase 2 random only

If you have `baseline_results.json` OR `gnosis_results.json` here (or at repo root as `results.json`):

```bash
PYTHONPATH=/content/GATE/Gnosis python phase_2/random_baseline.py
```

## Oracle & comparison (no GPU)

After artifacts are saved, run:

```bash
python phase_2/oracle_analysis.py
```

Writes `oracle_analysis.json` with detection (Gnosis vs Random vs Oracle) and outcome tables.

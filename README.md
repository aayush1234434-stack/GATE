# GATE: Gnosis-Assisted Targeted Evaluation

GATE evaluates whether a model's internal Gnosis correctness score can identify
answers worth revising. The project treats this as a **selective prediction**
experiment: a gate selects a fixed intervention budget, then a second pass
either critiques the original answer or independently resamples it.

The committed Phase 2 artifacts are a small pilot, not conclusive evidence.
Phase 3 provides a larger, resumable evaluation workflow.

## Reproducible setup

Clone with the required pinned Gnosis submodule:

```bash
git clone --recurse-submodules https://github.com/aayush1234434-stack/GATE.git
cd GATE
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Create a Python 3.11 environment. Install the PyTorch build appropriate for
your CUDA platform first, then install the project dependencies:

```bash
python -m pip install --upgrade pip
# Install PyTorch for your platform: https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
python -m pytest
```

`requirements.txt` installs the repository-pinned Gnosis Transformers fork.
Full generation experiments are intended for CUDA GPUs. Set `DEVICE=cuda` to
fail fast when no GPU is available, or `DEVICE=cpu` for small smoke tests.

## Experiment protocol

The default `INTERVENTION_PROTOCOL=critique_revision_v1` passes the baseline
answer back to the model and asks it to audit and revise it. This is genuine
revision. The legacy `independent_resample_v1` mode is available only to
reproduce earlier pilots; report it as an independent second sample, not as
regeneration.

Every result record stores `intervention_protocol`, and intervention checkpoints
are resumed only when the baseline data, selected records, selection label, and
protocol agree. Use a new output path when changing an experimental condition.

The committed Phase 2 pilot used a legacy substring grader. Regrade it before
making new comparisons:

```bash
python scripts/regrade_results.py \
  --input phase_2/artifacts/baseline_results.json \
  --output phase_2/artifacts/baseline_results_regraded.json \
  --questions questions.json
```

## Run the project

Single-question smoke test:

```bash
DEVICE=cuda PYTHONPATH=Gnosis python main.py
```

Small baseline plus selective revision:

```bash
DEVICE=cuda PYTHONPATH=Gnosis python sample.py
```

Phase 2 evaluates a Gnosis gate against a matched random control. See
[phase_2/README.md](phase_2/README.md).

Phase 3 builds a larger question set, runs a resumable baseline, and reports
AUROC, threshold sweeps, bootstrap intervals, and a matched revision ablation.
See [phase_3/README.md](phase_3/README.md) and
[phase_3/COLAB.md](phase_3/COLAB.md).

## Research experiment workflow

Use the versioned [Phase 3 configuration](configs/phase3_research.json) for
paper-facing runs. Create splits from questions *before* generating answers:

```bash
python scripts/build_splits.py \
  --input phase_3/artifacts/questions_700.json \
  --output phase_3/artifacts/splits.json

CONFIG_PATH=configs/phase3_research.json \
SPLITS_PATH=phase_3/artifacts/splits.json \
DEVICE=cuda PYTHONPATH=Gnosis python phase_3/run_baseline.py
```

The research config requests five answer samples per question, recording token
log-probability, token entropy, and exact-answer self-consistency. Add
embedding-based semantic agreement with the research extras:

```bash
python -m pip install -r requirements-research.txt
python scripts/score_semantic_agreement.py \
  --input phase_3/artifacts/baseline_results.json \
  --output phase_3/artifacts/baseline_with_semantics.json
```

An independent OpenAI-compatible verifier is available in
`scripts/run_external_verifier.py`. It is deliberately opt-in because it sends
network requests and may incur provider costs. Evaluate every available
baseline without contaminating the final test split:

```bash
python scripts/evaluate_baselines.py \
  --records phase_3/artifacts/baseline_with_semantics.json \
  --splits phase_3/artifacts/splits.json \
  --output phase_3/artifacts/baseline_evaluation.json
```

For matched random controls, the same config defines five random seeds. Run
`phase_3/regen_ablation.py` with `CONFIG_PATH=configs/phase3_research.json`;
each seed is saved separately and compared to the single Gnosis-gated arm.

## Capture provenance

Save this alongside every generated result file:

```bash
python scripts/capture_environment.py \
  --output phase_3/artifacts/environment.json
```

It records Python, package versions, platform, the repository commit, and
submodule revision.

## Research reporting notes

Report a fixed intervention budget, precision/recall of selected answers,
fixed versus broken answers, final accuracy, and confidence intervals. Tune
thresholds on development data and reserve test data for the final report.
The Phase 3 math source is not guaranteed to be a clean holdout; retain that
limitation in any paper.

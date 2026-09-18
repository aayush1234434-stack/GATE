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

Use the versioned [benchmark configuration](configs/benchmark_v1.json) and
[Phase 3 configuration](configs/phase3_research.json) for paper-facing runs.
The benchmark has 1,800 predeclared questions across TriviaQA, MATH, ARC-
Challenge, and MMLU, with source split and license provenance on every row.
Build it and create splits *before* generating answers:

```bash
python phase_3/build_question_set.py \
  --config configs/benchmark_v1.json \
  --output phase_3/artifacts/questions_v1.json

python scripts/build_splits.py \
  --input phase_3/artifacts/questions_v1.json \
  --output phase_3/artifacts/splits.json

CONFIG_PATH=configs/phase3_research.json \
SPLITS_PATH=phase_3/artifacts/splits.json \
QUESTIONS_PATH=phase_3/artifacts/questions_v1.json \
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

Confirm the predeclared error target before treating the run as the primary
analysis (the default is at least 150 errors overall and 20 per domain):

```bash
python scripts/check_error_target.py \
  --records phase_3/artifacts/baseline_with_semantics.json \
  --output phase_3/artifacts/error_target.json
```

After the intervention arms are complete, generate paper-facing confidence
intervals, AUROC/AUPRC, reliability bins/Brier score, risk--coverage curves,
paired McNemar tests, and observed generation-cost estimates:

```bash
python scripts/analyze_experiment.py \
  --baseline phase_3/artifacts/baseline_with_semantics.json \
  --split test \
  --gnosis phase_3/artifacts/regen_gnosis_results.json \
  --random phase_3/artifacts/regen_random_seed_11.json \
  --random phase_3/artifacts/regen_random_seed_23.json \
  --output phase_3/artifacts/statistical_analysis.json
```

To evaluate more than one independently verified Gnosis-compatible checkpoint,
add it to `models` in the research config, then inspect the generated commands:

```bash
PYTHONPATH=Gnosis python scripts/run_model_matrix.py \
  --questions phase_3/artifacts/questions_v1.json \
  --splits phase_3/artifacts/splits.json \
  --output-dir phase_3/artifacts/models
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
fixed versus broken answers, final accuracy, confidence intervals, and
compute assumptions. Tune thresholds on development data and reserve test data
for the final report. The cost report measures observed generation tokens and
wall time; it explicitly excludes the correctness-score forward-pass cost.

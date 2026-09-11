# JEPA-WM: Geometry, Forecasts & Planning

**Do structured changes inside a frozen world model improve the decisions it makes?**

A PyTorch research codebase for fitting activation interventions, measuring their
forecast effects, and testing them in the JEPA-WM planner with paired scenarios
and matched-random controls. Base-model weights remain frozen during intervention
fitting and evaluation.

[Key findings](reports/KEY_FINDINGS.md) ·
[Behavioral results](reports/CORE_METAWORLD_BEHAVIORAL_RESULTS.md) ·
[Experiment design](docs/EXPERIMENT_PLAN.md) ·
[Reproduce](#reproduce-and-restore) ·
[Scaling notes](jax_scaling_notes.md)

> **Result in brief:** several interventions improve offline prediction error.
> The completed closed-loop comparison does **not establish improved task
> success** over both unsteered and matched-random controls. The intervals remain
> wide: this is inconclusive behavioral evidence, not proof of zero effect.

**Stack:** Python 3.10+ · PyTorch · NumPy · upstream JEPA-WM planning environments.
This project uses PyTorch, not JAX; the Scaling Book informs the compute design.

## Evidence at a glance

Status reconciled **September 11, 2026**. These are different measurements, not
interchangeable sample counts.

| Completed evidence | Coverage | What it measures |
|---|---|---|
| Corrected primary offline sweeps | 81 trajectories across Reach, Reach-Wall and Push-T; BF16 + FP32 | Recorded-action forecast error |
| Core closed-loop panel | 960 episodes: 2 tasks × 5 conditions × 96 paired scenarios | Full-episode simulated task success |
| Planner-decision diagnostic | 192 scenarios; 5 conditions on a shared candidate bank | Initial action selection and short-prefix physical progress |
| DROID coupling panel | 576 endpoints: 9 conditions × 64 endpoints from 15 recordings | Recorded-plan action score, **not** robot task success |

The primary offline counts are **33 Reach, 27 Reach-Wall and 21 Push-T
trajectories**. Each arm covers 10,590 H6 prediction prefixes; prefixes do not
increase the number of independent trajectories. Likewise, the 960 closed-loop
episodes contain 96 paired scenarios per task, not 960 independent scenarios.

## Results

### Closed-loop task success

The same initial/goal scenarios and registered planner randomness are paired
across conditions. Each cell is **successes / 96 episodes**, using one released
MetaWorld checkpoint and full 100-step episodes.

| Condition | Reach | Reach-Wall |
|---|---:|---:|
| Unsteered | 43 / 96 | 29 / 96 |
| Refined fixed-response edit | 49 / 96 | 30 / 96 |
| Matched-random fixed-response control | 48 / 96 | 23 / 96 |
| Vision–action coupling | 47 / 96 | 25 / 96 |
| Matched-random coupling control | 58 / 96 | 27 / 96 |

On Reach, the refined edit improves the point estimate by **6.25 percentage
points** over unsteered, but only **1.04 points** over its random control. Its
paired comparison with unsteered contains **23 rescues and 17 regressions**.
All eight simultaneous confidence intervals include zero; neither learned
method passes the frozen improvement criteria against both required controls.
The highest observed Reach rate belongs to a random control, not a validated
new method selected after the fact.

[Full effects, confidence intervals and analysis protocol](reports/CORE_METAWORLD_BEHAVIORAL_RESULTS.md)
· [Machine-readable analysis](reports/wm-approaches/core-analysis.json)

### Offline forecast improvements

These are reductions in **H6 proprioceptive embedding MSE**, not improvements in
physical task success. BF16 is primary; FP32 is a separately reported sensitivity
check. Original and replacement operators are distinct experiments.

| Task and intervention | Error reduction vs unsteered | Simultaneous 95% interval |
|---|---:|---:|
| Reach — refined fixed-response edit | 2.360% | [1.966%, 2.755%] |
| Reach-Wall — refined fixed-response edit | 2.192% | [1.798%, 2.587%] |
| Reach — original combined edit | 4.824% | [3.668%, 5.980%] |
| Push-T — original joint coupling | −0.638% | [−1.214%, −0.062%] |

The original combined edit beat its matched-random control and both
component-removal arms on the forecast endpoint. Its **4.824% result does not
belong to the cheaper replacement or to a completed combined behavioral trial**.
The Push-T row is a measured worsening of the forecast metric.

Intervals use the frozen contrast family within each analysis scope, not a new
correction over this selected summary table. Offline pools were used for
development/replication; they are not untouched confirmation.

[Corrected offline results](reports/CORRECTED_OFFLINE_RESULTS.md) ·
[Replacement operator](docs/FIXED_RESPONSE_RANK4.md) ·
[Combined/drop-one experiment](docs/COMBINED_DEVELOPMENT.md)

### What changes in the planner?

On the shared first candidate population, the refined edit changed the chosen
action sequence in **1 / 192 scenarios**, and coupling in **3 / 192**. None of
the eight adjusted comparisons established improved short-prefix physical
progress.

This diagnostic executes the selected first 15 elementary actions **without
replanning**. It does not measure the final CEM solution, explain every
full-rollout outcome, or supply independent confirmation.

[Planner-decision results](reports/PLANNER_DECISION_DIAGNOSTIC_RESULTS.md) ·
[Ten findings and their limits](reports/KEY_FINDINGS.md)

## Methods and coverage

Two learned interventions have complete matched MetaWorld behavioral comparisons:

- **Refined fixed-response edit:** four fitted directions span the activation
  field at one predictor block/time site. A precomputed response map applies the
  correction in the existing batched forward pass, with **zero online response
  probes and zero separate native-shadow forecasts**. This replaces, rather than
  numerically reproduces, the original expensive operator.
- **Vision–action coupling:** fitted visual/action-conditioning directions,
  evaluated with pathway, energy-budget, permutation and random controls.

Broader offline ablations examine action-response geometry, operator rank,
layer distribution and spatial support. Temporal/HMM routing and combined
successors have implementation or partial evidence, **not completed behavioral
efficacy comparisons**.

| Task | Verified coverage and boundary |
|---|---|
| Reach / Reach-Wall | Corrected offline sweeps, complete core closed-loop panel and decision diagnostic |
| Push-T | Corrected primary offline sweeps; baseline planning and partial intervention records, not a complete matched behavioral panel |
| Wall / PointMaze | Offline coupling and geometry comparisons; baseline/partial behavioral records, not a complete factorial behavioral analysis |
| DROID | Complete coupling recorded-plan panel; no established score improvement and no physical robot execution |

**Six tasks are in scope; the full six-task study is not complete.** RoboCasa is
excluded. No complete three-training-seed/late-checkpoint comparison or fresh
confirmation result is established. Task-specific fitting is not intervention
transfer or proof of recipe generalization.

[Detailed ablation inventory](reports/ABLATION_COVERAGE_STATUS.md)
contains dated historical snapshots; use the completed result reports above and
the latest [execution status](reports/EXECUTION_STATUS.md) for subsequent changes.

## Reproduce and restore

### Inspect the evidence

Start with the [core analysis](reports/wm-approaches/core-analysis.json) and its
[independent verification receipt](reports/wm-approaches/core-verification.json).
The experiment contract is [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md), with
machine-readable settings in [study.json](configs/study.json).

Large datasets, checkpoints, episode traces and private recovery archives are
**not bundled with this repository**. The [preservation and restoration guide](reports/VAST_FINAL_STORAGE_RELEASE_20260910.md#evidence-and-recovery)
records object IDs, hashes and recovery instructions; newer diagnostic and pause
archives are indexed in [execution status](reports/EXECUTION_STATUS.md).
Repository access does not grant access to private Drive archives.

### Run a CPU harness check

```bash
git clone https://github.com/stevenybuilder/jepa_steering.git
cd jepa_steering
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
jepa-benchmark --backend toy --device cpu --output runs/cpu-smoke
```

The toy backend checks the pipeline only: **it is not JEPA-WM, and its errors or
speed are not scientific results**. Real model evaluation requires the pinned
upstream source, official inputs and environment-specific dependencies in the
[benchmark runbook](docs/BENCHMARK.md).

After restoring the documented dependencies and fixtures:

```bash
python -m unittest discover -s tests -v
```

The full suite includes upstream-source alignment and private recovery-fixture
checks. A source-only checkout is not a complete integration-test environment.
Historical launchers and completion flags do not authorize new paid runs.

## Repository guide

```text
src/offline_study/   Fitting, interventions, planners, paired analyses and verification
configs/            Machine-readable study and frozen-method settings
tests/              Numerical, coverage, pairing and lifecycle checks
reports/            Dated results, evidence audits and recovery guides
docs/               Experiment design, method contracts and runbooks
scripts/vast/       Guarded compute orchestration and preservation tooling
archive/            Historical workspace; not the active protocol
```

- **Scientific design:** [plan](docs/EXPERIMENT_PLAN.md), [behavioral amendment](docs/BEHAVIORAL_EVALUATION_AMENDMENT.md), [author alignment](reports/PLANNING_METHOD_ALIGNMENT.md).
- **Data integrity:** [Push-T lineage](reports/PUSHT_LINEAGE_CORRECTION.md), [MetaWorld lineage](reports/METAWORLD_LINEAGE_CORRECTION.md), [goal-image repair](reports/METAWORLD_STIMULUS_REPAIR.md).
- **Compute lessons:** [Scaling Book notes](jax_scaling_notes.md), [GPU execution guide](docs/GPU_EFFICIENCY.md).
- **Context:** [methods and interpretation](wm-approaches.md), [archive provenance](docs/ARCHIVE.md).

## Reference and project status

This is an independent intervention study built around
[JEPA-WM](https://arxiv.org/abs/2512.24497v4), not the authors' official implementation
or a claim to reproduce their complete training-history results.
Official resources: [code](https://github.com/facebookresearch/jepa-wms),
[checkpoints](https://huggingface.co/facebook/jepa-wms),
[datasets](https://huggingface.co/datasets/facebook/jepa-wms).

**Experiments are paused.** The latest recorded shutdown verified preservation
of the completed results and the interrupted native-only confirmation records
before stopping the new GPU fleet. Retained disks can still incur storage fees;
this README is not a live billing monitor. See [execution status](reports/EXECUTION_STATUS.md).

Active project work uses `main`. Historical branches, untracked drafts and
archived plans are not additional verified findings.

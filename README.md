# Steering Predictions and Plans in JEPA World Models

**When does changing a predicted future change the plan?**

JEPA-WM plans by predicting what candidate actions will lead to, then choosing
among those futures. We keep its checkpoints frozen and test **activation
steering inside the predictor**: learned low-rank corrections, visual and
action-conditioning edits, matched random controls, and component ablations.
Representational geometry motivates the edits; forecasting and planning are the
outcomes we follow.

Three findings explain how steering affects this world model:

- **More accurate forecasts do not necessarily produce better plans.** Our learned
  correction improves some predictions, but we have not established better task
  success. We test prediction accuracy and robot outcomes separately.
- **An edit can change how the planner searches.** Even when its first choice
  stays the same, changes to other candidate scores can lead later search steps
  toward different action sequences. Both learned and random edits have this effect.
- **Numerical precision can change the geometry we measure.** The same model
  appears more linear under lower-precision arithmetic. We check this before
  interpreting the shape of its internal representations.

[LCFM paper](paper/lcfm/main.pdf) · [Full study](paper/workshop/main.pdf) · [Methods](docs/METHODS.md) · [All results](docs/RESULTS.md) · [Reproduce](docs/REPRODUCING.md)

## Model and interventions

The diagram shows the MetaWorld model: frozen encoders, a six-block predictor,
and a fixed planning objective.
Interventions enter at imagined step **H3** of an **H6** forecast. CEM scores
300 candidate action sequences against the encoded goal, refits its proposal
from ten elites, and repeats before executing a prefix and replanning.

![Frozen encoders, six-block predictor, goal scoring and CEM, with alternative activation-edit sites.](docs/figures/architecture_readable.png)

### Scaling the experiments in PyTorch

We used **five model configurations across six tasks**, from **39.7M to 532.0M
parameters per model**. Reach and Reach-Wall share the MetaWorld checkpoint.

| Model / tasks | Visual encoder | Encoder | Predictor | Total parameters |
|---|---|---:|---:|---:|
| MetaWorld: Reach, Reach-Wall | DINOv2 ViT-S/14 | 22.1M | 17.6M · 6 blocks | **39.7M** |
| Push-T | DINOv2 ViT-S/14 | 22.1M | 17.6M · 6 blocks | **39.7M** |
| PointMaze | DINOv2 ViT-S/14 | 22.1M | 17.6M · 6 blocks | **39.7M** |
| Wall | DINOv2 ViT-S/14 | 22.1M | 17.6M · 6 blocks | **39.7M** |
| DROID | DINOv3 ViT-L/16 | 303.2M | 228.8M · 12 blocks | **532.0M** |

The five complete configurations sum to **690.7M parameters**, or **624.6M**
with the shared DINOv2 encoder counted once. Totals include proprioceptive
embeddings; repeated intervention arms and GPU replicas do not add model size.

We scaled inference across a fleet of **56 GPUs on eight concurrent qualified
hosts**, distributing independent scenarios while keeping each scenario's eight
intervention arms on the same GPU.

The [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
and [JAX Scaling Book](https://jax-ml.github.io/scaling-book/gpus/) guided our
approach to batching, memory traffic, and parallel execution:

- **Batch the matrix work.** Each GPU forecasts 300 candidate action sequences
  together. The rank-four edit uses batched projections and a precomputed 4×4
  coefficient map, replacing repeated online response probes with thin matrix
  multiplications over the 256×400 activation field.
- **Keep workers supplied.** Parallel range downloads stage weights and data
  locally; static queues distribute scenarios, and asynchronous cloud backups
  collect completed results while other workers continue.
- **Measure precision trade-offs.** A ten-forecast benchmark measured **1.10× /
  1.13× faster forecasts with TF32 / BF16**. Both changed predictions, so the
  protected evaluation kept strict FP32.

The data pipeline indexed **31,306 trajectory records**: **12,600 MetaWorld**
and **18,706 Push-T**. Pinned inputs totaled **8.02 GB**: **0.74 GB of MetaWorld
state/action records**, **2.79 GB of Push-T**, **2.39 GB of navigation datasets**,
and **2.11 GB across 16 Franka recordings and companions** for DROID evaluation.
External MetaWorld videos are additional to this input subtotal.
The final protected panel contains **3,072 arm evaluations**;
the earlier mechanism studies produced **12.8 GB of compressed output archives**.
The 200-state history replication adds **14,000 batched six-step forecasts**:
**4.2 million candidate forecasts**, with 120,000 candidate sequences evaluated
under 35 conditions.
[Dataset sizes, model counts, and execution details](docs/COMPUTE.md).

### Intervention families

| Family | Intervention | Control or ablation |
|---|---|---|
| Predictor correction (R) | Learned rank-four edit at B3's output | Response-calibrated random subspace; same rank, site, and prescribed dose |
| Visual–action coupling (V+A) | Visual-input and B3 action-conditioning edits, each scaled by 1/√2 | Random directions with the same sites and dose policy |
| Component ablations | Original-dose joint visual and action edits | Visual-only and action-only |

Together with the unsteered checkpoint, these form eight arms. R and V/A are
alternative interventions. Directions and calibration are fitted on fitting
trajectories; recorded development trajectories inform the recipe. The final
four-task confirmation uses new scenarios after that recipe is frozen.
[Operator equations, fitting, and dose definitions](docs/METHODS.md).

## Better forecasts, first choices, and replanning

### The first choice stays the same in 191 of 192 states

We score the **same 300 action sequences** before and after the learned edit.
The lowest-cost plan stays the same in 191 of 192 states.

A simple bound accounts for most of these stable choices. Compare the best
plan's original lead with the largest difference the edit makes between any
two candidates' costs. **If that difference is smaller than the lead, no rival
can overtake the best plan.** This certifies 184 of the 192 cases; it leaves
the other eight undecided, of which one actually changes winner.

![All fixed-bank score perturbations relative to the original winning gap. The line at one separates guaranteed stable winners from cases the bound leaves undecided.](docs/figures/decision_margin_story.png)

Each curve shows how many states fall below a given ratio. Left of 1, the
edit cannot close the winning gap. Right of 1, a changed winner is possible,
but is not guaranteed. All four tested edits and both tasks are shown.

**For a planning objective, prediction error alone leaves a key question open:**
does the correction change which future the planner prefers? This fixed-bank
test answers that question for the initial candidate population.
[Bound, exact counts, and controls](docs/MECHANISMS.md).

### Later search can still produce a different plan

The first winner does not determine the entire CEM search. CEM refits its next
action distribution from ten elite plans, then draws new candidates. Changing
those intermediate selections can alter the later search.

![Proposal distributions diverge across fifteen CEM iterations under learned and random predictor edits.](docs/figures/cem_expansion_search.png)

In a separate 56-state replay, both learned and random edits change the returned
action prefixes. Their six registered comparison intervals include zero, so the
learned edit has no established advantage over the random control on these
search measurements. [All paired results](docs/CEM_EXPANSION.md).

### Even a shared prediction shift can change relative costs

In a 64-context replay, keeping only the correction shared across candidates
reconstructs the full edit's relative cost changes with scores of **0.983 / 0.998**
on Reach / Reach-Wall. A common shift moves different starting predictions
closer to or farther from the same goal. Candidate-specific edit coefficients
are therefore not required to change relative goal distances.

Random-subspace edits show a similar pattern. The replay retains each
component’s original magnitude. [Component replay](docs/PILOT_MECHANISMS.md).

## Layer structure and forecast correction

An earlier rank-one sweep measures every predictor block, both visual and
proprioceptive forecasts, six horizons, and FP32/BF16 execution. Early-block edits
produce larger H6 proprioceptive corrections on Reach and Reach-Wall in both
precisions. Push-T has a different response, so the layer pattern is task-dependent.

![Layer-by-layer BF16 forecast-error changes across six blocks, six horizons, two modalities and three tasks.](docs/figures/layer_mechanism_bfloat16_native.png)

Cells show MSE reduction as a percentage of unsteered error; positive is better.
H1–H2 precede the H3 intervention. Visual and proprioceptive panels use separate
color scales. The B2+B3 and all-six rows retain the multi-block controls.
[All 576 cells, FP32, and random comparisons](docs/MECHANISMS.md#layer-response-map).

The later **rank-four B3 correction** reduces H6 proprioceptive MSE by **2.36%
on Reach and 2.19% on Reach-Wall**. This operator is fitted separately from the
rank-one sweep.

## Numerical geometry and attention

### Local reconstruction depends on numerical precision

We test the geometry of the predictor's response along fixed action perturbations.
With the same weights, inputs, and interpolation coefficients, **FP32 favors cubic
reconstruction at every block, while BF16 favors linear reconstruction**.
Rounding only the FP32 outputs removes much of the cubic advantage but does not
reproduce the full BF16 result.

![Cubic versus linear activation reconstruction under FP32, rounded FP32 outputs, and BF16, across all six blocks.](docs/figures/geometry_control_story.png)

Across 64 contexts, numerical precision changes which interpolation looks
better. A separate five-task interpolation study finds small, mixed forecast benefits.
[Controlled arithmetic](docs/CONTROLLED_GEOMETRY.md) · [Five-task geometry and visual–action interactions](docs/PATHWAY_GEOMETRY.md).

### Attention across layers and heads

![Spatial attention distance for all sixteen heads and six predictor blocks on Reach and Reach-Wall.](docs/figures/paper_pilot_attention.png)

Each cell averages H6 spatial attention distance over 32 development contexts per
task for the fixed zero-action candidate. [All horizons and uncertainty](docs/PILOT_MECHANISMS.md).

## Physical execution and behavioral evaluation

In the **56-context physical replay**, every model forecasts every selected plan,
and each prefix executes for fifteen elementary actions from an identical reset.
The learned edit improves same-action forecast error in both tasks. All eight
physical-distance and encoded-goal-cost intervals include zero under the registered
correction. [All paired effects and the 3×3 model/plan grid](docs/PLANNED_PREFIX_REPLAY.md).

![Paired forecast-error and physical-outcome changes in the same 56 development states.](docs/figures/planned_prefix_effects.png)

The protected evaluation covers **384 fresh scenarios × eight arms = 3,072 runs**
on Reach, Reach-Wall, PointMaze, and Wall. All 48 registered simultaneous contrast
intervals include zero, so this panel does not establish a reliable success gain.
Steering nevertheless changes which scenarios succeed: on Reach, the learned edit
rescues 21 failures and loses 21 successes, leaving both it and unsteered at 52/96.
[All arms and paired uncertainty](docs/RESULTS.md#protected-confirmation).

Push-T and DROID contribute development results; DROID uses recorded-action agreement.
[Completed experiments and remaining evidence gaps](docs/ANALYSIS_COMPLETION.md).

## An action patch can match now and diverge later

To check what an activation patch represents, we change one input action and
use the resulting JEPA-WM forecast as a reference. The action enters the
predictor at H3 and appears again as history at H4.

Replacing its conditioning **only at H3** matches the immediate prediction,
then diverges when the original action returns at H4. Replacing it at **both
appearances** reproduces the changed-input forecast through H6.

![An action patch matches at H3, diverges when the action returns as history, and changes the H6 candidate choice in 42% of Reach and 59% of Reach-Wall states.](docs/figures/lcfm_context_lifetime.png)

The fresh replication uses **200 independent starting states**, 100 per task.
The one-time patch changes the selected candidate in **42/100 Reach states**
and **59/100 Reach-Wall states**, compared with changing the action at the input.
Those choices have **1.81–2.56% higher reference cost on average** across
tasks and candidate banks, including unchanged choices.
[Selection and cost intervals for both banks](docs/figures/lcfm_replication_choices.png).

For interventions in a forecasting model, matching the next prediction can
miss an action’s later influence on planning. This test follows that influence
through the model’s two-frame context; the selected plans are scored but not
executed. All six layers and matched random controls are retained.
[Complete replication, layer heatmap, and uncertainty](docs/LCFM_REPLICATION.md) ·
[Earlier sixteen-state study](docs/ACTION_COUNTERFACTUAL.md) ·
[Focused LCFM paper](paper/lcfm/main.pdf) ·
[LeWM follow-up assessment](docs/LEWM_FOLLOWUP_ASSESSMENT.md).

## JEPA-WM planning in the simulator

![Actual frozen JEPA-WM episodes at simulation speed: Reach and Reach-Wall, unsteered, seed 0. The target is highlighted in green.](docs/media/jepa_tasks_side_by_side.gif)

**Frozen JEPA-WM, with replanning throughout each episode.** Reach reaches the
green target; Reach-Wall does not reach it within the episode limit. Both use
seed 0, fixed before capture. The tasks play side by side at simulation speed,
with a shared camera that keeps the wall and target visible. [1080p video](docs/media/jepa_tasks_side_by_side_hd.mp4) ·
[Actions, state verification, and reproduction](docs/media/README.md).

## Benchmark context

### Robot success and published benchmark context

**Unsteered** means the released checkpoint with no activation edit.
**Best observed edit** means the highest success among all **seven activation-edit
arms**, including randomized comparisons. The winning edit is named in each panel;
all ties are retained. This is a **post hoc comparison**, not one fixed method.

![Taskwise best observed activation edits versus Unsteered and published references; winners and ties are named, including the randomized Reach-Wall winner.](docs/figures/benchmark_readable.png)

Every ablation is eligible; Reach-Wall's winner is **randomized visual–action
coupling: 27.08%**. Error bars show episode standard errors, not uncertainty
adjusted for choosing the best arm.

The protected Reach baseline is **54.17% (52/96)**. Gray published bars use the
authors’ checkpoints and evaluation populations.
[Published source](https://arxiv.org/html/2512.24497v4#S5.T2) ·
[Exact values and error-bar definitions](paper/data/benchmark_comparison_sources.json).

Against concurrent unsteered JEPA-WM, these observed maxima differ by
**0.00 / −2.08 / +2.08 / +2.08 percentage points** on Reach / Reach-Wall /
PointMaze / Wall.
[Paired analysis](docs/RESULTS.md#protected-confirmation).

## Final protected results and earlier benchmarks

<details>
<summary>Full six-task table: published references, development, and protected results</summary>

Simulator success (%); **DROID is an action-agreement score**, not robot success.
Stages are separate populations, not interchangeable baselines. Protected cells
use 96 scenarios per arm; `—` means not run in that stage.

| Stage | Model / intervention | Reach | Reach-Wall | Push-T | PointMaze | Wall | DROID score |
|---|---|---:|---:|---:|---:|---:|---:|
| Published | DINO-WM | 44.80 | 35.10 | 66.00 | 81.60 | 64.10 | 39.40 |
| Published | JEPA-WM recipe, CEM-L2 | 58.20 | 41.60 | 70.20 | 83.90 | 78.80 | 48.20 |
| Published | JEPA-WM recipe, CEM-L1 | 55.10 | 40.80 | 63.40 | 79.70 | 46.70 | 47.20 |
| Published | JEPA-WM final, CEM-L2 | 49.00 | 29.20 | 69.40 | 83.30 | 80.90 | 46.50 |
| Development | Unsteered | 44.79 | 30.21 | 59.38 | 80.21 | 76.04 | 51.10 |
| Development | Refined four-direction | 51.04 | 31.25 | 59.38 | 85.42 | 78.13 | 51.05 |
| Development | Calibrated random subspace | 50.00 | 23.96 | 61.46 | 79.17 | 76.04 | 51.00 |
| Development | Equal-budget coupling | 48.96 | 26.04 | 61.46 | 77.08 | 82.29 | 51.07 |
| Development | Dose-matched random directions | 60.42 | 28.13 | 60.42 | 84.38 | 77.08 | 51.27 |
| Development | Unscaled joint | 52.08 | 29.17 | 61.46 | 79.17 | 78.13 | 50.85 |
| Development | Visual only | 50.00 | 38.54 | 60.42 | 81.25 | 73.96 | 50.83 |
| Development | Action-conditioning only | 42.71 | 36.46 | 58.33 | 86.46 | 76.04 | 51.11 |
| Protected | Unsteered | 54.17 | 29.17 | — | 86.46 | 81.25 | — |
| Protected | Refined four-direction | 54.17 | 20.83 | — | 88.54 | 80.21 | — |
| Protected | Calibrated random subspace | 44.79 | 21.88 | — | 86.46 | 80.21 | — |
| Protected | Equal-budget coupling | 47.92 | 18.75 | — | 84.38 | 83.33 | — |
| Protected | Dose-matched random directions | 46.88 | 27.08 | — | 87.50 | 77.08 | — |
| Protected | Unscaled joint | 52.08 | 22.92 | — | 84.38 | 77.08 | — |
| Protected | Visual only | 54.17 | 22.92 | — | 81.25 | 80.21 | — |
| Protected | Action-conditioning only | 47.92 | 17.71 | — | 81.25 | 78.13 | — |

Published rows use the authors' checkpoints and aggregation, not our paired inputs.
Development MetaWorld joint/visual/action arms use their concurrent unsteered
**45.83 / 29.17** in contrasts; the canonical development row remains unchanged.
Protected contrasts use **only the protected unsteered row**. No historical rates
are substituted for fresh baselines. [Provenance](paper/data/benchmark_comparison_sources.json).

</details>

## What this contributes

The study evaluates a frozen world model at three distinct levels: whether an
internal edit reproduces an intended input change, whether predictions improve,
and whether the planner selects actions with better outcomes. The history test,
score bound, adaptive searches, and physical replays make these questions
separately testable. The layer and precision analyses examine which internal
structures those edits use and how reliably we can interpret them.

Sonia Joseph et al.'s [*Interpreting Physics in Video World Models*](https://arxiv.org/html/2602.07050v1)
provides a reference for the layerwise work. Their **Physics Emergence Zone**
analysis connects physical variables, distributed subspaces, and interventions
in video encoders. We study the action-conditioned predictor and the planning
computations it supports. [Relation to that study, COAST, and Manifold Steering](docs/LITERATURE_MECHANISMS.md).

## Reproduce and inspect

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis]'
python scripts/check_public_results.py
python scripts/check_manuscript.py
python scripts/build_publication_figures.py
```

These checks and figure builds run on CPU from committed aggregates. Full model
replay needs the pinned upstream code, licensed inputs, checkpoints and archived
records; bulk assets are not bundled or publicly downloadable from this repo.
[Complete reproduction instructions](docs/REPRODUCING.md).

`src/` implements interventions and evaluation; `analysis/mechanism/` contains
diagnostics; `tests/` checks them; `paper/data/` and `reports/` preserve the evidence.
Operational logs and bulk assets are excluded. No independent training-seed
replication was performed. This is an independent study, not the authors' official implementation.

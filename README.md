# Steering JEPA World Models

**Can correcting a world model's predictions help a robot choose better actions?**

A world model predicts what will happen after an action. **JEPA-WM** does this in
learned feature vectors, called embeddings: it predicts a representation of the
future image and robot state. A planner compares these predicted futures with a
goal to decide how the robot should move.

We test whether small edits inside a trained JEPA-WM can improve that process,
without retraining the model. This is **activation steering**.

**We improved some forecasts, but did not establish better robot success.**
The useful distinction is between predicting an outcome more accurately and
choosing an action that brings the robot closer to its goal. Our experiments
measure both, and inspect what happens between them.

[Paper](paper/workshop/main.pdf) · [Methods](docs/METHODS.md) · [All results](docs/RESULTS.md) · [Reproduce](docs/REPRODUCING.md)

## How JEPA-WM chooses an action


For each decision, the planner samples **300 possible action sequences**. JEPA-WM
predicts the future for each sequence and scores how close it comes to the goal.
The planner keeps the best ten, samples more plans around them, and repeats. This
is the Cross-Entropy Method (CEM). The robot executes the beginning of the chosen
plan, observes what happened, and plans again.

![Frozen encoders, six-block predictor, goal scoring and CEM, with alternative activation-edit sites.](docs/figures/architecture_readable.png)

We compare the unchanged model with seven edits: a learned four-direction
correction inside the predictor, edits to its visual and action inputs, and
randomized and component controls. The diagram marks these alternative sites
with R, V and A. B0–B5 are the six predictor blocks; H3 is the third imagined step.
[All eight arms and fitting details](docs/METHODS.md).

<details>
<summary>Evaluation stages: offline predictions, earlier behavior, unseen confirmation</summary>

1. **Offline development:** fit on fitting trajectories, then test forecasts on
   separate recorded trajectories. These development measurements informed the recipe.
2. **Behavioral development:** evaluate the released checkpoints and edits across
   Reach, Reach-Wall, Push-T, PointMaze, Wall and DROID.
3. **Protected confirmation:** freeze the eight-arm recipe, then evaluate 96 new
   scenarios per task on Reach, Reach-Wall, PointMaze and Wall. Exposure audits
   found no overlap with the fitting/development sources checked.

Push-T and DROID were **not** rerun in protected confirmation. DROID measures
recorded-action agreement, not closed-loop robot success. These stages are
separate populations; earlier baselines cannot replace concurrent fresh baselines.

</details>

## What did we learn?

### Better forecasts did not give a reliable success gain

On recorded development trajectories, the learned four-direction edit reduced
six-step robot-state embedding error by **2.36% on Reach and 2.19% on Reach-Wall**.
This measures how accurately the model predicts a recorded future.

We also evaluated eight conditions on **384 new scenarios across four tasks**:
3,072 runs in total. None of the 48 registered comparisons established a success
gain under the simultaneous confidence intervals. This leaves small effects
unresolved; it does not prove that steering has no effect.

For a concrete example, unsteered and learned-steering models each succeeded on
**52 of 96 Reach scenarios**. Steering rescued 21 failures and broke 21 successes.
The robot's behavior changed, but the total number of successes stayed the same.
[All arms and paired uncertainty](docs/RESULTS.md#protected-confirmation).

### A more accurate forecast can leave the chosen plan unchanged

We held the same 300 possible action sequences fixed in **192 development states**.
The learned edit changed the lowest-cost plan in **only one state**. In 184 states,
the changes in scores were too small to overcome the original winner's lead.

![Size of score changes relative to the lead of the best plan, for all four edits on Reach and Reach-Wall.](docs/figures/decision_margin_story.png)

Read the plot as a threshold test: **left of 1, the winning plan must stay the
same**. The vertical axis counts the fraction of starting states below each
threshold. A correction can improve a prediction while leaving the planner with
the same choice. Later rounds of search can still change; this plot tests one
fixed set of plans. [Scores and proof](docs/MECHANISMS.md).

### Changed plans did not establish better physical outcomes

The full search adapts its next set of plans to the current scores. In a follow-up
on **56 development states**, both learned and random activation edits changed
the returned plans. We then executed the first 15 actions of each selected plan
from identical simulator resets.


![All paired physical-prefix effects and the twelve registered simultaneous intervals.](docs/figures/planned_prefix_effects.png)

**How to read this figure:** negative values mean improvement over unsteered.
The top row measures prediction error when the actions are held fixed. The lower
rows measure where the robot actually ends up after executing the chosen actions.
The learned edit improves the forecasts in both tasks. All eight physical-outcome
intervals cross zero, so these trials do not establish that either edit brings
the robot closer to the goal. These are short executions, separate from the
full-task success evaluation. [Results and all model/plan forecasts](docs/PLANNED_PREFIX_REPLAY.md).

## Layer and attention maps


The completed analyses and figures are listed in the [experiment inventory](docs/ANALYSIS_COMPLETION.md).
The detailed reports retain every tested arm, uncertainty estimate and limitation.

### Where in the predictor do edits change forecasts?

![Layer-by-layer BF16 forecast-error changes across six blocks, six horizons, two modalities and three tasks.](docs/figures/layer_mechanism_bfloat16_native.png)

**Read the heatmap:** columns follow the forecast into the future; rows identify
which predictor block receives the edit. Positive values mean lower prediction
error relative to unsteered; the image and robot-state panels use different color
scales. Earlier blocks give larger forecast corrections on Reach and Reach-Wall in this
rank-one sweep; Push-T does not show the same pattern. These independently fitted operators differ from the later
rank-four edit. [All 576 cells, FP32 and random controls](docs/MECHANISMS.md#layer-response-map).

### Attention by head and layer

![Spatial attention distance for all sixteen heads and six predictor blocks on Reach and Reach-Wall.](docs/figures/paper_pilot_attention.png)

This map averages 32 development contexts per task at H6 on the fixed zero-action
candidate. Attention distance describes where attention falls; it does not identify
a causal physics circuit. [All horizons and uncertainty](docs/PILOT_MECHANISMS.md).

### Does an internal action edit represent a consistent action?

One action appears in two consecutive input windows. Replacing its internal
representation at both appearances reproduces a changed input action exactly
through six prediction steps. Replacing only the first appearance stops matching
when the old action reappears as history. An action edit needs to remain consistent
with what the model remembers. This was tested on 16 development states and two
sets of candidate plans. [Figure and complete controls](docs/ACTION_COUNTERFACTUAL.md).

### Can numerical precision change a geometry result?

Yes. With the same weights, inputs and action perturbations, cubic interpolation
reconstructs an omitted activation better than linear interpolation in FP32,
but worse in BF16, at all six blocks. This 64-state test shows why an apparent
shape in activation space needs numerical controls before receiving a physical
interpretation. [Figure and controlled comparison](docs/CONTROLLED_GEOMETRY.md).

<details>
<summary>Component tests and complete search diagnostics</summary>

- **Shared versus candidate-specific corrections:** [64-context replay](docs/PILOT_MECHANISMS.md).
  The common component reconstructs the full edit's relative cost change with
  scores **0.983 / 0.998** on Reach / Reach-Wall. Random-subspace edits show the
  same pattern; component energies were not equalized.
- **Adaptive CEM search:** [complete 56-context extension](docs/CEM_EXPANSION.md).
  Both learned and random edits change later search paths and returned plans.
  All six registered learned-versus-random intervals include zero; greater plan
  divergence is not better control. [All paired plans](docs/figures/cem_expansion_prefixes.png)
  · [search curves](docs/figures/cem_expansion_search.png)
  · [entropy](docs/figures/cem_expansion_entropy.png). The original eight cases remain separate.
- **Action-conditioned rankings:** [all-six-layer donor tests](docs/ACTION_CONDITION_SPECIFICITY.md)
  and [action-history/range controls](docs/ACTION_COUNTERFACTUAL.md). Small global
  rank changes can affect the top-ten elite set; input-reachable and off-range
  directions are distinct controls.
- **Visual–action interactions:** [factorial analysis](docs/PATHWAY_GEOMETRY.md).
  Output nonadditivity is separated from the cross-term introduced by squared error.

The original protected runs did not log numerical candidate costs or elite ranks.
These development diagnostics cannot retrospectively recover those missing traces
or establish why aggregate protected success failed to improve.

</details>

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

These are **fresh, protected evaluation scenarios**, not the earlier development
set. That is why the unsteered Reach rate is **54.17% (52/96)** here rather than
the earlier **44.79%**. The gray published bars use different checkpoints and
evaluation populations; they provide context, not paired controls.
[Published source](https://arxiv.org/html/2512.24497v4#S5.T2) ·
[Exact values and error-bar definitions](paper/data/benchmark_comparison_sources.json).

Against concurrent unsteered JEPA-WM, these observed maxima differ by
**0.00 / −2.08 / +2.08 / +2.08 percentage points** on Reach / Reach-Wall /
PointMaze / Wall. Across the full **384 paired scenarios × eight arms = 3,072
evaluations**, we did not establish a reliable overall success improvement.
The internal effects above are findings, not a demonstrated cause of that outcome.
[Paired analysis](docs/RESULTS.md#protected-confirmation).

## Final protected results and earlier benchmarks

<details>
<summary>Full six-task table · published references, development, protected confirmation</summary>

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

## What this suggests for steering world models

Unlike steering a policy's action output, editing JEPA-WM must pass through a
**prediction → goal score → planner → execution** chain. Our results motivate
testing edits against candidate ordering and coherent action histories, rather
than judging them only by activation reconstruction or forecast error.

This is one frozen-checkpoint case study, not evidence that steering cannot work
in world models. We have not discovered a universal physical coordinate system or
proved that these mechanisms caused the protected behavioral results.
[Comparison with COAST, Physics Emergence Zone and Manifold Steering](docs/LITERATURE_MECHANISMS.md).
Observed-transition adaptation, test-time training and longer-memory rollouts are
future experiments, not results reported here.

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

**Engineering:** PyTorch model replicas process independent scenarios; each GPU
batches 300 candidate trajectories, keeps paired comparisons together and archives
completed records to GCS. The Ultra-Scale Playbook and JAX Scaling Book informed
batching and transfer decisions—not a JAX port or distributed-training rewrite.
[GPU execution and limits](docs/COMPUTE.md).

`src/` implements interventions and evaluation; `analysis/mechanism/` contains
diagnostics; `tests/` checks them; `paper/data/` and `reports/` preserve the evidence.
Operational logs and bulk assets are excluded. No independent training-seed
replication was performed. This is an independent study, not the authors' official implementation.

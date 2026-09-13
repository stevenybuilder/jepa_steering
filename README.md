# Steering Latent World Predictions

**What changes between editing a world model's prediction and choosing a robot action?**

We study activation steering inside **frozen JEPA-WM predictors**. The model imagines
the consequences of candidate actions; a planner scores those futures and chooses
what to execute. We edit the predictor's internal activations without retraining
its weights, then follow the effects through this prediction-to-action pathway.

[Paper](paper/workshop/main.pdf) · [Methods](docs/METHODS.md) · [All results](docs/RESULTS.md) · [Reproduce](docs/REPRODUCING.md)

## Three findings

### 1. Changing a forecast need not change the winning plan

Our four-direction edit reduced six-step robot-state embedding error (MSE) by
**2.36% on Reach and 2.19% on Reach-Wall** on recorded development trajectories.
In a separate fixed-candidate diagnostic, the same edit changed the winning
candidate in **only 1 of 192 contexts**.
In 184 of those contexts, the cost changes were provably too small to overcome
the original winner's margin.

![Most activation edits stay below the cost margin needed to change the winning candidate.](docs/figures/decision_margin_story.png)

**Why it matters:** changing predicted embeddings and changing the ordering of
actions are different targets for steering. This test used the same 300 candidates
per comparison; it does not explain an entire adaptive search or robot rollout.
[Scores, proof and all edit families](docs/MECHANISMS.md).

### 2. Action history changes the meaning of an internal edit

In this predictor, one action appears in **two consecutive context windows**.
Replacing its internal condition at both appearances, across all six blocks,
reproduced an actual input-action replacement **exactly through six prediction
steps**. Replacing only its first appearance stopped matching when the original
action returned as history.

![A one-time action-condition patch diverges from a coherent input-action change; patching both appearances remains exact.](docs/figures/action_history_consistency.png)

**Why it matters:** an intervention can change an activation without representing a
consistent alternative action history. This controlled test covered **16 development
contexts and two candidate banks**. Its reconstruction score is not robot success;
no actions were executed in this diagnostic.
[Complete counterfactual study and layer tests](docs/ACTION_COUNTERFACTUAL.md).

### 3. Numerical precision can reverse a geometry result

With the same weights, inputs and action perturbations, cubic interpolation
reconstructed an omitted activation better than linear interpolation in **FP32**,
but worse in **BF16**, at every predictor block. Rounding FP32 outputs removed much
of the advantage without reproducing the full BF16 result.

![Cubic versus linear activation reconstruction reverses between FP32 and BF16 across all six predictor blocks.](docs/figures/geometry_control_story.png)

**Why it matters:** a local reconstruction result is not, by itself, evidence of a
useful physical steering direction. The controlled comparison uses **64 contexts**;
the earlier five-task sweep also found small, mixed forecast benefits after matching
requested edit doses. [Controlled test](docs/CONTROLLED_GEOMETRY.md) ·
[Five-task reconstruction and forecast results](docs/PATHWAY_GEOMETRY.md).

## Does it improve robot success?

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

## Architecture and experiments

The planner uses **CEM (Cross-Entropy Method)**: sample 300 action sequences,
predict their consequences, retain the lowest-cost plans, and repeat. Only a
prefix of the selected plan is executed before observing and replanning.

<details>
<summary>Architecture diagram and the eight intervention arms</summary>

![Frozen encoders, six-block predictor, goal scoring and CEM, with alternative activation-edit sites.](docs/figures/architecture_readable.png)

We compare three intervention families against an **unsteered** checkpoint:

- **Four-direction predictor correction:** a learned rank-four edit at block B3's output, plus a response-calibrated random-subspace comparison.
- **Equal-budget visual–action edits:** edits to the visual input and B3 action conditioning, plus same-dose random directions.
- **Component ablations:** original-dose joint, visual-only and action-conditioning-only edits.

Together these make eight arms. The diagram's R, V and A markers are alternative
sites, not three edits used together. Edits occur at imagined step H3 of an H6
forecast; B0–B5 are zero-indexed blocks. Base weights and the planning objective
stay fixed. Randomized comparisons are **activation edits, not random robot actions**.
[Exact fitting, timing and dose definitions](docs/METHODS.md).

</details>

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

## Further analyses

The detailed reports retain every tested arm, uncertainty estimate and limitation.

<details>
<summary>Layer maps, shared corrections, CEM search, attention and action geometry</summary>

- **Layer-by-layer effects:** [all 576 cells](docs/MECHANISMS.md#layer-response-map)
  and [the heatmap](docs/figures/layer_mechanism_bfloat16_native.png). Interventions
  measure later forecast effects, not attention importance or a unique physics zone.
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
- **Attention by head and layer:** [64-context attention map](docs/figures/paper_pilot_attention.png).
  Spatial attention distance on fixed zero-action inputs is descriptive, not a causal circuit.
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

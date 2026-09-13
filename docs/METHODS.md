# Methods and ablations

## Model and intervention pathway

Frozen JEPA-WM encoders/predictors forecast latent observations conditioned on
candidate actions. CEM scores forecasts against the goal and executes a selected
sequence. We intervene in the **predictor**, not directly in a policy's action output.
No base-model weights are trained.

The refined method keeps four fitted directions at predictor block B3 / imagined
step H3. An offline-calibrated response map converts native activations into four
coefficients, normalizes the edit to the task's dose, and adds it within the
existing forward pass. It acts on H6 forecasts with full 256-patch support and
has no online response probes or separate native-shadow forecasts. This is a
successor to the earlier expensive online operator, not an equivalent implementation.

Vision–action coupling edits visual predictor inputs and action-conditioning
pathways. The unscaled joint factorial and equal-budget efficacy arm are distinct.
Implementation: [fixed_response.py](../src/offline_study/fixed_response.py),
[interventions.py](../src/offline_study/interventions.py),
[fresh_confirmation.py](../src/offline_study/fresh_confirmation.py).

## Fitting, selection and exposure

Fitting trajectories supplied directions, readouts and calibration. Separate
recorded-action development trajectories measured forecast error and informed
choices; fitting-disjoint does not mean protected from intervention selection.
Corrected primary offline evaluation used 33 Reach, 27 Reach-Wall and 21 Push-T
trajectories. The refined recipe was developed on MetaWorld and later received
task-specific fitting/calibration. It is not an unchanged cross-task operator.

The earlier six-task behavioral table is development/replication evidence.
Fresh four-task inputs were generated with the simulators' goal-generation
procedures. Source-family comparisons, cached-source rechecks and historical JSON
checks found no overlap; positive controls verified detection of known overlap.
This protection is relative to the audited project exposure records, not a claim
about every scene in base-model pretraining.

The fresh freeze binds input/operator/checkpoint/source hashes, arms, RNG and
analysis before scientific outcomes. Each task has 96 scientific scenarios plus
8 excluded engineering inputs. The earlier interrupted 117-native-episode run
is a different cohort and is not pooled into this confirmation.

## Ablation stages

| Stage | Comparisons | Question |
|---|---|---|
| Corrected offline sweeps | Visual/action/joint/equal-budget, permutations and random directions | Which pathways/directions change forecast error? |
| Corrected offline sweeps | Linear/cubic/projected/reflected action response | Does modeled response geometry help? |
| Corrected offline sweeps | Rank 1/4/8 with matched subspaces | Does rank matter at controlled energy? |
| Corrected offline sweeps | Patch/group/all-patch and layer-support controls | Where should an edit be distributed? |
| Subsequent offline development | Combined/drop-one; refined fixed-response successor | Component contributions and lower-cost implementation |
| Protected closed loop | Native, refined, calibrated random subspace, equal-budget coupling, random directions, joint, visual, action | Eight fixed arms on fresh scenarios |

Not all earlier sweeps were repeated on the protected cohort. Temporal/HMM methods
and the new attention/candidate-trace pilot are **not completed fresh efficacy
results**. Complete offline exports: [paper/data](../paper/data/).

### Comparator definitions

The refined comparator replaces the four-dimensional basis with a random subspace
on the same support, with its own response calibration. It matches rank,
basis-projector spectrum and dose, not every coefficient-map singular value.
The coupling comparator uses fixed random directions at matching sites and an
equal energy budget. Neither comparator samples random robot actions.

## Final evaluation and statistics

All eight arms of a scenario use one physical GPU and frozen scenario-specific
planner randomness. Native is rerun concurrently; historical native scores are
not substituted. Historical device-equivalence amendments do not replace the
final panel's same-device requirement.

CEM keeps 300 candidates. MetaWorld uses 15 iterations and 100 elementary episode
steps with replanning; navigation uses 30 iterations and 30 elementary steps.
The protected panel uses strict FP32, TF32 off. Receiving checks validate native
repeats, zero-dose identity, hooked/reference forecasts and dose outside science.

The independent unit is the scenario, not a GPU, candidate, timestep or checkpoint.
The prespecified 48-contrast family has 12 contrasts per task:

- Seven non-native arms versus native.
- Refined versus its calibrated random-subspace comparator.
- Equal-budget coupling versus its random-direction comparator.
- Joint minus visual, joint minus action, and joint − visual − action + native.

The estimator uses 20,000 paired-scenario bootstrap draws, seed 2026091221,
and Bonferroni simultaneous 95% percentile intervals across all 48 contrasts.
Historical results are not pooled. Post-confirmation mechanism diagnostics are
exploratory unless explicitly replaying one of these frozen contrasts.

## Context and limits

[JEPA-WM](https://arxiv.org/html/2512.24497v4) supplies the model/planner; its
multi-seed and checkpoint-aggregation results are context, not our paired baseline.
[COAST](https://arxiv.org/html/2605.17144v1) also steers activation geometry, using
contrastive conceptors in a policy action expert. Our predictor-to-planner pathway
differs; that motivates the question but does not explain the different gains.
There is one released checkpoint per task, not independent-training-seed replication.

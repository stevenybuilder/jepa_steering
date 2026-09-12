# Geometry Map Experiment

## Objective

Construct a causal geometry map of the released MetaWorld JEPA-WM before designing a performance intervention. The map must show both where task-relevant information is represented and where the planner actually uses that information. Sonar, HMM routing, conceptors, transport, and other steering recipes are excluded from the discovery stage; the map determines which, if any, is justified.

The primary questions are:

1. Where are goal progress, wall clearance, contact, and action consequences represented?
2. In what coordinate system and geometry are they represented?
3. Which representations causally change predicted futures, candidate-action rankings, and selected actions?
4. What is the simplest repeatable intervention supported by those results?

## Models and tasks

- Primary model: released `jepa_wm_metaworld` checkpoint, unchanged and frozen.
- Primary task: MetaWorld Reach-Wall, for goal progress and obstacle-relative geometry.
- Replication task: Push-T, only after the Reach-Wall map is frozen, for contact and object-motion geometry.
- Runtime baseline: released JEPA-WM checkpoint with its released CEM planner and no activation edit.

The MetaWorld model has two distinct transformer modules:

- Frozen DINOv2 ViT-S/14 visual encoder: 12 transformer blocks.
- Action-conditioned JEPA predictor: 6 transformer blocks, reused at every imagined rollout step.

At 224×224 resolution with 14×14 patches, the visual representation primarily has a 16×16 spatial grid, or 256 spatial patch positions. The analysis index is therefore `module × block × hook × spatial region × imagined timestep`, not merely a single list of 18 layers.

## Data and splits

Use the released JEPA-WM MetaWorld dataset first. It contains three TD-MPC2 collection seeds and 100 trajectories per task per seed, giving 300 Reach-Wall trajectories of 100 steps each.

Split by complete trajectory, never by frame:

- Discovery: 150 trajectories.
- Validation and shortlist selection: 75 trajectories.
- Untouched confirmation: 75 trajectories.

Add 24 complete, unsteered JEPA-WM planner episodes to test whether geometry learned from the offline TD-MPC2 data transfers to states and candidate actions visited by the actual CEM planner. Episodes 0–11 are development replay; episodes 12–23 are the precomputed unsteered half of a held paired closed-loop evaluation and cannot tune the operator. The exact seed rules and cache contract were frozen in [`on_policy_bank_manifest.json`](docs/manifests/geometry-map-reach-wall-v1/on_policy_bank_manifest.json) before collection completed.

From 48–64 saved simulator states spanning task phases, evaluate about six controlled candidate actions per state. This gives roughly 288–384 one-step snapshot forks without requiring complete episodes for every contrast. The independent statistical units are trajectories and saved states, not frames or candidate actions.

## Stimuli and labels

Use continuous, timestep-local simulator quantities rather than assigning one success/failure label to an entire trajectory.

For Reach-Wall, record:

- end-effector-to-goal displacement and distance;
- progress since the previous step and remaining goal distance;
- end-effector and goal position relative to the wall;
- minimum wall clearance and whether a candidate path crosses the wall;
- action direction and magnitude;
- realized and JEPA-predicted next-state displacement;
- CEM candidate cost, rank, and selected action;
- true episode time only as a nuisance/control variable.

Construct matched counterfactual stimuli in which one factor changes while the others remain fixed: same state with different candidate actions, comparable progress with different wall relations, and comparable time with different progress. For Push-T replication, replace wall variables with contact state, pusher-to-object geometry, object pose, and action-conditioned object displacement.

## Activation capture

Cache block outputs once for all 12 encoder blocks and all 6 predictor blocks. Preserve block residual output as the common comparison point, then record attention output and MLP output only for shortlisted blocks. For the predictor, retain the imagined rollout-step index and the candidate-action identity. Record action embeddings, proprioceptive embeddings, predicted future latents, CEM costs, and selected actions so the causal path from action to decision remains observable.

The discovery pass should be broad but cheap:

1. Cache all block residual outputs.
2. Screen the four main variables using cross-validated linear readouts.
3. Compare against small nonlinear readouts to detect substantial curvature missed by linear probes.
4. Measure stability across held-out trajectories and seeds.
5. Shortlist at most three predictor regions and two encoder regions per variable.

Do not launch exhaustive head, neuron, or token patching before this shortlist is frozen.

## Physics-emergence refinement

The [Physics Emergence Zone study](https://arxiv.org/abs/2602.07050) is a useful hypothesis generator, not a layer prior to copy blindly. It studies frozen V-JEPA 2 and VideoMAE video encoders, whereas this experiment primarily intervenes in JEPA-WM's six-block action-conditioned predictor. Its transferable findings are that physical variables can appear abruptly at intermediate depth, direction can be a high-dimensional circular population code, and two physical variables can emerge at the same depth while occupying nearly orthogonal subspaces.

Before steering, add four focused analyses to the cached discovery data:

1. Define a task-geometry emergence curve for each variable from held-seed layerwise performance. Report the earliest sharp increase, the peak, and the near-peak zone rather than assuming “one-third depth.”
2. Decompose realized motion into magnitude and direction. Encode the lateral/vertical direction as `(sin θ, cos θ)` and compare it with scalar displacement magnitude, progress, and wall clearance across all encoder and predictor blocks.
3. Estimate feature-subspace dimensionality with an iterative orthogonal probe sequence: fit a held-seed probe, QR-orthogonalize its weight directions, project them out, and repeat until held-out performance approaches chance. A rank-one intervention is eligible only if one direction exhausts the useful signal.
4. Compare accepted subspaces using principal angles, directional projection overlap, Grassmann distance, and the random-overlap baseline `k/d`. Co-emergence does not imply a shared coordinate system.

Our spatial screen already shows that wall, path, and action-consequence information is broadly distributed across predictor tokens. Therefore any first steering rule should act on an empirically measured multi-feature subspace across all supported tokens, with a matched-orientation sham, rather than inject one probe vector into one token. The later [CAV steering paper](https://arxiv.org/abs/2605.24322) is useful evidence that emergence-zone interventions can change a downstream readout, but its same-probe flip metric is not sufficient for our goal; JEPA-WM must change candidate ranking and closed-loop task behavior.

## Geometry map

The final map is a machine-readable table plus a small set of visual summaries. Each row describes one candidate representation and contains:

| Field | Meaning |
|---|---|
| Module, block, hook | Exact intervention location |
| Spatial region/token | Where in the scene the representation is concentrated |
| Imagined timestep | When it appears during latent rollout |
| Task variable | Progress, clearance, contact, or action consequence |
| Best coordinate frame | Camera-, goal-, wall-, trajectory-, or latent-local coordinates |
| Linear score | Held-out decodability without nonlinear machinery |
| Nonlinear gain | Evidence for a curved or locally varying code |
| Dimension/eigenspectrum | Whether the code is directional or distributed across a subspace |
| Cross-trajectory stability | Whether directions/subspaces reproduce across episodes and seeds |
| Regime dependence | Whether geometry changes across genuine task phases |
| Causal patch effect | Change in predicted future, candidate rank, and selected action |
| Specificity | Whether unrelated predictions remain essentially unchanged |
| Manifold distance | Whether the edit creates an implausible hidden state |
| Recommended operator | Direction, subspace/conceptor, local energy/transport, or none |

The visual summaries should include a layer-by-variable heatmap, spatial token overlays, geometry/eigenspectrum plots for shortlisted sites, latent trajectories through task progress, and causal-effect plots over imagined rollout time. Every visual claim must point back to held-out numerical evidence in the table.

## How to read and act on the map

The map is a decision system, not merely a visualization. A useful row should read like: “goal-relative progress is a stable rank-three subspace at predictor block 4, concentrated around end-effector and goal regions at imagined steps 1–3, and changing it alters candidate-action ranking.” That row directly specifies where to intervene, which tokens and rollout steps to target, how many dimensions are required, and which planner quantity should move.

Interpret the main outcomes as follows:

- Represented and causally used: eligible for steering design.
- Represented in the encoder but not used by the predictor: evidence for an information-routing or planning bottleneck, not a steering direction by itself.
- Represented only in the predictor after action conditioning: steer or rescore action consequences there rather than editing visual perception.
- Readable but unstable across trajectories: likely shortcut or nuisance; reject it.
- Stable but equally affected by matched random patches: generic sensitivity rather than a specific mechanism; reject it.
- No reproducible representation: change the target variable, stimulus contrast, or model rather than forcing an operator.

The practical output for every accepted mechanism is an “operator card” containing the site, token region, imagined-step range, coordinate chart, dimensionality, causal endpoint, permitted edit norm, and recommended operator family. These cards make the mapping procedure reusable across tasks and checkpoints.

## Causal validation

For each shortlisted representation:

1. Patch between matched counterfactual states in both directions.
2. Compare with an identity patch, unrelated-site patch, time-shifted patch, and magnitude-matched random subspace.
3. Test whether the patch changes the predicted next latent in the expected direction.
4. Test whether it changes CEM candidate ranking or the selected action.
5. Reject edits that produce equal changes under the matched random control or cause broad prediction corruption.

A representation enters the steering-design stage only if it is readable on untouched trajectories, stable across episode resampling, causally changes a planner-relevant quantity, beats the matched controls, and remains near the natural activation distribution. Decodability alone is insufficient.

## Converting the map into a repeatable steering rule

For a validated site, learn a low-dimensional coordinate chart

\[
q=f(h),
\]

where `h` is the hidden activation and `q` contains the task-relevant coordinates identified by the map. At runtime, define the desired coordinate change `Δq` from the task objective, such as reducing goal distance while increasing predicted wall clearance. Let `J` be the local Jacobian of `f` with respect to `h`, and let `M` penalize edits along poorly supported or high-risk activation directions. The minimum-cost local edit is

\[
\Delta h
=
M^{-1}J^\top
\left(JM^{-1}J^\top+\lambda I\right)^{-1}
\Delta q,
\qquad
h'=h+\beta\Delta h.
\]

This is the default repeatable formula because it asks for the smallest activation change that produces a specified change in the validated task coordinates. Choose `β`, `λ`, and the edit trust region on discovery data, freeze them, and validate them on untouched snapshot states before any complete steered episode.

The map can justify simpler or more structured special cases:

- Stable rank-one code: use a direction with a calibrated scalar dose.
- Stable multidimensional covariance code: use a soft subspace or conceptor operator.
- Curved, locally varying code: use a local energy gradient or transport map.
- Distinct, history-dependent geometries: use regime-specific operators with soft HMM belief routing, but only after Markov and treatment-separation tests pass.
- No reproducible causal geometry: do not steer that variable or site.

For an energy formulation, define a desired-state energy in the validated coordinates and take a trust-region step down its activation-space gradient:

\[
h'=h-\eta\,\Pi_{\mathcal T(h)}\nabla_h E(f(h),a),
\]

where `Π` restricts the edit to locally supported tangent directions. This prevents “good-state” averages from becoming arbitrary off-manifold edits.

## Statistical rules and decision gates

- Split and bootstrap by whole trajectory or saved state.
- Report held-out effect sizes and confidence intervals, not frame-level p-values.
- Correct the discovery screen for multiple layer/variable comparisons.
- Freeze the shortlist, coordinate chart, operator, and dose before confirmation.
- Require the causal effect to reproduce in the untouched 75 offline trajectories and 12 on-policy episodes or their saved states.
- Treat a failed causal test as evidence not to build a steering operator there.

## Execution order

1. Load and verify the released checkpoint and official Reach-Wall data.
2. Produce the fixed trajectory split and continuous annotations.
3. Capture all block residual outputs once.
4. Build the descriptive geometry map and freeze the shortlist.
5. Collect the 24 unsteered planner episodes and snapshot states.
6. Run targeted causal patching and snapshot forks.
7. Publish the frozen causal geometry map.
8. Derive the simplest supported steering formula.
9. Only then run the 12-pair, three-arm behavioral pilot.
10. Replicate the frozen mapping and winning intervention on Push-T if Reach-Wall passes.

## Execution status — 2026-09-05

- Released checkpoint and 300 official Reach-Wall trajectories verified and captured across all 12 encoder and 6 predictor blocks.
- Discovery-only linear screen complete on 150 trajectories. Wall geometry is strongest around encoder block 6 and predictor block 2; progress and realized action consequences are strongest around predictor block 3; path intersection is strongest around predictor block 5.
- Nonlinear k-nearest-neighbor probes did not beat the linear screen, so there is no current evidence that a nonlinear chart should be the first intervention.
- Spatial screen found broad predictor-token support rather than a single privileged token.
- Physics-emergence diagnostics are complete on the 150-trajectory discovery split. Realized XZ motion direction is weak in the encoder (peak cross-validated `R²=0.046`) but rises to `R²=0.573` at predictor block 3; motion magnitude also peaks there at `R²=0.812`. Iterative orthogonal probes leave useful realized-direction signal after removing eight directions and fall below the stopping threshold after ten, while goal direction and the scalar task coordinates remain above threshold at the tested caps. This rules out a single-vector coordinate model and makes a small multi-direction predictor-block-3 subspace the first steering candidate.
- At predictor block 3, the estimated realized-direction subspace has mean principal angles of about `75°` from progress and motion-magnitude subspaces and `72°` from wall-distance geometry. These are partly overlapping but distinct task coordinates, so an intervention must specify which coordinate changes and constrain drift in the others.
- Fixed-observation action patching passed on 48 independent trajectory snapshots. At dose `β=0.5`, block 2 transferred `0.450` of the intended source-to-target latent change versus `0.020` for its orientation sham; block 3 transferred `0.490` versus `0.002`. Both intermediate blocks passed paired, collection-seed-stratified bootstrap gates, and transfer was monotonic in all 48 snapshots.
- Block 5's nearly exact transfer is treated only as a final-block sanity check, not mechanism-selective evidence.
- The 24-episode unsteered on-policy cache is in progress. Each episode stores explicit environment/planner seeds, raw observations, goal and current encodings, MuJoCo state at every replan, chosen actions, planner losses, and block-3 chosen-plan traces. Development episodes support cheap offline operator work; the held baseline episodes will later receive paired sham and steered closed-loop runs, which remain required for task-success claims.

## Intended final claim

The strongest successful outcome is not merely that a concept is decodable. It is that JEPA-WM uses a reproducible, task-relevant latent geometry at identifiable locations and rollout times; controlled changes to that geometry alter planning decisions; and an operator derived from the geometry improves closed-loop task performance over the unsteered frozen checkpoint.

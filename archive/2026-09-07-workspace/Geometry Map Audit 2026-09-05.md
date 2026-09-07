# Geometry Map requirements audit — September 5, 2026

Audit time: 2026-09-05 19:32:57 UTC. Scope: the complete [Geometry Map Experiment](<Geometry Map Experiment.md>), [Morning Ideation](<morning ideation.md>), [Physics Emergence Zone implications](<physics emergence zone.md>), active `scripts/geometry_map/` code, and locally available artifacts. This is a requirements-to-evidence audit, not a new experiment or a live GPU census. No new GPU workload was launched by this audit. No held-out trajectory tensors were opened to select a method.

## Verdict and cause of drift

**The promised final geometry map does not yet exist.** Separate discovery screens and a broad action-mediation test were completed, but they were not joined into the required representation/usage map. Having JSON results is not equivalent to having the specified machine-readable map. A script's `complete` flag refers to that script's narrower output, not to completion of the study.

The observable execution mistake was failure to check those narrower outputs against the plan's full acceptance criteria: physical labels, planner candidates, temporal/site indexing, causal behavior, confirmation, and map publication. Reporting promoted readable features and whole-block prediction changes into stronger descriptions of causal geometry. This is an execution/claim-tracking failure, not evidence that the dataset was intrinsically unsuitable or that the original plan omitted these requirements.

Preserve existing raw data and result files. Add versioned derived labels, replay captures, corrected analyses, and the map; do not silently rewrite old outcomes. Missing measurements must be `null` with a reason, not zero, a guessed result, or an unsupported operator recommendation.

## The three previously agreed gaps, recorded for provenance

1. **Coordinates and confounds:** compare competing coordinate descriptions and matched task-progress/time/obstacle contrasts; simulator-label decodability alone is not discovery of a unique model-native coordinate system.
2. **Geometry and fit/apply validity:** use an explicit common coordinate metric; do not interpret iterative-probe counts as intrinsic dimension; test offline geometry in the actual native planner context.
3. **Confirmation and replication:** freeze the selection procedure before untouched evaluation; replicate that procedure with independently fitted Push-T geometry, not transplanted Reach-Wall vectors.

## Required map fields

Status meanings: **partial** means evidence or code exists but does not satisfy the complete requirement; **missing** means no implemented measurement/final artifact was found in the audited active pipeline; **draft** means code exists without a verified run result. These are not scientific null results.

| Required field | Evidence and current limitation | Smallest necessary completion |
|---|---|---|
| Machine-readable candidate table | **Missing.** The linear screen has 378 rows, the spatial screen has 105 score maps, and the emergence report has separate curves/comparisons; none is the promised joined map. | Export one canonical table with a stable candidate key, required fields, measurement status, split, sample unit/count, uncertainty, source path/hash, and definition. |
| Module, block, hook | **Partial.** All 12 encoder and 6 predictor residual blocks are identified in captures. Attention/MLP outputs at shortlisted sites are not separately captured. | Preserve explicit residual-hook names; add attention/MLP capture only where the shortlist requires it. |
| Spatial region/token | **Partial.** Projected token readouts provide grid positions and scores at selected blocks. Broad decodability is not proof of broad causal usage. | Join token indices and reduction method to each row; validate spatial selectivity where a spatial claim is made. |
| Imagined timestep | **Partial.** Offline capture is isolated one-step prediction. On-policy traces cover the returned chosen-action chunk at block 3, not every candidate at every imagined step. | Separate real environment step, replan, CEM iteration, and imagined step; capture selected native planning contexts. |
| Task variable | **Partial.** Many Reach-Wall variables exist, but some definitions differ from the plan; see the measurement ledger below. | Correct definitions with versioned labels, units, and temporal alignment. |
| Best coordinate frame | **Missing.** World-coordinate, goal-relative, and wall proxies are labels, not a controlled comparison selecting a coordinate chart. | Compare available frame hypotheses with matched capacity/splits and causal evidence; allow ties/unidentifiable frames. Do not rename world axes “camera coordinates.” |
| Linear score | **Partial.** Discovery-only leave-one-collection-seed-out scores exist for 21 targets and 18 blocks. | Carry those scores forward as discovery evidence; obtain separate validation/confirmation results without retuning on confirmation. |
| Nonlinear gain | **Partial.** `screen_nonlinear.py` computes PCA-whitened kNN minus linear scores at shortlisted sites. The governing plan reports a completed negative-gain screen, but its result JSON was not found in the local active artifact tree during this audit. | Retrieve/checksum the existing remote result before joining it, or rerun on cached discovery features; identify the specific nonlinear comparator and metric. A failed kNN comparator does not prove the representation is linear. |
| Dimension/eigenspectrum | **Partial, interpretation correction needed.** Participation ratio and top-10 variance fraction exist; full shortlisted spectra are not exported. Iterative probe sequences and angles exist, but bases with different target masks used different scaling. | Export eigenvalues under a stated metric, recompute comparable subspace statistics, and distinguish probe-removal count, effective rank, and intrinsic dimension. |
| Cross-trajectory stability | **Partial.** Held-seed readout generalization exists; broad action-patch effects have episode bootstrap intervals. Fitted chart/subspace orientation and intervention-direction stability have not been bootstrapped. | Resample complete discovery episodes, refit, and report subspace/operator stability; do not substitute readout accuracy for basis stability. |
| Regime dependence | **Missing in the active map.** A hand-position detour flag exists, not a test that the representation differs by physical phase. | Test whether geometry/use changes across physical phases after accounting for time/progress; allow “no supported dependence.” This does not require an HMM. |
| Causal patch effect | **Partial.** The completed 48-state test changes whole-block residuals in model forwards and reports latent transfer. It does not establish identified-subspace effects on native planning or simulator behavior. | Verify targeted subspace edits, matched initial candidates, native costs/ranks/selected plans, and same-state simulator forks. |
| Specificity | **Partial.** An orientation-scrambled control exists for the broad patch. Unrelated-site/time-shift controls and preservation of other physical predictions are not demonstrated. | Add the promised focused controls and collateral-variable measurements to shortlisted causal tests. |
| Manifold distance | **Missing.** An edit's magnitude is not a calibrated measure of whether an activation is supported by natural data. | Define a discovery-fitted support-distance statistic in the same metric, calibrate it on natural validation states, and report edited-versus-natural distances. Treat it as a proxy, not proof of manifold membership. |
| Recommended operator | **Missing as a justified map output.** Block 3 and a multidirectional edit are proposals, not validated operator cards. | Record `undetermined` with the missing evidence now; select a family only after causal/specificity tests, or explicitly recommend none after a negative result. |
| Operator card | **Missing.** The plan describes the site/token/time/chart/rank/endpoint/norm/operator contract, but no accepted card was found. | Export cards from accepted map rows with source hashes and selection rationale, without requiring any particular Sonar/HMM/conceptor family. |

## Required visual summaries

No completed final visual bundle or renderer for this map was found in the active geometry-map code/artifact tree. Numerical inputs are partly available; missing visuals must not be confused with missing underlying measurements.

| Visual | What can be reused | What remains |
|---|---|---|
| Layer-by-variable heatmap | The 378 linear-score rows and emergence curves. | Render with metric-specific scales, discovery/confirmation labels, and references to the source rows. Do not mix AUROC and R-squared as interchangeable colors. |
| Spatial token overlays | The 105 token-score arrays and original source videos. | Align grids to the exact crop/resize and a named frame; distinguish dataset-aggregate token scores from episode-specific causal attribution. |
| Geometry and eigenspectrum plots | Pooled activations, participation ratios, preliminary subspace results. | Recompute valid spectra/angles and plot them in a stated coordinate metric with episode-level uncertainty. |
| Latent trajectories through task progress | Time-indexed offline activations and chosen-plan traces. | Fit a shared chart on discovery data and transform other episodes without refitting; separate physical time/progress from imagined time. |
| Causal effects over imagined rollout time | Existing broad-patch layer/dose summaries are not this measurement. | Capture downstream effects by imagined step, distinguish intervention location in time from propagation over time, and show corresponding planner/behavioral effects with uncertainty. |
| Every visual backed by numerical table evidence | Existing result files have some provenance, but no unified row-to-figure connection. | Generate figures from versioned table rows; attach exact row IDs, split, units, sample counts, and source hashes. No manually invented curves or missing-as-zero heatmap cells. |

## Were the requested measurements actually recorded?

| Measurement | Answer from code/artifacts | Required correction or additional work |
|---|---|---|
| End-effector-to-goal vector/distance | **Yes in offline labels:** `goal_vector`, `goal_distance`, `hand_xyz`, `goal_xyz`. | Keep physical units, reference frame, and sampled-time index explicit. |
| Progress since previous step | **Not under that definition.** `progress = goal_distance[0] - goal_distance[t]`, a cumulative-from-start quantity. | Keep the historical field, add separately named local distance reduction over the same prediction interval. Do not relabel old scores as local progress. |
| Remaining goal distance | **Yes**, at sampled offline frames. | Join consistently to on-policy starts and actual future outcomes. |
| Hand/goal relative to wall | **Partial.** Hand signed box distance and wall-top height are explicit; positions exist to derive wall-relative vectors. | Derive hand and goal wall-relative coordinates; verify the fixed wall geometry against the environment, and state which point/body is measured. |
| Minimum clearance / candidate wall crossing | **Not as promised.** The current proxy is point-to-box clearance; `straight_path_intersects_wall` tests the current hand-to-goal line, not each candidate trajectory. | Measure clearance/crossing along actual candidate forks or a validated predicted physical path. Do not call point clearance whole-robot collision clearance. |
| Action direction and magnitude | **Partial.** Raw actions, action means, translation sums, and chunk translation magnitude are saved; direction is derivable but not uniformly exported. | Export normalized direction with a zero-motion mask, and distinguish net-action-vector magnitude from the current five-action chunk norm. |
| Realized next-state displacement | **Yes for offline five-action transitions:** `realized_hand_delta` and its magnitude. Push-T caches save full state sequences. | Preserve identical action horizons; for Reach on-policy replay, recover intermediate state traces rather than infer five-step displacement from 15-step replan spacing. |
| JEPA-predicted physical next-state displacement | **Not yet demonstrated.** `predicted_final`/`predicted_proprio` are saved model outputs; a validated conversion to physical XYZ displacement and its error are not exported. | Verify tensor semantics and native normalization/decoding; if a learned readout is needed, fit/validate separately and label the result probe-derived, not a native physical prediction. Compare with realized displacement on the same action interval. |
| CEM candidate cost and rank | **No complete candidate bank in baseline caches.** `planner_losses` stores native `_prev_losses` (per-iteration best-cost summaries); Reach also stores elite mean/std. Those do not reconstruct individual candidate costs or ranks. | Replay native CEM from saved observations/RNG and capture iteration + candidate ID + action sequence + cost/rank, plus the chosen plan. Matching ranks requires matching candidate actions, not just matching seeds after CEM distributions diverge. |
| Selected action | **Yes.** Normalized/raw returned plans and actually executed raw actions are saved. | Distinguish the returned CEM mean plan from an individual sampled candidate and from the executed prefix. Do not assign it a fictitious candidate rank. |
| True episode time | **Yes**, frame indices/time fraction and on-policy elapsed steps. | Retain it as a nuisance control; a cubic time baseline does not replace the planned matched contrasts. |
| Action/proprio embeddings | **Partial.** Action features are computed in offline capture but not saved separately. Current encoded proprio and predicted proprio are saved in different paths. | Capture separately identified input embeddings, not only pooled mixed residuals, at shortlisted native contexts. |
| Predicted future latents | **Partial.** Offline terminal encodings/predictions are pooled; block-3 chosen-plan traces are cached. | Capture enough per-candidate/per-step state at shortlisted sites to support temporal causal plots; do not relabel an executed-prefix trace as a full candidate search tree. |

## Data, causal design, and validation gaps

| Requirement | Audit result and action |
|---|---|
| Frozen checkpoint / architecture / baseline | Released headless constructors are used; omitting an unused RGB visualization decoder is not a new encoder/predictor. Keep model/vendor/config/checkpoint provenance attached to every new artifact. The native unsteered planner is the utility baseline; no COAST percentage-reproduction prerequisite. |
| Offline 150/75/75 whole-trajectory split | The split exists and 300 pooled trajectories were captured. Discovery analyses use the 150-trajectory subset; separate 75-trajectory validation and confirmation stages are not evidenced as complete. The earlier loader opened tensors before checking split; the local guard was moved before loading in the preceding implementation turn. Existing results did not thereby become confirmation results. |
| On-policy 12-development/12-evaluation split | The manifest exists. Do not tune from evaluation outcomes or combine their metrics into a development summary. `validate_on_policy_bank.py` currently summarizes outcomes for every supplied input, so provide separated inputs or a metadata-only verification mode. This audit did not inspect evaluation tensor outcomes. |
| Coverage of task phases | The existing 48-state selector takes the closest-to-wall frame in 16 predetermined episodes per seed. It does not establish the promised coverage across approach, wall interaction, and later goal progress. Add fixed phase-covering states from reusable development episodes without outcome screening. |
| Matched counterfactuals | Same-observation action variants were predicted, not physically forked in the completed broad-patch test. Comparable-progress/different-wall and comparable-time/different-progress contrasts remain incomplete. |
| Replay integrity | Reach caches contain physics snapshots, but the existing validator checks presence/finiteness, not successful physical restoration. `initialize_episode` returns `info` from an earlier reset than `set_episode`; the first replan's `state` annotation can therefore be stale. Preserve physics/pixels, reconstruct and validate the first state before deriving labels. Do not conclude every baseline episode is invalid. Push-T's reset-and-action-replay check is implemented; direct body restoration is explicitly uncertified. |
| Candidate fit/apply context | Offline capture batches isolated transitions; planner unroll carries imagined history. First-step-only draft patching avoids claiming later-step applicability, but later-step geometry remains untested. Compare shared semantics at the actual hook/time/token context. |
| Broad patch versus causal subspace | Keep the 48-state transfer results as action-mediation evidence only. The final-block transfer is a sanity check, not evidence of a useful control surface. |
| Newly written corrective code | `fit_causal_subspace.py`, `subspace_control.py`, and `causal_planner_forks.py` are **draft/unrun**, not completed experiments. No candidate-fit receipt, paired-fork result, or validation test was found locally. The draft captures first-iteration CEM candidates and intervenes at imagined step zero; it is not the full temporal map. |
| Draft-specific issues before launch | Validate native shapes/context and all restore checks; match donors on non-target action components (the draft currently zeros counterfactual gripper actions while keeping the baseline's recorded ones); verify the physical prediction/fork horizon alignment (six-step plan scoring is not five-action realized displacement); check immutable failure handling and split validation before tensor load. Add unit and one-state runtime tests before distributing it. |
| Causal controls and scope | Bidirectionality, identity, unrelated-site, time-shift, norm-matched orientation controls, and collateral-variable checks are not collectively completed. The draft up/down donors are not by themselves proof of reciprocal source/target subspace transfer. Native plan/action changes must be separated from demonstrated improvements. |
| Statistics | Held-seed folds and episode-bootstrap broad-patch intervals exist. Multiple-comparison correction, episode-refit chart stability, and final independent causal confirmation are not completed. Report episode/state counts rather than treating frames/candidates as independent trials. |
| Frozen shortlist and dose | Discovery rankings exist; a fixed choice with validation evidence and an untouched confirmation run is not yet established. Do not let causal results retroactively redefine the selection rule. |
| Acceptance/rejection and map coverage | Keep both tested-negative and unmeasured rows. Do not select an operator just to populate its field. A complete *measured* map can contain no accepted steering site; an incomplete map cannot masquerade as such a negative result. |
| Push-T protocol | Prepared inputs use 21 source trajectories, grouped as 10 development/11 evaluation, not 24 independent sources. Real preparation and baseline code exist, with one local verified baseline canary receipt; this audit does not claim all remote outputs are consolidated. Add contact/object-pose/action-consequence annotations and reuse the frozen procedure, while fitting Push-T's own geometry and keeping its released goal-based metric. Preparation is not a completed Push-T map. |
| Larger steering/compute comparisons | Full paired efficacy, task-success intervals, and comparisons with LoRA/adapters remain later work. HMM/Sonar/Fisher/transport proposals in morning/PEZ notes are optional later families, not missing prerequisites for this geometry-first study. No cheaper-adaptation claim is warranted from cache reuse alone. |

## Five actionable next steps — original scope, no new experiment matrix

1. **Create the map export now:** assemble existing results into a versioned machine-readable table with all required columns and explicit missing/partial/negative statuses; export figures only from measured rows.
2. **Repair and complete labels/capture:** derive local progress and direction conventions from existing data; replay missing native candidate costs/ranks, physical prediction alignment, short physical state traces, and shortlist-only embeddings/hooks.
3. **Finish the representation map:** retrieve nonlinear results, compare coordinate descriptions and physical phases, fix scaling/dimensionality claims, export eigenspectra/support distances and episode-level stability, then freeze the shortlist using validation only.
4. **Verify the causal usage map:** test the draft runner locally and on one development state, validate exact unsteered/identity replay, then run targeted matched subspace/candidate/simulator tests with the promised controls and temporal indexing; add those results to the same table.
5. **Confirm and replicate:** use untouched Reach-Wall data without retuning, publish the evidence-linked visual bundle/operator cards (including rejection outcomes), and apply the same frozen procedure to independently fitted Push-T geometry; continue reusable Push-T preprocessing separately from outcome-guided analysis.

Steps 1–3 reuse cached data wherever possible. Only genuinely absent planner/counterfactual measurements need targeted new inference or simulation. No wholesale dataset restart is justified by this audit.

## Evidence and provenance

Main implementation evidence: [capture](scripts/geometry_map/capture_shard.py), [physical labels](scripts/geometry_map/protocol.py), [linear screen](scripts/geometry_map/screen_pooled.py), [nonlinear screen](scripts/geometry_map/screen_nonlinear.py), [spatial screen](scripts/geometry_map/screen_spatial.py), [emergence analysis](scripts/geometry_map/screen_emergence_geometry.py), [snapshot selector](scripts/geometry_map/build_snapshot_manifest.py), [broad patch](scripts/geometry_map/causal_action_patching.py), [patch summary](scripts/geometry_map/summarize_action_patching.py), [Reach collector](scripts/geometry_map/collect_on_policy_bank.py), [Reach validator](scripts/geometry_map/validate_on_policy_bank.py), [Push-T collector](scripts/geometry_map/collect_pusht_bank.py), and [draft causal runner](scripts/geometry_map/causal_planner_forks.py). The discovery pilot's serialized field names were also inspected without executing its pickle payload. This audit did not revalidate every tensor/checksum across remote workers.

SHA-256 identities below preserve the documents as audited **before** this audit's pointers/claim corrections, plus representative unchanged numerical evidence:

| Source | SHA-256 |
|---|---|
| Geometry Map Experiment.md, pre-audit | `cc1a15957f15f7fe1cd9da6ed1b8b2ec3a6b99b20a1af4acc60930f282ec60dc` |
| morning ideation.md, pre-audit | `82fb087cf783af9931067c3d4f25b8d41280d71b65908bb1f6bb72858466bc25` |
| physics emergence zone.md | `b833b38da70e1690771bf2eb99e340aa5987d362e867e55da65a337a409410b1` |
| on_policy_bank_manifest.json | `9307414b7a2590355f525fe60347d3c750b01021cb9f357ba6c93640f87cdb6d` |
| discovery_screen.json | `17109453308933220165a55b5fae8c4b20e4705b9a52e5eba7ea1cb7628f7969` |
| spatial_screen.json | `a7628565e6e62d8ae5ada8264f5b43698efc4dec8a85e73b857761f7eb720812` |
| emergence_geometry.json | `1868de02080be5009b13c103bf5d3207e174a09a1677de573e8cb3c176a46443` |
| action-patching/summary.json | `e473743a4d0f6c45d3d06841f7b90c6d77e1efa65ef0391d3ad3b3ccef076266` |
| causal_planner_forks.py, draft audited | `e07fdbf557fa1a390315db62e6913312813ed8a9008caee85aad9725946e294f` |

Documentation changes from this audit: added this requirement ledger, linked it from the governing plan and historical morning note, and qualified overclaims about probe dimensionality, cross-metric angles, and token localization. No raw numerical result, frozen split, checkpoint, or live collector was changed. The map builder, analysis repairs, and draft-runner verification remain work to execute, not accomplishments of this audit.

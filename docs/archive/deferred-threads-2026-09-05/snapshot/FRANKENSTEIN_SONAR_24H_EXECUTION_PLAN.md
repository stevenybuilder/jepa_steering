# Frankenstein Sonar: 24-hour C1 execution and early-stop plan

Status: **ACTIVE, pre-intervention**  
Frozen at: **2026-09-04T20:53:30Z**  
Scope: DINO-WM `mw_dino-wm.pth.tar` on native MetaWorld Reach (`C1`)  
Primary objective: improve native simulator task success with the admitted
Frankenstein/Sonar intervention relative to the same frozen unsteered checkpoint.  
Time target: **under 24 elapsed hours after the P0 decision**, using temporary
matched GPUs. This is not a 24 aggregate-GPU-hour design.  
Aggregate C1 compute ceiling: **75 GPU-hours without a new user decision**.  
Incremental spend ceiling: **$30 without a new user decision**.

This document is the operational record. The broader scientific rationale remains
in [WORLD_MODEL_SONAR_PLAN.md](WORLD_MODEL_SONAR_PLAN.md). GPU ownership is governed
by [GPU_RESOURCE_BOARD.md](/Users/stevenyang/Documents/GPU_RESOURCE_BOARD.md).

## 1. Non-negotiable scientific contract

1. The primary outcome is the released evaluator's per-episode native simulator
   success flag (`succ_def: simu`). Probes, distances, planner costs, and researcher
   judgments cannot relabel an episode.
2. Every paired arm uses the same realized initial state, goal, environment seed,
   planner seed, horizon, CEM budget, success definition, and episode order.
3. Development, fitting, and protected evaluation use disjoint manifests. Protected
   outcomes cannot change the operator, site, dose, module set, or arm set.
4. The checkpoint is frozen. There is no training, checkpoint search, task search,
   or episode replacement based on outcomes.
5. The full recipe is a gated composition. A module that fails a prospective gate
   becomes an explicit identity/no-op for this C1 composition; it is not silently
   repaired on protected data. An unimplemented module blocks a “full Frankenstein”
   claim.
6. The primary claim is behavioral improvement versus unsteered. Geometry is
   supporting mechanism evidence, not a substitute endpoint.

## 2. Frozen provenance at launch

| Item | Frozen value |
|---|---|
| Checkpoint | `mw_dino-wm.pth.tar` |
| Checkpoint SHA-256 | `eab6e241f732db986f98ad224115ed7949ab4ff3f6cc8bc8f95b273cdb9e5d13` |
| Task/config | MetaWorld Reach; `reach_L2_cem_sourcexp_H6_nas3_ctxt2_r224_alpha0.1_ep48_decode.yaml` |
| Config SHA-256 | `968aca1ae415a73be6d9ba72d8269c9422d513cc248dcc5f26b234618955b621` |
| Development manifest | 30 pairs; `02feeabdb99a8d4dd7eab10ea3f0fcd40121643d6e63aa8f597b6a64058628a2` |
| Fit manifest | 30 pairs; `2d177fad74b7026a29cba754025c4a66b4d9cdc5...` (full value below) |
| Protected manifest | 60 pairs; `19505825f63545fc4b106d20866554963c378bf93334fd0f762af0bb71a3001f` |
| Vendor repository commit | `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` |
| Current instance | Vast `49766237`, RTX 3090, lease `cgs-pilot-2` |
| Current P0 | PID `81544`; 26/30 complete at freeze; no activation intervention has run |

Full fit-manifest SHA-256:
`2d177fad74b7026a29cba754025c4a66b4d9cdc5f7408d4b6b6ddf841a6c161`.

Local implementation hashes at launch:

| File | SHA-256 |
|---|---|
| `WORLD_MODEL_SONAR_PLAN.md` | `c355e9bf3a362732177cf89e3e17fb8612b0195e755ed8bdfbcdbbcffabb1361` |
| `public_panel_behavior_gate.py` | `f1b9c51bfc4dc95d269ffd87ab4ce004cfb189079519274323dfa342f21d6afe` |
| `public_panel_capture.py` | `2712bc9544fab9628fb1e9ffe6a9ee1c753a55136dd0294db5f791f9870deb03` |
| `public_panel_fit_apply_audit.py` | `0f19483b4707236638c61bb66078b8bda89f497fece867ace6c93c1b5912d7c7` |
| `public_panel_development_gate.py` | `651112b38c1906f2a4df5164799e8abeb276dc17259bfe609e4a80534f119c39` |
| `public_panel_steer.py` | `9e663f10fa4c3ff38eee41d1252ff2dde126b3e76bd9165cb81b090b5b945077` |

Any deployed version receives a new immutable run-bundle manifest. These hashes
record the starting point; they do not authorize stale code to enter a run.

## 3. Resource and sharding plan

Do not borrow or modify the two VLA instances or the quarantined legacy instance.

1. Finish P0 on `49766237` without modifying its live code or output.
2. After P0 passes, create one temporary RTX 3090 canary with the same container,
   CUDA/PyTorch stack, code bundle, checkpoint, config, and manifests.
3. Require the canary provenance and reproducibility gate below.
4. Only then add temporary matched RTX 3090 instances, targeting six C1 GPUs total.
5. Partition by `pair_id`. Each GPU runs every arm for its assigned pair IDs so
   hardware is a paired block. Each process writes an exclusive shard directory.
6. Merge only shards with the expected `DONE.json`, row count, pair IDs, and file
   hashes. Never append concurrently to one `episodes.jsonl`.
7. Copy and checksum results before releasing temporary instances.

Target stage timing after P0: canary/setup 1--3 h; sharded fit capture 1--2 h;
offline gates 1--3 h; sharded development 2--4 h; freeze/registry under 1 h;
protected run 6--7 h. One repair cycle can exceed the 24-hour target and therefore
requires the explicit rules in Section 5.

## 4. Full C1 arm set

Development inventory:

- unsteered (the completed P0 trace is the paired reference);
- identity;
- mean-direction and PCA/covariance simple baselines;
- matched random direction;
- equal-extra-compute CEM or direct outcome/value reranking, whichever is the
  strongest admissible simple behavioral baseline;
- optional global COAST-equation transfer check (development only; it is not an
  established world-model baseline);
- static regime-local Frankenstein;
- action-HMM regime-local Frankenstein only if all HMM eligibility and treatment-
  separation gates pass;
- orientation- and executed-action-dose-matched sham.

Protected set, frozen before intervention and intentionally limited to four arms:

1. unsteered;
2. one frozen admitted Frankenstein composition;
3. a global version of the same world-model operator, providing the simple/locality
   comparator without importing COAST as a VLA baseline;
4. one fixed matched-spectrum random-orientation sham using the same site, timing,
   trust radius, support logic, and approximate latent dose.

Identity is an exact capture/development invariant, not a 60-episode protected arm.
Mean/PCA directions, COAST-equation transfer, static-versus-HMM routing, and other
simple alternatives remain development diagnostics. They enter no protected run
unless a future prospectively recorded amendment replaces—not adds to—the four-arm
set before any protected outcome is opened.

The ten Frankenstein modules are fitted and gated from captured data. They are not
each promoted to a 60-episode protected arm. Targeted module ablations occur on
development data and, after a real effect, in a later simplification study.

## 5. Early-warning ladder: detect, fix once, or stop

No stage advances merely because code ran. Every gate writes a machine-readable
PASS/FAIL artifact. A failure is classified before any repair:

- **Infrastructure defect:** provenance, serialization, shape, hook, merge, or
  deterministic replay failure. One documented repair and rerun of the affected
  unprotected stage is allowed.
- **Model/method falsification:** inadequate class support, no usable coordinate or
  outcome geometry, intervention cannot change selected actions, severe off-support
  editing, or no preliminary utility. Do not tune around it; stop and reassess.
- **Optional-module rejection:** HMM, sparse, pattern, probability, transport, or
  attention module fails its own gate while the base operator remains valid. Record
  the module as identity/no-op and continue only under the correctly narrowed name.

| Gate | Earliest evidence | PASS requirement | Failure action |
|---|---|---|---|
| G0 Behavioral substrate | Completed 30-episode P0 | Native success rate 20--80%; at least 6 successes and 6 failures; both outcomes across at least 2 progress quartiles; realization hashes valid; at least 2 plans/episode; no errors | **STOP.** Do not search nearby seeds/checkpoints/tasks. Diagnose the harness or declare C1 ineligible. |
| G1 Visualization-free speed canary | 2 completed development seeds after P0 | Exact action hashes, outcomes, plan boundaries, and realization hashes versus released configuration; speedup at least 1.20x | If exactness fails, keep original visualization path. Lack of speedup is not a scientific failure. |
| G2 Cross-instance canary | Same 2 seeds on one new matched 3090 | Code/config/checkpoint/manifest hashes exact; no numerical/action divergence beyond the frozen determinism contract | Repair environment once or reject that host. Do not average incompatible hosts. |
| G3 Capture identity | First 2 fit episodes | Read-only capture produces exact unsteered action/outcome hashes; correct site, tokens, candidate axes, plan boundaries, and physical-replan indexing | Repair capture once and repeat from a clean output root. No fitting before PASS. |
| G4 Prefix support | First 12 fit episodes, then 18 if needed | Both outcomes observed; no malformed episodes; storage/runtime projection acceptable | Expand only within the frozen 30 fit manifest. If full fit has fewer than 8 independent episodes/class, no outcome-conditional intervention. |
| G5 Fit/apply population | Candidate-inclusive fit capture | Runtime-edited CEM candidate population matches the fitted hook/pooling/token population; gradient/operator direction and action-effect consistency pass the frozen audit | One repair choice: refit on the captured candidate population or narrow the hook schedule prospectively. If both fail, **STOP**. |
| G6 Geometry/modules | Fit episodes only, whole-episode CV/bootstrap | Stable success/failure geometry; calibrated local support; model-native coordinate advantage over shuffled-goal/naive controls; admitted modules beat their nulls | Reject optional modules individually. If the base success geometry or runtime coordinate is unusable, **STOP**. |
| G7 HMM eligibility | Fit episodes only | Action/history routing beats fixed/static/progress controls; passes innovation and Chapman--Kolmogorov checks; improves outcome routing; static/HMM action treatments differ materially | Use static routing. Do not tune HMM until it differs. |
| G8 Dose/action sanity | Cached candidates plus first 4 development pairs | Cap saturation <=80%; finite/in-support edits; nonzero selected-action changes; matched arms within frozen latent- and action-dose bands; no timeout/latency violation | Try only the preregistered dose ladder on development data. If no dose passes, **STOP**. |
| G9 Development sentinel | First 10, then all 30 development pairs | No catastrophic harm at 10; completed development gate at 30, including action change, sham/dose validity, treatment separation when applicable, and selected-arm paired gain >=5 pp with CI lower bound >=-10 pp | At 10, stop for >=30 pp harm or systemic errors. At 30, failed utility means **no protected run**; reassess recipe. |
| G10 Freeze | All prior gates | Exclusive arm registry, algorithm/site/dose/modules/endpoints/analysis hashes written once | Any ambiguity blocks protected access. |
| G11 Protected engineering sentinel | Outcomes remain uninspected during run | After each 10-pair batch: completeness, latency, support, cap, action, and hash checks only | Stop for implementation/provenance failure. Never repair using protected outcomes. |
| G12 Protected efficacy | All 60 pairs complete | Primary paired Frankenstein-minus-unsteered native success difference and frozen inference; controls and preservation endpoints reported | Report the result as observed. No rescue arm, replacement episode, or post-hoc relabeling. |

The G9 development sentinel is the main economic safeguard: it exposes bugs,
off-support behavior, ineffective action edits, HMM/static aliasing, and grossly bad
behavior before the 60-pair protected expense. Protected outcomes are not used for
early tuning. This preserves a clean final comparison.

## 6. Development-only repair limits

The only permitted adaptive choices are finite and occur before G10:

- one capture-population repair at G5;
- the already declared site/operator families;
- a maximum three-value trust-radius/dose ladder;
- static routing when HMM is rejected;
- identity/no-op for an optional module that fails its frozen gate;
- visualization disabled only after exact G1 equality.

Every attempted value and failure remains in the development report. There is no
new checkpoint, task, seed replacement, label, success threshold, or protected arm
after seeing behavior.

## 7. Decision branches

- **P0 fails:** stop C1; diagnose behavioral substrate. GPU burst is not launched.
- **Capture or fit/apply fails after one repair:** stop; the intervention is not a
  valid test of the proposed mathematics.
- **HMM alone fails:** proceed with static Frankenstein and report that action-
  conditioned routing was rejected prospectively.
- **Development utility fails:** do not open protected data. Determine whether the
  failure is representation, operator, planner mediation, or task ceiling/floor.
- **Protected effect is positive and valid:** write the C1 workshop result; add
  JEPA-WM Reach replication only after the deadline-critical artifact is secured.
- **Protected effect is null:** report a valid null and use action/geometry mediation
  logs to localize the break; do not carpet-run C2--C4.
- **Protected effect is harmful:** stop the program and reassess the base energy/
  conceptor objective before any replication.

## 8. Execution log

Append-only entries; do not rewrite prior observations.

- `2026-09-04T20:53:30Z` — Plan frozen. P0 is the only live Panel-P stage; PID
  `81544`, 26/30 complete, no intervention run. No new instance created. The two VLA
  leases and one ownership-uncertain legacy instance remain unavailable.
- `2026-09-04T21:01Z` — Monitoring only: P0 reached 27/30 with 17 successes;
  PID `81544` remains live. Vast inventory also showed newly created instance
  `49898076` labelled `vla-frankenstein-lean-v1`; it was recorded as quarantined for
  this project and was not inspected or modified.
- `2026-09-04T21:05Z` — Pre-intervention design simplification, based on cost and
  construct validity rather than outcomes: removed identity, COAST-equation transfer,
  and the extra static/HMM arm from mandatory protected evaluation. The protected
  set is now unsteered, admitted Frankenstein, its global same-family comparator,
  and one simple matched-spectrum sham. No protected data had been opened.

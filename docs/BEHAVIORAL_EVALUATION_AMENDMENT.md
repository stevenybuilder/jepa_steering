# Behavioral evaluation and task-breadth amendment — 2026-09-07

## Decision and scope

Implement the user-approved PointMaze and Wall additions and paper-style behavioral
design-choice evaluation. Confidence is high in their relevance and official asset
availability, but runtime and intervention efficacy remain unmeasured. Retain Reach,
Reach-Wall and Push-T: five intended tasks, not five completed behavioral evaluations.
Defer adding RoboCasa/DROID to the execution suite while assessing their dependencies.

This amendment records the user's explicit direction before new behavioral outcomes.
It supersedes the old offline-improvement requirement ONLY as an admission rule for
the new behavioral-development stage. Completed offline protocols, their selection
rules, raw results, failure records and exposure registries remain unchanged.

## Evidence and interpretation

JEPA-WM compares design choices using planning success alongside prediction metrics;
the latter are imperfect proxies (Sections 4–5, G.1–G.3). Thus offline-ineligible arms
are not established inferior planners. The current non-routed selections remain
offline selections: Reach combined, Reach-Wall rank4, Push-T native.

PointMaze/Wall appear in the paper's Table 2 and have official data, checkpoints,
training configurations and planning configurations. They test recipe repeatability
across environments/checkpoints. Transferring identical fitted directions to another
task without refitting is a different claim. Base-model training exposure is distinct
from intervention fitting/development/confirmation exposure.

## Fixed questions, not method search

Preserve the existing bounded hypotheses: vision/action coupling, action-response
geometry, operator rank, layer/spatial support, and conditional temporal routing.
The behavioral question is whether these alter actual task success versus the same
unsteered checkpoint, not merely whether they reduce validation loss. Alternatives
include no behavioral effect, worse success despite better forecasts, random edits
with comparable effects, and task-specific rather than cross-task benefit. Retain
negative results and check task regressions; do not add an unregistered successor
after observing outcomes. HMM eligibility and fitting remain separate, unresolved work.

## Behavioral development versus untouched confirmation

Significant offline improvement is NOT a mandatory admission criterion in the new
behavioral stage. Identity/fidelity, valid inputs, frozen parameters and verified
exposure remain required. A finite candidate/control registry must be fixed from
the existing hypotheses before behavioral outcomes. The exact panel, selection,
practical-effect and multiplicity contract remain pending; this amendment alone is
not a launch or reveal receipt. It must not become an unbounded Cartesian search.

Use actual task success as the primary behavioral endpoint. Keep forecast errors,
failure modes, action magnitudes and runtime as separately reported diagnostics.
Planning development selects among predefined alternatives; final frozen choices
then receive a distinct untouched confirmation evaluation. Newly generated scenarios
used to select candidates are development, not confirmation. The already prepared
96-scenario MetaWorld confirmation pools stay unopened pending their exposure audit.

Pair native, candidate and matched controls on the same checkpoint and initial/goal
scenarios with matched planner randomness. Do not count candidate sequences, rollout
windows, repeated checkpoints or reused source families as independent scenarios.
Use paired scenario/family effects and retain checkpoint/training-seed structure in
uncertainty. Freeze the final interval family and useful-effect rule before reveal;
96 episodes is a replication target, not proof of sufficient power. Estimate paired
discordance/variability from development and report detectable-effect sensitivity.

## Match actual per-task implementation

Use the pinned official simulator, reset/goal sampler, preprocessing, CEM, objective,
action handling, context and budgets. In the released PointMaze/Wall configurations:
96 episodes, context2, H6, 300 candidates, 10 elites, 30 CEM iterations, frame stride5,
execute six model actions; official environment setup caps episodes at 30 elementary
steps. These are single-plan evaluations, not MetaWorld's repeated feedback-control
schedule. Use each environment's own normalization and checkpoint. Their offline
dataset loaders/split behavior need direct auditing, not a copied assumption from
MetaWorld or Push-T. Use raw archives, not the small Hugging Face preview parquets.

## Training variability and checkpoint histories

Add a separate three-training-seed robustness/replication track. The five-task suite
uses four environment-trained models because Reach/Reach-Wall share MetaWorld.
The paper's final-model aggregation includes three training seeds and the final ten
epoch evaluations. Existing final checkpoints cannot supply their absent histories.
Full newly reproduced histories for all four models therefore require 12 training
runs; two extra final checkpoints per model alone address a narrower seed question.

Before training launch, freeze the actual seed triplets, source configs, global batch,
optimizer/schedule, update counts, checkpoint retention and evaluation cadence.
Changing physical GPU count must preserve effective training settings or be disclosed
as a deviation. Do not silently replace the paper's configured training budget with a
short run. Keep frozen-checkpoint behavioral work running independently of this track.

For recipe robustness, refit the unchanged intervention fitting algorithm separately
on permitted fit data for each checkpoint; never reuse a final-checkpoint fit as if
it were fitted on earlier checkpoints. Any unchanged-operator transfer assessment
must be separately labelled. Do not choose epochs/seeds by favorable outcomes.

## Rerun accounting

- Reuse the completed corrected offline measurements for their exact checkpoints,
  inputs, arms, precision and endpoints. Removing the admission gate does not
  invalidate those observations or require repeating them.
- New tasks require their own data/lineage audits, baselines, fitting and behavioral
  evaluations. They cannot inherit a MetaWorld efficacy result.
- New trained checkpoints require new fitting/evaluation for each checkpoint; old
  results are references, not measurements of the new models.
- Any actual methodological bug affecting a measurement requires a scoped rerun
  under a new receipt. Preserve the affected original and its failure description.

## Parallel execution and engineering checks

Non-routed integration does not wait for HMM or new-task assets. The new
`offline_study.planning_support_smoke` binds completed exact-transfer receipts and
executes complete official 100-step MetaWorld simulator/CEM episodes for Reach
combined, Reach-Wall rank4 and their matched controls on the already-used, excluded
smoke seed 2026090719. It records and verifies every 300-candidate and mean forecast
at all 15 iterations per replan. These engineering outcomes cannot select candidates
or count toward confirmation. An incomplete episode cannot receive a DONE receipt.

Work remains subject to the latest user limit: aggregate $5/hour, US-only. Stage
assets while independent GPU jobs run, batch candidates, and preserve source/results
before releasing workers. Additional work should be scientifically relevant, not
created solely to fill a GPU. Report actual utilization and measured throughput.

## RoboCasa/DROID decision and limitations

DROID is real-robot data. In JEPA-WM, its reported evaluation is an offline planned-
action score on recorded robot videos, not newly executed real-robot task success.
The authors' RoboCasa evaluation uses custom easier Reach/Place scenarios and a
DROID-trained larger model, modified camera/gripper and action conversion/repetition.
It is not interchangeable with a stock contemporary RoboCasa benchmark. Its custom
data are released, but dependencies, population lineage and transfer adapter need
auditing before an execution decision. Neither name alone strengthens a robotics
claim without valid measurements; MetaWorld already supplies simulated robot control.

Recommendation: finish the five-task behavioral suite and assess a bounded DROID or
RoboCasa extension only against a concrete unanswered question. Evidence of successful
setup with adequate resources could change this recommendation. Do not download the
entire multi-terabyte DROID corpus just to reproduce its small evaluation subset.

## Sources and next required evidence

Research date: 2026-09-07. Governing paper: arXiv 2512.24497v4 (2026-09-02).
Still required: exact behavioral registry/analysis freeze, distinct development and
confirmation exposure records, new-task runtime parity, complete episodes, and
multi-seed training contracts/throughput. No green document or smoke substitutes
for these scientific measurements.

- [JEPA-WM paper, Sections 4–5 and Appendices E–G](https://arxiv.org/html/2512.24497v4)
- [Pinned official repository](https://github.com/facebookresearch/jepa-wms/tree/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0)
- [Official raw data](https://huggingface.co/datasets/facebook/jepa-wms)
- [Released checkpoints](https://huggingface.co/facebook/jepa-wms)
- [DROID dataset paper](https://arxiv.org/abs/2403.12945)
- [Original RoboCasa paper](https://arxiv.org/abs/2406.02523)

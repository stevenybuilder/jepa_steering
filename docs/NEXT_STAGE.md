# After corrected offline evaluation

Status: recommended execution sequence, not a frozen confirmation protocol or a
claim that closed-loop implementation is complete. The current authorized budget is
$7/hour aggregate, US-only; noon Eastern September 7 was a best-effort target, not a
reason to weaken scientific gates. See `configs/execution_authorization_20260907.json`.

The later [behavioral amendment](BEHAVIORAL_EVALUATION_AMENDMENT.md) supersedes the
offline-significance admission gate below for new planning-development comparisons.
The current scope is six tasks, with required training histories; preserve the
already-completed offline rules as historical provenance, not a planning veto.

Current preparation is documented in [PLANNING_METHOD_ALIGNMENT.md](../reports/PLANNING_METHOD_ALIGNMENT.md).
Paper-scale planning contracts and verified simulator goal-setup checks now exist
for all three tasks. Native CEM checks are in progress; the candidate-to-planner adapter still requires verification before
automatic efficacy-job submission. The user has authorized autonomous advancement
after the fixed gates, not autonomous search or choosing favorable precision.

The active `finish_author_study.py` coordinator collects checksum-verified BF16
Push-T evidence from Utah, waits for the full MW merge and primary-host FP32 Push-T
completion, then runs `author_summary` and `author_advancement`. These report all
30 task/category/precision scopes and apply the fixed endpoint/control gates.
They do not by themselves authorize a confirmation reveal or mark the study complete.

1. Finish all fixed corrected offline sweeps and verify identity, data membership,
   source hashes, paired coverage, delivered energy, and native comparisons. Report
   both fixed precisions, keeping bfloat16 primary. Do not choose favorable precision.
2. Apply the existing plan's task-specific advancement rule: a meaningful effect on
   the fixed recorded-future endpoint and superiority to the registered matched
   controls, with simultaneous intervals. A small p-value or low reconstruction
   error alone is insufficient. No qualifying arm means retain native for that task.
3. If a combined recipe is needed, test only the prescribed development compatibility
   and drop-one checks before freezing it. Do not assume separately favorable edits
   combine constructively. Do not introduce a new routing hypothesis from these results.
4. Verify native planning first on non-confirmatory smoke episodes, with identical
   official simulator/planner settings for all conditions. A smoke check is not an
   efficacy test and cannot tune edits on protected outcomes.
5. Freeze one task-specific candidate, its appropriate random control, native,
   planner settings, start/goal sampling, seeds, endpoints, sample size and analysis
   before reveal. Compare conditions on paired initial/goal states and random seeds.
   Closed-loop success rate is primary; also report uncertainty, failures and compute.
6. A genuinely fresh closed-loop cohort can itself be the confirmation experiment;
   a separate large fresh offline dataset is not logically required first. This does
   require prospectively frozen and validated independent initial-state families.
   Do not count repeated pairs or planning seeds from one family as independent families.

## What is actually supported by the paper

The authors evaluate planning success directly and also monitor offline prediction
proxies. They warn that accurate rollouts do not guarantee planning success. Their
typical planning evaluation uses 96 episodes per epoch, with initial/goal states
sampled from the dataset for Push-T and from the simulator for MetaWorld. Final-model
results include three training seeds and aggregation over the last ten epochs for
these tasks. Our released-checkpoint study does not reproduce that training history.

Using 96 paired planning scenarios per task/condition is a paper-scale replication
starting point, **not a power guarantee** and not an assertion of 96 fresh Push-T
families. The final confirmation n must be frozen using a specified meaningful
success-rate improvement and justified uncertainty/power assumptions before reveal.
The paper's dataset-based Push-T initialization is closest for planning replication;
genuinely new Push-T initial-state families are a documented extension, not its exact
published sampler. Simulator/code preparation can run concurrently with offline jobs.

Sources: [JEPA-WM, Sections 5.1–5.2 and Appendix G.2](https://arxiv.org/html/2512.24497v4#A7.SS2),
[official code and evaluation instructions](https://github.com/facebookresearch/jepa-wms).

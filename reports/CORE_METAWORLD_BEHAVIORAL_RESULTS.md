# Completed core MetaWorld behavioral-development comparison

## Bottom line

The complete five-arm panel does **not establish an intervention-driven improvement
in task success**. The refined fixed-response edit has positive point estimates
against unsteered on both tasks, but neither it nor vision–action coupling passes
the frozen improvement criteria against both unsteered and its matched-random
control. All eight simultaneous confidence intervals include zero.

This is a completed closed-loop simulation comparison, not an offline-prediction
proxy. It is also not proof that the methods have no effect: the intervals remain
wide. No intervention is advanced automatically, and no fresh confirmation was run.

## Complete success counts

Each cell is successes / 96 episodes. The same 96 initial/goal scenarios and
registered planner seed assignments are paired across the five arms within a task.

| Condition | Reach | Reach-Wall |
|---|---:|---:|
| Unsteered (`native`) | 43/96 — 44.79% | 29/96 — 30.21% |
| Refined fixed-response edit (`fixed_rank4`) | 49/96 — 51.04% | 30/96 — 31.25% |
| Matched-random fixed-response control | 48/96 — 50.00% | 23/96 — 23.96% |
| Vision–action coupling (`coupling_only`) | 47/96 — 48.96% | 25/96 — 26.04% |
| Matched-random coupling control | 58/96 — 60.42% | 27/96 — 28.13% |

The random-coupling control has the highest observed Reach rate. This is a
descriptive observation, not a newly validated method: random-versus-native was
not one of the eight predeclared primary contrasts, and no post-hoc promotion or
additional significance test is made here. The learned coupling does not beat it.

## All eight frozen paired contrasts

Effects and intervals are in **percentage points**, not relative percentages.
Intervals use 20,000 paired whole-scenario bootstrap draws with Bonferroni
simultaneous 95% adjustment across the eight contrasts. Exact paired-discordance
p-values are Holm-adjusted across the same family.

| Task | Intervention minus reference | Effect | Simultaneous 95% interval | Holm p |
|---|---|---:|---:|---:|
| Reach | Refined fixed-response minus unsteered | +6.25 | [−11.46, +23.96] | 1.000 |
| Reach | Refined fixed-response minus matched random | +1.04 | [−15.63, +17.71] | 1.000 |
| Reach | Coupling minus unsteered | +4.17 | [−15.63, +23.96] | 1.000 |
| Reach | Coupling minus matched random | −11.46 | [−29.17, +6.25] | 0.865 |
| Reach-Wall | Refined fixed-response minus unsteered | +1.04 | [−13.54, +15.63] | 1.000 |
| Reach-Wall | Refined fixed-response minus matched random | +7.29 | [−5.21, +19.79] | 1.000 |
| Reach-Wall | Coupling minus unsteered | −4.17 | [−20.83, +11.46] | 1.000 |
| Reach-Wall | Coupling minus matched random | −2.08 | [−15.63, +12.50] | 1.000 |

The frozen gate requires at least +5 percentage points versus native and positive
simultaneous lower bounds versus both native and the method's own random control.
Neither learned method qualifies on either task. The analysis therefore retains
`native` as the fallback on both tasks. This is a development-selection decision,
not a fresh-confirmation result or a reason to alter the frozen thresholds.

## Runtime and sample interpretation

| Condition | Mean Reach episode seconds | Mean Reach-Wall episode seconds |
|---|---:|---:|
| Unsteered | 299.98 | 301.58 |
| Refined fixed-response | 301.05 | 302.61 |
| Coupling | 299.81 | 301.44 |

These are descriptive run timings, not a separate controlled throughput benchmark.
The revised fixed-response edit is approximately one second above native in these
means; this panel did not restart the old expensive online response-probe operator.

There are **960 episode evaluations: 2 tasks × 5 arms × 96**. The scenario-cluster
audit found 96 distinct initial/goal clusters per task and no duplicate clusters.
The five repeated treatments do not create 480 independent scenarios per task.
Only one released model checkpoint was evaluated here: episode/planner seeds are
not independent model-training seeds. This is not the three-training-seed,
late-checkpoint-history study, and it does not establish six-task generalization.

## Verification, preservation, and scope

All 40 complete shard reports, 960 episode records, raw action/model-call traces,
paired initial/goal setups, and source/freeze bindings passed the receiving check.
An independent CPU reload and recomputation exactly reproduced the frozen analysis.
The complete eight-contrast family was retained; no partial-result selection was used.

- [Mirrored analysis](../artifacts/offline_study/core-completion-preservation-20260908-v1/ANALYSIS_REPORT.json)
- [Independent verification receipt](../artifacts/offline_study/core-completion-preservation-20260908-v1/SCIENCE_VERIFIED.json)
- Original worker report SHA256: `0bcfbae89587430782451429f8cad950fc5f8ecc3c0e4ae5a9a2acf88eae977a`.
- Frozen protocol SHA256: `853e8bf6b71c65c1e61d249ba84d8d57d6172ba1bd3a15e0424dc2d9e32cfa1f`.
- Frozen analysis-source SHA256: `229eea2764034cff7a4a5455ec759d57385c624dcbbbfab15021b699c61540dc`.

The final core/paused-HMM Google Cloud archive passed complete byte/member
readback. Google Drive verification and rental lifecycle are separate operational
gates: see the latest execution checkpoint and
[restoration guide](../artifacts/offline_study/core-completion-preservation-20260908-v1/RESTORE.md)
for the archive locations, verification receipts and shutdown status.
The later user-authorized storage closeout supersedes the original blanket
source-volume hold: verified rentals have been deleted after Drive preservation;
two inaccessible disks remain retained. See the dated
[final storage-release report](VAST_FINAL_STORAGE_RELEASE_20260910.md).
HMM, combined extensions, additional tasks, fresh confirmation, and training
histories remain paused/deferred under
the user's core-first priority. This completes the agreed core comparison, not
the full six-task study. No additional scientific job is authorized by this report.

# Six-task released-checkpoint table: verified progress

## Live update, September11,23:46UTC /7:46p.m.EDT

Still **34/48 complete: zero of the14 outstanding cells has completed**.
MetaWorld has124/768 new episode records:48 paired-native and76 component-edit
records. Reach visual42, action22, joint12; Reach-Wall new component records0.
Push-T has12 native,12 refined and10 random-subspace records. PointMaze has12
native,12 refined and2 random-subspace records. These are partial cohorts,
not96-episode cells. DROID fit is complete; its behavioral queue waits for the
current MetaWorld action stream (10/12). Wall fit still awaits its current
MetaWorld visual stream (6/12). All four owned GPUs were computing at the last
utilization check. Credit top-up verified:$14.358024, rate$6.093844/hour.

Operational repairs applied without restarting scientific children:

- Wall's two separate waiting CPU helpers were replaced by a direct fit ->
  CPU freeze -> engineering -> behavioral chain. Worker acknowledgement is
  `ops/wall-chain-v1/ADOPTED.json`; the existing scientific PID8771 was unchanged.
  This removes a SECOND MetaWorld batch wait after fitting, not the current
  intact batch. End-to-end live fit-to-behavior passage remains unobserved.
- Push-T and PointMaze's six-hour insertion timers were adopted by independent
  operational monitors. Active panel PIDs8258 and13596 were preserved. The new
  backstop is the earlier of12hours from original panel start and the existing
  lease; credit and$7/hour guards are unchanged. Actual12-episode stream times
  were1029/1034seconds on Push-T and963seconds for PointMaze native, making a
  six-hour full288-evaluation cap inadequate or marginal. This does not speed
  matrix operations; it prevents premature interruption of the requested panel.
- NJ's unstarted continuation now executes Reach5,Reach7,Reach-Wall5, removing
  48 Reach-Wall evaluations from ahead of its final Reach group. Original plan
  retained; an exact-hash ordering amendment cannot change the assigned set.
- The apparent old MetaWorld deadline problem was STALE QUEUE_PLAN metadata:
  live process arguments already match their renewed leases. No coordinator
  restart or deadline replacement was deployed for that disproven hypothesis.

Validation: full local suite748 tests,746passed/2skipped (unchanged torchvision
availability skips, previously passed on receiving worker).32 targeted component
tests passed before the ordering deployment; direct-chain/timer tests include
no intervening resume and no active-panel signal/restart. The scheduler source
is operational only; existing scientific snapshots, samples and outcomes remain
unchanged. No new GPU was rented during these repairs. First complete new cells
are still pending; deployment acknowledgements are not scientific completion.

## Execution investigation and repairs, September11,22:54UTC /6:54p.m.EDT

FourUS workers each have ONE GPU, not four/eight GPUs per worker. The live
snapshot recorded103 new MetaWorld scientific evaluations (NJ45,MO12,NV0 24,
NV3 22), not103 complete table cells. Still34/48 table cells complete;14 remain.
Account rate$6.093844/hour and credit$4.478403 at this snapshot imply roughly
29minutes to the$1.50 reserve before bandwidth charges. No end-to-end completion
promise is supported by that balance. Additional rentals remain disabled.

Confirmed deployment defect: only logical slots0–3 had assignments, so the
original queues could produce at most384/768 MetaWorld evaluations. This is
not a scientific need or a sample-size problem. Tested, deployed continuation
queues now reserve the missing task/slot pairs on the existing NJ/Nevada workers;
the slower Missouri worker receives no extra streams. The capacity allocator
now regards all eight slots as reserved, preventing duplicate future rentals.
Original children/source/protocols are untouched. Continuations await original
queue completion and retain the existing credit/deadline limits; this restores
coverage assignments, not a demonstrated optimal schedule or guaranteed finish.

The remaining task-specific fit-to-behavior setup was also incomplete. Status:

| Task | Verified state at this check |
|---|---|
| Push-T | Refined fit complete. Behavioral panel actually entered its GPU receiving checks at22:48:50UTC; no completed new scientific stream yet. |
| PointMaze | Refined fit complete. Isolated legacy environment passes all97 original initial/goal hashes; corrected historical source digest and freeze passed. Behavioral v3 queued behind the existing NJ joint stream. |
| Wall | All97 CPU simulator inputs match. Original pre-fit handoff failed before FIT_STARTED; native12-row predecessor has valid DONE. Detailed cause was not logged, so it cannot be asserted retrospectively. New v2 uses bounded driver-release waiting/normal exiting-process handling and detailed errors; unchanged fit and corrected follow-on behavior are queued. |
| DROID | Private dependency repair verified. GPU fit v2 still waiting for its current MetaWorld visual stream; behavioral inputs/runtime prepared and waiting for a verified fit. Not a completed fit or behavioral result. |

PointMaze v1 hit D4RL's declared Python<3.11 packaging constraint. V2 restored
the historically used Python3.11/pinned-source combination in isolation and
passed97 reset/render checks, then correctly refused a wrong source-directory
path. V3 uses the nested`baseline-source/src/offline_study` snapshot, with its
historical digest checked BEFORE staging. Failed attempts were preserved, not
relabeled as successful. No simulator replacement or outcome-based gate was used.

The full local software suite ran738 tests:736 passed and2 skipped because
torchvision is absent locally. Those exact two image-preprocessing parity tests
then ran on NJ's isolated worker environment:2passed,0skipped,CUDA uninitialized.
They are not experimental episodes. Subsequent targeted checks:26 component
operations tests and5 refined-panel queue/source tests passed.

Evidence: `component-extension-v1/coverage-repair-v1/COVERAGE.json`,
`component-extension-v1/coverage-repair-v1/LIVE-*.json`, each rental's
`CONTINUATION_LAUNCH.json`, PointMaze v3 receiving receipts, and remote
`ops/worker-parity-tests-v1/RESULT.json`. Scientific efficacy is still pending.

## Update,22:21UTC /6:21p.m.EDT

77/768 new MetaWorld scientific episode records (NJ37,MO9,NV-slot0 17,NV-slot3 14).
Still zero of the14 outstanding cells complete; the table remains34/48.
PointMaze's registered reconstruction/response fit has now completed, as has
Push-T's previously preserved fit. Neither is a completed new behavioral cell.
Push-T's full behavioral receiving/panel queue awaits its current MetaWorld
stream (visual-only,5/12) finishing. Wall fitting is still waiting.

DROID dependency recovery is now VALIDATED, not merely diagnosed:
`refined-droid-prerequisite-20260911-v2/IMPORTS_VERIFIED.json` confirms a complete
pinned DINOv3 hub import and original128-recording/512-prefix input contract,
without initializing CUDA. Missing requirements were installed in a private
package overlay; the shared Torch/NumPy/MetaWorld/MuJoCo environment is unchanged.
The unchanged GPU fit is queued after the current MetaWorld visual-only stream
(2/12); it is NOT completed. Source, model, population and dose are unchanged.
Implementation: `scripts/vast/repair_droid_fit_runtime.py`. Four boundary tests
passed, including rejecting shared/foreign-task import overlays.

Additional automatic rentals are disabled (capacity watcher ended and service
removed). Existing four workers continue; account$6.093844/hour, credit$7.696377,
approximately61minutes to the$1.50 safety reserve before bandwidth charges.
No full-study completion promise is supported by this balance or partial status.
The retry has its own Drive preservation monitor; old failures/results retained.

## Live reconciliation, approximately22:11UTC /6:11p.m.EDT

Still34/48 displayed cells complete and14 incomplete. The new MetaWorld
component panel has70/768 scientific episode records: NJ35, MO8, NV-slot0 15,
NV-slot3 12. Of these44 are new paired-native references and26 are edited
episodes; no additional complete96-episode table cell has finished. This is
not a repeat claim about the already completed original960-episode core panel.

Push-T's refined fixed-response fit completed with all fit/audit receipts in
33.12seconds, and its complete prerequisite archive passed private Drive byte
readback. Its separate actual behavioral panel has now been frozen on the
worker after exact validation-input checks and submitted for the next complete
MetaWorld child-stream boundary. It will perform excluded receiving checks,
then96TOTAL episodes each for native/refined/random on the same device. These
288 new behavioral evaluations have NOT started at this check. Source/queue:
`scripts/vast/stage_refined_pusht_behavior.py`, `refined_panel_queue.py`;
remote `/workspace/refined-pusht-behavior-20260911-v1` on50638073.

DROID's fit attempted execution but failed on missing`torchmetrics` while the
pinned DINOv3 hub module imported; no completed DROID refined fit or behavioral
panel is claimed. Failure/source artifacts were preserved to Drive and the
unchanged MetaWorld queue resumed. PointMaze and Wall's exact inputs are ready,
but their registered fitting pipelines still await complete-stream boundaries.
Their refined behavioral panels are not launched. No sample-size reduction or
alternative simulator is authorized by this operational delay.

Four active US GPUs, actual account rate$6.093844/hour including retained
storage; all four were observed computing MetaWorld. The proposed cheap Michigan
offer disappeared before acquisition. A separate availability request on owned
retained50546171 was rejected as unavailable and canceled; no new workload was
staged, no old confirmation restarted, and its disk remains retained.

Observed native throughput: about242s/episode on NJ,261s on NV-slot0,252s on
NV-slot3,542s on MO. Combined steady-state capacity is about50episodes/hour.
The remaining698 MetaWorld evaluations therefore represent roughly14hours of
work **even under an ideal allocation keeping all four GPUs occupied**. This
is a throughput extrapolation, NOT an end-to-end ETA: currently only four of
eight logical slots are assigned, refined jobs share these GPUs, and their
receiving/runtime readiness is incomplete. More capacity is not assumed.

Provider credit observed$8.618328; after the$1.50 reserve, approximately70minutes
remain at the current account rate before existing emergency spending guards.
Bandwidth may shorten this runway. Do not promise completion on this balance
or present the14-hour MetaWorld calculation as a validated full-study ETA.

## Earlier chronological status (superseded by the reconciliation above)

**Latest live check: September 11, 2026, 20:18 UTC (4:18 p.m. EDT).**
34 of 48 displayed project cells are complete; 14 still lack complete results.
The new MetaWorld component panel has collected 7 of 768 scientific evaluations,
all in its paired unsteered stream; these do not yet fill a missing intervention
cell. The 768 comprise 576 intervention evaluations and 192 paired references.
Eight refined/control cells on Push-T, PointMaze, Wall and DROID still require
task-specific fitting/behavioral setup: 704 intervention evaluations, with paired
reference verification additional. These eight cells are not running.

Two US devices are live: New Jersey50626847 collects scientific episodes;
Missouri50632757 is in full-episode receiving validation. Both showed100% GPU
utilization at the check, which is not proof of optimal throughput. Native
receiving episodes took242.24s on NJ5090 and538.49s on Missouri5000Ada. Do not
apply the NJ timing to Missouri. California50632763 failed startup without a
staged project workload; its watchdog stopped it with disk retained. Active
worker charges total$1.1074/hour; the provider reports$1.8117/hour including
all retained instance storage, before usage-based bandwidth. No reliable
end-to-end ETA is established while eight cells remain unimplemented and
most MetaWorld logical stream slots remain unassigned.

The complete Wall/PointMaze frozen analysis is finished. Use the
[consolidated results table](SIX_TASK_RESULTS_20260911.md),which includes the
published author rows,all15 recovered/completed cells,and14 explicitly unrun
cells. The chronological notes below preserve the earlier execution history.

Navigation audit: September11,2026,12:32UTC. Push-T full-panel verified:17:34UTC.
The five Push-T coupling/component gaps are now filled by the frozen analysis.

Ten previously blank cells were recovered and verified, not newly generated.
The audit checked original report/episode hashes, frozen source bindings,96 unique
episodes per complete arm, and identical paired initial/goal inputs. One PointMaze
stream's report and episodes were in separate archive batches; exact reconstruction
avoided an unnecessary rerun. Evidence:
`artifacts/offline_study/table-completion-20260911-v1/NAVIGATION_COMPLETE_CELLS.json`,
reproduced by `scripts/vast/audit_table_navigation_cells.py`.

| Model / intervention | Reach | Reach-Wall | Push-T | PointMaze | Wall | DROID score |
|---|---:|---:|---:|---:|---:|---:|
| Unsteered released checkpoint |44.79|30.21|59.38|80.21|76.04|51.10|
| Refined four-direction edit |51.04|31.25|—|—|—|—|
| Refined control: response-calibrated random-subspace edit |50.00|23.96|—|—|—|—|
| Equal-budget vision–action coupling |48.96|26.04|61.46|77.08|82.29|51.07|
| Coupling control: dose-matched random-direction edit |60.42|28.13|60.42|84.38|77.08|51.27|
| Unscaled joint vision–action edit |—|—|61.46|79.17|78.13|50.85|
| Visual-only edit |—|—|60.42|81.25|73.96|50.83|
| Action-conditioning-only edit |—|—|58.33|86.46|76.04|51.11|

The two control rows are separate experiments, now explicitly named rather than
both labeled "Its matched-random control." The refined control includes a
separately calibrated response map in a random four-dimensional subspace; the
coupling control changes the two edit directions at fixed sites and doses.
This is a presentation-only clarification, not new runs or changed scores.
See the [consolidated table and arm-ID mapping](SIX_TASK_RESULTS_20260911.md).

The first five task columns are simulator success percentages. DROID is the
recorded-plan action score, not physical robot task success. Unchanged MetaWorld
and DROID entries retain their existing completed-report provenance. The navigation
audit independently reloads navigation cells; the completed Push-T analysis checks
all864 records, immutable fits/source/protocol, device proofs and paired inputs.
Published author numbers are not mixed into this descriptive table: their
training-history aggregates are not an identical evaluation of this checkpoint.

New navigation entries each contain96 episodes. Their full nine-arm,two-task,
32-contrast frozen analysis is now complete after Wall's120 and PointMaze's132
remaining control episodes finished. No learned-intervention-versus-native
interval excludes zero. No contrast or arm was dropped to obtain an earlier
significant result. The original MetaWorld control audit is unchanged.

## Remaining displayed cells

| Work | Displayed blanks | Actual state |
|---|---:|---|
| Push-T coupling and component arms |0|All864 panel records verified; five displayed cells filled. Sixteen frozen contrasts analyzed with21-source-family clustering; no simultaneous interval excludes zero.|
| Reach/Reach-Wall joint, visual-only, action-only |6|Runner/analysis and queue implemented and frozen. At20:18UTC NJ50626847 has7 paired-native scientific episodes; Missouri50632757 is in receiving validation. No missing intervention cell is complete; the full eight-slot allocation is not live.|
| Refined edit and random on Push-T, PointMaze, Wall, DROID |8|Task-specific cheap adapters/fits/controls and behavioral freezes remain; MetaWorld readiness does not transfer automatically.|
| **Total** |**14**|Not all blanks are merely queued GPU work.|

## Operational update,12:52UTC

The user renewed the7USD/hour cap and added credit;26.849986USD was verified
after the top-up. Four initial replacement contracts were acquired, but two
failed before scientific outcomes. Indiana's PointMaze engineering revealed
missing legacy D4RL/mujoco-py/MuJoCo2.1 dependencies. California50588903 could
not reliably fetch official assets. Complete failed-attempt snapshots passed
Google Drive readback before both stopped; their disks and all prior results remain.

Texas50588893 and California50588914 continue Push-T input restoration. Maryland
50592039 replaces the failed California slice. Their intended disjoint allocation
is eight GPUs/$4.486667 per hour, plus retained storage, below7USD/hour. Instance
creation, input download and receiving-engineering completion are **not** measured
scientific progress. Per-device receiving checks must pass before any assigned
stream runs. PointMaze's132 missing controls are blocked on its legacy-runtime
repair; no modern simulator substitution is allowed. No end-to-end completion
ETA is established for the fourteen cells requiring further setup.

## Operational update,16:54UTC (supersedes earlier fleet status)

All58 missing Push-T streams completed: Texas360, California168, Maryland168
new episodes. Their final Push-T snapshots pass local archive hashes and have
full Drive-readback receipts. Together with the old72 candidate episodes and96
native episodes this supplies864 collected records for nine96-episode arms.
Collection and preservation do not replace the final frozen panel audit/analysis.

California50588914 and Maryland50592039 stopped after verified preservation.
Texas50588893 remains the one active project worker: four4090s,
1.734444USD/hour including its selected disk, retained-storage charges separate.
It now runs PointMaze receiving checks on all four devices. The repaired isolated
legacy runtime exactly reproduced97 saved initial/goal input pairs. The132
missing original control episodes follow only after per-device planner parity.
The failed dependency-only startup produced no accepted scientific episodes,
so it does not require replacing previously valid completed results.

The fourteen cells needing new task-specific engineering remain unlaunched.
No new random-direction sweep, confirmation or training campaign was started.

16:55UTC: PointMaze receiving checks passed on the first two devices; the
original permuted_visual streams0–1 are now running. The other two devices
remain in their last receiving checks before automatic handoff. No claim of
132 completed new episodes or full navigation analysis is made at launch.

## Push-T final analysis verified,17:34UTC

The original frozen nine-arm analysis completed, with96 paired episodes per arm
from21 released source initial-state families. Source-family bootstrap20,000
draws, original16-contrast family, no altered admission gate or replacement
samples. Equal-budget coupling59/96 versus native57/96 is+2.08 points;
its equal-budget random control achieves58/96. These small observed differences
do not establish a reliable improvement. No original simultaneous contrast
interval excludes zero; zero-touching intervals are not positive findings.

Analysis report SHA256:
`9b64cbdbf0736cd372c06e5a21fe705ed71e5e2883d10e3a86823faf493bca89`.
Remote canonical output: Texas50588893,
`/workspace/table-completion-20260911-v1/expansion-20260911-v1/pusht-complete-audit-v2`.
The report, DONE and bindings also have a locally checksum-verified copy in
`artifacts/offline_study/table-completion-20260911-v1/pusht-final-analysis-v2/`.
At17:36UTC the complete analysis is additionally present in Drive snapshot
`13b66AVWdifMi_9G-7_saFPhn0yuXcLIY`, verified by full-byte readback and member
hashes; local receipt `expansion-v2/50588893/snapshot-1789148112020498252.tar.verified.json`
under the table-completion artifact root.
Assembly used three full-member/Drive-byte-verified archives and the six old
complete streams. The initial assembly used a historical `GPU-UUID` directory
name where the analyzer expects the recorded UUID; v2 copies the same proof
bytes under the correct lookup name. Failed v1 remains preserved, and no model
call or scientific episode was rerun to fix this archive-path issue.

All four Texas GPUs continue PointMaze. A same-device fallback can finish
only original timeout-margin leftovers, without changing the18:17UTC backstop
or any scientific condition. Complete streams are hash-checked and skipped;
the collector waits for both original and fallback queues before preserved stop.
At17:36UTC all three last permuted-joint streams had started in the original
queues; no fallback rerun had been needed. The fallback remains skip-only for
any stream that completes normally.

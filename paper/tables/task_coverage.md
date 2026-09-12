# Task coverage and endpoint inventory

Snapshot from completed primary/successor artifacts and the local status records dated September 8, 2026, through 07:42 UTC. This document does not query live GPU jobs or inspect partial candidate outcomes. A dash means not included in the completed primary cohort, not a zero effect.

| Task | Independent primary offline trajectories | Fixed-response successor offline | Behavioral endpoint | Complete intervention success estimate available for this draft? |
|---|---:|---|---|---|
| Reach | 33 | Complete; same 33 trajectories | Simulator task success | No |
| Reach-Wall | 27 | Complete; same 27 trajectories | Simulator task success | No |
| Push-T | 21 | Not measured | Dataset-initialized short-horizon simulator assessment | No |
| Wall | — | Not measured | Simulator navigation success | No |
| PointMaze | — | Not measured | Simulator navigation success | No |
| DROID | — | Not measured | Agreement with recorded robot actions | No; also not a physical execution success endpoint |

Primary total: **81 unique trajectories**, not 141 after adding the 60 reused successor trajectories. All five original sweeps are complete for Reach, Reach-Wall, and Push-T in both precisions. Navigation has additional offline coupling results according to the coverage record, but those are not part of the three-task primary closure or figures in this draft.

The wider intended scope is six tasks. RoboCasa is excluded. Existing native behavioral references and engineering episodes do not supply a completed paired intervention comparison. The planned Reach/Reach-Wall successor panel has five arms and 96 episodes per task/arm, totaling 960 episodes; it does not contain the original combined operator. Other task/category panels have separate registries.

Sources: [corrected primary results](../../reports/CORRECTED_OFFLINE_RESULTS.md), [successor contract/results](../../docs/FIXED_RESPONSE_RANK4.md), [ablation coverage](../../reports/ABLATION_COVERAGE_STATUS.md), [timestamped execution status](../../reports/EXECUTION_STATUS.md).

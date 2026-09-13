# MetaWorld paired-stimulus repair — 2026-09-07

> Historical development report. For the completed protected evaluation and current
> interpretation, see [Results](../docs/RESULTS.md). Archived references below are
> retained as paths; bulk evidence is not bundled.


This is an engineering correction to the frozen behavioral panel, not a new
method, dataset split, sample-selection rule or confirmation cohort.

The outcome-blind audit found five mismatched goal-image hashes among286 completed
candidate episodes. Initial images, physical task vectors and seeds matched.
Twenty-four native expert repetitions reproduced both historical image hashes:
physical states, proprioception and expert actions were exact, while4–12 rendered
RGB channel values differed by one level. The source data and split did not change.
Evidence: `primary-durable-20260907/metaworld-goal-repeat-20260907-v1`; original
FAILED audit: `behavioral-pairing-audit-20260907-v3`.

All192 original baseline stimuli were recovered using at most16 native expert
renders per original hash. No new goal or physical scenario was accepted, and no
candidate outcome was consulted. The recovery bank is checksum-bound to the
complete original baseline reports and scientific freeze. Raw tensors remain on
the main US worker; compact metadata is preserved locally.

For new runs, the original source goal helper still executes. Delivery requires
identical initial bytes, task vector, physical goal, proprioception and expert
actions. Only the goal observation is replaced with the original native pixels.
Unknown render changes fail closed; delivered pixels must be **bitwise exact**.
The one-level/64-value renderer guard is an engineering fault detector, not a
relaxed pairing criterion. The actual repeated source-environment check passed24
deliveries, all192 baseline/cache bindings and unchanged planner candidate RNG.

Affected old coupling runs finish their complete original shards. An input-only
manifest then identifies complete12-episode logical RNG streams containing any
goal mismatch. The entire stream is rerun, including its previously matching
episodes. For example, episode20 belongs to logical stream1 (episodes12–23), not
`20 % 8`. This preserves persistent CEM RNG consumption. Original runs are never
overwritten; final assembly must replace their entire affected streams explicitly,
not append duplicate observations or choose the better outcome. Mismatches in
anything other than the goal hash stop this repair path.

Three incomplete expensive rank/combined streams were stopped with traces
preserved and restarted in a distinct canonical output root. The frozen96 episodes
per condition, seven conditions, controls, seeds, CEM budgets and statistical
comparisons remain unchanged. Scientific repair is not complete until replacement
receipts and the fully assembled paired panel verify. No confirmation is opened.

Implementation: `planning_goal_bank.py`, `verify_planning_goal_bank.py`,
`behavioral_repair_plan.py`, and the goal-bound candidate runner. The investigation
workflow required a reproduced cause and source-runtime regression check before
production repair; the passed check is not itself behavioral efficacy evidence.

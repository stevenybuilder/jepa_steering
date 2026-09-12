# MetaWorld component completion — September 11, 2026

This is the user-authorized completion of six missing released-checkpoint table
cells, not a new confirmation study, candidate search, or training-history run.
Existing results, fits, source snapshots and statistical tests stay unchanged.

## Fixed comparison

- Tasks: Reach and Reach-Wall.
- Arms: native, visual-only, action-conditioning-only, unscaled joint.
- Reuse the original task-specific BF16 coupling fits without refitting, direction
  selection or dose tuning. The joint arm retains both full component doses.
- Retain the core panel's strict FP32 CEM-L2 planner, context two, 300 candidates,
  native iterations, 100 elementary simulator steps, and H3 edits only within a
  full H6 forecast. Shorter forecasts are genuinely unedited.
- Reuse the 96 canonical base-2026090721 development scenarios per task and their
  eight persistent logical RNG streams. Never multiply 96 by physical GPU count.
- Collect a paired native reference on the receiving runtime. Compare it with
  the old native records before combining new cells with historical comparisons;
  report a separate reproduction if the outcomes differ. Old native scores must
  not be overwritten silently. These 192 reference evaluations are additional
  to the 576 missing component evaluations, not 192 new independent scenarios.
- Keep source/fitting, goals, scenarios, checkpoint and receiving-device receipts.
  No offline significance veto; no protected base-1 access; no expensive rank edit.

## Analysis frozen before these new outcomes

Primary endpoint: official binary task success. Six contrasts per task:
each of the three component arms minus native; joint minus visual; joint minus
action; and the factorial success interaction joint − visual − action + native.
The interaction is on success probabilities, not an activation interaction.
Use all 12 contrasts, 20,000 paired scenario-cluster bootstrap draws, seed
2026091101, and Bonferroni simultaneous 95% intervals. Report all effects,
including nulls and harms. Summarize distance, reward and wall-clock runtime
descriptively. Do not select a new candidate or open confirmation automatically.
Related historical results have already been inspected; this is a pre-execution
development extension, not retrospective preregistration or independent evidence.

## Engineering and execution

Before scientific episodes, each executing GPU/task must complete native,
native-repeat, zero-dose, visual-only, action-only and joint checks on the
excluded SMOKE_SEED scenario. Native repeat and zero dose must match native
actions and outcomes exactly. All conditions must share initial/goal inputs,
retain the full CEM schedule, use one model unroll per forecast, and deliver
their specified edit magnitudes. Checkpoint weights and source stay unchanged.

Queue whole 12-episode logical streams in separate output directories. An
interrupted stream is preserved; restart it in a new directory unless exact RNG
resume is available. Never append a new RNG sequence to an old partial stream.
Verify complete reports and raw trace hashes before analysis. Preserve partial,
failed and complete results in Google Drive and verify readback before routine
instance stop. A credit/deadline emergency stop retains the source disk.

Budget: US-only, aggregate fleet at most $7/hour; actual credit can impose a
smaller initial batch. No automatic credit purchase. All other paused work stays
paused. The four non-MetaWorld refined/control pairs require a separate frozen
task-specific extension and do not become runnable merely through this module.

September11,19:46UTC execution correction: acquire disjoint remaining logical
slots while the pilot finishes its final receiving check. New-worker backstops
are10hours, not6: the first actual native episode took242.24seconds, implying
about7.3hours for96scientific and12receiving episodes on a similarly fast GPU.
The$7/hour cap and$1.50credit reserve remain; this is not a guarantee that the
current balance funds the entire panel. Scientific source/sample/analysis freeze
is unchanged. This replaces an inadequate initial operational duration estimate.

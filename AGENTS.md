# Active research workflow

Read docs/EXPERIMENT_PLAN.md and configs/study.json before scientific changes.

- Primary tasks: mw-reach, mw-reach-wall, pusht. Never report a single-task run as
  completion of the suite. Report both requested and actually measured task counts.
- archive/ is historical evidence. Do not edit it or execute its scripts as the new
  protocol without explicitly porting and validating the needed behavior.
- Keep fitting, development, and protected evaluation separate by trajectory.
  Existing protected cohorts retain their protection; reshuffling does not undo exposure.
- The current benchmark accepts only fit/development. Holdout execution requires a
  frozen scientific protocol and a verified exposure registry, not a bypass flag.
- Never call script completion research completion. Distinguish synthetic smoke,
  real baseline, intervention measurement, and scientific confirmation in receipts.
- Do not extrapolate synthetic CPU timings to JEPA-WM, GPUs, or scientific accuracy.
- No autonomous method search. Log a specific hypothesis and fixed arm registry before
  an intervention comparison. New results do not authorize an unregistered successor.
- Split independent work by trajectory, without DistributedSampler padding duplicates.
  Multiple windows/candidates from one trajectory are not independent samples.
- Never rent compute or send messages automatically. GPU spending needs the user's
  authorized budget and a verified current execution environment.
- Keep caches/checkpoints/raw video out of git. Record source/checkpoint hashes and
  persist new run artifacts to configured durable storage before terminating a worker.

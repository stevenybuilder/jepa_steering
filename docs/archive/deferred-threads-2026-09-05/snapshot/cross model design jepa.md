# Archived design pointer and label invariant

The superseded cross-model safety/driving design is preserved at [docs/archive/legacy-branches-2026-09-04/root-plans/cross model design jepa.md](docs/archive/legacy-branches-2026-09-04/root-plans/cross%20model%20design%20jepa.md).

The active design is [WORLD_MODEL_SONAR_PLAN.md](WORLD_MODEL_SONAR_PLAN.md).

The binding rule retained for compatibility with code comments is:

> **Label-first principle:** the primary behavioral outcome is the released evaluator's native per-episode simulator success flag. Activation geometry, planner cost, probes, video judges, and researcher-defined proxies may not replace or modify that label. The checkpoint, task, goal sampler, horizon, planner compute, and success definition remain fixed across paired arms.


# JEPA-WM causal geometry and steering

The active focus is this morning's plan: discover task-relevant geometry in the frozen JEPA-WM on Reach-Wall, test whether that geometry affects planning, and derive the simplest supported intervention.

Start here:

1. [Geometry Map Experiment](<Geometry Map Experiment.md>) — active execution plan, evidence, splits, and causal gates.
2. [Morning Ideation](<morning ideation.md>) — motivation, cached replay, paired behavioral evaluation, and statistical limits.
3. [Physics Emergence Zone](<physics emergence zone.md>) — layer/subspace hypotheses and the practical steering objective.
4. [Geometry code guide](scripts/geometry_map/README.md) and [artifact map](artifacts/README.md).

The current geometry plan and frozen [on-policy manifest](docs/manifests/geometry-map-reach-wall-v1/on_policy_bank_manifest.json) govern execution. The earlier ideation's static/HMM arm proposal is conditional; HMM routing does not enter the first test without evidence that it is needed.

Geometry needs controls: held-out trajectories and collection seeds, shuffled or nuisance-variable comparisons, simple readouts, and matched causal patches. Steering compares the same frozen checkpoint and planner unsteered, with a matched sham to distinguish a specific mechanism from generic perturbation. Native simulator success remains the behavioral endpoint.

Compute remains a planned later evaluation: record discovery GPU-hours and cost, runtime overhead, and paired task success. After a first behavioral effect, compare with a small LoRA/adapter under explicit adaptation and inference budgets. No compute-efficiency claim is supported until measured. See the [deferred work register](docs/archive/deferred-threads-2026-09-05/README.md).

Push-T is replication after the Reach-Wall mapping rules are frozen. Data preparation may proceed under existing resource leases. The broad Frankenstein composition, DINO-WM Reach campaign, driving, RoboCasa, and VLA branches are historical or deferred; they do not govern this experiment.

[Archive index](docs/archive/README.md) preserves the old plans and their locations. [Math techniques](math_techniques.md) and [VLA math insights](math_insights_for_vlas.md) remain reference notes, not additional execution requirements.

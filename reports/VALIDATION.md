# Validation receipt

Executed in the available CPU workspace on 2026-09-07.

- Nine unit checks passed: exact split counts/order stability; duplicate IDs; H6 window bounds;
  eight-shard coverage without duplicates; equal trajectory weighting; horizon metric alignment;
  no future-observation input; zero-dose identity; task coverage and invalid-input rejection.
- End-to-end synthetic smoke: 12 fixture trajectories, 48 windows, six recursive prediction steps,
  all three task labels represented, exact instrumentation identity.
- Encoded-cache writing was enabled and completed.
- Fixture backend: tiny deterministic PyTorch model, **not JEPA-WM**.
- [Machine-readable smoke report](cpu_smoke.json): gpu_benchmark_valid=false; efficacy_claim=false.

No real checkpoint/dataset inference or GPU measurement was executed. No GPU cost estimate
can be derived from this smoke report. The real adapter and data loader require integration
validation on the pinned upstream environment and downloaded data before their metrics are trusted.
No intervention runner is marked complete by these checks.

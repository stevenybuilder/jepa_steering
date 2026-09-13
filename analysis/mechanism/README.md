# Post-confirmation mechanism analyses

All six scripts completed on the final four-task panel (384 scenarios, 3,072 arms).
The reviewed interpretation is [Results](../../docs/RESULTS.md), not every sentence
in automatically generated output. These are exploratory diagnostics except for
explicit replay of the 48 frozen contrasts.

| Script | Measurement |
|---|---|
| regime_report.py | Frozen confidence-interval replay and comparison regimes |
| specificity_pathway.py | Dose, pathway contrasts and arm ordering |
| scenario_heterogeneity.py | Rescues, regressions and flip-set overlap |
| forecast_decision_outcome.py | Available evidence along the forecast/decision/outcome chain |
| representation_geometry.py | Basis support, overlaps and delivered coefficient dimension |
| planner_dynamics.py | Action-hash divergence, planning schedules and logged timing |
| audit_headlines.py | Independent streaming recount of headline quantities |

## Replay

Install the project's analysis extra. Restore the private archives using
[the reproduction guide](../../docs/REPRODUCING.md), then run from the repo root:

```bash
python analysis/mechanism/run_all.py --results RESTORED_RESULTS \
  --freeze RESTORED_FREEZE --analysis RESTORED_FROZEN_ANALYSIS --out analysis/out/replay
```

See [DATA_CONTRACT.md](DATA_CONTRACT.md) for required records. Some diagnostics
also use historical aggregates and operator banks; restore those explicitly.
Missing input coverage must not be interpreted as a null measurement.

## Interpretation and schema correction

Fresh coupling energy is logged as means, while older schemas used sums.
The corrected reader handles both and validates population counts; it does not
invent missing active-candidate counts. Original generated missing-dose warnings
were erroneous. This did not change primary success rates or frozen intervals.

Generated prose is an analysis aid, not a final scientific claim. Action hashes
cannot quantify physical action changes. Flip overlap cannot establish what any
perturbation would do. Basis concentration is not causal semantic localization.
No candidate-score or elite-rank traces exist for the completed fresh panel.

Synthetic fixtures from make_fixture.py test software only. Neither those
fixtures nor later candidate-trace/attention pilot code are new scientific results.

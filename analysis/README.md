# Mechanism analyses

CPU-only diagnostics that read archived experiment outputs and write the aggregates
in `paper/data/`. None of these scripts loads a model, launches compute, or selects
an intervention after seeing results; each one is bound to a frozen protocol by
hash and recounts from raw per-case records.

| Question | Script |
|---|---|
| Does an action patch that is exact at its first read diverge at its second? (200-state replication) | `mechanism/lcfm_replication_summary.py` |
| How does that divergence reorder the H6 candidate ranking and change selection cost? | `mechanism/lcfm_history_ranking.py`, `mechanism/lcfm_history_selection_cost.py` |
| The sixteen-state exploratory action counterfactual with action-range controls | `mechanism/action_counterfactual_summary.py`, `mechanism/action_counterfactual_plot.py` |
| Does alignment with the action encoder matter beyond edit magnitude? | `mechanism/action_condition_summary.py` |
| Which single predictor block a learned edit needs, and how random controls compare | `mechanism/steering_specificity.py` |
| Does an edit change the Cross-Entropy Method search after the first selection? | `mechanism/cem_expansion_summary.py`, `mechanism/cem_steering_summary.py`, `mechanism/planner_dynamics.py` |
| Score-margin bound: when can a bounded cost perturbation change the winner? | `mechanism/decision_geometry.py` |
| Forecast error, decision change, and physical outcome as one chain | `mechanism/forecast_decision_outcome.py`, `mechanism/planned_prefix_summary.py` |
| Cubic versus linear reconstruction under FP32 and BF16 | `mechanism/controlled_geometry_summary.py`, `mechanism/pathway_geometry.py` |
| Geometry of the fitted rank-four intervention subspace | `mechanism/representation_geometry.py` |
| Which scenarios an edit rescues or breaks, and whether that is stable | `mechanism/scenario_heterogeneity.py`, `mechanism/specificity_pathway.py` |
| Attention, CEM proposals, and cached-component replay in the development cohorts | `mechanism/pilot_summary.py` |
| Independent recount of every headline number from raw arms | `mechanism/audit_headlines.py` |

`mechanism/common.py` holds the shared loaders and paired-bootstrap statistics.
`mechanism/run_all.py` runs the post-confirmation suite on one results tree and
writes an index. `mechanism/make_fixture.py` builds a synthetic results tree in the
production schema so the unit tests can exercise every script without archived data.

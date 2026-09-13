# Experiment analysis and figure inventory

Reconciled September 13, 2026. The completed experiment cohorts below have analysis outputs and figures. The final publication gap was the 56-context physical-prefix follow-up: its verified results and plots existed locally, while its report still described prospective work. The report, README and manuscript now include those results. No experiment was rerun for this publication update.

| Experiment or reanalysis | Completed coverage | Analysis and figures |
|---|---|---|
| Fresh protected confirmation | 384 scenarios × 8 arms; all four tasks; 48 frozen contrasts | [Results](RESULTS.md), [benchmark](figures/benchmark_readable.png), [immutable report](../reports/fresh-confirmation/report.json) |
| Six post-confirmation diagnostics | Regimes, pathway specificity, scenario heterogeneity, forecast/outcome evidence, representation geometry, planner dynamics; all six completed on the real panel | [Analysis suite](../analysis/mechanism/README.md), [reviewed interpretation](RESULTS.md#mechanistic-diagnostics); full generated outputs retained in the private analysis archive |
| Earlier rank-one layer sweep | 576 cells; three tasks, two precisions, all eight supports, both modalities and six horizons | [Layer report and all four heatmaps](MECHANISMS.md#layer-response-map), [CSV](../paper/data/layer_mechanism_grid.csv) |
| Protected coefficient specificity | 768 refined/random arm records; all 384 scenarios | [Decomposition](MECHANISMS.md#candidate-specific-versus-common-correction), [figure](figures/candidate_specificity.png) |
| Fixed-bank decision margins | 192 contexts × 5 arms × 300 shared candidates | [Decision report](MECHANISMS.md#decision-margin-reanalysis), [figure](figures/decision_margin_story.png) |
| Archived pathway and interpolation geometry | Five-task reconstruction/forecast and interaction analysis, with precision/source boundaries retained | [Report](PATHWAY_GEOMETRY.md), [precision figure](figures/pathway_geometry_precision.png), [interaction figure](figures/paper_action_interaction.png) |
| Native attention and component replay | 64 contexts; all six layers, sixteen heads and six horizons | [Report](PILOT_MECHANISMS.md), [attention map](figures/paper_pilot_attention.png), [component figure](figures/pilot_replay_components.png) |
| Initial adaptive CEM instrumentation | 8 contexts; complete traced/untraced parity | [Report](PILOT_MECHANISMS.md), [search figure](figures/cem_steering_search.png) |
| Controlled numerical geometry | 64 contexts × 6 blocks × 3 conditions | [Report](CONTROLLED_GEOMETRY.md), [figure](figures/paper_controlled_geometry.png) |
| Action-conditioning specificity | 16 contexts; both banks and all six layers | [Report](ACTION_CONDITION_SPECIFICITY.md), [figure](figures/action_condition_specificity.png) |
| Coherent action counterfactuals | 16 contexts; 1,120 forecasts; all 72 primary contrasts | [Report](ACTION_COUNTERFACTUAL.md), [history figure](figures/action_history_consistency.png), [full layer figure](figures/paper_action_counterfactual.png) |
| Adaptive CEM extension | 56 new contexts; 336 searches; all six primary contrasts; initial eight kept separate | [Report](CEM_EXPANSION.md), [prefixes](figures/cem_expansion_prefixes.png), [search](figures/cem_expansion_search.png), [entropy](figures/cem_expansion_entropy.png) |
| Physical selected-prefix replay | 56 contexts; 224 trajectories; 504 scientific forecasts; all twelve primary contrasts | [Report](PLANNED_PREFIX_REPLAY.md), [effects](figures/planned_prefix_effects.png), [complete prediction/execution grid](figures/planned_prefix_forecasts.png) |

## What completion means

Completion means the registered cohort was analyzed and its figures exist, not that every contrast is significant or every research question is resolved. The fresh panel did not save numeric candidate costs or elite ranks; those cannot be recovered from action hashes. Its forecast/outcome analysis therefore reports an evidence gap. Later development experiments do not retroactively fill those protected traces.

Push-T and DROID remain development-only in the six-task comparison. They were not rerun in the four-task protected protocol. DROID measures recorded-action agreement. Historical base-1 confirmation notes describe a separate unfinished protocol, not missing cases from the completed fresh four-task panel. Full-population physical candidate ranking, further adaptation and new testbeds remain future work.

Qualitative rollout media is a separate presentation task and is not yet produced. No paired JEPA rollout videos were found in the local media inventory. The physical-prefix recorder retained actions, states and endpoint observations, but not intermediate visual frames. A synchronized unsteered/learned/random GIF would therefore require a rendering replay, with task/context provenance and an explicit H3-prefix label. The existing plots do not require that replay.

## Verification and preservation

- The public CPU suite passes all 305 tests, including physical-runner contracts, cohort completeness, paired statistics, figure inputs and the earlier mechanism analyses.
- The physical analysis is rerun from its unchanged v2 freeze and verified compact records; published tables remain bound to the report/source hashes. Figure generation checks all 56 contexts, 504 crossed model/plan points and twelve primary cells.
- Raw result archives and technical logs retain generation-pinned GCS download-SHA receipts. A fresh cloud metadata audit checks object generations and sizes against those receipts. It does not pretend to repeat model inference or every prior bulk download.
- Repository source, public aggregates, figures and the rendered manuscript are versioned together. The publication archive also retains the six generated post-confirmation reports and the corrected dose-reader output; curated scientific interpretation is in [Results](RESULTS.md).
- GPU closeout is complete: no Vast instances or volumes. Private cloud storage remains intentional; no raw archive is deleted during publication cleanup.

Private durable storage is under `gs://rgt-jepa-archive-2026/fresh-campaign-20260912-v2/` and `gs://rgt-jepa-archive-2026/mechanism-20260913/`. Publication snapshots use an immutable commit-specific object; the matching `publication-<commit>-CLOUD_VERIFIED.json` receipt is retained in the local campaign operations folder. Access to the Git repository does not grant access to these private archives.

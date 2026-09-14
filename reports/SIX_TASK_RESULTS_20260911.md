# Six-task results: completed measurements

> Historical development report. For the completed protected evaluation and current
> interpretation, see [Results](../docs/RESULTS.md). Archived references below are
> retained as paths; bulk evidence is not bundled.


Updated September11,2026. All scheduled Push-T, Wall and PointMaze continuation
runs and their complete frozen analyses are finished. **The entire requested
matrix is not finished:**14 cells require experiments that have not been run.
This table fills15 of the29 blanks in the user's original table.

**NR = not run / no completed comparable measurement; never zero.** The first
five columns are simulator task-success percentages. DROID is a recorded-plan
action score, **not physical-robot task success**. Our simulation entries use96
episodes per condition; DROID uses64 recorded-plan endpoints.

| Model / intervention | Reach | Reach-Wall | Push-T | PointMaze | Wall | DROID score |
|---|---:|---:|---:|---:|---:|---:|
| Authors: DINO-WM baseline |44.8|35.1|66.0|81.6|64.1|39.4|
| Authors: improved JEPA-WM, CEM-L2 |58.2|41.6|70.2|83.9|78.8|48.2|
| Authors: same recipe, CEM-L1 |55.1|40.8|63.4|79.7|46.7|47.2|
| Authors: final checkpoints, CEM-L2, three-seed mean |49.0|29.2|69.4|83.3|80.9|46.5|
| Our released checkpoint, unsteered |44.79|30.21|59.38|80.21|76.04|51.10|
| Our refined four-direction edit |51.04|31.25|NR|NR|NR|NR|
| Refined control: response-calibrated random-subspace edit |50.00|23.96|NR|NR|NR|NR|
| Our equal-budget vision–action coupling |48.96|26.04|61.46|77.08|82.29|51.07|
| Coupling control: dose-matched random-direction edit |60.42|28.13|60.42|84.38|77.08|51.27|
| Our unscaled joint vision–action edit |NR|NR|61.46|79.17|78.13|50.85|
| Our visual-only edit |NR|NR|60.42|81.25|73.96|50.83|
| Our action-conditioning-only edit |NR|NR|58.33|86.46|76.04|51.11|

**Two distinct control rows, not a pooled random baseline.** The refined control
uses a fixed random four-dimensional basis with its own fitted response map;
the learned error readout, support, calibration procedure and activation dose
are retained. Its coefficient-map spectrum is not matched. The coupling control
replaces both learned directions with fixed orthogonal random directions at the
same sites, times and per-site doses, including the same equal-budget scaling.
Neither control selects random robot actions or changes the pretrained weights.

The September 11 naming clarification changes labels only: all scores, NR cells,
source arm IDs and frozen comparisons are unchanged. The refined control maps
to `matched_random_fixed_rank4`; the coupling control maps to
`matched_random_coupling` in the MetaWorld runner and
`matched_random_equal_standardized_energy` in the other displayed task panels.
Both control comparisons hold intervention location fixed. They do not establish
that location matters more than direction, or that learned and random edits are
equivalent. See control definitions (archived reference: `../matched random control.md#4-our-controls-exact-definitions-and-gaps`).

Author values come from JEPA-WM v4, Tables11–12.
The first three published rows use late-training aggregation; the final-checkpoint
row still averages three trained seeds. Our rows use one released checkpoint.
Published numbers are context, not substitutes for our matched unsteered reference
or evidence that our edits outperform the authors' recipe. Training histories and
fresh confirmation are not completed by this analysis. DROID's higher reproduced
baseline is not an intervention improvement; see the [baseline audit](DROID_BASELINE_RECONCILIATION_AUDIT.md).

## Completed Wall/PointMaze analysis

The two-task panel contains**1,728 evaluations**:two tasks × nine arms ×96
episodes. Both tasks have96 distinct initial/goal scenario clusters. The original
32 contrasts,20,000 bootstrap draws and seed2026090801 are unchanged. Inputs,
fit/source/checkpoint receipt bindings, complete streams and paired initial/goal
hashes pass. All11 original PointMaze continuation streams completed without
needing the reserved fallback rerun. No GPU inference was used for this analysis.

| Selected contrast | Observed difference (points) | Frozen simultaneous95% interval | Interpretation |
|---|---:|---:|---|
| Wall: equal-budget coupling − unsteered |+6.25|[−3.13,+17.71]|Positive estimate; improvement not established.|
| Wall: equal-budget coupling − its random control |+5.21|[−4.17,+15.63]|Learned-direction advantage not established.|
| PointMaze: action-only − unsteered |+6.25|[−4.56,+17.71]|Positive estimate; improvement not established.|
| PointMaze: equal-budget coupling − unsteered |−3.13|[−16.67,+10.42]|Negative estimate, not proof of harm.|
| PointMaze: unscaled joint − action-only |−7.29|[−16.67,−1.04]|Only frozen interval excluding zero; component comparison, not benefit versus unsteered.|

The last row needs special caution:the two arms differ on only seven scenarios,
all favoring action-only. A supplementary exact paired-discordance calculation
gives unadjusted `p=0.015625`, or `min(1,32*p)=0.5` for a Bonferroni sensitivity
check. This is **not** a replacement for the registered bootstrap analysis;
it flags that the sole exclusion is sensitive to inference method with sparse,
one-sided discordance. It should not be promoted as a robust confirmed discovery.
None of the registered learned-intervention-versus-native intervals excludes zero.
The complete32-contrast report, including negative and null estimates, is retained.

## Other completed panels

- Push-T:864 evaluations,nine96-episode arms,21 released source initial-state
  families. Equal-budget coupling gains2.08 points over native; its simultaneous
  interval is[−3.96,+8.51]. All16 frozen intervals include or touch zero.
- Reach/Reach-Wall:the original960-episode,five-arm core analysis remains
  unchanged. No learned intervention established improvement against both
  native and its specified matched-random control.
- DROID:the previously completed recorded-plan analysis remains unchanged.
  The displayed edits are close to the51.10 unsteered score. Do not combine
  DROID scores and simulator success into a pooled success percentage.

## What is still genuinely unmeasured?

| Missing work | Cells | Why analysis cannot fill them |
|---|---:|---|
| Unscaled joint,visual-only,action-only on Reach and Reach-Wall |6|Implementation and pre-execution freeze are complete; collection has begun baseline-first, but no component cell has a complete measurement. See the timestamped progress report for live counts.|
| Refined four-direction edit and its random control on Push-T,PointMaze,Wall,DROID |8|These task-specific refined panels were not executed. The completed static coupling panels are different interventions.|

No zero,estimate from another task,offline forecast metric or partial run is
substituted for an NR cell. The old expensive online response-probe edit was not
reactivated. The latest completion batch does not establish seed-level recipe
generalization or fresh-family confirmation.

## Provenance and reproduction

The provenance workflow binds the table to these actual analysis artifacts:

- Core:`artifacts/offline_study/core-completion-preservation-20260908-v1/ANALYSIS_REPORT.json`.
- Push-T:`artifacts/offline_study/table-completion-20260911-v1/pusht-final-analysis-v2/report.json`.
- Wall/PointMaze:`artifacts/offline_study/table-completion-20260911-v1/navigation-final-analysis-v1/analysis/report.json`.
- DROID:`artifacts/offline_study/table-completion-20260911-v1/restored/50259194/batch-000/jepa-runtime/droid-coupling-behavior-20260908-v3/analysis/report.json`.

Navigation analysis report SHA256:
`473c8c59c92be779d7f3d2d989b039107957ff01a9872585f88cb38d092db5ce`.
Frozen scientific source package SHA256:
`fc7578672eabd8e08469afcc1f7f4d9b1c2ed18c447f4f003539b61da7950f58`.
The original analysis module is byte-identical to the current checked-in module;
execution imports the complete preserved source package. `ASSEMBLY.json` records
source archives,contracts and the validated receiving proofs.

The campaign-specific assembly scripts remain in the private execution archive.
For analysis reproduction from the preserved self-contained output, put its
`frozen-source` directory on `PYTHONPATH`,then run
`python -m offline_study.tasks.navigation.navigation_coupling_analysis --root <output>/panel --reference <output>/reference --output <new-analysis>`.
No model/checkpoint download or simulator launch is required for that calculation.

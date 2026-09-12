# Evidence and interpretation appendix

## Evidence sources

The plotting script verifies all 30 primary report hashes against the completed closure manifest, plus the two combined and four successor report hashes against their completion receipts. It does not rerun bootstrap analyses or evaluate a model. The resulting [figure provenance manifest](data/figure_provenance.json) binds source paths, source bytes, software versions, and generated figures. [Reported contrasts](data/reported_contrasts.csv) retains all contrasts from the included scopes, including companion visual endpoints that are not highlighted in the main figures.

- Original primary results: [corrected results](../reports/CORRECTED_OFFLINE_RESULTS.md) and [closure manifest](../artifacts/offline_study/primary-durable-20260907/three-task-offline-closure-20260907/metrics/report.json).
- Original combination: [frozen combined-stage protocol and results](../docs/COMBINED_DEVELOPMENT.md).
- Successor: [fixed-response protocol](../docs/FIXED_RESPONSE_RANK4.md), [Reach BF16 report](../artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/reach/bfloat16/report.json), and [Reach-Wall BF16 report](../artifacts/offline_study/fixed-response-20260908-v1/offline-analysis-v2/reach-wall/bfloat16/report.json). The corresponding FP32 reports are included in the figure manifest and contrast export.
- Implementation: [fixed_response.py](../src/offline_study/fixed_response.py) and [original support operator](../src/offline_study/support_operator.py).
- Identity, exposure, fitting, and expansion rules: [active experiment plan](../docs/EXPERIMENT_PLAN.md), [planning alignment](../reports/PLANNING_METHOD_ALIGNMENT.md), and [methodology comparison](tables/methodology_alignment.md).

The manifest establishes the provenance of the plotted report summaries. It is not a new independent audit of every upstream video, model weight, or raw shard; those larger audits are documented in the source reports.

## Statistical scope

The positive effect convention is 100 times the paired error reduction divided by the observed native mean. Report intervals are intervals on paired differences scaled by that mean, not confidence intervals for a population ratio parameter. They are simultaneous within each report's declared contrast/endpoint family. Putting several reports in one figure does not create global familywise coverage. Cross-study visual comparison does not establish noninferiority or equivalence.

Thirty original scopes represent five sweeps, three tasks, and two precisions. Repeated observations reuse 81 trajectories. The successor revisits the same Reach and Reach-Wall pools; it does not add 60 independent evaluation trajectories. Data exposure and narrative selection limit confirmatory interpretation even where individual contrasts have multiplicity-adjusted intervals.

Rank-1 Reach efficacy and rank-1 equivalence are different questions. Rank 1 is eligible against native and its random control but does not meet the smallest-equivalent-to-best rule. Rank 4 is selected under that rule; a nonsignificant pairwise difference alone was not treated as equivalence.

## Interpretation boundaries

- Rank describes four distributed correction directions, not four scalar coordinates, four layers, or a four-dimensional physical world.
- Spatial permutation tests patch assignment while preserving broad support; it does not identify the semantic content of the edit.
- The depth and spatial experiments are separate fixed-rank sweeps. Their best point estimates do not authorize combining newly selected sites and supports.
- Original combined component removals also remove energy. They do not establish equal-energy synergy. Equal-energy vision/action coupling alone also does not establish superiority over visual-only steering on Reach.
- Local action-response reconstruction is different from recovering a physical manifold or improving a recorded future forecast. At the symmetric center of the four-anchor setup, cubic and quadratic least-squares interpolation coincide; this is not evidence of uniquely cubic physical dynamics.
- The successor's calibration estimates responses to recorded actions, not simulator ground-truth counterfactuals for all CEM candidates. Testing action-ranking behavior remains necessary.
- One native/edited engineering episode timing pair cannot establish population latency equivalence or estimate a speedup. Reported calibration time excludes inherited basis/readout fitting and research-development compute.
- A useful recipe would need prospectively tested task adaptation. No cross-task transfer, HMM benefit, or completed physical-robot execution benefit is claimed.

## Figure policy

Use the existing completed contrast intervals, show matched controls, and label unmeasured cells explicitly. Do not draw zero-valued success bars for pending results. Do not invent environment screenshots or counterfactual rollout images. The current figures use a conceptual pipeline and quantitative offline evidence; task imagery can be added later from source-bound observations with appropriate captions.

## Fitted-basis visualization

The two fixed-response bank files are checked against their existing DONE receipts before tensor loading with PyTorch's weights-only reader. Figure 6 reduces each direction's 400 feature loadings to spatial squared mass and also shows the four-direction mean. No fitting, model execution, semantic labeling, or trajectory outcome analysis is added. Individual basis axes are rotation-dependent; the averaged subspace mass is invariant to orthogonal basis changes.

## Broader research questions

The [research synthesis](research_synthesis.md) connects each original hypothesis to its controls, explains the teacher/EMA distinction, and compares circuit analysis, PEZ, manifold steering, smaller JEPA methods, and future expert routing. It also identifies relevant PhD students and researchers from psychology, neuroscience, physics, and mathematics using public sources. Proposed validation there remains future work.

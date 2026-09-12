# Workshop publication assessment

Research checked September 8, 2026. This reviews the current extended abstract and the completed evidence in [findings.md](../findings.md), including the afternoon additions. It is an assessment, not an acceptance prediction, experiment authorization, or change to running jobs.

## Verdict and confidence

**Narrow development-results study: publishable after minor revisions, with moderate confidence; competitive strength is borderline and depends on workshop fit.** This means a defensible short submission after the concrete revisions below, not that a reviewer must accept it. The existing draft already disclaims behavioral improvement and broad transfer. It contains enough controlled observations to support a modest empirical contribution without waiting for every proposed experiment.

**Practical robotics steering method: major revisions and new evidence required.** We have not established task-success gains, representative full-planner overhead, or transfer. Completing the existing core behavioral comparison is the highest-value next evidence for that claim. A reliable null can support an evaluation paper, but a wide inconclusive interval does not demonstrate ineffectiveness.

The strongest legitimate case is the combination of separately measured forecast gains for an inexpensive successor, spatial/pathway controls, and a precision-dependent failure of a concrete reconstruction diagnostic. The strongest objection is that standard activation adaptation produces small, task-specific development-set changes, while the general lesson that prediction and control differ is already known. Large experiment counts and large reconstruction ratios do not settle novelty.

## Claim ledger

| Claim and scope | Evidence and status | Alternative, assumption, or falsifier |
|---|---|---|
| A fixed distributed correction reduces forecast error in two released reaching models. | **Supported on exposed development pools:** successor BF16 gains 2.36% and 2.19%, with positive contrasts against its random controls. | Assumes the audited pairing, readout, and endpoint contract. Failure on a prospectively selected cohort would weaken generalization; existing results alone do not establish that broader claim. |
| Spatial assignment and pathway distinguish intervention effects. | **Supported within the registered comparisons:** Reach permutation contrast and opposing Push-T pathway effects. | Patch layout and activation statistics are competing explanations; physical variables and circuits are unidentified. Semantic mechanism requires a targeted causal test. |
| The tested reconstruction diagnostic can mislead intervention selection. | **Supported locally:** precision reversals on Push-T, Wall, PointMaze; large FP32 reconstruction advantages without useful forecast gains. | Quantization and interpolation conditioning may explain the result. Those are plausible numerical explanations, not discovered physical structure. Absolute errors and delivered energy must accompany ratios. |
| Steering has low deployment overhead. | **Partially supported:** zero online probes and a real GPU forecast timing check; combined overhead about 12.9% in that check. | One fitting stimulus and three repeats cannot establish full-planner or population latency. Representative episode cost could weaken this claim. |
| Steering improves task success. | **Not yet established:** registered simulated comparisons remain incomplete; completed DROID action-score comparison is inconclusive. | Planner-distribution shift, action-ranking changes, or an irrelevant embedding endpoint could defeat forecast gains. DROID scores cannot substitute for robot-success rates. |
| The method recovers a physical manifold or generalizes as a reusable JEPA interface. | **Unsupported at this scope.** | A PCA basis with a fitted map is not a semantic atlas. Untested MoE/HMM/generalization directions must remain future work. |

## Prior art and the novelty boundary

[JEPA-WM Appendix G.3](https://arxiv.org/html/2512.24497v4#A7.SS3) already investigates prediction metrics versus success, validation/planning objective mismatch, out-of-distribution proposed actions, and planner optimization limits. A generic loss/success mismatch is therefore not our contribution. The opening is controlled evidence about interventions inside retained action-conditioned predictors, including their specific diagnostic failures and application cost.

[COAST](https://arxiv.org/abs/2605.17144) already fits activation subspace operators for frozen policies and evaluates task outcomes across multiple policy architectures. We cannot claim low-rank frozen steering itself as new or imply comparable behavioral validation. Editing imagined consequences before a planner is a distinct intervention location; location alone still needs an informative empirical result.

Our best concise claim is: **A compact fixed correction improves some frozen-JEPA forecasts, but spatial/pathway controls and precision-sensitive geometry diagnostics constrain when and how those results can be interpreted.** The contribution is the controlled evidence and documented method, not new PCA or inverse mathematics.

## Blocking claims, major limitations, and minimum repairs

1. **Behavioral or transfer headline — blocker for a methods claim.** Finish the existing core registered panel and report paired success effects and uncertainty versus native and matched random. Do not change selection after partial results. A practical-positive acceptance test is the existing frozen useful-gain/control rule; a null report needs to distinguish ruled-out gains from unresolved ones. A stronger repair is prospective scenario and model-seed replication, rather than automatically expanding every category now.
2. **Novelty and numerical interpretation — major limitation.** State which hypothesis each control resolves and include absolute reconstruction MSE, precision, forecast endpoint, and realized-energy caveats. The minimum repair is possible from completed artifacts. Acceptance test: the conclusion remains informative without the thousands-fold headline and does not claim that numerical sensitivity is itself a new general phenomenon. A stronger repair would be one predeclared test of a specific explanation, not a broad new sweep.
3. **Exposed data and limited model replication — major limitation.** Label the study exploratory/development, show actual independent trajectory groups, and preserve all registered contrasts in supplementary material. Do not present successor reuse as additional independent data. Narrow claims to the measured checkpoints; independent replication is needed only for the stronger reproducibility/generalization claim.
4. **Deployment cost — major limitation for practicality.** Retain the measured forecast benchmark with its device, stimulus, repeat count, and scope. Report complete episode timing alongside efficacy when available. The arithmetic and byte-exact checks support implementation claims, not a population cost estimate.
5. **Submission packaging — minor revision.** Update the draft with completed navigation/DROID results, remove stale status language, prioritize one coherent question, and put essential figures/tables inside the venue's page limit. Our current four text pages plus separate figures are working artifacts, not automatically a compliant four-page submission.

## Workshop fit and current availability

| Venue or precedent | Verified scope and timing | Assessment for us |
|---|---|---|
| [CoRL 2026: Bringing Physics Simulation and World Models Together for Robotic Manipulation](https://corl26ws-physwm.github.io/index.html#call-for-papers) | Organizer CFP welcomes completed/ongoing work on fidelity, planning, and controllable representations; up to four pages excluding references; September 30, 2026 AoE deadline; non-archival. | A concrete current target. Frame as a controlled predictor-intervention/evaluation study; a complete behavioral comparison would improve relevance. The CFP does not require us to build a new real robot experiment. |
| [NeurIPS 2026: World Models in Physical AI](https://www.worldmodels-physicalai.com/) | Includes JEPA embeddings, world-model evaluation, and downstream control; up to eight pages excluding references. Published extended deadline was September 5. | Strong topical fit, but already closed as of this review. Do not treat it as an available submission without a new official extension. |
| [CoRL 2026: Do Robots Need World Models?](https://do-robots-need-world-models.github.io/) | Organizer page explicitly invites in-progress, comparative, negative, and surprising results; four pages. Dates and submission link remain TBD. | Good conceptual fit for a specific diagnostic/control result; availability is not yet verified. |
| [NeurIPS 2025 Mechanistic Interpretability CFP](https://mechinterpworkshop.com/neurips2025/cfp/) | Historical, closed precedent welcoming rigorous negative results and falsifiable empirical hypotheses. | Shows that task-success improvements are not a universal workshop requirement. This is not a current submission opportunity or an assurance of acceptance. |

The [official NeurIPS 2026 workshop announcement](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/) confirms several relevant world-model and interpretability themes, but each workshop sets its own scope and deadlines. For example, [Interpretability as a Science](https://interpscience.github.io/) explicitly centers LLM understanding, so geometry terminology alone does not make our JEPA study a fit.

## Recommendation

Prepare one focused workshop short paper from the completed evidence and prioritize finishing the already defined core behavior panel. A positive result would support practical correction; a sufficiently informative negative result could sharpen the diagnostic paper. Do not spend the whole remaining study budget merely to make the experiment table longer.

**Would I advise a respected colleague to publish this as written?** Not the exact current draft: it needs the status/evidence updates, a tighter novelty argument, and venue-compliant presentation. After those revisions, yes, as a clearly scoped development-results workshop paper. I would not endorse a headline claiming demonstrated general-purpose or task-success-improving steering on today's evidence.

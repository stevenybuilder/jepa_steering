# World-model paper-to-code audit — 2026-09-04

## Verdict

**Major revisions and new evidence required before the first behavioral activation intervention.**

**Status correction:** this is a technical snapshot, not the current execution plan. The later JEPA-WM Reach-Wall fit completed at 1/15 and failed class support. The active replacement is the unsteered DINO-WM MetaWorld Reach P0 screen under namespace `panel-p-frankenstein-dino-reach-v1`. See [`WORLD_MODEL_SONAR_PLAN.md`](../WORLD_MODEL_SONAR_PLAN.md). No Panel-P activation intervention has run.

This is not a verdict that Sonar is mathematically incoherent. Several numerical primitives are strong: the Gaussian-mixture gradient is exact, the anisotropic solve uses `G^{-1}g`, the capped quadratic step preserves gradient magnitude below its trust cap, Gaussian OT maps fitted moments correctly, and the fixed HMM uses forward-backward joint transition posteriors. The problem is end-to-end identification: until population alignment, regime stability, outcome support, dose, and routing are validated, a correct local formula can still be the wrong experiment.

No Panel-P JEPA-WM/DINO-WM activation intervention has run. Protected evaluation remains sealed.

## Transfer boundary

Only mechanisms with a real analogue in the frozen world-model predictor are in scope.

| Source idea | World-model status |
|---|---|
| COAST conceptor resolvent, success/failure Boolean contrast, multiplicative gate | Adapted as an equation reference in the frozen factor coordinates |
| Othello model-native coordinate reframing | Adapted as an outcome-free absolute-versus-relational physical readout audit |
| JEPA/HMM state-space claim ladder | Adapted as falsification tests and a conditional router; not an architectural equivalence claim |
| Gaussian energies, mixture responsibilities, Mahalanobis support, Gaussian OT | Directly relevant to latent world-model states |
| GIFT angular/log-magnitude split and relational frames | Angular/radial/joint ablation and coordinate hypothesis only; no teacher losses or retraining |
| LAVLA GR00T cross-attention weighting | Excluded: JEPA-WM/DINO-WM have no homologous action-decoder cross-attention or diffusion clock |
| Action Atlas VLM/action-expert conclusions | Excluded; only its generic warning about token pooling transfers |
| ACT CVAE/action-decoder semantics | Excluded; CEM control chunks are recorded only as actions between physical replans |
| Rectified LpJEPA target law | Training objective excluded; its sparse-distribution hypothesis is preserved as a gated post-hoc SAE/dictionary branch requiring reconstruction, prediction, and action fidelity |
| CAFT | Excluded because it changes weights and answers a different question |
| FailBench VLM judging | No VLM judge is used; native simulator success is primary |

Both JEPA-WM and DINO-WM in the paired MetaWorld comparison are latent world models. DINO-WM is not a pixel-space control.

## Strongest defensible contribution now

The strongest current contribution is the controlled composition, not a novel primitive:

1. outcome-agnostic, episode-held-out world-model coordinates;
2. a shared local-regime definition;
3. success/failure covariance or density fitted only afterward;
4. support- and action-sensitive bounded edits;
5. native paired closed-loop outcomes with identity and structure-matched controls.

The clean mechanism comparison is

\[
D_r=C_{s,r}\wedge\neg C_{f,r},
\qquad
D_t^{\rm static}=\sum_r q_t(r)D_r,
\qquad
D_t^{\rm HMM}=\sum_r b_t(r)D_r.
\]

Static and HMM arms must use the exact same `D_r`; only the causal temporal prior may differ. HMM is therefore a Sonar router, never a standalone edit.

## Claim-to-code ledger

| Claim or formula | Code state | Verdict |
|---|---|---|
| Episode-equal PCA whitening and predictive-factor comparison | `public_panel_factor_gate.py` | Implemented; predictive factors can fail back to global/PCA rather than being forced |
| One Gaussian vs GMM vs progress vs fixed HMM | `public_panel_factor_gate.py` | Implemented; progress control is up to ten exact replan bins, HMM EM uses `xi`, and unstable `K=2` falls back to a real `K=1` artifact |
| Action-conditioned transition `A_phi(a_t,dt)` | `public_panel_action_hmm.py` | Implemented with softmax action/progress transitions and exact Baum-Welch `xi`; DINO-WM admission pending |
| Innovation/history sufficiency after action conditioning | `public_panel_factor_gate.py` | Implemented with whole-episode held-out ridge and RBF tests after current state plus executed control |
| Empirical direct-vs-composed path consistency | `public_panel_action_hmm.py` | Implemented as a declared empirical adequacy diagnostic; not misreported as a theorem because intervening emissions update belief |
| Shared regimes for static and HMM local operators | Coordinate artifact → `public_panel_sonar_fit.py` | Corrected; no independent Sonar GMM when a factor-gate regime artifact is present |
| Conceptor class centering | `public_panel_sonar_math.py` | Corrected: success and failure are centered separately during covariance fitting |
| Moore-Penrose Boolean conceptor algebra | `public_panel_sonar_math.py` | Implemented without hidden jitter/truncation/clipping; invalid spectra fail |
| COAST runtime map `h'=hM^T` | `public_panel_steer.py` | Corrected to act about the origin |
| Global/static-local/matched-spectrum/label-shuffled conceptors | fitter and runtime | Implemented; deployed checksums match local files and the remote torch math smoke passes |
| Proper mixture-energy gradient | math and runtime | Implemented and finite-difference tested |
| Capped anisotropic quadratic step | math and runtime | Corrected; boundary-normalized form is a separately named ablation |
| Angular/radial/joint comparison | runtime | Corrected to metric-norm-match nonzero components |
| Gaussian OT | math, fitter, runtime | Implemented as a named comparator, not mislabeled a conceptor |
| Exact executed-control action metric | capture → fitter | Corrected; planner proposals are rejected, and the readout must predict actions on held-out episodes |
| Fit/apply population equivalence | capture + `public_panel_fit_apply_audit.py` | Corrected to exact runtime pooling and every retained physical replan; requires a fresh `all_plans` runtime-population recapture |
| Othello-style relational coordinates | `public_panel_coordinate_gate.py` | Corrected to a hash-bound reach-wall adapter, matched world-vs-changing-goal labels, fixed-affine and shuffled-goal controls, and aligned replans only |
| CKA | `public_panel_geometry.py` | Restricted to matched rows/cross-model anchors; not used between unmatched regimes |
| Concept/operator episode stability | factor gate + Sonar fitter | Implemented separately for regimes/beliefs, conceptors, energy gradients, precision matrices, and OT fields |
| HMM-vs-static outcome routing | Sonar fitter | Implemented leave-one-episode-out with identical fold-fitted local densities and causal filtering; it is necessary but not sufficient for HMM deployment |
| Native outcome and paired stimuli | evaluator/manifest | Implemented; exclusive-write hash-bound arm-registry tooling is implemented, but no registry can be frozen before development gates pass |
| Paired development admission | `public_panel_development_gate.py` | Implemented fail-closed: complete development runs, exact identity actions/outcomes, realized-state pairing, nonzero action treatment, cap saturation, declared dose matching, selected-arm utility, and static/HMM treatment separation |

## Pre-intervention blockers

### 1. Fresh outcome and regime support is unknown

The active DINO-WM Reach cell must first pass its 30-episode P0 gate: at least six native successes and six failures, a 20--80% success rate, valid realization hashes, and both outcomes spread across the episode order. Only then may the separate 30-episode fit cohort be captured. The fit cohort and every retained local regime must independently provide adequate episode-level class support; repeated states within one episode do not increase the number of independent episodes. Episodes cannot be replaced after seeing outcomes. Failure makes this cell ineligible.

### 2. Fit and hook populations may not match

The fitter uses selected-action extra-unroll summaries. The hook edits CEM candidate forwards. Source inspection found that the earlier candidate extractor flattened leading axes and averaged all predictor tokens, while the runtime hook preserves the top-level candidate batch and edits the final `n_spatial` token block. It also stored candidates only for the first plan call even though the hook would run at every replan.

The corrected capture mirrors runtime pooling, tags bounded candidate populations by physical plan call, and offers `--planner-capture-scope all_plans`. The audit requires every replan in the frozen pre-outcome window and collapses repeated replan measurements within episode. The historical JEPA-WM first-plan-only capture remains valid for its outcomes and selected-action geometry but cannot authorize an all-replan hook. A fresh DINO-WM runtime-population capture is mandatory after P0. If the gate fails, do not tune the threshold: refit on tagged candidates or explicitly narrow and revalidate the hook schedule.

### 3. HMM deployment is implemented fail-closed but not yet admitted

The fixed HMM is a valid sequence model, but deployment requires action-conditioned transitions, strong progress controls, innovation sufficiency, empirical path composition, better outcome routing than static responsibilities, and nontrivial treatment separation. `public_panel_action_hmm.py` now fits softmax action/progress transitions using Baum-Welch joint transition posteriors and episode-equal weighting, and compares them out of episode with fixed, progress-only, and action-shift controls. The runtime computes candidate-specific beliefs without advancing state during CEM and commits one belief only from an unedited chosen-action unroll after each physical plan call.

`sonar_coast_hmm` remains fail-closed unless the serialized sequence/action/composition and outcome-routing evidence passes; the final registry separately requires treatment separation. If the HMM branch fails, the experiment can still test global and static-local Sonar.

### 4. Behavioral dose is not frozen

Latent metric norm is insufficient. `public_panel_development_gate.py` now produces the required hash-bound evidence from paired development runs: it requires a nonzero executed-action treatment, checks declared action-dose ratios, rejects cap-hit fractions above 0.80, and withholds static-vs-HMM inference unless their normalized action separation and bootstrap lower bound pass. The code is ready; no behavioral dose has yet been measured.

### 5. The final arm registry is not yet frozen

The existing stimulus manifest was created before the COAST-hybrid correction and its metadata contains generic arm names such as `factor_sonar`. Stimulus rows are arm-independent and remain immutable. `public_panel_arm_registry.py` now freezes a separate exclusive-write registry and requires hashes of the stimulus files, operator artifacts, fit/apply, identity/action-hash, action-dose, development-utility, exact arm parameters, and HMM-specific evidence when applicable. It must be run only after the development winner is frozen and before protected evaluation; the original manifest is never overwritten.

### 6. Remote evaluator-dose validation is pending

Local Panel-P tests pass (`71 passed, 1 skipped`; the skip is the laptop's missing torch). The complete skipped runtime file passes separately on the GPU (`16 passed`) under CUDA PyTorch `2.7.1+cu128`, including action-HMM belief lifecycle and executed-action alignment, causal-mask-aware bounded Q/K capture, exact paired attention restoration, and offline/runtime sparse-code agreement. The new code has not been copied over the live unsteered P0 process. An evaluator-level identity/action-hash smoke, the fresh all-replan runtime-population capture, and measured action-space dose for every arm remain mandatory; tensor-level/runtime unit tests do not substitute for closed-loop calibration.

## Remaining empirical gaps that block a full-Frankenstein claim

These gaps do not all block P0/P1 or a separately named Sonar-core experiment. The recipe modules are now implemented, but these gaps block any statement that the complete recipe has been admitted or behaviorally tested on DINO-WM Reach.

1. **Gaussian adequacy.** Held-out density-family selection and empirical tail calibration are now implemented by factor block and regime. They must still pass on DINO-WM Reach; if they fail, the response is a preregistered robust-density extension or rejection of energy steering, not threshold tuning.
2. **Credit assignment.** The common pre-outcome window and episode weights reduce leakage, but every early state still inherits the final episode label. Early/late stability and divergence-time analysis must determine whether the signal precedes behavior.
3. **Token aggregation.** Uniform final-spatial-token pooling is the active representation. A LAVLA-style cross-attention copy is invalid, and the existing capture is insufficient for a per-token self-attention/planner-sensitivity tournament. This branch requires a new development capture and should open only if pooling fails.
4. **Model-native coordinate causal use.** Source verification found and corrected two semantic bugs before execution: slots `4:7` are a randomized cylinder rather than the wall, and the saved boundary-zero simulator row can reflect stale pre-`reset_warmup` `info`. Captured predictor rows are post-action endpoints, so the hash-bound adapter now aligns replan `t` to simulator boundary `t+1`; replan zero is valid under that endpoint convention. It validates the active goal, compares matched labels, uses fixed-affine and shuffled-goal controls, and requires a stable goal-relative readout row space across episode folds. Runtime row-space projection and a dose-matched complement control are implemented. Better readout is still nomination, not causal use; their behavioral comparison remains pending.
5. **Cross-model transport.** Procrustes/Grassmann comparisons belong after independent JEPA-WM and DINO-WM positives on paired stimuli. They cannot rescue a null primary task.
6. **Ambient COAST comparator.** The clean factor-space global operator is capacity-matched to local Sonar. Any ambient-width COAST equation reference must be reported separately rather than treated as a fair capacity-matched method comparison.
7. **Subspace-specific distribution adequacy.** `public_panel_frankenstein_gate.py` now selects isotropic, diagonal, or shrinkage-full density per factor block using whole-episode held-out log score, measures empirical 90/95% tail coverage and regime symmetric KL, and serializes passed 99% block/regime support boundaries for runtime abstention. DINO-WM Reach evidence is pending.
8. **Sparse superposition.** The outcome-agnostic overcomplete dictionary, deterministic offline/runtime ISTA encoder, `P(S|z)` statistics, PCA/random controls, predictive/action fidelity, stability gate, and runtime support abstention are implemented. DINO-WM Reach may still reject the module; collateral effects alone are not labeled superposition.
9. **Pattern separation.** Cross-episode matched lures, action-cluster Fisher geometry, permutation/stability tests, and an admitted runtime protection projection are implemented. When the hash-bound coordinate artifact is supplied, lures are matched using audited hand/goal geometry offline; otherwise the report explicitly labels its weaker progress-only fallback. DINO-WM Reach admission remains an empirical decision.
10. **Attention/Hopfield mediation.** DINO-WM capture now records bounded per-head Q/K moments, exact attention entropy, and key-frame probability mass. The gate tests regime separation before/after Q/K projection; the runtime has an exact paired clean-forward attention-output restoration ablation. These are implemented but unevaluated on the active capture, and remain distinct from HMM beliefs and VLA action-decoder weighting.
11. **Transport distinction.** Gaussian OT remains the within-coordinate distribution mapper. Relational Procrustes now has a matched-factor, leave-one-episode-out gate plus `public_panel_transport_bind.py`, which produces a runnable target-coordinate operator while retaining target routing/support/dose geometry. Cross-model evidence is pending.

## Minor/documentation gaps

- The old `run_public_panel_stage23.sh` describes an obsolete auto-running COAST table. It must not be used for the active experiment.
- Older `public_panel_geometry.py` CEM-order regime results remain optimizer-dynamics diagnostics and cannot support a physical-regime claim.
- The fitter now serializes rank, shrinkage, covariance floor, support quantile, aperture/cutoff, and metric weights; the selected energy step size and trust radius still enter only at development-dose freeze and must appear in the final arm registry.
- Generalized-eigen directions remain diagnostics until eigengap and bootstrap stability pass; high variance is not semantics.

## Binding acceptance sequence

1. Finish and pass the immutable DINO-WM Reach P0 behavioral cohort.
2. Capture the separate fit cohort, enforce global/per-regime class support, and run the factor/regime gate only on the common pre-outcome physical-replan window.
3. Verify or adapt the model-native coordinate audit for DINO-WM Reach, treating it as outcome-free interpretation only.
4. Fit one shared regime family, outcome densities, factor-space conceptors, OT, action metric, and all controls.
5. Pass the separate episode-bootstrap regime/belief, conceptor, density-gradient, precision, and OT stability gates.
6. Collect the corrected all-replan runtime population and pass the fit/apply audit; otherwise refit or narrow the hook schedule.
7. Keep HMM disabled unless every transition, innovation, composition, routing, and lifecycle gate passes.
8. Retain the passed remote intervention-math smoke and run evaluator-level identity/action-hash smokes on the same deployed hashes.
9. Run `public_panel_development_gate.py` on completed paired development arms to measure action-space dose, support, cap saturation, selected-arm utility, and treatment separation; retain its exclusive-write evidence files.
10. Freeze one winning utility operator, its matched controls/comparators, and a separate arm registry using the exclusive-write registry tool, which now requires the development-utility evidence as well as identity/dose evidence.
11. Confirm prospective power without opening evaluation outcomes, then open paired protected episodes once, with unsteered native success as the primary utility baseline.

## Publication-risk verdict

The experiment would still be vulnerable to a major-revision review if treatment separation, fit/apply equivalence, or independent class support fails on the real DINO-WM capture. The controller and auxiliary modules are now executable and fail-closed, but no implementation result substitutes for behavioral evidence. A cleanly failed gate is a valid result and should trigger reassessment rather than another post-hoc operator search.

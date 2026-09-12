# JEPA rep geometry

Model-native representational geometry for locating an action-conditional hazard state in a latent world model (JEPA-WM DROID), written after the wave-1 null of the CGS egg pilot. Companion to `experiment_design.md` (the frozen pilot) and `claude_handoff/HANDOFF.md` (state). Reference syntheses from the VLA project, read-only: `/Users/stevenyang/Documents/mechinterp-vla/coast.md`, `/Users/stevenyang/Documents/mechinterp-vla/EBM representational geometry.md`. All work for this document lives in `rep_geometry_transcoder/`.

Date: 2026-09-02. Status: preregistered addendum, executed on discovery seeds only; confirmation seeds stay sealed until one candidate and one intervention rule are frozen.

## 0. Where we are, in one paragraph

On the hazard × candidate-action factorial (fragile egg on/off the descent path × gentle/aggressive action; 21 validated held-out scenes; H0′ null placement on 14), JEPA-WM reproduces the action effect (Recovered Fraction 0.63) but its predicted hazard×action interaction is small in raw latent coordinates (cosine 0.24, NMSE 0.96) and not hazard-specific (Rec_H0 ≈ Rec_H1). A linear leave-one-scene-out localization with max-T correction nominated one site (`L07.attn_out`, step 0, route tokens), and real-donor patching found it — and every other single component site — causally dead (≤1% recovery). A curved-manifold pass (local-direction tests, RSA vs bearing, kNN/kernel/Jacobian readouts) found no rotating or nonlinear code. A COAST soft-conceptor arm shows the interaction subspace is source-specific (beats matched-spectrum random) but adds nothing over a rank-one direction for contact/force, and its multiplicative intervention is running. The coordinate frame in which the hazard variable is simplest is path-relative ("distance from egg to the descent line", 24/36 sites), but readability is not use. Everything above is on RoboCasa renders that lie beyond the DROID p95 for a DROID-only checkpoint (no in-distribution JEPA-WM checkpoint exists in the release; `jepa_wm_robocasa` shares the DROID weights).

## 1. The structural fact that reorganizes the search

The action enters the predictor in exactly one way: `adaLN_modulation(action_encoder(a))` produces (shift, scale, gate) for attention and MLP in each of 12 blocks, broadcast identically to all 256 tokens. There are no action tokens. Two consequences:

1. Any change of egg-token content produces a nonzero "hazard×action interaction" at every nonlinear site by elementwise algebra (`MLP(scale(a)·x_egg + shift(a))`). Within-token statistics are contaminated by this artifact no matter how they are corrected. This is what the H0′ null and the token-group analysis were for, and it is why the egg-token "hits" were discounted.
2. A relational state (egg-on-the-route × action) can only be created **across tokens**: route-token outputs depending on egg-token content, mediated by attention **after** an action-modulated block. Cross-token derivatives are artifact-free by construction. This is the right shape of the problem, and it is a different shape from the VLA case (where the action expert reads a multimodal prefix through cross-attention). The Othello lesson applies as ontology: the relational target must be expressed in the model's own frame (route-relative), and the readout must be relational (bilinear), not linear in a concatenation.

## 2. Math tricks (what is standard, what is ours)

### 2.1 Cross-token relational Jacobian sonar (ours: the assembly; pieces standard)

Let `P(z, a)` be the one-step predictor, `z_e` the egg-token content, `g` the route (gripper + corridor) tokens, `a₀` the gentle action. The predictor's response to an infinitesimal action "ping" is
`J_g(z, a) = ∂P_g(z, a)/∂a ∈ ℝ^{(n_g·d)×7}` (7 forward-mode passes; rank ≤ 7 because it factors through the action encoder).
The relational signal is the change of that response with hazard content, with the same-magnitude off-route change subtracted:
`ΔJ_rel = [J_g(z_H1) − J_g(z_H0)] − [J_g(z_H0′) − J_g(z_H0)]`,
i.e. the finite-difference form of the mixed derivative `∂²P_g/∂a∂z_e · δ_e`. Statistics: `S_F = ‖ΔJ_rel‖_F/‖J_g(z_H0)‖_F` (scale-free), the principal angles between `col J_g(z_H1)` and `col J_g(z_H0)` (how far the ≤7-dimensional action-response subspace rotates toward the hazard; a Grassmann distance, not a norm), and the projected statistic `⟨ΔJ_rel(a₁−a₀), I_true⟩` that ties the sonar to the true-future interaction vector. Pattern/value split: recompute with attention probabilities detached; the difference isolates the attention-pattern path (the AtP* Q/K correction in derivative form). Every Jacobian hit is confirmed by a finite difference `a₀ → a₁` on the same tokens (second-order attribution errors are dominated by downstream nonlinearity, arXiv 2606.09899).

Nulls: H0′ (design), norm-matched random `δ_e` in the egg-token subspace (Hutchinson-style, isotropic, too easy in 1024-d but reported), sign-randomized same-construction `δ` (harder; alignment leakage per arXiv 2608.24335), and quartet label shuffles within scene. Unit = scene; sign-flip permutation; max-T across sites. Implemented: `scripts/cgs_pilot/action_jacobian_sonar.py` (forward-AD, site mediation by tangent injection, mixed-derivative check).

Grounding: A/B Jacobians and controllability rank as world-model diagnostics (arXiv 2607.01736); reachability Gramians and Jacobian amplification (2607.10362); counterfactual error bounded by the action-excitation margin (2607.22430) — our matched-vs-swapped margin is 5–15% of the action separation, i.e. the low-margin regime; contact sensitivity confined to collision windows in toy transition models (2606.28455); Integrated Hessians for mixed-partial attributions (2002.04138).

### 2.2 Action-entry path patching (standard method, new target)

There are only 12 × 3 = 36 places where the action exists inside the predictor. Run block `b` with (shift, scale, gate) computed from `a₁` and every other block from `a₀`; measure the hazard dependence (H1 vs H0 vs H0′) of the route-token change and its cosine with `I_true`; also take `∂P_g/∂mod_b` directly (3·1024 dims per block) to bypass the rank-7 bottleneck. This is causal mediation on the conditioning pathway (Vig et al., 2004.12265) and is cheap enough to be exhaustive. A single block that recovers ≥30% of `I_true` with H0′ < 10% localizes the entry of the relational computation; "only the full set recovers" is a distributed-entry result and is informative. To be implemented: `modulation_patch.py`.

### 2.3 Energy-landscape identity (ours: the identity; the pieces standard)

JEPA is an energy-based framework; the planner minimizes `E(a) = ‖P(z, a) − z_goal‖²`. Define the hazard-effect energy `E_hz(a) = ‖P(z_H1, a) − P(z_H0, a)‖²`. Then
`dE_hz/da · (a₁ − a₀) = 2 (P_H1 − P_H0)ᵀ [J(z_H1) − J(z_H0)] (a₁ − a₀)`,
so the energy view and the sonar coincide: the interaction is the projection of ΔJ onto the hazard displacement. Report it with the H0′ term subtracted. In the planner's own currency use the finite-candidate energy gap `G = ‖P(z_H1,a₁) − z_true(H1,a₁)‖² − ‖P(z_H1,a₁) − z_true(H0,a₁)‖²` (does the prediction sit closer to the true contact future than to the no-contact counterfactual future; the IntPhys surprise logic of 2502.11831 / 2506.09849 carried to action-conditioned prediction) and the candidate-ranking flip rate under H1 vs H0 relative to H0′. Do not report Hessians of the CEM cost (flat in `a` for an action-insensitive model; the planner sees rankings, not gradients).

### 2.4 Bilinear relational probe (standard, missing from our tournament)

A linear probe on concatenated tokens cannot express "egg content × route content". The minimal relational readout is `ŷ = z_gᵀ W z_e` with rank-1/2 `W`, LOSO, quartet-shuffled null. Probe capacity dominates on some physics tasks (linear 49% vs attentive 94% on MVP for V-JEPA, arXiv 2606.09646) and not others; we have not tested the relational case at all.

### 2.5 Dose-response along the model's own conditioning curve (standard idea, new path generator)

Patches taken from `h_site(z, a(t))`, `a(t) = a₀ + t(a₁ − a₀)`, and from `h_site(z(s), a)` with `s` interpolating the egg pose H0 → H1 **in the simulator** (real intermediate stimuli). A monotone dose-response against `I_true` is stronger than an endpoint patch and stays on the activation manifold (2605.05115; geodesics under a pulled-back metric, 2605.24942). The pose interpolation costs stimulus generation and is scheduled for loop 2.

### 2.6 Higher-order distributional sonar and moving geometry (from the EBM doc)

Cramér–Wold sliced-W2 on frozen projections (covariance eigenvectors, random unit vectors with a frozen seed, LOSO discriminants), before and after matching the first two moments — separation that survives moment matching is geometry the conceptor ellipsoid cannot see; nHSIC vs the relational labels with train-fold bandwidth; principal angles / Grassmann distance and orthogonal Procrustes transport of the rank-4 interaction basis across layers and imagined steps (a code that transfers only after transport rotates while preserving meaning); two-sided token × hidden operators (Tucker-2) to test whether the interaction is spatially distributed; spectral entropy / participation ratio / coordinate concentration as descriptives. Implemented in `geometry_higher_order.py` (running on the wave-1 dump).

### 2.7 Conceptor (COAST) arm

Soft PCA projector `C = R(R + α⁻²I)⁻¹` on residualized within-scene DiD rows, `C_safety = C_int ∧ ¬C_action ∧ ¬C_null`, quota/similarity diagnostics, multiplicative strengthen/suppress with β sweep, matched-spectrum random and rank-one controls, band application, selectivity H1 vs H0/H0′ with equivalence margins. A numerical note that matters: the textbook AND `(A⁺ + B⁺ − I)⁺` explodes for soft non-commuting conceptors where `C_int ≈ 0` but `¬C_action < 1`; use `(A + εI)⁻¹` so every eigenvalue stays in [0, 1]. Implemented (`geometry_conceptor.py`, `conceptor_patch.py`); intervention running.

### 2.8 Not used as search tools

Diffusion-geometry similarity at n ≈ 20 scenes (dominated by scene identity), CEM-cost Hessians, pullback geodesics on the predictor (defined, `G(z) = J_zᵀJ_z`, useful only as an edit-magnitude diagnostic).

## 3. Tao's constraints, operationalized

From *Mathematics in the age of AI* (arXiv 2608.16753) and *Machine-Assisted Proof* (Notices AMS, Jan 2025):

1. Ungrounded optimization: proxies (quota, overlap, probe accuracy, steering distance, p-values) separate from the objective. Every candidate must pass a machine-checkable specification, not a p-value: recovery ≥ 30% on discovery, cosine with `I_true` ≥ 0.5, all nulls < 10%, then the same on sealed confirmation seeds. A significant p with recovery 0.000 (observed twice) is reported as an artifact.
2. Verification that does not depend on the author: the gate scripts are the specification of evidence; confirmation seeds are the checker; no reselection after opening them.
3. Proof scarcity → abundance: screen cheaply (7-column Jacobians at every site; 36 entry sites), verify expensively (donor and conceptor patching) on few.
4. Small verifiable lemmas rather than one heroic site: L1 the action enters at block(s) `b`; L2 egg content reaches route tokens through attention at layers ≥ L; L3 that dependence changes with the action (ΔJ_rel ≠ 0 against all nulls); L4 the change predicts `I_true` and the measured force DiD; L5 intervening on it changes the prediction and the candidate ranking selectively. Each lemma has its own null.
5. A verified result nobody understands is not the goal: the final claim must be a compact rule ("the route tokens' action-response subspace rotates by θ toward the egg displacement at block b, and suppressing that rotation removes the predicted contact"), stated before the confirmation run.

## 4. Preregistered hypotheses and decision rules

Egg family (wave-2 discovery seeds, ≥ 16; Jacobian statistics prefer ≥ 24):

- **E1 (sonar).** At imagined step 0 some route-token site has `S_F ≥ 0.05` and top principal angle ≥ 10°, exceeding the H0′, random-δ and sign-randomized nulls at max-T p < 0.05. Falsified: no site clears all three → the route tokens' action response is insensitive to egg content even infinitesimally.
- **E2 (entry).** A single modulation block swap `a₀→a₁` recovers ≥ 30% of `I_true` on route tokens with H0′ < 10% and cosine > 0.5. Falsified: every single block < 10% while the full swap ≥ 30% (distributed entry).
- **E3 (energy).** Per-scene `dE_hz/da·(a₁−a₀)` minus its H0′ counterpart is positive at sign-flip p < 0.05 and correlates with the measured force DiD (Spearman ≥ 0.5, CI excluding 0).
- **E4 (relational readout).** A rank-2 bilinear probe on (route, egg) tokens beats the best linear/kernel readout of the interaction by a CI excluding 0 at some site, LOSO, against the quartet-shuffled null.
- **E5 (conceptor/band intervention).** Strengthen/suppress with `C_safety` moves the H1 prediction toward/away from the true contact future (≥ 30% recovery at some β) while H0 and H0′ stay within the equivalence margin and the matched-spectrum random conceptor does not.

Loop-1 verdict rule: a mechanism claim requires E5 (or a donor patch) to pass on discovery and then on confirmation, with E1–E4 supplying the lemmas. If none of E1–E5 passes on the egg family, loop 1 ends as a structured null for the registered geometry classes on this model/stimulus pair — consistent with the action-insensitivity literature (Delta-JEPA 2606.31232; CoCo 2608.04653; 2607.22430) — and loop 2 starts.

Multi-patch families (cup, bowl; own calibration seeds; protocol v0.4):

- **C1 (scaling).** Raw interaction cosine and NMSE improve monotonically egg < cup < bowl, with bowl cosine ≥ 0.5 and NMSE ≤ 0.7 on ≥ 16 scenes. Falsified: bowl within the egg CI → the null is not a stimulus-size problem.
- **C2 (site emergence).** Some route-token attn_out site at step 0 reaches ≥ 30% recovery with unrelated-site and H0′ < 10%.
- **C3 (continuous mine/theirs).** With two distinct descent lines (a lateral-offset action variant), the action-conditional path-relative variable is decoded from route tokens with held-out R² ≥ 0.15 above the world-frame variable at a band site and transfers across paths only in the path frame. Prerequisite: the current design has world ≡ path (vertical descent), so the frame test is not yet possible.

## 5. Execution plan (autonomous, in order)

1. Wave 1 (discovery, 12 scenes): conceptor intervention (running) → `action_jacobian_sonar.py` relational + mediation → `geometry_higher_order.py` (running) → bilinear probe in the tournament → `modulation_patch.py` (36 entry sites) → energy-gap and ranking endpoints in the gate. All on discovery seeds.
2. Wave 2 (≈ 20 discovery / ≈ 20 confirmation from seeds 201–400): gate with `Rec_H1 − Rec_H0` as the primary number, localization v2 with the H0′ dump, step-0 donor sweep, then the same nonlinear arms. Freeze a band or record the structured null; only then open confirmation with zero refitting.
3. `LOOP1_SUMMARY.md`: numbers re-derived from the JSON records, claim-ladder level per result (L0–L5), what was ruled out, citation corrections applied.
4. Loop 2 if warranted: cup and bowl families (generated), lateral-offset action family (unseen actions; enables C3), relocation/camera nuisance, protocol v0.4, then gate → sonar → entry patching → intervention; V-JEPA 2-AC as a second architecture if time permits. `LOOP2_SUMMARY.md`.

## 6. Claim ladder (from the EBM doc; enforced in every summary)

L0 activation statistic differs — descriptive. L1 held-out relational prediction beats nuisance baselines — representation candidate. L2 candidate is upstream and transfers across scenes/families — model-native coordinate candidate. L3 selective donor-free or donor-based intervention changes the prediction and candidate ranking on fresh scenes — causal relational computation. L4 comparison across checkpoints/architectures — attribution to training. L5 selective repair with preservation — applied result. Wave 1 sits at L0 (with one L1 near-miss on egg tokens); nothing above.

## 7. Citation hygiene

Corrections from the literature pass are recorded in HANDOFF.md ("Citation corrections"): 2608.22092, 2511.04638 and 2510.00845 were over-attributed; the sign-flip/max-T machinery is Westfall–Young / 2403.02065 / 2410.13032; Mahalanobis and "band not index" are our choices. Key references for this document: 2607.01736, 2607.10362, 2607.22430, 2606.09899, 2606.27510, 2606.28455, 2606.09646, 2605.05115, 2605.24942, 2605.17144, 2608.24335, 2602.07050, 2606.31232, 2608.04653, 2608.11601, 2502.11831, 2506.09849, 2309.00941, 2608.16753.

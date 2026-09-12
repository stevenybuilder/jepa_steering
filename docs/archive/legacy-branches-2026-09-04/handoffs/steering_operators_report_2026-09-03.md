# Steering operators report — causal-metric minimum-distortion edit and AdaLN modulation operator

Date: 2026-09-03 (America/New_York). Author: implementation subagent (autonomous; user away).
Scope: design-doc "Registered additions" (a) and (b) in `cross model design jepa.md`; Step 4 conditional-transformation
candidates, Step 8 donor-free causal steering, "Statistical design and power". Under the binding **Label-first principle**
(design doc, user decision ~18:00 UTC, amendments 1–2): the open-loop steered-planner table in which these operators run is a
**calibration instrument**, not an endpoint; the closed-loop MetaDrive-native label (`arrive_dest` vs native failure
predicates, `closed_loop_rollout.py`) decides claims. Nothing in this report is a causal claim.

## 0. What was delivered

| Item | Path | Status |
|---|---|---|
| Operator 1 module (fit, hook, pullback check, self-test) | `scripts/cgs_pilot/causal_metric_steer.py` | done; self-test PASSED locally (3 envs) and on box 2 |
| Operator 2 module (fit, runtime steerer, self-test) | `scripts/cgs_pilot/modulation_operator.py` | done; self-test PASSED locally and on box 2 |
| Integration as modes `min_distortion` / `modulation` | `scripts/cgs_pilot/steered_planner_ranking.py` (flags `--min-distortion-dir`, `--modulation-operator-dir`, `--modulation-component`, `--modulation-rank`, `--modulation-gate-group`) | done; frozen tests 8/8 bit-for-bit, new tests 5/5 |
| New tests | `tests/test_steering_operators.py` | 5 passed |
| Box-2 real-data driver | `scripts/cgs_pilot/run_steering_operators_box2.sh` | launched 17:55 UTC, waiting for the seed-0 dumps (see §6) |
| Self-test records | `artifacts/steering_operators/self_test/artifacts/{causal_metric_self_test,modulation_operator_self_test}/self_test.json` (+ `.log`) | written |

Not modified (as instructed): `run_drive_arms.sh`, `run_coast_table.sh`, `run_wave_mech_drive.sh`, `geometry_cross_arm.py`,
`localize_interaction.py`, `modulation_patch.py`, `claude_handoff/HANDOFF.md`. `modulation_patch.py` is imported (`ModulationHooks`,
`capture_modulations`, `modulation_jvp`, `_predict`); `geometry_cross_arm.py` is imported (`load_arm`, `load_site`, `token_sets_for`);
`geometry_localize.pool_site`, `sonar_metrics.svd_subspace`, `cgs_stats` / `stats_utils` are reused.

## 1. Grounding (what each design choice is taken from)

- **Causal inner product / whitening / intervention representation** (`representational_geometry_paper_concepts.md` §1): the metric
  is part of the representation hypothesis; causally separable concepts must be orthogonal under ⟨x,y⟩_C = xᵀCov⁻¹y; whitening by
  M^{1/2} turns it into a dot product; an intervention representation "changes the concept value while leaving causally separable
  concepts unchanged". → Operator 1 measures the edit and the relational energy in the nuisance-whitened metric and constrains the
  edit's projection on the hazard-free action subspace and on the receiver distribution.
- **Design doc Step 4**: "nuisance-whitened metric ⟨x,y⟩_N = xᵀ(Σ_N + λI)⁻¹y estimated on fitting scenes from independently varied
  non-relational factors (distance, speed, offset in hazard-free and off-lane cells)" and "pullback metric G_ℓ = JᵀW J + ρ(Σ_N+λI)⁻¹";
  **Step 8**: "maximize Δ relational energy subject to ‖δ‖_N ≤ ε, D_action ≤ τ, D_local ≤ τ_d (Mahalanobis to the receiver
  distribution, already reported as 0.81 for the L03 donor)"; smallest calibrated dose; bidirectionality; frozen grid {0.25, 0.5, 1.0}.
- **Conceptors / Boolean algebra / matched-spectrum control** (§6, §12.4): the relational target is the second moment of the same
  relational DiD rows that the transport export softens into `rel_*` conceptors (C = R(R+α⁻²I)⁻¹ shares R's eigenvectors); the
  random control is a matched frame with the same norm and constraint weights (§12.2: "must outperform an appropriately matched
  Gaussian baseline").
- **Mahalanobis ellipsoid** (§12.1): C⁻¹ penalises low-variance axes; whitening maps the ellipsoid to a sphere — the reason the
  minimum-distortion direction is found in whitened coordinates (see §2 below).
- **Mante selection vector, dynamical intervention** (§11, §12.5, §12.6): the operative representation is the update rule
  ḣ = F(h,x,c); a context-dependent selection vector decides which input component pushes the state along the slow mode;
  "dynamical intervention changes F, not a single state". → Operator 2 edits the AdaLN modulation (the only place the action enters),
  gated by hazard content read at the block input, rather than the token state.
- **Evidential cautions** (§14): variance is not semantics (→ nuisance-whitened, not PCA, target); decodability is not use (→ the
  selection score and the LOSO alignment are descriptive; use is read from the steering table and, ultimately, the closed-loop label);
  a global vector can hide structured residuals (→ rank-r edit subspace, r ∈ {1,2,4}, not a single mean vector).
- **Statistical design**: unit = scene; fitting on discovery scenes only, full-data export for the operator (as the conceptor export)
  with LOSO alignment reported as a diagnostic; calibration on discovery by the frozen rule in `steered_planner_ranking.calibrate`;
  sealed confirmation through `--calibration-file` with zero refitting; the operator's own control set must fail at the same dose.

## 2. Operator 1 — minimum-distortion edit under the causal metric (`causal_metric_steer.py`)

### 2.1 Nuisance covariance Σ_N (per site × step × token group, fitting scenes only)

Displacement rows produced by varying **non-safety** factors only (no in-lane solid identity anywhere):

| group | rows | what it carries |
|---|---|---|
| `scene` | hazard-free cells (levels 0 sidewalk, 2 = H0′ mirror pose) centred within (level, action) | scene identity, hazard pixel position; on v0.9 stimuli distance / prefix throttle / lateral offset |
| `action` | h[l,1] − h[l,0], l ∈ {0,2}, uncentred | the generic action main effect |
| `null` | h[2,a] − h[0,a], uncentred | H0′ position/appearance displacement |
| `factors` | ridge coefficient per 1 SD of each v0.9 factor (`hazard_dist_m`, `prefix_throttle`, `hazard_lateral_offset_m`) read from the stimulus manifest, when its range is non-degenerate | explicit factor directions (only when present; on v0.7/v0.8 stimuli `factors.used = false`) |
| `token` | within-cell token deviations of hazard-free cells (≤ 4000 rows) | token-position variation (the edit is written per token) |

Σ_N = Σ_g w_g (1/n_g) Σ_i n_{g,i}n_{g,i}ᵀ + λI, groups equally weighted (a token group with thousands of rows must not swamp the factor
contrasts), λ = λ_rel·tr(Σ_N)/d with λ_rel = 0.1 (ridge floor for directions no nuisance factor ever varied; n_rows ≪ d = 512).
M = Σ_N⁻¹; ‖δ‖_N² = δᵀMδ.

### 2.2 Relational target and the energy in the causal inner product

Relational rows r_i = DiD_solid,i − DiD_null,i (H0′ subtracted where present), projected off the rank-k (k = 4) hazard-free action
subspace Q_act — the `geometry_cross_arm.relational_rows` convention. R = (1/S)Σ r_i r_iᵀ, m̄ = mean r_i.
Relational energy of an edit in the causal inner product: E_N(δ) = (1/S)Σ_i ⟨δ, r_i⟩_N² = δᵀ (M R M) δ =: δᵀAδ.

### 2.3 Closed form (stated in the module docstring)

maximise δᵀAδ s.t. δᵀMδ ≤ ε², δᵀBδ ≤ τ² (B = Q_actQ_actᵀ), δᵀS_rδ ≤ τ_d² (S_r = diag(1/var_r)/d, the diagonal receiver
Mahalanobis of `patch_site.mahalanobis_score`, D_local² = mean_d z²).
Lagrangian L = δᵀAδ − α(δᵀMδ−ε²) − β(δᵀBδ−τ²) − γ(δᵀS_rδ−τ_d²); stationarity Aδ = α(M + w_B B + w_S S_r)δ with w_B = β/α,
w_S = γ/α (KKT: w = 0 for an inactive constraint) — the **generalised eigenproblem** A v = λ M_tot v. Whitening δ̃ = M_tot^{1/2}δ gives
the ordinary eigenproblem of Ã = M_tot^{−1/2} A M_tot^{−1/2}; W = M_tot^{−1/2}Ṽ_r (M_tot-orthonormal), Λ = WᵀAW = diag(λ_1..λ_r).
Every feasible edit is δ = Wc, ‖c‖ ≤ ε:

- **strengthen** (signed by m̄): maximise 2b̄ᵀc + cᵀΛc with b̄ = WᵀA m̄ → boundary solution c_j = b̄_j/(ν−λ_j), ν > λ_max, ‖c‖ = ε
  (secular equation; the "hard case" b̄_1 = 0 is handled). State-independent additive edit; r = 1 gives ±εW_1, the top generalised
  eigenvector.
- **suppress** (remove the token's own relational energy): minimise 2b(h)ᵀc + cᵀΛc, b(h) = WᵀA(h−μ) → c_j = −b_j/(λ_j+ν), ν ≥ 0,
  ‖c‖ ≤ ε; ν = 0 is whitened projection removal (the causal-metric twin of conceptor suppression), otherwise the ball binds.
  Solved per token, vectorised (40 bisection steps) in the hook.

Multipliers: w_B, w_S found by log-bisection so that the worst-case values over the dose ball, ε₀²λ_max(WᵀBW) and
ε₀·sqrt(λ_max(WᵀS_rW)), meet τ² and the Mahalanobis increment bound (triangle inequality on the diagonal Mahalanobis norm).
Unit dose ε₀ = median_i ‖r_i‖_N (β = 1 writes an edit as large, in nuisance units, as the relational DiD itself); τ =
0.1·median_i‖Q_actᵀa_i‖ (10 % of the hazard-free action effect, the equivalence-margin spirit); Mahalanobis increment ≤ 0.25
(the L03 donor sat at 0.81 total).

**Why the metric matters (and why the naive reading fails).** With ‖δ‖_N = δᵀΣ_N⁻¹δ alone and a Euclidean energy δᵀRδ, the
generalised eigenproblem would favour high-nuisance-variance directions even more than PCA does. The energy must be measured with
the same causal inner product: E_N = ⟨δ, r_i⟩_N². If r_i = ρ_i u* + ν_i with nuisance leakage ν_i ~ Σ_N, then in whitened
coordinates the leakage is isotropic, R̃ = M^{1/2}RM^{1/2} has top eigenvector M^{1/2}u*, and δ* = M^{−1/2}ṽ recovers u*; the
Euclidean twin (Σ_N → I·tr(Σ_N)/d, same trace) has top eigenvector along the strongest leaked nuisance axis whenever the leaked
variance exceeds ρ². This is exactly the self-test (§5).

### 2.4 Pullback metric (JVP) — implemented as a dose cap, model needed

The `pullback` sub-command uses the existing forward-AD injection (`action_jacobian_sonar.downstream_response`) to measure, on the
hazard-free cells, the route-token JVP of each operator direction relative to the JVP of the action reference direction, and writes
`eps_cap_beta` per variant into the operator file (`pullback.json`). Reverse-mode gradients are unavailable because
`EncPredWM.unroll`/`forward_pred` are `@torch.no_grad()`-decorated in the vendored model, so a full d×d G_ℓ = JᵀW J is not estimated;
D_action is enforced in closed form through Q_act and checked in the pullback on the r fitted directions. Runs on the box only; not
yet executed on real activations (§6).

### 2.5 Interface and files

`MinDistortionOperator.torch_fn(mode, beta, variant, device) -> fn(h)` on `[B, n_tokens, d]`, identical to what
`steered_planner_ranking.TorchRunner._transform` returns; sham (β = 0) returns the input tensor itself (bit-identical).
Variants: `main`, `euclid` (Euclidean twin), `random` (matched-norm random M_tot-orthonormal frame). Files
`<site>__s<step>__<group>.npz` with `mean`, `m_bar`, `Q_act`, `receiver_mean/var`, per variant `W [d,r]`, `B_read [d,r]`, `Lam [r]`,
`c_star [r]` (unit-ε strengthen step), `eps0`, `A_rel` (d ≤ 1024) and a JSON `meta` (constraints, unit-edit diagnostics,
twin comparison, LOSO alignment). `fit` sub-command: `--dump --stimulus --discovery-seeds --solid-level --sites --steps --groups
--rank --k-action --lam-rel --tau-action-rel --mahal-incr [--no-token-nuisance] [--no-manifest-factors] [--no-loso]`.

LOSO diagnostic (`meta.loso`): Σ_N and the rank-1 direction refitted without the held-out scene; held-out cosine (in the variant's
metric) with the sign from the training-fold mean. Under the null of no consistent direction the signed statistic is negatively
biased (leave-one-out sign choice), so the criterion is a positive mean; `abs_mean` is the unsigned capture.

## 3. Operator 2 — AdaLN modulation operator (`modulation_operator.py`)

m′ = m[I + g(a, h_hazard) U diag(γ) Vᵀ] on the modulation output of one block, restricted to a component set.

- **Where**: m_b = adaLN_modulation(action_encoder(a)) ∈ R^{6D} (shift/scale/gate × attn/MLP), broadcast to all tokens
  (`FWAdaLNBlock.forward`, vendored `AdaLN_vit.py`: the modulation is computed first, then attn, then MLP — so a block pre-hook sees the
  block input before the modulation hook fires; the steerer uses exactly this order). Only the LAST frame's modulation vector is edited.
- **Read/write directions**: U = V = unit throttle-brake contrast of each D-slice c of the component set (u_c ∝ Δ̄_c = mean_i
  (m_b(a1) − m_b(a0))|_c). The read (m − m_ref)·u_c is the amount of throttle in slice c (= "g read from the AdaLN vectors
  themselves"; it is ≈ 0 for the brake chunk, so the operator is action-gated by construction). m_ref = mean brake modulation.
- **Selection score (the modulation-path DiD)**: with `modulation_patch.modulation_jvp`, the route-token response to a unit tangent
  along u_c on the solid in-lane context minus the H0 context (minus the H0′−H0 difference when present), averaged over the a0 and a1
  linearisation points, projected on the TRUE consequence I_true on route tokens (recovery units as E2). Unit = scene: bootstrap CI,
  sign-flip p, max-T over candidates. This is Mante's selection vector in operator form: which modulation component is allowed to push
  the state along the relational mode, conditional on context.
- **Rank grid** r ∈ {1,2,4}: the r slices with the largest |mean score| (attn/mlp have 3 candidates, so r = 4 → 3, recorded as
  `rank_requested`); γ_c = mean s_c / max|mean s_c| ∈ [−1,1] (sign = whether more throttle in that slice drives the consequence), so
  β = 1 at full gate writes an edit as large as the throttle-brake contrast of the top slice.
- **Context gate** g_h(h_hazard) = clip((w·x̄ − c0)/(c1 − c0), 0, 1), w = unit mean (x̄_hazard(solid) − x̄_hazard(H0)) at the block
  input (last-frame hazard tokens averaged), c0/c1 = mean projections of hazard-free / solid cells. The gate's values on ghost, H0,
  H0′ cells are reported (`gate.ghost` etc.): if the model's hazard-token content does not separate the identities the gate fires on
  the ghost too, and identity selectivity must then come from downstream — a diagnostic, not a hidden assumption.
- **Runtime rule**: m′_last = m_last + s·β·g_h·Σ_c γ_c((m_last − m_ref)·u_c)u_c, s = ±1 (strengthen/suppress).
- **How it differs from a residual-stream conceptor**: the conceptor rewrites the token STATE at one site (h′ = mean + (h−mean)M);
  the modulation operator rewrites the UPDATE RULE (the coefficients with which the block writes hazard evidence into every token),
  conditionally on the action (read from m) and on the hazard content (read at the block input). It touches no token state directly,
  is the same for all tokens of the frame, is zero for the brake chunk, and its effect on the prediction goes through the block's own
  nonlinearity — §12.5 "dynamical intervention", §12.6 "a small direction can be mechanistically important if it strongly changes the
  subsequent trajectory".
- **Controls**: `sham` (β = 0, bit-identical), `ungated` (g_h frozen at its mean hazard-free value — the literal twin; near-sham when
  that value ≈ 0, which the fit records), `ungated_on` (g_h ≡ 1; the informative twin, must break H0/H0′ preservation if the gate
  matters), `random_matched` (random unit directions inside the same slices, same γ), `wrong_block` (same operator written at block
  b + offset), `wrong_group` (gate reads the complementary token group). Files `L<bb>.adaln__<comp>__r<r>.npz` (+ `.json`);
  `ModulationSteerer` has the `PredictorSteerer` interface (`__enter__/__exit__`, `.applied`, `.expected`).

## 4. Integration into `steered_planner_ranking.py`

- New source flags (exactly one of `--conceptor-dir / --transported-conceptor / --min-distortion-dir / --modulation-operator-dir`);
  `--modulation-component {attn,mlp,all}`, `--modulation-rank 1 2 4` (one family per rank, source `modulation:<comp>:r<r>`),
  `--modulation-gate-group hazard` (the modulation run's `--group` becomes the gate group; `wrong_group` = its complement, `corridor`).
- `SteerConfig` gains `operator: str = "conceptor"` (default unchanged, so every existing config name / family / control set is
  identical; `asdict` gains the additive key `operator`; `coast_table` rows gain the column `operator`; `steering.operator_kind` and
  a per-operator `steering.operator` string are written). `OPERATOR_CONTROLS`, `CONTROL_KINDS` (superset used by `aggregate`'s
  "controls must fail" rule), `VARIANT_CONTROLS`, `SITE_SHIFT_CONTROLS` added; `build_configs(..., operator=)`; `--controls` default
  is now the operator's own set (identical to `CONTROLS` for the conceptor).
- `TorchRunner(..., operators=, mod_operators=)`; `_transform` dispatches to `_transform_min_distortion` (same `fn(h)` contract, edit
  RMS logged, sham bit-identical); `_steerer` returns a `ModulationSteerer` for modulation configs (sites `L<bb>.adaln`; the wrong-site
  machinery `unrelated_site` keeps the hook, so `wrong_block` = the adaln site `offset` layers away).
- Numerics of the conceptor path: untouched. Verification: `tests/test_steered_planner_ranking.py` 8/8 under the torch venv and under
  `/usr/local/bin/python3.9` (numpy 2.0.2); the sham-reproduction test (`reproduces_unsteered`) and all frozen expectations pass
  unchanged. `tests/test_predictor_hooks.py` 29/29 unchanged. `tests/test_steering_operators.py` (new, 5 tests): frozen config names/
  schema, new families and control kinds, analysis-core control accounting for the new kinds, both self-tests through pytest, file
  round trips, bit-identical shams through the real hook classes on a fake AdaLN predictor.

## 5. Self-tests (planted mechanisms; numbers from the JSONs listed in §0)

**Operator 1** (`causal_metric_self_test/self_test.json`): d = 64, 24 scenes, 6 nuisance axes (σ 3…1) plus 0.05 isotropic,
relational rows r_i = ρ_i u* + ν_i with full-strength nuisance leakage (leaked variance along the top axis 9 ≫ ρ² = 1).
cos(edit, u*): causal 0.919, Euclidean twin 0.083, random 0.120; cos with the top nuisance axis: causal 0.022, Euclidean 0.155;
signed edit on u*: +1.55 (sign follows m̄). Constraints at unit dose: D_action 0.0017 ≤ τ² 0.0408; Mahalanobis increment 0.2466 ≤
0.25 (this constraint was active — w_S found by bisection). sham identical (numpy `array_equal`), strengthen equals the closed form,
suppress reduces the causal energy of every row and stays inside the budget. LOSO alignment: causal mean 0.439 [0.401, 0.477], t 22.1,
sign-flip p 2e-4; Euclidean twin −0.372 [−0.534, −0.208] (negative = the direction has no consistent sign across scenes; |cos| 0.49
reflects nuisance capture). Torch hook on a fake AdaLN predictor through `PredictorSteerer`: sham `torch.equal` to the unsteered
rollout with the hook fired; strengthen and suppress change the output; torch suppress matches the numpy reference (atol 1e-3).
`SELF_TEST_PASSED` under the scratch torch venv (numpy 1.26.4, torch 2.2.2), `/usr/local/bin/python3.9` (numpy 2.0.2, numpy-only part),
and box 2 `/opt/conda/bin/python` (numpy 2.2.6, torch 2.7.1+cu128).

**Operator 2** (`modulation_operator_self_test/self_test.json`): fake 3-block AdaLN predictor, block 1 plants
x += 2·relu(hazard content)·(gate_mlp·u_gate)·v_rel (consequence only when hazard content AND throttle-like gate_mlp); shift/scale
carry hazard-independent action effects. 8 scenes. Selection: top slice `gate_mlp`, score 0.943, max-T p 0.0078; every other block's
largest |score| 0.014. Gate: solid 0.96, H0 0.038, H0′ 0.038, separation d′ 11.1. Applying the r = 1 mlp operator at β = 1:
relational recovery gain +1.17 on the solid throttle cell, 0.00 on the H0 cell (gate closed); suppress −1.16; `ungated_on` moves the
H0 cell by 0.21 (the gate is what preserves hazard-free cells); random matched −0.066; wrong block −0.007; sham bit-identical with the
hook fired. Edit relative RMS on the modulation vector 0.30. `SELF_TEST_PASSED` locally and on box 2 (CPU).

## 6. Real-data check on box 2 (development; open-loop = calibration only)

Status at the time of writing: the seed-0 activation dumps do not exist yet — the identity-geometry job's first run failed (raw
factorial cells had not arrived; `logs/drive_identity_geometry.log` `FileNotFoundError … drive_factorial/armA/seed_3/cells/…`) and is
re-queued by `/root/requeue_identity_geometry.sh` behind the `drive_factorial` rsync (`/root/pull_box1.log`: `PULL_MODELS_DONE`
present, the raw-cell rsync still running at 17:55 UTC). No dump was launched by this agent (coordinator instruction).

The driver `run_steering_operators_box2.sh` (synced, md5 verified both sides, launched 17:55 UTC with `setsid nohup`, log
`/root/cgs-pilot/logs/steering_operators.log`, driver log `logs/steering_operators_driver.log`) waits for `arm X seed 0: slimmed` in
`logs/drive_identity_geometry.log`, copies each arm's slimmed dump immediately to `artifacts/steering_operators/dump_arm{A,B}`
(`run_identity_geometry.sh` deletes the activations in its cleanup stage), then runs:

1. `causal_metric_steer.py fit` per arm (CPU; all non-adaln sites, step 0, groups `hazard_corridor` + `hazard`, common discovery seeds
   `drive_factorial_merged/discovery_seeds_common.txt`, solid level 1 for A / 3 for B) → `artifacts/steering_operators/arm{A,B}_seed0/min_distortion/`;
2. `modulation_operator.py fit` per arm (GPU, ≤ 16 discovery scenes, all 6 blocks, components attn/mlp/all, ranks 1 2 4, n_boot/n_perm 500)
   → `…/arm{A,B}_seed0/modulation/`;
3. `steered_planner_ranking.py` on the first 12 common discovery seeds, frozen β grid, both modes, progress + brake goals, the
   operator's own control set: `--min-distortion-dir` at the arm's band (A: `L03.attn_out+L03.mlp_out`, the frozen patch band; B:
   `L02.mlp_out+L03.mlp_out`, no band exists for B — development choice), and `--modulation-operator-dir` at `L02.adaln`, `L03.adaln`
   (component all, ranks 1 2 4) → `…/arm{A,B}_seed0/steer_{min_distortion,modulation}/discovery/{steered_ranking.json,coast_table.md}`
   + `calibration.json`. Markers `SO_DUMPS_COPIED`, `SO_FIT_MD_DONE`, `SO_FIT_MOD_DONE`, `SO_STEER_DONE`, `STEERING_OPERATORS_DONE`.

Results: **see §6.1 below (filled in when the JSONs exist; otherwise "pending")**. Any number quoted there is read from the JSON files
named above, never from memory. These are 12-scene / 16-scene discovery smoke runs: development evidence for whether the operators run,
what dose calibrates, and whether the controls fail — not confirmation, not a claim.

### 6.1 Results (updated 21:45 UTC; numbers read from the JSONs on box 2 with `read_steering_results.py`)

**Stage markers** (`logs/steering_operators.log`): `SO_DUMPS_COPIED` 20:04 UTC (arm A copied 19:25, arm B 20:04; 809 MB, 18 sites
each), `SO_FIT_MD_DONE` 20:39 UTC (arm A 1107 s, arm B 1010 s; 36 operator files per arm = 18 sites x {hazard_corridor, hazard}).
The arm-A modulation fit then crashed (`modulation_operator.py` built the JVP tangent with the ENCODER latent width 1024 -> 6144
instead of the predictor width 512 -> 3072); fixed (D read from the captured modulation vector; regression check with latent
width != predictor width added to the local check), self-test re-passed, resynced (md5 verified), driver relaunched 21:35 UTC
(it skips the finished stages).

**Operator 1 fit, both arms (53 common discovery scenes, `arm{A,B}_seed0/min_distortion/causal_metric_fit.json`).**
Nuisance rows per site: scene 212, action 53, null 106, token 4000; `factors_used = false` on this (v0.7) stimulus; nuisance
effective rank 3.35–17.8; λ = 0.001–0.107. The decisive descriptive finding: at 36/36 arm-A and 27/36 arm-B (site, group) cells
**the causal-metric direction and the Euclidean twin coincide** (cos(main, euclid) > 0.999, both with cos ≈ 1.000 to the mean
relational row); held-out LOSO alignment 0.985–0.997 (A) / 0.986–1.000 (B) for the causal variant and 0.971–0.998 / 0.986–1.000
for the Euclidean twin (sign-flip p 2e-4 everywhere). This is the algebra of a rank-1 target:
when R ≈ m̄m̄ᵀ the whitened top eigenvector is M^{1/2}m̄ and δ* = M^{−1/2}(M^{1/2}m̄) = m̄ for every metric — the relational DiD rows
of this stimulus are dominated by the shared consequence template already reported as F17 (hard rank 1, quota ≈ 0.002; PC1 0.54–0.67).
So on this data the min-distortion edit degenerates to the (budgeted) mean-DiD direction, i.e. the conceptor family's `rank_one`
control with a dose in nuisance units, and **the Euclidean twin is not a discriminating control here**; the twin only separates
where a constraint binds (the 9 arm-B cells: L00.mlp_out cos 0.922 / 0.835 (hazard_corridor / hazard), L00.resid_post 0.970 /
0.883, L01.resid_post 0.990 / 0.926, L03.attn_out 0.933 / 0.863, L04.mlp_out-hazard 0.994 — every one with the Mahalanobis-increment
multiplier w_S > 0 (0.29–22.7), and D_action at its bound at L03.attn_out (w_B 0.052 / 0.057) and L04.mlp_out-hazard (0.003)).
Unit doses ε₀ (nuisance units) 1.29–16.2 (A), 1.51–25.7 (B); worst-case Mahalanobis increment at unit dose 0.037–0.25 (A),
0.054–0.25 (B); the 0.25 bound is active exactly at the cells above. The self-test regime (nuisance leakage spread over several directions, rank > 1 target) is therefore not the regime
of the v0.7 driving stimulus; the v0.9 randomised-factor stimulus (distance / prefix throttle / lateral offset; `factors_used` would
become true) is where the metric can matter. Steering rows for this operator (stage 3) are reported below when they land.

**Operator 2 fit, arm A (16 discovery scenes, 6 blocks, 54 operator files, 1033 s; `armA_seed0/modulation/modulation_operator_fit.json`).**
Selection (linearised relational recovery per UNIT tangent, solid minus H0 minus the H0′ DiD; unit = scene): the gate slices win
at every block — `gate_mlp` 0.0097 (L00), 0.0106 (L01), 0.0030 (L02), 0.0025 (L03); `gate_msa` 0.0033 (L02), 0.0035 (L03),
0.0015 (L04); all sign-consistent across the 16 scenes (sign-flip / max-T p 0.002 = 1/(500+1)), while shift/scale slices sit at
±0.001 or below. Scaled by the throttle-brake contrast norm ‖Δ_c‖ (1.1–3.1), a full-contrast edit (β = 1) along the top slice has a
linearised relational gain of 0.032 (L01 gate_mlp), 0.022 (L00), ≤ 0.009 elsewhere — i.e. ≤ 3 % of I_true (the full 6-block
modulation swap of E2 recovers the whole interaction): no single slice carries the consequence linearly, consistent with the
distributed / nonlinear E2 result in HANDOFF. Hazard gate at the block input: solid 0.90–0.99, H0 ≤ 0.006, H0′ 0.08–0.44,
**ghost (cone) 0.19–0.69** (d′ 3.2–12.9): the block-input hazard readout is only partly identity-selective, so any selectivity of
this operator must come from downstream, not from the gate — recorded as a diagnostic, as designed.
**Operator 2 fit, arm B (16 discovery scenes, 1073 s; `armB_seed0/modulation/modulation_operator_fit.json`).** Same picture,
gate slices first: `gate_mlp` 0.0141 (L00), `gate_msa` 0.0092 (L01), 0.0038 (L02), 0.0066 (L03), max-T p 0.002; L04 ≈ 0 (p 1.0);
scaled full-contrast gains 0.036 (L00), 0.023 / 0.018 (L01), 0.016 (L03) — again ≤ 4 % of I_true. The block-input hazard gate is
markedly MORE identity-selective in arm B: solid (cone) 0.984–0.994, ghost (pedestrian) **0.07–0.20**, H0 0.000, H0′ 0.04–0.10,
d′ 14–28 (arm A: ghost 0.19–0.69, d′ 3–13). Descriptive asymmetry worth recording next to the F5/T1c′ asymmetries: arm B's
hazard-token content separates its solid cone from the pedestrian better than arm A separates its solid pedestrian from the cone.
Local copies of every finished JSON (fits, arm-A steering, driver log): `artifacts/steering_operators/box2/`.

**Operator 1 steering, arm A, 12 discovery scenes (seeds 3,4,7,9,10,14,15,17,18,19,20,21), band L03.attn_out+L03.mlp_out,
33 configurations, 1560 s; `armA_seed0/steer_min_distortion/discovery/steered_ranking.json`, `calibration.json`.**
Progress goal (primary): unsteered safe-choice h1 = 1.00 (12/12), h0 = h0′ = h3 = 0.00 — the solid identity is already at ceiling on
these scenes, so strengthen has no room and the informative direction is suppress. **No configuration flips any scene**: every
safe-choice DiD is 0 [0, 0] (main, euclidean_twin, random_matched, wrong_site, wrong_group, sham), `calibrated beta = None` for both
families, `selective = False`. On the continuous cost margin (unsteered h1 +0.467, h0 −0.480, h0′ −0.468, h3 −0.703) the edits move
the margins by |Δ| ≤ 0.007: strengthen β = 1 h1 −0.0020 [−0.0022, −0.0018] (wrong direction), h0 −0.0062, h0′ −0.0062, h3 −0.0041
(margin DiD +0.0043 driven by the hazard-free cells, not by the solid); suppress β = 1 h1 +0.0001 (no removal effect), h0 −0.0021.
The Euclidean twin is numerically identical to main for strengthen (same direction, §6.1 above) and differs only for the per-token
suppress; the matched random frame moves the margins by the same order (h1 +0.0027 at β = 1). Edit size: relative RMS 0.027 / 0.053
/ 0.107 at β = 0.25 / 0.5 / 1; receiver Mahalanobis of the edited tokens 1.015 → 1.024 at β = 1 (increment +0.009; suppress
−0.005). Brake goal (secondary): unsteered h1 = h3 = 1.00, h0 = h0′ = 0.083; again no flips, |Δmargin| ≤ 0.007.
Reading: at the frozen dose grid the minimum-distortion budget (β = 1 ≈ the relational DiD's own size in nuisance units, ≈ 10 %
of the activation norm at L03) is one to two orders of magnitude below what moves the planner currency at this band — the L03
donor that recovered 72 % replaced the whole corridor state. Hazard-free preservation holds trivially. A dose-response beyond
β = 1 (or ε₀ expressed in donor-patch units) is needed before the operator can be judged; see §8.

**Per-cell continuous margin table, arm A min-distortion, 12 scenes (coordinator request (a); every number from
`steer_min_distortion/discovery/steered_ranking.json`, results[progress]; Δmargin = steered − unsteered normalised cost margin
Δ_H/(cost_A0 + cost_A1), scene-bootstrap 95 % CI; "DiD CI excl. 0" = `did_margin_ci_excludes_zero_expected`).**

Goal `progress`, n_scenes = 12, unsteered safe-choice: h0 0.00, h0prime 0.00, h1 1.00, h3 0.00; unsteered margin: h0 -0.480, h0prime -0.468, h1 +0.467, h3 -0.703.

| family | mode | beta | kind | h1 Δmargin | h3 Δmargin | h0 Δmargin | h0′ Δmargin | DiD margin (solid−h0) | DiD CI excl. 0 (exp. sign) | safe-choice DiD | edit RMS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0 | sham | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] p=1 | 0 | +0.00 [+0.00, +0.00] | - |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0.25 | main | -0.0004 [-0.0005, -0.0003] | -0.0011 [-0.0011, -0.0010] | -0.0015 [-0.0017, -0.0013] | -0.0015 [-0.0015, -0.0014] | +0.0011 [+0.0009, +0.0013] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.027 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0.25 | euclidean_twin | -0.0004 [-0.0005, -0.0003] | -0.0011 [-0.0011, -0.0010] | -0.0015 [-0.0017, -0.0013] | -0.0015 [-0.0015, -0.0014] | +0.0011 [+0.0009, +0.0013] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.027 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0.25 | random_matched | +0.0008 [+0.0006, +0.0009] | +0.0003 [+0.0003, +0.0003] | -0.0001 [-0.0002, -0.0000] | +0.0000 [-0.0001, +0.0002] | +0.0009 [+0.0007, +0.0010] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.033 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0.25 | wrong_group | +0.0001 [+0.0001, +0.0001] | -0.0005 [-0.0006, -0.0005] | -0.0015 [-0.0016, -0.0014] | -0.0012 [-0.0013, -0.0011] | +0.0016 [+0.0015, +0.0017] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.028 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0.25 | wrong_site | -0.0003 [-0.0003, -0.0003] | -0.0001 [-0.0002, -0.0001] | -0.0002 [-0.0003, -0.0001] | -0.0004 [-0.0005, -0.0004] | -0.0001 [-0.0002, +0.0000] p=0.37 | 0 | +0.00 [+0.00, +0.00] | 0.051 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0.5 | main | -0.0009 [-0.0010, -0.0007] | -0.0021 [-0.0022, -0.0020] | -0.0030 [-0.0035, -0.0027] | -0.0030 [-0.0031, -0.0029] | +0.0022 [+0.0018, +0.0026] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.053 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0.5 | euclidean_twin | -0.0009 [-0.0010, -0.0007] | -0.0021 [-0.0022, -0.0020] | -0.0030 [-0.0035, -0.0027] | -0.0030 [-0.0031, -0.0029] | +0.0022 [+0.0018, +0.0026] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.053 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0.5 | random_matched | +0.0015 [+0.0012, +0.0017] | +0.0006 [+0.0006, +0.0006] | -0.0003 [-0.0004, -0.0001] | +0.0001 [-0.0001, +0.0003] | +0.0017 [+0.0014, +0.0020] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.066 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0.5 | wrong_group | +0.0002 [+0.0001, +0.0002] | -0.0011 [-0.0011, -0.0010] | -0.0031 [-0.0033, -0.0028] | -0.0025 [-0.0027, -0.0023] | +0.0032 [+0.0030, +0.0035] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.056 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 0.5 | wrong_site | -0.0005 [-0.0006, -0.0005] | -0.0003 [-0.0003, -0.0003] | -0.0004 [-0.0006, -0.0001] | -0.0009 [-0.0010, -0.0008] | -0.0002 [-0.0005, +0.0000] p=0.186 | 0 | +0.00 [+0.00, +0.00] | 0.102 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 1 | main | -0.0020 [-0.0022, -0.0018] | -0.0041 [-0.0043, -0.0040] | -0.0062 [-0.0071, -0.0055] | -0.0062 [-0.0064, -0.0060] | +0.0043 [+0.0035, +0.0052] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.107 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 1 | euclidean_twin | -0.0020 [-0.0022, -0.0018] | -0.0041 [-0.0043, -0.0040] | -0.0062 [-0.0071, -0.0055] | -0.0062 [-0.0064, -0.0060] | +0.0043 [+0.0035, +0.0052] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.107 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 1 | random_matched | +0.0027 [+0.0023, +0.0031] | +0.0013 [+0.0012, +0.0014] | -0.0004 [-0.0007, -0.0001] | +0.0002 [-0.0002, +0.0006] | +0.0032 [+0.0026, +0.0036] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.133 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 1 | wrong_group | +0.0002 [+0.0001, +0.0002] | -0.0021 [-0.0022, -0.0020] | -0.0063 [-0.0069, -0.0058] | -0.0052 [-0.0055, -0.0048] | +0.0065 [+0.0060, +0.0071] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.112 |
| min_distortion:band:L03.attn_out+L03.mlp_out | strengthen | 1 | wrong_site | -0.0011 [-0.0011, -0.0010] | -0.0006 [-0.0006, -0.0005] | -0.0006 [-0.0010, -0.0001] | -0.0019 [-0.0020, -0.0018] | -0.0005 [-0.0010, -0.0001] p=0.0322 | 0 | +0.00 [+0.00, +0.00] | 0.203 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0 | sham | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] p=1 | 0 | +0.00 [+0.00, +0.00] | - |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0.25 | main | +0.0002 [+0.0001, +0.0003] | -0.0003 [-0.0004, -0.0003] | -0.0015 [-0.0017, -0.0013] | -0.0015 [-0.0016, -0.0015] | +0.0017 [+0.0015, +0.0019] p=0.000488 | 0 | +0.00 [+0.00, +0.00] | 0.027 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0.25 | euclidean_twin | +0.0004 [+0.0004, +0.0005] | -0.0002 [-0.0003, -0.0002] | +0.0006 [+0.0004, +0.0009] | -0.0000 [-0.0001, +0.0000] | -0.0002 [-0.0005, +0.0001] p=0.233 | 0 | +0.00 [+0.00, +0.00] | 0.027 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0.25 | random_matched | -0.0008 [-0.0009, -0.0007] | -0.0002 [-0.0003, -0.0002] | +0.0001 [+0.0000, +0.0002] | -0.0001 [-0.0002, +0.0001] | -0.0010 [-0.0011, -0.0008] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.033 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0.25 | wrong_group | +0.0001 [+0.0001, +0.0001] | -0.0002 [-0.0003, -0.0002] | -0.0011 [-0.0013, -0.0009] | -0.0008 [-0.0008, -0.0007] | +0.0012 [+0.0010, +0.0014] p=0.000488 | 0 | +0.00 [+0.00, +0.00] | 0.028 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0.25 | wrong_site | -0.0002 [-0.0002, -0.0002] | -0.0002 [-0.0002, -0.0001] | -0.0004 [-0.0005, -0.0004] | -0.0003 [-0.0003, -0.0002] | +0.0002 [+0.0002, +0.0003] p=0.000488 | 0 | +0.00 [+0.00, +0.00] | 0.051 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0.5 | main | +0.0001 [-0.0000, +0.0002] | -0.0005 [-0.0006, -0.0004] | -0.0028 [-0.0032, -0.0024] | -0.0028 [-0.0029, -0.0027] | +0.0029 [+0.0024, +0.0034] p=0.000488 | 0 | +0.00 [+0.00, +0.00] | 0.053 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0.5 | euclidean_twin | +0.0008 [+0.0007, +0.0009] | -0.0005 [-0.0005, -0.0004] | +0.0015 [+0.0010, +0.0020] | +0.0001 [+0.0000, +0.0002] | -0.0007 [-0.0012, -0.0002] p=0.0151 | 1 | +0.00 [+0.00, +0.00] | 0.053 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0.5 | random_matched | -0.0017 [-0.0019, -0.0014] | -0.0004 [-0.0005, -0.0004] | +0.0002 [+0.0000, +0.0004] | -0.0001 [-0.0003, +0.0001] | -0.0020 [-0.0022, -0.0016] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.066 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0.5 | wrong_group | +0.0002 [+0.0001, +0.0002] | -0.0003 [-0.0004, -0.0003] | -0.0019 [-0.0024, -0.0016] | -0.0013 [-0.0014, -0.0012] | +0.0021 [+0.0018, +0.0026] p=0.000488 | 0 | +0.00 [+0.00, +0.00] | 0.057 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 0.5 | wrong_site | -0.0004 [-0.0004, -0.0003] | -0.0003 [-0.0003, -0.0002] | -0.0009 [-0.0011, -0.0008] | -0.0005 [-0.0006, -0.0004] | +0.0005 [+0.0004, +0.0007] p=0.000488 | 0 | +0.00 [+0.00, +0.00] | 0.101 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 1 | main | +0.0001 [-0.0000, +0.0003] | -0.0002 [-0.0003, -0.0001] | -0.0021 [-0.0023, -0.0020] | -0.0022 [-0.0023, -0.0021] | +0.0023 [+0.0020, +0.0025] p=0.000488 | 0 | +0.00 [+0.00, +0.00] | 0.093 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 1 | euclidean_twin | +0.0008 [+0.0006, +0.0010] | -0.0004 [-0.0005, -0.0003] | +0.0034 [+0.0027, +0.0042] | +0.0005 [+0.0004, +0.0007] | -0.0026 [-0.0033, -0.0019] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.105 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 1 | random_matched | -0.0037 [-0.0042, -0.0032] | -0.0007 [-0.0008, -0.0006] | +0.0004 [+0.0000, +0.0008] | -0.0002 [-0.0006, +0.0003] | -0.0041 [-0.0046, -0.0034] p=0.000488 | 1 | +0.00 [+0.00, +0.00] | 0.131 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 1 | wrong_group | +0.0005 [+0.0004, +0.0006] | +0.0004 [+0.0002, +0.0005] | -0.0009 [-0.0010, -0.0007] | +0.0000 [-0.0002, +0.0002] | +0.0014 [+0.0013, +0.0015] p=0.000488 | 0 | +0.00 [+0.00, +0.00] | 0.108 |
| min_distortion:band:L03.attn_out+L03.mlp_out | suppress | 1 | wrong_site | -0.0009 [-0.0010, -0.0009] | +0.0002 [+0.0002, +0.0003] | -0.0017 [-0.0020, -0.0015] | -0.0000 [-0.0003, +0.0002] | +0.0008 [+0.0005, +0.0011] p=0.000488 | 0 | +0.00 [+0.00, +0.00] | 0.189 |

What the table shows (coordinator points (1)–(2), confirmed): the discrete indicator is saturated (h1 = 1.00, everything else 0.00)
so no row can differ on safe choice; on the continuous margin the DiD "significance" flag is set for main AND for the Euclidean
twin, the matched random frame and wrong_group at strengthen β = 1 (wrong_site is the only 0), and for suppress it is 0 for main
but 1 for the twin and the random frame — **no specificity**. The margin displacements themselves are ≤ 0.007 in absolute value
against unsteered margins of ±0.47–0.70, and strengthen moves the solid cell the wrong way (h1 −0.002) while lowering the
hazard-free cells (h0/h0′ −0.006) — the "DiD" is a hazard-free artefact of a tiny additive shift, not a relational effect.

**Plain statement (coordinator point (c)).** On the v0.7 stimulus the minimum-distortion edit degenerates to the mean-DiD
direction: the relational rows are rank-1 (the shared consequence template, F17), so the causal-metric and Euclidean directions
coincide (36/36 arm-A cells, 27/36 arm-B cells; §6.1), the operator has **no discriminating power here**, and its Euclidean twin
is not a control. Its test belongs on the v0.9 randomised-factor stimulus, where the manifest factors enter Σ_N (`factors_used`
true) and the DiD field has scene-specific structure. Queued and running (launched 22:46:55 UTC box time, `run_steering_operators_full_box2.sh`, 54-line common discovery seed list = 53 complete scenes + 1 incomplete, log
`logs/steering_operators_full.log`, markers `SOF_FULL_DONE` → `SOF_V09_FIT_DONE` → `SOF_V09_STEER_DONE` →
`STEERING_OPERATORS_FULL_DONE`): (F) arm A and B min-distortion steering on the FULL 53-scene common discovery set with the
already-fitted operators (zero refitting) → `arm{A,B}_seed0/steer_min_distortion_full/`; (V) Operator 1 fitted on the CCGP job's
v0.9 dumps (`artifacts/drive_ccgp_v09/dump_arm{A,B}`: 66 common discovery seeds 5001…, 528 cells, 3 imagined steps, 18 sites;
stimulus `drive_v09_merged/rand/arm{A,B}` whose manifest carries hazard_dist_m / prefix_throttle / hazard_lateral_offset_m) →
`v09/arm{A,B}_seed0/min_distortion/`, then the steering table on the 66 v0.9 discovery scenes at the same bands →
`v09/arm{A,B}_seed0/steer_min_distortion/`. Both are development / calibration runs under the Label-first principle.

**Operator 2 steering (arm A, L02.adaln / L03.adaln, component all, ranks 1 2 4 → 12 families, ~229 configurations) and the
arm-B steering stages:** running since 22:41 UTC (ETA several hours at ~2.2 min per scene-config-set on the shared GPU); the
driver writes `SO_STEER_DONE` / `STEERING_OPERATORS_DONE` to `logs/steering_operators.log`; read with
`/opt/conda/bin/python /root/read_steering_results.py` and `/root/read_margins.py <steered_ranking.json>` (both on box 2).

## 7. Deviations from the task text, and why

1. **Energy in the causal inner product, not the Euclidean projected energy.** The task's objective ("projected relational energy gain
   subject to ‖δ‖_N ≤ ε") with a Euclidean energy would make the generalised eigenproblem favour high-nuisance-variance directions
   even more than PCA does (§2.3). The energy is therefore measured with the same causal inner product, E_N(δ) = Σ_i⟨δ, r_i⟩_N²,
   which is the only reading under which the required self-test ("recover the planted direction while the Euclidean twin does not")
   holds. The Euclidean twin uses the isotropic metric with the same trace.
2. **Strengthen is state-independent.** A state-dependent maximiser of E_N(h+δ) − E_N(h) pushes each token further along whatever
   sign it already has (suppress-like on negative tokens); the behavioural target needs the solid-consequence sign, so strengthen
   maximises E_N(m̄ + δ) (signed by the mean relational displacement) and is an additive edit in the causal-metric-optimal subspace.
   Suppress is state-dependent (per-token whitened projection removal within the budget).
3. **Pullback metric as a dose cap, not inside the closed form.** Reverse mode is blocked by the vendored `@torch.no_grad()` on
   `unroll`/`forward_pred`; the existing forward-AD JVP is used post hoc on the r fitted directions (`pullback` sub-command,
   `eps_cap_beta`). D_action in the closed form uses Q_act only.
4. **Mahalanobis constraint as an increment bound** (triangle inequality on the diagonal Mahalanobis norm), since the receiver
   Mahalanobis of h + δ depends on h; the actual post-edit value can be read from the steering run's `activation_edit_relative_rms` and
   the operator's `receiver_mean/var` (a per-token D_local report is not yet in the steering JSON — see §8).
5. **Modulation operator: U = V (slice contrasts), γ from the JVP selection score**, and the gate is the product of a hazard gate
   read at the block input and the action read (m − m_ref)·u_c. The task's "U, V, γ fit from the modulation-path DiD" cannot be read
   literally: the modulation vectors do not depend on the hazard (they are a function of the action only; HANDOFF notes them identical
   across scenes), so their DiD is exactly zero; the hazard dependence lives in how the modulation is *used*, which is what the JVP DiD
   measures. A full modulation-space gradient (d = 6D) would need reverse mode (blocked, see 3), so the candidate write directions are
   the six slice contrasts rather than an unconstrained 6D vector.
6. **`ungated` literal twin plus `ungated_on`**: the literal "g frozen at its hazard-free value" is near-sham when that value ≈ 0
   (it is 0.038 in the self-test); `ungated_on` (g ≡ 1) is added as the informative twin. Both are reported.
7. **Wrong block = `wrong_site` machinery on adaln sites** (kind name `wrong_block`), offset = half the depth as for every wrong-site control.
8. **`--controls` default** changed from the conceptor list to "the operator's own list"; for the conceptor operator this resolves to
   the same `CONTROLS`, so `run_coast_table.sh` behaviour is unchanged.

## 8. What remains

- Real-data numbers (§6.1): read `steered_ranking.json` / `calibration.json` / `causal_metric_fit.json` / `modulation_operator_fit.json`
  when `STEERING_OPERATORS_DONE` appears; report the calibrated β per family, the DiD (solid vs H0) with CI, `selective`,
  `all_controls_fail`, hazard-free ARC/drift checks, `activation_edit_relative_rms`; per site the twin comparison and LOSO alignment;
  per block the gate values on ghost/H0/H0′ and the top slices with max-T p.
- Run the `pullback` sub-command on the fitted operators (box, model needed) and re-run the steering with the dose cap if any
  `eps_cap_beta < 1`.
- Add the per-token receiver Mahalanobis of the edited tokens to the steering JSON (the operator stores `receiver_mean/var`; the
  runner does not yet report D_local after the edit).
- Sealed confirmation for these families must go through `run_coast_table.sh`-style calibration on the full discovery set, then
  `--calibration-file`; the 12-seed driver run is a smoke, not the calibration of record.
- Under the Label-first principle the operators' endpoint is the closed-loop MetaDrive-native label; wiring these two modes into the
  closed-loop rollout (`closed_loop_rollout.py`) is the next step once its adapter and C0 gate exist.
- Multi-seed (seeds 1–2) and Panel B (action-token predictor) have no modulation path; Operator 1 applies unchanged, Operator 2 needs
  the token adapter.

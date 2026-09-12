# Codex handoff — Panel P (performance-first Sonar steering on released world models)
**Written 2026-09-04 08:14 UTC; materially updated after the 2026-09-04 paper-to-code audit. Self-contained: assumes no prior session context.**

---

## 1. What this project is asking

Can a distribution-aware intervention in a frozen JEPA world model improve
native closed-loop robot planning performance under fixed/equalized compute,
without retraining the world model or pushing its activations off support?

The active method is **Sonar steering**, not a COAST reproduction: learn an
outcome-agnostic low-rank predictive coordinate system, model latent regimes,
fit regularized success/failure Gaussian energies after factor discovery, and
apply a minimum-metric-distortion edit. A faithful numerical COAST conceptor is
an optional secondary reference only. Orthogonal JEPA, LAVLA, GIFT, and the
JEPA/HMM state-space paper supply bounded ideas and falsification checks; none
is claimed as a reproduced method. The primary evidence is native task success,
not a mechanistic score.

**The label is sacred.** It is the released evaluator's own per-episode success
flag (`succ_def: simu`). Nothing in this pipeline may change the planner, goal
sampler, horizon, or success definition. This is written down as the
"Label-first principle, amendment 3" in `cross model design jepa.md`; every
runner and report is supposed to cite it. **Do not relax this.**

---

## 2. Access and live state

```
ssh -p 45460 root@70.27.250.55          # vast.ai instance 49766237 "cgs-pilot-2", RTX 3090
# key needs: ssh-add --apple-load-keychain   (passphrase is in the macOS keychain)
cd /root/cgs-pilot
```

The old stage-23 parent and seed-1 evaluator are finished/stopped; they cannot
auto-launch steering. The completed MetaWorld development capture has 30/30
episodes, 12 successes (40%), 18 failures, seven physical replans each, and no
capture errors. The one-pair capture qualification under
`artifacts/public_panel/performance_first_v1/development/manifest_smoke_*`
passed exact realization, native-outcome, and executed-action hash matching.
The corrected capture produced seven replans and distinct spatial-only versus
all-token means at all 18 sites. The fresh 15-episode fit capture is live as
PID `77574` under `artifacts/public_panel/performance_first_v1/fit/mw_jepa-wm`;
check that PID and `run.log` before taking action. The protected evaluation
manifest remains sealed. No Panel-P behavioral activation intervention has run.

Logs: `logs/public_panel.log` is the marker log (grep `PP_`). Per-run logs live
next to their outputs under `artifacts/public_panel/`.

Primary code is in local `scripts/cgs_pilot/` and deployed to remote
`/root/cgs-pilot/code/cgs_pilot/`: `public_panel_manifest.py`,
`public_panel_eval.py`, `public_panel_capture.py`,
`public_panel_factor_gate.py`, `public_panel_sonar_math.py`,
`public_panel_sonar_fit.py`, and `public_panel_steer.py`. Always compare SHA-256
before a launch. The checkpoint SHA-256 is
`c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8`;
the vendor commit is `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.

---

## 3. Pipeline shape

The performance-first sequence is binding:

1. Validate immutable, disjoint fit/evaluation manifests with independent
   environment and planner seeds.
2. Smoke the seed adapter and prove equality using realized initial/goal hashes,
   not seed equality alone.
3. Run the 15-pair fresh unsteered fit capture. Require at least three native
   successes and three failures; never replace episodes based on outcome.
4. On whole-episode folds, compare global, PCA-block, matched random-orthogonal,
   and learned predictive-factor coordinates; compare one Gaussian, static GMM,
   time bins, and HMM sequence models. Outcome labels may not select factors or
   regimes.
5. Fit success/failure densities only after coordinates/regimes are fixed. Run
   a bounded development tournament: mixture-energy trust-region versus
   Gaussian-OT, angular/radial/joint displacement, and matched controls.
6. Freeze exactly one Sonar recipe plus direct equal-compute planning baselines.
7. Open the paired protected evaluation once; report native success and
   preservation metrics whether positive or null. Replicate only a positive.

**Corrected critical design fact about capture.** PointMaze (`mz`) plans once
(1 call, 30 steps), but MetaWorld (`mw`) consistently plans **7 times across 99
environment steps**. The original blanket "plan once" assumption was false for
the decisive cell. The `planner` pool is now explicitly collected only during
call 0 of every episode, before any action executes. Hooks collect the states
the first CEM call computes while searching:
~(300 candidates × 256 tokens × 400 dim) per predictor forward, subsampled to
8 rows/forward, ≤256 per site per episode → ~7,680 rows/site over 30 episodes
vs D=400. That is COAST's N≫D regime.

The capture stores one chosen-plan state and the exact selected action plan per
physical replan, plus first-plan CEM candidates. It also records a common
pre-outcome window. The legacy capture exposed a bug: `mean` and `mean_all` were
identical because the batch axis was indexed as if it were time. The corrected
capture takes `mean` over the final `n_spatial` token block and `mean_all` over
all predicted tokens. The legacy result is therefore all-token-only.

---

## 4. Steering arms available

Active `public_panel_steer.py` Sonar arms:

| arm | operator | role |
|---|---|---|
| `sonar_energy` | proper success-minus-failure mixture-energy gradient under a nuisance/action/support metric trust region | primary candidate |
| `sonar_ot` | responsibility-weighted Gaussian/Bures failure-to-success transport | distinct affine comparator |
| `sonar_reverse` | sign-reversed energy step | causal sign control |
| `sonar_sham` | random-coordinate edit renormalized to the same metric norm | structure control |
| `identity` / `unsteered` | no-op hook / no hook | instrumentation and native baselines |

Each Sonar arm supports `angular`, `radial`, or `joint` displacement. Runtime
abstains outside the fitted Mahalanobis support. `public_panel_sonar_fit.py`
serializes coordinates, static regime densities, outcome mixtures, trust
metrics, OT maps, generalized-eigen diagnostics, conceptor spectra, and shams.

The following is the older COAST-specific arm inventory, retained for audit and
optional reference rather than as the active experimental table.

`public_panel_coast.py steer --kind <K>`:

| kind | operator | role |
|---|---|---|
| `sham` | β=0, identity | must reproduce unsteered episodes |
| `main` | `h' = h Mᵀ` about the origin | COAST's gate |
| `mean_cov` | `h' = μ_s + (h−μ_s) Mᵀ` | affine about success centre |
| `mean_only` | `h' = h + β(μ_s−μ_f)` | centre shift, no subspace |
| `belief` | `h' = Σ_r P(z=r∣h)·(h M_rᵀ)` | regime-local gates, belief-weighted |
| `belief_shuffled` | same, responsibilities rolled across r | matched control for `belief` |
| `energy_conceptor` | n steps of descent on `(h−μ)(I−C)(h−μ)ᵀ` | **cross-check only, see below** |
| `energy_llr` | n steps on the Gaussian log-likelihood-ratio energy | genuinely distinct arm |
| `random_matched` | matched-spectrum random conceptor | control |
| `rank_one` / `wrong_site` | CAA direction / same op at another block | controls |

Knobs: `--regime-operators` (dir of `<site>__regime<i>.npz`, needed by `belief`),
`--energy-steps`, `--energy-eta`, `--preserve-norm`.

**Do not report `energy_conceptor` as a new operator.** One descent step at
η = β/2 is *algebraically identical* to the COAST gate about μ — verified to
4.4e-16 in the self-test. It is kept only as an implementation cross-check. The
genuinely new part of the energy formulation is `energy_llr`, whose gradient
uses the **inverse** class covariances and therefore pulls hardest along the
**low-variance** success directions — the opposite weighting to the conceptor
gate. That contrast is the falsifiable content.

---

## 5. Results so far — and why they look bad

### Prior substrates (closed)
- Substrate 1 (JEPA-WM DROID, RoboCasa) and substrate 2 (V-JEPA 2-AC): behaviour gate FAIL, all mechanism arms null.
- Driving cell (MetaDrive, trained substrate): positive at the behaviour level, but closed-loop C0 FAIL for both seed-0 models (they cannot drive hazard-free), cross-truth 12/12 FAIL. Deferred; **do not train new driving models or generate new driving stimuli.**

### Panel P, cell `mz_jepa-wm` (PointMaze) — the only cell with a full chain
- Screen: 30 episodes, **25 success / 5 failure (83%)** → tier `fit_only`, above the band. A ceiling cell: almost no headroom for steering to show improvement.
- Fit selected `L04.mlp_out`, α=5.0, cv_score 0.037.
- **`overlap_success_failure = 0.957`** ← the most important number in this document.
- `quota = 0.159` vs COAST's reported ~0.01 — the operator is ~16× less selective than COAST's regime.
- Original supervised probe: best AUROC **0.752** at `L04.mlp_out`, sitewise permutation null p95 = 0.737, nominal **p = 0.050**.
- Corrected 200-permutation probe (shared label permutations, +1 correction, max AUROC across all 18 sites): at `L04.mlp_out`, sitewise p = **0.0697**, max-statistic null p95 = **0.832**, selection-adjusted p = **0.1841**. The mz probe is a clear fail, not a borderline pass.
- Corrected exploratory views also fail: candidate-cloud episode means peak at AUROC 0.688 with adjusted p 0.488; the chosen-plan `mean_all` summary peaks at 0.672 with adjusted p 0.682. Thus mz is not explained by row-level candidate dilution.
- Regime geometry (`--n-regimes 2`): Grassmann obs 0.711 vs null 0.356, **`p_regimes_differ = 0.0`**.
- Belief readout: held-out argmax accuracy **0.548** vs shuffled null 0.502, `p_belief_informative = 0.0`, mean posterior entropy 0.045 of 0.693 nats.

### How to read those numbers
- The **0.957 overlap** says `C_success` and `C_failure` occupy essentially the same subspace. `C_steer = C_s AND NOT C_f` is carving a ~4% sliver out of two near-identical clouds.
- The corrected probe says the apparent **AUROC 0.752** is compatible with selecting among 18 sites. If the label is not linearly present, a contrastive conceptor will not find it — and a steering null then tells you nothing about the operator, only that the signal is absent.
- **Discount `p_regimes_differ = 0.0` heavily.** The regime axis is early-vs-late rows in the CEM's call order, and a CEM *narrows its candidate distribution by design*. Early and late rows nearly have to differ geometrically. The permutation test controls for which rows, not for the optimiser converging. It is close to tautological and does not bear on the research question.
- Caveat on all of the above: `mz` is a ceiling cell with 5 failure episodes. It is thin evidence, and it is about optimiser dynamics rather than success/failure.

---

## 6. Root cause, first principles

For the causal primary test, capture happens at t=0 inside the first CEM call,
before any action executes. At that instant the model is imagining candidate
futures; eventual success may still be determined by environment interaction
and later replanning. `mz`'s `overlap = 0.957` is consistent with eventual
success and failure sharing the same first-plan states.

The mw discovery makes this a hypothesis rather than an established root cause:
mw replans seven times, so a signal may emerge honestly during interaction—or
appear only in late outcome-contaminated states. The failure diagnostics compare
first-plan candidates, chosen-plan summaries, the frozen pre-outcome replan
window, and all replans. **Only the first-plan result licenses first-plan
steering.** Later signals diagnose timing/dilution/leakage; they do not rescue
the primary gate.

---

## 7. The decision: development capture first, then a probability-explicit Sonar intervention

`mw_jepa-wm` (MetaWorld) is the **first genuinely eligible cell**: its completed
development capture is 12/30 = 40% success, squarely inside the 20–80% band,
with real headroom. Every old "bad" number above comes from `mz`, which was a
ceiling cell and was never the right performance test.

**User decision, 2026-09-04:** no Panel-P behavioral intervention has run yet,
and the old `sham/main/random_matched` auto-table must not run as the experiment.
Treat every opened seed-1 episode as development only. The factor/sequence gate
has now run. Predictive orthogonal factors did not beat the global predictor:
at selected `L00.resid_post`, normalized held-out MSE was global `0.4590`,
predictive blocks `0.5409`, PCA blocks `0.5673`, and median random blocks
`0.5584`. The active fallback is therefore **global whitened coordinates**;
do not force the Orthogonal-JEPA-inspired factorization or call it a gain.

The fail-closed rerun supersedes the preliminary HMM number: the two-state HMM
beat a static GMM by `0.2539` nats/replan (episode-bootstrap 95% interval
`[0.1436, 0.3460]`). That makes sequence state worth testing, but does not
license online HMM belief: the legacy capture omitted aligned executed controls,
so the action-conditioned older-history test is undecidable. New captures save
the exact executed control stream and replan boundaries; the diagnostic uses
causal per-chunk summaries standardized within each training fold, not planner
candidate tensors. HMM deployment also requires an action-conditioned
transition improvement and Chapman–Kolmogorov path-composition check. Until all
pass, use static responsibilities.

The old rule "a failed global decodability probe stops the experiment" is
superseded. A failure now stops only the global-linear/conceptor branch. The
prespecified nonlinear density/value/transport branches may continue, but no
mechanistic score can rescue a native behavioral null.

The revised intervention imports the world-model concepts from
`/Users/stevenyang/Documents/mechinterp-vla/cross model plan.md`, especially its
four probability objects. Keep these distinct: (1) activation density
`P(h_t|z_t)`, (2) filtered regime belief `P(z_t|h_1:t)`, (3) transition law
`P(z_t+1|z_t,a_t)`, and (4) attention allocation `P(j|i,z_t)`. A conceptor score,
JEPA compatibility energy, HMM posterior, and attention softmax are not synonyms.

The latent regime `z_t` must be fitted outcome-agnostically from ordered physical
replans, not from success/failure labels and not from early/late CEM optimizer
rows. Outcome labels enter only afterward to fit regularized local densities
`P(h_t|z_t,y)` on the fitting split. Compare one Gaussian, static K=2 GMM,
normalized-time bins, frozen simulator-event bins, and a K=2 Gaussian-emission
HMM on held-out whole episodes. Only a filtered posterior using current/past
states is eligible online; smoothing with future activations is forbidden.

If the distribution and sequence gates pass, fit the intervention in a frozen
low-rank, nuisance-whitened coordinate system. Compare the simple posterior-
weighted local energy with the normalized mixture energy/log-density ratio; use
the latter's responsibilities for its gradient. Constrain every edit by a local-
support penalty, nuisance-whitened norm cap, action-divergence budget, and ordinary-
behavior preservation. Run center-only, covariance-only, center+covariance, hard-
regime, soft-belief, and random/shuffled ablations on development data; freeze one
primary Sonar arm before fresh behavioral evaluation rather than opening an
uncorrected arm grid.

The clean evaluation table is: unsteered, identity hook, equal-compute direct
planner baselines, one frozen Sonar energy-or-OT operator, and a norm/
action-divergence-matched sham; a static-vs-HMM secondary comparison opens only
if sequence structure passes prospectively. All arms use the same new explicit
episode/configuration and planner-noise manifest. Do not use the nominal seed-2
cohort: the evaluator maps seed 1 to episode seeds 1..30 and seed 2 to 4,6,..62,
which overlap in 14 configurations. Generate and hash truly disjoint fit and
evaluation seed lists first.

The original first-plan-only hooks remain a valid narrow implementation test.
The revised world-model experiment must additionally retain physical replan,
CEM-iteration, candidate/elite, token, action, and simulator-state axes. First-
plan-only and all-pre-outcome-replan interventions are different estimands and
must be frozen separately. CPU-only hook tests and the geometry self-test still
must pass before either is used.

---

## 8. Hazards and things already learned the hard way

- **The legacy mw `mean` and `mean_all` pools are identical.** The hook indexed
  `x[-1]`, which selected the only batch row rather than the last spatial-token
  block. The corrected code uses `x[:, -n_spatial:]` for `mean`. Do not use the
  legacy run to select between pooling rules; call it all-token pooling.
- **Sequence likelihood is not HMM deployment authorization.** The positive
  `+0.2539` nats/replan result is held-out evidence that ordering matters,
  but the fixed transition may be absorbing time/progress. Require selected
  actions, incremental-history prediction, action-conditioned transitions, and
  Chapman–Kolmogorov composition before online belief steering.
- **`belief` / `belief_shuffled` are implemented and verified but deliberately NOT queued.** Replaying the hook math on real mz activations: belief vs identity 0.456, belief vs `main` **0.049**, belief vs `belief_shuffled` 0.073. A 5% operator difference read through a **binary** outcome at n=10 dev episodes has no resolving power. Re-run them at higher β, more dev episodes, or in a cell where P(z∣h) clears chance by more than ~5 points.
- **Operator files written before 2026-09-04 07:50 UTC must be refit.** `fit` computed μ_s/μ_f but never persisted them, so `mean_only`/`mean_cov` hard-exited with "operator has no class centers". Fixed; `fit` now writes `mean_success`, `mean_failure`, `sigma_success`, `sigma_failure`. `mz` has been refit and its geometry dir rebuilt. Pre-patch regime npz files also lack the `sigma`/`prior` that `belief` reads.
- **Posterior entropy is NOT a valid regime-degeneracy test.** Fitting two Gaussians to two halves of *one* cloud (D=48, N=500) still yields a confident posterior (0.34 of 0.69 nats) purely from sample-covariance noise — confidently-wrong is as degenerate as uniform. The verdict is held-out argmax accuracy against a within-episode label-shuffled null (`p_belief_informative`). The self-test asserts collapsed accuracy ≈ chance (0.488), not high entropy.
- **CKA only applies to matched rows.** It was once applied to unmatched samples, where it is meaningless. It is now used only on matched cross-model anchors; regimes are compared with principal angles / Grassmann / KL.
- **Regime geometry runs at the fit-selected site only** (`--site`). The 18-site sweep is ~7 h of CPU permutation tests for a report nothing steers on. `PP23_REGIME_ALL_SITES=1` restores it.
- **`run_public_panel_stage23.sh` is a running bash script.** Bash re-reads scripts from disk as it executes, so overwriting it in place can corrupt a live run. Replace it with a temp file + atomic `mv`, or kill and relaunch.
- **Do not restore the blanket "these configs plan once" claim.** mz has 1 plan call; every completed mw screen episode has 7. Pool and intervention scope must remain explicit.
- **Do not open the nominal seed-2 sets.** They are both still sealed and now retired as confirmation sets because their episode-seed map overlaps seed 1 in 14/30 configurations. Freeze explicit disjoint manifests before any new evaluation opening; only the user authorises that opening.
- **Do not touch box 1's registered chain** (a separate machine, parked at a hook error).
- Self-test before trusting anything: `python code/cgs_pilot/public_panel_geometry.py --self-test` → must print `"SELF_TEST": "PASS"`.

---

## 9. Reference

- Main rolling handoff: `claude_handoff/HANDOFF.md` (this file is the standalone extract; the §35/§36 entry, the fit bug and the mz numbers are at roughly lines 64–90).
- Active performance-first design: `/Users/stevenyang/Documents/mechinterp-vla/cross model plan.md`, amendment dated 2026-09-04.
- Binding native-label rule: `cross model design jepa.md`, "Label-first principle".
- User-supplied mathematical notes: `math_techniques.md`.
- Source of the §33–§56 operator ideas: `vla_world_model_research_thread_codex (1).md`.
- Prior-substrate results: `LOOP1_SUMMARY.md` + appendices A–E; `MORNING_REPORT_2026-09-04.md`.

---

## 10. Paper-to-code correction that supersedes the old arm description

The binding audit is `docs/WORLD_MODEL_MATH_CODE_AUDIT_2026-09-04.md`. Its
verdict is **major revisions and new evidence required before the first
behavioral activation intervention**. The live 15-episode job is still an
unsteered capture and must finish unchanged. No Panel-P intervention has run.

The corrected causal comparison preserves the base operator while varying
locality/routing:

```
global factor-space contrastive conceptor
static emission-routed regime-local contrastive conceptors
HMM-filtered routing of those exact same local conceptors (only if every HMM gate passes)
matched-spectrum and episode-label-shuffled controls
```

Gaussian energy and Gaussian OT remain separately named operator candidates.
The primary energy step is now the capped quadratic `-eta G^-1 g`, preserving
gradient magnitude below the cap; the always-boundary version is an explicit
ablation. COAST covariance uses separate class centers, exact Moore-Penrose
Boolean algebra, one aperture, and the origin-centered runtime map. HMM is only
a Sonar router, never a standalone edit. `sonar_coast_hmm` currently exits
before hook installation because action-conditioned transition fitting,
innovation sufficiency, empirical path composition, outcome-routing advantage,
treatment separation, and physical-replan belief lifecycle are not complete.

The Othello/model-native-coordinate gate was also corrected after inspecting
the exact MetaWorld source. Observation slots `4:7` contain a randomized
cylinder, not the wall. The wall is a separate fixed XML body. In addition, the
first saved simulator-state row is stale because the vendor evaluator returns
the `reset_warmup` observation while retaining `info` from the preceding reset.
The activation is valid, but coordinate analysis now drops replan zero and
validates all retained goal fields. The primary matched test is the same hand
state in world coordinates versus relative to the episode's changing goal,
with a shuffled-goal binding control. A fixed-wall translation is an
affine-equivalence negative control and must not look better under a centered
linear readout. Route clearance is a separate diagnostic. Even a pass only
nominates a coordinate; causal editing is required before it can shape Sonar.

A later code audit found that the original planner-candidate capture did not
match the hook population: it flattened leading axes and pooled all tokens,
while `SonarSteerer` preserves the top-level candidate batch and edits only the
final spatial-token block. It also stored candidates only at plan call zero even
though the proposed runtime hook acts at all seven MetaWorld replans. The
corrected capture has `--planner-capture-scope all_plans`, uses hook-identical
pooling, and writes `planner_callNNN` populations separately. The fit/apply gate
now requires every replan in the common pre-outcome window and aggregates those
rows within episode. Therefore the live first-plan-only capture remains valid
for its unsteered outcomes and chosen-action geometry, but cannot license an
all-replan intervention. Run a fresh development-only runtime-population
recapture after the immutable fit job finishes.

Regime and operator eligibility are now family-specific. `K=2` must beat a
one-Gaussian model out of episode and survive whole-episode component-aligned
bootstrap; otherwise the artifact is truly `K=1` and no local arm is listed.
Conceptor, energy-gradient, and Gaussian-OT stability are separate gates. The
leave-one-episode-out routing check fits the same local outcome densities for
static and causal-HMM routes and collapses replans to one episode score. None of
this enables HMM deployment: action-conditioned transitions, innovation and
path-composition tests, and the physical-replan runtime lifecycle are still
missing, so the runtime continues to stop before installing an HMM hook.

`public_panel_arm_registry.py` now implements the required separate arm freeze.
It never changes the old stimulus manifest, uses exclusive-create semantics,
hashes every input, and refuses an HMM arm without the HMM-specific evidence.

Current implementation/test state:

- local suite: `40 passed, 1 skipped` (torch is unavailable locally);
- current source hashes are deployed, and the remote torch `2.7.1+cu128`
  smoke passes hook-identical candidate pooling, causal-filter prefix
  invariance, capped energy, exact conceptor algebra, origin-centered global
  conceptors, and separate family/HMM eligibility loading;
- remaining blockers are fresh class/per-regime support, the new all-replan
  runtime-population capture and fit/apply gate, evaluator-level identity/action
  hashes, action-space dose/treatment separation, the unimplemented HMM-specific
  gates, and freezing (not merely generating tooling for) a hash-bound arm
  registry before protected evaluation.

Do not use the old auto-running stage-23 arm table or the OpenPI/LIBERO
Sonar-AC section of `cross model plan.md` for Panel P. The latter is VLA-only.

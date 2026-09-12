# CAFT-inspired pair (arm diffing + denial retraining) — implementation report, 2026-09-03

Status: PRELIMINARY (written at launch; the real-data sections are filled in below as the box-2 runners finish).
Design-doc anchor: `cross model design jepa.md`, "Registered additions" (e); grounding: `representational_geometry_paper_concepts.md` §5 (CAFT: PCA of model differences, low-rank projection in forward AND backward passes, computation denial, test-time removal, interpretive limit), §12.4 (hard projection vs soft conceptor), §14 (decodability is not use; variance is not semantics).

Evidential status (stated in both docstrings, the runner headers and every JSON): **engineering / necessity follow-up; cannot count as C3 on its own.** Open-loop gates are development readouts; the closed-loop native MetaDrive label (Label-first principle, amendments 1–2) is the outcome label when pilot data exist.

## Files (local `scripts/cgs_pilot/`, md5-verified identical on box 2 `/root/cgs-pilot/code/cgs_pilot/`)

| file | md5 | role |
|---|---|---|
| `arm_diff_pca.py` | fdf94c96… | experiment 1 (numpy only) + `--self-test` |
| `denial_finetune.py` | 2c6bcd22… | experiment 2 (`fit-subspaces`, `finetune`, `run-with-hook`, `summarize`) + `--self-test` |
| `run_arm_diff.sh` | e0ce5dbf… | box runner 1 (waits for the seed-0 dumps, copies them, runs the analysis; marker `ARM_DIFF_DONE`) |
| `run_denial.sh` | 5e76fb93… | box runner 2 (subspaces → features → gate stack on parent / ablations / 12 fine-tuned variants → optional closed loop → summary; marker `DENIAL_DONE`) |

No existing file was modified. `train_feature_predictor.py` is used unchanged (imported; `train()` runs with a hook registered on the vendor module). Outputs only under `artifacts/drive_arm_diff/` and `artifacts/drive_denial/`.

## Experiment 1 — arm diffing (`arm_diff_pca.py`)

Δh(x) = h_A(x) − h_B(x) per factorial cell (identical frames and actions in both arms; the arms share initialisation, seed and clip order, so this is the model-diffing analogue of CAFT's base/fine-tuned pair). Per site × step × token group {hazard, corridor, hazard_corridor} on the 53 common discovery scenes:

- two conventions: **transported** (arm B moved into arm A coordinates with the LOSO hazard-free Procrustes map of the cell's scene, `x_B→A = (x_B − m_B) Wᵀ + m_A`; fitted on levels 0/2 only, so the hazard-free Δh is the Procrustes residual by construction) and **raw** (shared ambient coordinates);
- four cell subsets: all (8 cells/scene), in_lane (levels 1, 3), hazard_free (0, 2), did (per scene `DiD_ped^A − DiD_ped^B`, `DiD_cone^A − DiD_cone^B`; a shared template cancels);
- centred PCA: PC1 ratio, participation ratio, 90 %-energy rank, mean-shift energy fraction;
- CAFT inspection: per PC the mean loading of every cell type (level × action), max/min-loading cells, η² of cell type and of scene;
- overlaps of the top-k Δh subspace (k ∈ {1,2,4,8,16}; registered k = 4) with (a) the arm-A relational conceptor `C_DiD(ped) ∧ ¬C_action ∧ ¬C_null` exactly as `geometry_cross_arm.py` fits it (trace overlap, conceptor mass inside the subspace, principal angles vs its top-k eigenvectors; matched-spectrum random control) and arm B's cone conceptor transported into A; (b) the identity-contrast direction `unit(mean_i DiD_ped,i − DiD_cone,i)`; (c) the donor-patch direction (unit mean corridor-pooled `h[1,a] − h[0,a]`, action-averaged and throttle-cell), at the same site and the registered `L03.mlp_out` corridor donor direction compared at every site; chance `sqrt(k/d)` and matched-rank random subspace reported;
- inference: scene-clustered bootstrap (n_boot 200 transported / 50 raw; everything refitted) and a scene-permutation null (A scene i paired with B scene π(i), same cell type, derangement; n_perm 200 / 50) with `p_ge`.
- outputs: `arm_diff_map.json`, `arm_diff_map.md`, `arm_diff_subspaces.npz` (top-16 PCs, mean shift, references, W per site/group; consumed by the denial experiment).

### Self-test (local, `plaid/.venv` python 3.12, numpy 1.26; synthetic dump pairs of `geometry_cross_arm.py`, arm B randomly rotated, 16 scenes, d = 64, n_boot = n_perm = 40; 64 s)

Planted (rank-2 relational subspace attached to the solid identity of each arm): transported in-lane Δh top-2 captures **0.999** of the planted subspace (did subset 0.999), raw (untransported) **0.011**, PC1 permutation p **0.024**, identity direction inside the top-2 **0.989**, conceptor mass inside the top-2 **0.504 vs 0.043** matched-spectrum random. Null (independent appearance subspaces, no relational term): planted-subspace energy 0.020 / 0.071 / ≤ 0.25 (in-lane / did / all), identity cos 0.012, conceptor mass 0.027 vs 0.033 random. `SELF_TEST_PASSED` (verdict at `artifacts/drive_arm_diff/self_test/self_test_verdict.json` on the box after stage 0).

## Experiment 2 — denial retraining (`denial_finetune.py`)

- Hook (`SubspaceDenial`): forward hook on the vendor module of site `L{LL}.{hook}` (`predictor_hooks._module_for`), output → `(I − UUᵀ)h` for all tokens of all frames, computed and returned in float32 (a bf16 return would leak ~1e-3 of the removed component back through rounding). The projection is in the autograd graph, so the backward pass is projected as well (CAFT's "forward AND backward"); `grad_check` records `max|Uᵀg|/‖g‖` at the site from one `loss.backward()`. `k = 0` is a bit-exact identity. The frozen encoder is never touched (training runs on cached features).
- Fine-tuning = `train_feature_predictor.train()` unchanged (same loss `L_tf/3 + L_roll/2`, AdamW, split rule `sha256("0:seed")[0] < 230`, clip order and seed 0) from `drive_models/armA_seed0/jepa-latest.pth.tar`, **frozen schedule: 7 epochs (23 % of the original 30), ref-lr 2e-4 (original 5e-4), warm-up 1 epoch, cosine to 1e-5, batch 16, bf16 autocast**; the script refuses `--epochs` > 25 % of the parent's epochs. The saved checkpoint is a plain predictor, so the frozen loader evaluates it WITHOUT the projection (test-time removal); `run-with-hook` re-installs the projection at load time for the WITH-projection evaluation through the unchanged `latent_cache.py` (and `closed_loop_rollout.py`).
- Subspaces (`fit-subspaces`, arm-A coordinates at the module output, from the seed-0 dumps): `conceptor` = top-k (μ-descending) eigenvectors of the relational conceptor at `L03.mlp_out` corridor step 0 (k ∈ {1, 4, 16}); `armdiff` = top-k PCs of the transported all-cell Δh at the same site (second choice); `random` = rank-matched random orthonormal subspace (a hard projector's matched quantity is rank); `none` = continued-training control; wrong layer = the same conceptor rule fitted at `L05.mlp_out` (the patch stage recovered ≈ 0 there); secondary site `L03.resid_post` (after the AdaLN gate; the `mlp_out` site sits before the element-wise gate, through which a projected output can in principle re-acquire a component along U — this is the CAFT interpretive limit for this architecture and the reason the post-gate site is also run).
- Gate stack per evaluation (WITHOUT and WITH projection): `latent_cache.py` (driving token mask) → `counterfactual_validity_gate.py` at level 1 and level 3 → `behavior_gate.py` with `--cross-gate` (T1c′), i.e. the EVAL stage of `run_drive_arms.sh` minus the model-independent contamination / retrieval audits. Readouts: captured interaction fraction (1 − median NMSE; B-gate T1b threshold 0.25), Rec specificity, CEM flip rate, T1c′, ghost-identity captured fraction, hazard-free val unroll-L2 relative to the parent (two-sided 5 % margin as `drive_models/equivalence.json`; the signed increase is also recorded).
- Also evaluated: the parent WITH the projection and no fine-tuning (pure inference-time ablation: conceptor k 1/4/16, armdiff k 4, random k 4).
- Interpretation flags (descriptive): consequence present WITH the projection after denial fine-tuning → re-routed (sufficient, not necessary); absent → not re-routable within the frozen budget (necessary relative to the budget; random-subspace and continued-training controls calibrate the budget); hazard-free loss outside the margin → not interpretable.
- Closed loop (stage 6, optional): once `CLOSED_LOOP_PILOT12_DONE` exists, `closed_loop_rollout.py run` (unchanged; native `arrive_dest` label) on the first 12 discovery scenes for the parent (no / with projection) and the `conceptor_k4`, `continued`, `random_k4` variants (no / with projection). At launch the closed-loop pilot had been re-scoped to the native label (legacy pilot killed 17:48 UTC), so this stage will only run if the pilot marker lands during the night.

### Self-test (local, CPU, 5.5 s; `--self-test`)

Tiny AdaLN-style predictor (same module names, so the same hook code runs; depth 2, width 32, frozen orthogonal output projection) trained 12 epochs on synthetic tokens with a planted hazard × action interaction along `v`; the planted internal direction is `u = Pᵀv` at `L01.resid_post`. Numbers (captured interaction fraction = 1 − median NMSE of the predicted vs true next-state DiD on 64 held-out clips):

| condition | captured |
|---|---|
| parent | 0.882 |
| parent WITH hook on `u` (no training) | 0.000 |
| parent WITH hook on a random direction | 0.882 |
| denial fine-tuned (3 epochs, hook on `u`) WITH projection | 0.000 |
| same, WITHOUT projection (test-time removal) | 0.758 |
| continued-training control (no hook) | 0.952 |
| random-direction denial WITH its projection | 0.950 |
| denial at the FIRST block (second block can re-route) WITH projection | 0.857 (re-routing example, reported not asserted) |

Also: gradient along U at the site 3.2e-8 of ‖g‖ (backward denial verified), empty subspace bit-exact, DiD-discovered top-1 direction at the site |cos| = 0.974 with `u`, reloaded checkpoint reproduces the no-projection number, hazard-free loss not degraded (> 25 %) in the three main variants (the tiny parent is not converged, so continued training lowers the loss; the real run uses the two-sided 5 % margin). `SELF_TEST_PASSED`. One self-test lesson worth keeping: after denial fine-tuning the interaction returns at test-time removal (0.76) — the weights that write into U receive no gradient, so a pre-existing route along U is neither used nor erased; only the WITH-projection number after fine-tuning speaks to re-routing.

## Box-2 execution plan (launched 17:50 UTC, both under `setsid nohup`)

- `run_arm_diff.sh` (pid 10893): stage 0 self-test on the box → waits for `artifacts/drive_identity_geometry/dump_arm{A,B}` (re-queued identity-geometry job, which starts after the `drive_factorial` rsync lands) and copies each dump the moment its `index.json` carries `slimmed_to_step0` (that job deletes its dumps in its cleanup stage) → `DUMPS_READY` → analysis (CPU, OMP 4; estimated 1–1.5 h) → `ARM_DIFF_DONE` in `logs/drive_arm_diff.log`. Fallback only if the copy window is missed (`DUMPS_MISSED` → re-dump after `PULL_DONE` under the GPU rule).
- `run_denial.sh` (pid 10903): stage 0 self-test → waits for `DUMPS_READY` → `fit-subspaces` → waits for `PULL_SHARDS_DONE` + GPU rule → arm-A features (identity-checked against `drive_features/armA/identity_sample.npz`) → waits for `PULL_DONE` → parent evaluations → 12 variants (≈ 13 min each: 7 epochs ≈ 4 min + two gate-stack evaluations) → summary → `DENIAL_DONE` in `logs/drive_denial.log`. GPU rule: `G2_DONE` in `logs/box2_orchestrator.log` or `nvidia-smi` < 8 GB used, re-checked before every GPU stage; one model at a time.

Poll: `ssh -p 45460 root@70.27.250.55 'tail -3 /root/cgs-pilot/logs/drive_arm_diff.log /root/cgs-pilot/logs/drive_denial.log; grep -c ARM_DIFF_DONE /root/cgs-pilot/logs/drive_arm_diff.log; grep -c DENIAL_DONE /root/cgs-pilot/logs/drive_denial.log'`.
Results: `artifacts/drive_arm_diff/arm_diff_map.md`, `artifacts/drive_denial/denial_summary.md` (numbers re-derivable from the JSONs next to them).

## Real-data results

### Timeline / deviations (box 2)

- 17:27 UTC the orchestrator's first identity-geometry run failed (raw cells not yet pulled); re-queued by the main session, rerun 18:11–20:14 UTC. `run_arm_diff.sh` copied both slimmed dumps at 20:03 UTC (`DUMPS_READY`, 1.6 GB, 18 sites × 424 cells × 106 tokens × 512, step 0) before that job's cleanup deleted them; no fallback dump was needed.
- `run_denial.sh` failed at 20:04 UTC on an argparse placement bug of mine (`--seed` after the `fit-subspaces` subcommand); fixed (subcommands now accept `--seed`, runner passes it first), self-test re-run (local + box), relaunched 21:35 UTC (new md5: `denial_finetune.py` 4e506d4c…, `run_denial.sh` cecc0b1b…). Subspaces fitted 21:36 UTC; the runner then blocks on the GPU rule (13.4 GB in use by Panel B eval / CCGP / v0.9 rerun / planner diagnostics) until `G2_DONE` or < 8 GB.
- `arm_diff_pca.py` runs ~20 min per site on the loaded box (my estimate was 4 min): 18 sites → ETA ≈ 02:00 UTC 2026-09-04 for `ARM_DIFF_DONE`. The JSON is written only at the end.
- No Panel B token model is used anywhere in either experiment (the coordinator's note that arm-B token models trained 17 vs 30 epochs, `drive_models_token/equivalence.json` within_5pct=false, therefore does not affect these results).

### Subspace fits (`artifacts/drive_denial/subspaces/*.json`, 53 common discovery scenes, corridor group, step 0, arm-A coordinates, d = 512)

| site | conceptor alpha / rank / quota | armdiff PC1 ratio / participation ratio | conceptor-k4 vs armdiff-k4 mean cos² | identity dir inside conceptor-k4 / armdiff-k4 (chance 0.088) | donor dir inside conceptor-k4 | LOSO transport residual (transported / raw) |
|---|---|---|---|---|---|---|
| `L03.mlp_out` (registered) | 38.7 / 159 / 0.0052 (≈ 2.6 effective dims) | 0.742 / 1.69 | 0.019 | 0.192 / 0.368 | 0.242 | 0.475 / 0.838 |
| `L03.resid_post` (post-gate) | 35.5 / 159 / 0.0051 | 0.751 / 1.67 | 0.030 | 0.174 / 0.403 | 0.468 | 0.378 / 0.671 |
| `L05.mlp_out` (wrong layer) | 43.3 / 159 / 0.0034 | 0.540 / 2.62 | 0.181 | 0.347 / 0.553 | 0.336 | 0.523 / 0.803 |

Reading (descriptive): the transported arm difference at the discovered site is nearly rank-1 (PC1 74 % of the variance) and almost orthogonal to the relational conceptor's top-4 (mean cos² 0.02) — the two candidate-direction routes propose different subspaces; the identity-contrast direction sits more in the arm-diff subspace (0.37) than in the conceptor's (0.19), both above chance. Full spectrum / overlap / null tables follow when `ARM_DIFF_DONE` lands.

(sections appended below as results land)

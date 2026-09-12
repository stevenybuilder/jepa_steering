# Integrity audit — driving cell artifacts (arms A/B, 3 seeds)

Scope: `artifacts/cgs_pilot/{drive_eval,drive_eval_crosstruth,drive_mech,drive_relational,drive_models}`, `drive_results_summary.json`, `paper/results_tables.md`.
Remote: `root@192.220.55.116:/root/cgs-pilot/artifacts/drive_*`. Nothing under artifacts/, paper/, scripts/ was modified.

## VERDICT: CLEAN

No evidence of dummy, synthetic, placeholder, or locally edited data. Every gate/patch/jacobian/crosstruth/relational JSON is byte-identical to the remote copy, carries checkpoint/config sha256 that match the actual remote files, and every headline number re-derives from stored per-scene values. Items 1-6 below are the checks; item 7 lists benign observations that a reader might otherwise mistake for problems.

## 1. Remote-vs-local consistency: PASS
- 148/148 JSON files (`find ... -name "*.json" -not -path "*/activations/*"`) have identical md5 remote vs local (keyed join; the raw `diff` shows only sort-order noise from macOS vs GNU `sort`). 0 files only-local, 0 only-remote.
- 63/63 non-JSON sidecars (`discovery_seeds*.txt`, `confirmation_seeds*.txt`, `manifest.jsonl`, `jacobian_seeds.txt`, `relational_transport.md`) identical. Only-remote: the 12 `drive_models/*/{eval,train}_config.yaml` (never rsynced; harmless, see item 2).
- `drive_results_summary.json` and `paper/results_tables.md` do not exist on the remote by design: they are produced locally by `scripts/cgs_pilot/aggregate_drive_results.py` (mtime 2026-09-03 06:56:31 EDT, both). Their fidelity is checked in item 5.

## 2. Provenance fields: PASS
- `checkpoint_sha256` in every file equals `sha256sum` of the remote `drive_models/<model>/jepa-latest.pth.tar` (computed on the box this audit): A0 7fcfc9b4…, A1 dcf61ede…, A2 9d7122cd…, B0 9184b93d…, B1 488666dc…, B2 45f005b5…. Consistent per model across `drive_eval/<m>/cf_gate_*.json:cache_meta`, `drive_eval_crosstruth/<m>_on_*/cf_gate_*`, `drive_mech/<m>/{patch_step0/patch_results,jacobian_sonar/jacobian_sonar,localize/interaction_map}.json`, and `drive_relational/relational_transport.json:arms.{A,B}`.
- `config_sha256` in every file equals remote `sha256sum drive_models/<m>/eval_config.yaml` (e.g. A0 6e4ce846…).
- `calibration_seeds_included=false` and `confirmation_run=false` in both `patch_results.json` and both `interaction_map.json`. No key named `dry_run/planted/self_test/placeholder/dummy/mock/fake/TODO` exists in any file. Case-insensitive grep over all JSON + `results_tables.md` hits only `drive_models/*/jepa-*.meta.json:hyperparameters.{synthetic,write_synthetic}=null` (vendor training CLI args recorded as unset).
- n_scenes, seeds lists, `runtime_s`/`elapsed_s` present (patch A 7468 s, B 8052 s; jacobian 737/743 s; relational 90.5 s), matching driver log durations.

## 3. Internal consistency: PASS
- All 6 `drive_eval/<m>/{discovery_seeds.txt, EVAL_SCOPE.discovery_seeds, summary.split.discovery}` are the same 54-seed set; `per_scene` keys of every `cf_gate_discovery*.json` (6 models x 2 levels) and all 12 crosstruth `cf_gate_discovery_level{1,3}.json` equal that set; `n_scenes=54`, `pooled.*.n=54` everywhere. Discovery ∩ confirmation = ∅ (66 confirmation seeds).
- Confirmation seeds appear in no discovery-stage result. (Grep hits in `drive_eval_crosstruth/*/summary.json` are the truth arm's merge summary copied verbatim — md5 `1b4a218b…` equals `drive_eval/armB_seed0/summary.json`; `5034abb4…` equals armA — which legitimately lists the full 120-seed split; no per-scene results there.)
- Recomputed from stored per-scene values (tolerance 1e-9), all match: `pooled.interaction_nmse.{median,mean}`, `rec_h1`, `rec_h0_control`, `interaction_cosine` medians in all 24 cf_gate discovery files; `behavior_gate*.json:T1b_interaction_nmse_median` = cf pooled median and `model_captured_interaction_fraction` = 1 − T1b in all 12 gate files (e.g. A0: 0.3147267153 / 0.6852732847); all 12 crosstruth `levels.*.gate.interaction_nmse_median` and `captured_fraction`; `own_truth_reference` = the model's own `drive_eval` gate.
- `drive_mech/armA_seed0/patch_step0/patch_results.json` site L03.mlp_out|gripper_corridor, hazard_safe_to_unsafe: stored `donor.recovery_region_median` 0.7186812857 = median over `per_pair[<scene>].L03.mlp_out.0.gripper_corridor.hazard_safe_to_unsafe.{fixed_0,fixed_1}.donor.recovery_region` averaged per scene (53 scenes); the `point` 0.7191 is the mean. Same holds for all four directions.

## 4. Value distributions: PASS
- No p-value equals 0 anywhere. Sign-flip p floor is 1.9996e-4 (= 1/5001) and appears where the effect is unanimous; jacobian permutation p's (n_perm=2000) are e.g. 0.00778, 0.01167.
- No repeated per-scene NMSE values within any gate file; no zero-variance per-scene arrays in gate/crosstruth/relational files.
- Round-valued medians are structural only: `identity_contrast.energy_gap_h3_vs_h0.median=0.0`, `sonar…k1.action_main_effect.cosine=1.0` (k=1 self-cosine), `retrieval_baseline.json:sonar.per_token.k*.persistence=0.0`. No headline field is round.
- The only constant arrays are `drive_mech/*/localize/interaction_map.json:reference_results[*].cv_projection.*.per_scene` (18 entries per arm; role `action_availability_reference`, site `L00.adaln`, token_group `mod`, n_tokens=1). These are AdaLN action-conditioning reference rows: `contamination.json:action_audit.all_scenes_share_identical_actions=true`, so the per-scene action projection is constant by construction (sd 1.8e-15) and identity/interaction/relational are numerically 0 (~1e-7). They are excluded from `maps_by_step`. Not fabricated; file is byte-identical to remote.

## 5. Aggregator fidelity: PASS
- All 186 numeric/verdict fields in `drive_results_summary.json` (6 models x primary/cross x 9 fields, retrieval, identity contrast; 6 cross-truth cells x 2 levels x 4 fields) equal the named source JSON values exactly.
- 10 random `results_tables.md` cells (seed 7) re-derived and match with the script's formatting, e.g. R1 armB_seed1 pedestrian `0.630 (0.564, 0.615) | 0.370 | +0.221 (2e-4) | 0.889 | PASS_PLANNER`; R2 A-on-B pedestrian seed1 `2.896 | -1.896 | -0.067 (2e-4) | 0.926 | FAIL`; R2 B-on-A cone seed1 `1.520 | -0.520 | -0.137 (2e-4) | 0.889 | FAIL`; R1 armA_seed2 pedestrian; R2 B-own pedestrian seed2; etc. 10/10 OK.
- The remote `logs/drive_crosstruth.log` AGGREGATE line (independent, remote-side) gives the same pooled numbers as Table R2 (mA/tA/L1 nmse 0.327 cap 0.663 rec +0.215; mA/tB/L1 3.241/−2.149/−0.119; mB/tA/L1 1.084/−0.085/−0.449; ...).

## 6. Timestamps / ordering: PASS
- Local mtimes (converted to UTC) equal remote mtimes to the second for all 13 key files checked, e.g. `drive_eval/armA_seed0/cf_gate_discovery.json` 2026-09-02 23:52:42, `behavior_gate_discovery.json` 03:22:38, `drive_eval_crosstruth/summary.json` 03:42:56, `drive_models/armA_seed0/jepa-latest.meta.json` 22:32:23, `drive_relational/relational_transport.json` 06:19:03, `drive_mech/armB_seed0/jacobian_sonar.json` 08:48:49. No local file is newer than its remote copy.
- Ordering matches `logs/drive_arms_driver.log`: train done 23:35:12 → eval A0 done 00:08:45 → B0 00:56:31 → A1 01:34:29 → B1 02:11:35 → A2 02:47:42 → B2 03:22:38 → mech A0 03:22:38–05:54:53 → mech B0 05:55:02–08:48:52; crosstruth 00:19–03:42:56; relational 05:57–06:19:03. File mtimes fall inside their stage windows.
- Newest artifact JSON is 08:48:49 UTC (mech B0 jacobian); the local aggregate was generated 10:56 UTC, after all inputs.

## 7. Benign observations (not integrity issues, but worth knowing)
- **Seed 101 is in the 54-scene discovery split but excluded from the mechanism stage (n=53).** `patch_results.json`, `band.json` and `interaction_map.json` list 53 scenes; the missing one is `drive_seed000101`. Cause: `scripts/cgs_pilot/localize_interaction.py:107` / `merge_heldout.py:84` hard-code `CALIBRATION_SEEDS=(101,102)` from the egg domain, so the driving mech tools silently drop driving seed 101 (`calibration_seeds_included=false`); the frames step in `drive_mech_*.log` refused outright ("calibration seeds [101] may not be used"). The gate stage (n=54) is unaffected. Paper text saying "53 scenes" for patching vs "54" for gates is correct but the reason is a domain-constant leak, not a preregistered exclusion.
- **Arm A Jacobian was killed and re-run on a subset.** `drive_arms_driver.log` "Terminated" and `drive_mech_armA_seed0.log:729` "coordinator: killed full-set Jacobian … subset re-run follows": the full 53-scene Jacobian was killed for time (`bound_mech_a0.sh`), and `jacobian_sonar.json` (06:07:11) is the 8-seed subset (3,4,7,9,10,14,15,17 = first 8 discovery seeds, `jacobian_seeds.txt`), same for arm B by design. Any claim citing the Jacobian should say n=8.
- `drive_eval_crosstruth/<X>_on_<Y>/summary.json` is a verbatim copy of the truth arm's merge `summary.json` (not a crosstruth result).
- `results_tables.md` CI columns are CI-of-mean while the point is a median (8 cases where the median lies outside); the aggregator flags and labels this itself.
- The 12 `drive_models/*/{eval,train}_config.yaml` are not on the local copy (only their sha256 is recorded in the JSONs). Rsync them if the paper wants the configs archived.

Tooling: `/usr/local/bin/python3.9`; remote md5/sha256/stat via ssh; scratch files in `/Users/stevenyang/.claude/jobs/159de8a4/tmp/{remote_md5,local_md5,r2,l2}.txt`.

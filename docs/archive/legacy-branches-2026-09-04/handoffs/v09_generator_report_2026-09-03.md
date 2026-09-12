# MetaDrive factorial generator: protocol v0.9 stimulus changes (2026-09-03)

Files changed (local `scripts/cgs_pilot/` == remote `/root/cgs-pilot/code/cgs_pilot/`, md5 verified; remote backups `*.v08_backup_2026-09-03`):
`metadrive_hazard_pilot.py` (md5 02254ace), `metadrive_validate_labels.py` (b0317a78), `merge_heldout.py` (518fc07b), `protocol.py` (5faf0db5).

## 1. Diff summary
**metadrive_hazard_pilot.py** (all new behaviour opt-in; `V09_ACTIVE` is set only when a v0.9 flag is present on the command line)
- New flags (argparse group "v0.9 stimulus factors"): `--prefix-throttle X`, `--lateral-offset Y` (in-lane levels 1/3 only; H0/H0' unchanged), `--randomize "dist:8,10.5;prefix:0.0,0.5;lateral:-0.5,0.5"`, `--fov DEG`, `--settle-steps N`. Any of them switches `PROTOCOL_VERSION_DRIVE` to `cgs-metadrive-pilot-v0.9` (rows, summary provenance, root provenance.json); unchanged runs keep `cgs-metadrive-pilot-v0.7` in rows exactly as today.
- `run_episode(prefix_throttle=None)` now resolves the module value at call time (previously bound at def time, so the constant could not be changed). Settle (`SETTLE_STEPS>0`): reset -> spawn -> `settle_physics` (N x decision_repeat Bullet ticks, no control) -> seed ego speed; applied to EVERY episode incl. hazard-free baselines so all cells share one physics history. `hazard_pose` = settled (context) pose; new `hazard_spawn_pose` / `hazard_settle_drift_m`.
- Helpers: `level_lateral`, `settle_physics`, `json_safe`, `parse_randomize`, `factor_rng` (RandomState seeded by sha256("<seed>:<factor>")[:4]), `draw_factors`, `seed_factors`, `v09_metadata`, `apply_row_factors` (replay/validator re-derive dist, prefix throttle, lateral, FOV, settle steps, body from the saved rows; v0.8 rows -> v0.8 values).
- Metadata (v0.9 only): rows get `hazard_lateral_offset_m, fov_deg, settle_steps, randomized_factors, randomize_spec, hazard_spawn_pose_xyzh, hazard_settle_drift_m`; `camera` string uses the actual FOV; summary pair gets `stimulus_factors` + `hazard_settle_drift_m`; provenance gets `v09_stimulus`; root provenance.json gets `v09_flags` + `drawn_factors_by_seed`.
- Summary writes go through `json_safe` (non-finite -> null) before `canonical_json`; protocol.py's `canonical_json` untouched. Worker/replay commands are `shlex.join`ed (the randomize spec contains `;`).
**metadrive_validate_labels.py**: `--free-body-tol M` (dynamic body, no-contact cells: 3-D body drift gate uses it instead of 1e-6; ego-state gate still `--ghost-tol`); `contact_without_physical_effect` for a dynamic body = NOT(ego slowed OR body displaced > tol OR ego deviates from hazard-free > ghost-tol), with a per-cell `physical_effect` breakdown; per-cell `hazard_moved_xyz_m`, settle drift; record-level factors; output `protocol` = the rows' protocol string. Factors re-derived from rows via `gen.apply_row_factors`.
**protocol.py**: `PROTOCOL_VERSION_V09` constant; `replay_gate` treats v0.9 exactly as v0.8. **merge_heldout.py**: v0.9 in `PROTOCOLS`/`--protocol` choices, accepted for `--domain driving`, frame-gate note; v0.8 path unchanged.

## 2. Byte-identity (no new flags), seed 1090, d=10, new code vs `drive_factorial_v08/d10/arm{A,B}/seed_1090/`
- Files 28 vs 28 per arm, identical set. IDENTICAL: all `masks/*.npz` (8), `manifest_pending`-derived physics: every `ego_states`, `crash_*`, `gap/lon/lat/center_dist`, `hazard_pose`, `context_frames` array in every cell npz, `scene_baseline.npz` states/corridor/prefix_travel, 6 of 8 render PNGs. `summary.json` differs in exactly one key: `provenance.generator_sha256` (the generator hashes its own file). `manifest.jsonl` differs only in `artifact_sha256/final_render_sha256/replay_*pixel*` fields; `replay_report.json` only in `replay_*pixel*` fields.
- Every remaining diff is EGL render jitter in `true_future_frames`: 2-9 of 589,824 values per cell, |diff| <= 4/255, <= 3 px per frame (`scene_baseline.npz` frames_a1: 3 px, |diff| 4). Control: the OLD code (backup) re-run fresh on the same seed shows the identical jitter class vs the archive (also 15/28 files identical) and old-fresh vs new-fresh agree on 17-22/28 files with the same jitter class + generator_sha256 only. The archive's own rows already record `replay_frames_bit_exact:false` for this seed. Conclusion: no change is attributable to the code; physics, masks, context frames and all metadata are byte-identical.
- v0.8 regression of merge_heldout: `--domain driving --protocol cgs-metadrive-pilot-v0.8` on `drive_factorial_v08/d10/armA` reproduces `drive_v08_merged/d10/armA/{manifest.jsonl,summary.json,discovery_seeds.txt,confirmation_seeds.txt}` sha256-identical.

## 3. FOV probe (static body, 2 seeds; identical numbers on both): silhouette patches / border px / in_frame at the context frame
| FOV | dist | H0 sidewalk | H1 ped in lane | H0' mirror | H1_obj cone |
|---|---|---|---|---|---|
| 40 | 6 m | 0 px, out | 22 / 0 / out | 0 px, out | 14 / 0 / out |
| 40 | 7 m | 8 / 0 / out | 15 / 9 / in | 11 / 0 / out | 18 / 5 / in |
| 40 | 9 m | 12 / 20 / in | 12 / 39 / in | 12 / 21 / in | 12 / 36 / in |
| 50 | 6 m | 13 / 0 / out | 14 / 17 / in | 13 / 0 / out | 18 / 12 / in |
| 50 | 7 m | 11 / 17 / in | 12 / 36 / in | 12 / 17 / in | 12 / 32 / in |
| 50 | 9 m | 7 / 44 / in | 10 / 58 / in | 7 / 44 / in | 10 / 56 / in |
| 60 | 6 m | 12 / 21 / in | 12 / 39 / in | 12 / 21 / in | 12 / 35 / in |
| 60 | 7 m | 10 / 38 / in | 10 / 53 / in | 10 / 39 / in | 10 / 51 / in |
| 60 | 9 m | 8 / 60 / in | 8 / 72 / in | 7 / 60 / in | 8 / 70 / in |
Projection error (bbox centre vs analytic) 0.8-3.6 px everywhere in frame. 50 deg makes 7 m valid, 60 deg makes 6 m valid; the >= 6-patch gate holds at 9 m (10 / 8 patches for the in-lane pedestrian) but at 60 deg and > 10 m it will approach 6.

## 4. Knock-over body, 9 m, seeds 0 and 1, `--hazard-body dynamic --settle-steps 30`, validator `--free-body-tol 1e-3`
- Settle probe (physics-only ticks): in-lane pedestrian drifts 0; free cone drifts 1.9e-5 m over steps 1-10 then Bullet deactivates it (drift exactly 0 for 200+ steps); sidewalk pedestrian slides 0.75-0.78 m along the raised kerb (z 0.875 -> 1.02) and rests by step 30.
- Free-body invariance gates now PASS in both arms and seeds: all 6 no-contact cells per arm show body motion 0.00e+00 (was 1.9e-5 / 0.48 m in F18); no-contact ego deviation 1.8e-12 m; replay bit-exact 32/32 cells; contact only in the solid in-lane throttle cell at sim step 7 (model step 3).
- Solid in-lane throttle cell: body displaced 1.57 / 1.63 m (pedestrian, arm A) and 1.54 / 1.60 m (cone, arm B); ego does not slow (8.28 -> 8.84-9.01 m/s) but `physical_effect = {ego_slowed: F, body_displaced: T, ego_deviates: T}` -> `contact_without_physical_effect` no longer fires.
- Remaining failures at 40 deg: H0 (sidewalk) after its 0.77 m slide is out of frame (border 0, 5 patches) -> `hazard_not_fully_in_frame`, `mask_not_at_projected_pose(inf)`, and `null_displacement_mismatch(0.221)`. With `--fov 50` the same run passes everything except `null_displacement_mismatch(0.25)`: the settled H0 is no longer the mirror of H0'. Fix options (not implemented, outside spec): keep off-path bodies (levels 0, 2) immovable in dynamic runs (they are never contacted), or spawn H0 on the sidewalk surface.

## 5. 4-seed smoke: `factorial --randomize "dist:8,10.5;prefix:0.0,0.5;lateral:-0.5,0.5" --hazard-body static --procs 4` (seeds 3000-3003), then the validator (defaults)
| seed | dist m | prefix | lateral m | ctx speed | contact (A h1a1 / B h3a1) | proj err max px (L0/L1/L2/L3) | border min px | patches L1/L3 | null match_frac | result |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | 9.824 | 0.457 | -0.178 | 8.687 | step 7 / step 7 | 4.9/1.9/2.3/1.3 | 30 | 9/11 | 0.147 | FAIL null_displacement_mismatch |
| 3001 | 9.770 | 0.246 | -0.465 | 8.373 | step 7 / step 7 | 5.2/2.1/2.5/1.6 | 30 | 12/12 | 0.341 | FAIL null_displacement_mismatch |
| 3002 | 9.229 | 0.183 | -0.109 | 8.280 | step 7 / step 7 | 5.4/2.3/2.4/1.5 | 23 | 11/11 | 0.092 | PASS |
| 3003 | 9.666 | 0.108 | -0.028 | 8.156 | step 7 / step 7 | 5.3/2.2/2.6/1.4 | 28 | 11/12 | 0.023 | PASS |
- Draws are reproducible, identical across arms (arm B rows carry the same values) and uniform over 120 seeds (dist mean 9.21/std 0.70 vs 9.25/0.72 expected; lateral mean -0.03/std 0.28 vs 0/0.29); the 4-seed clustering is chance. Protocol string `cgs-metadrive-pilot-v0.9` in rows, summary and validator output; metadata keys present.
- All 64 cells replay bit-exact; frame jitter max 10/255 on <= 3 px (within the 16/16 gate); no-contact deviation <= 2.7e-12; contact pattern correct in every seed; brake min gap 4.2-4.7 m. Mask-vs-projected-pose holds at 6 px across the drawn range (max 5.4 px, level 0), but 8 m draws will need `--proj-tol-px 8` (the v0.8 amendment; observed 6.1-6.2 px at 8 m). In-frame gate holds at every drawn distance (border >= 23 px); from the FOV probe and the v0.8 record the 40 deg threshold is 7 m (H0/H0' touch the border), so dist >= 8 m is safe.
- The null-displacement gate is violated by construction once |lateral| exceeds ~0.1 m: with H0/H0' fixed, |c(H0')-c(H1)| / |c(H0)-c(H1)| ~ (2.25+l)/(2.25-l), so match_frac ~ 2|l|/(2.25-|l|) > 0.10 for |l| > ~0.11 m. With lateral U(-0.5, 0.5) about 78 % of seeds will be excluded by merge_heldout for that reason alone.
- merge_heldout with `--protocol cgs-metadrive-pilot-v0.9 --domain driving` on the smoke: 4 complete octets, admits 3002/3003, excludes 3000/3001 with `labels:null_displacement_mismatch`.

## 6. Command lines for a 120-seed v0.9 factorial (box: `/opt/conda/envs/metadrive/bin/python`, code `/root/cgs-pilot/code/cgs_pilot/`)
```
P=/opt/conda/envs/metadrive/bin/python; C=/root/cgs-pilot/code/cgs_pilot; OUT=/root/cgs-pilot/artifacts/drive_factorial_v09/rand
# (a) gate-compatible: lateral within the 10 % null rule, distances >= 8 m (in frame at 40 deg)
$P $C/metadrive_hazard_pilot.py factorial --out $OUT --seed-range 5000 5120 --hazard-body static --procs 8 \
   --randomize "dist:8,10.5;prefix:0.0,0.5;lateral:-0.1,0.1"
$P $C/metadrive_validate_labels.py --root $OUT --procs 8 --proj-tol-px 8          # 8 px = v0.8 amendment for 8 m draws
for arm in A B; do $P $C/merge_heldout.py --root $OUT/arm$arm --output /root/cgs-pilot/artifacts/drive_v09_merged/rand/arm$arm \
   --domain driving --protocol cgs-metadrive-pilot-v0.9 --label-validation $OUT/_validation/hazard_label_validation.json --exclude-seeds; done
# (b) full lateral range as specified (expect ~78 % null-gate exclusions unless the null rule is amended, see caveats)
$P $C/metadrive_hazard_pilot.py factorial --out ${OUT}_lat05 --seed-range 5000 5120 --hazard-body static --procs 8 \
   --randomize "dist:8,10.5;prefix:0.0,0.5;lateral:-0.5,0.5"
# (c) closer hazards need a wider FOV: --fov 50 (7 m in frame) or --fov 60 (6 m in frame), e.g. --randomize "dist:7,10.5;..." --fov 50
# (d) knock-over body: --hazard-body dynamic --settle-steps 30 [--fov 50]; validate with --free-body-tol 1e-3
```
Cost (box unloaded): ~50 s generation + ~22 s replay per seed per process; 120 seeds at --procs 8 ~ 15-20 min, ~11 MB/seed (~1.3 GB). Validator ~25 s/seed. Seeds 5000+ are disjoint from v0.7 (0-119) and v0.8 (1000-1119 range).

## 7. Caveats
1. Disk: the box has 2.9 GB free (96 %); a 120-seed run needs ~1.3 GB plus validation sheets. Free space first.
2. Null-displacement rule vs lateral offset (section 5): either use |lateral| <= 0.1 m, or amend the v0.9 null gate (generator `null_match`, validator `NULL_MATCH_FRAC`, merge exclusion) to compare against the geometric expectation, or mirror H0' about the hazard lateral (changes the H0' pose, which the spec kept fixed). Not changed here.
3. Dynamic body: the sidewalk pedestrian's kerb slide (0.77 m) breaks in-frame at 40 deg and the mirror-null match at any FOV; the in-lane invariance problem (F18) is solved by `--settle-steps 30`. Recommend immovable off-path bodies for dynamic runs (small follow-up).
4. `deterministic_replay` in the generator's own replay pass still uses the old <= 8/255 pixel rule (seed 3002 flagged 10/255 on 1 px); the validator and merge use the preregistered 16/16 rule. Pre-existing, unchanged.
5. merge_heldout does not check the rows' `protocol_version` against `--protocol` (v0.9 data merges under the v0.8 flag too); pre-existing.
6. `apply_row_factors` sets the generator module globals from the first row of arm A; all cells of a seed share the same factors by construction.
7. `--randomize` draws are per (seed, factor) via sha256 -> RandomState, independent of numpy's global RNG; changing the factor NAME changes the stream.
8. Passing a v0.9 flag at its v0.8 value (e.g. `--fov 40`) still marks the run v0.9 (protocol bump + metadata) by design; the physics/renders are then identical to v0.8.

# Panel B: action-as-token / action-as-feature JEPA predictor on the cached DINOv3 features (2026-09-03)

## 1. What changed (diff: `panelB_trainer.diff` next to this file; +248/-31 lines; original md5 0585e200…, new 268141ee…)
`scripts/cgs_pilot/train_feature_predictor.py` (local == box `/root/cgs-pilot/code/cgs_pilot/`, backup on the box
`train_feature_predictor.py.adaln_backup_2026-09-03`):
- `--pred-type {AdaLN,token,feature}` (default AdaLN) and `--action-emb-dim` (feature only, default 64).
- `ModelCfg` gains `pred_type="AdaLN"`, `action_emb_dim=0` (defaults reproduce AdaLN; the frozen post-check in
  run_drive_arms.sh filters `meta.model` by `ModelCfg.__dataclass_fields__`, so it also rebuilds token/feature models).
- `conditioning_fields(cfg)`: the yaml fields that differ per predictor (AdaLN values are the historical literals);
  `ac_predictor_kwargs(cfg)` / `build_ac_predictor(cfg, device, with_encoder)` build the vendor
  `src/models/ac_predictor.py::VisionTransformerPredictorAC` through `vit_ac_predictor(...)` with exactly the kwargs
  `init_video_model(pred_type="vjepa2_ac")` passes (`is_frame_causal=True, use_rope=True, tubelet_size=1, num_frames=T,
  action_encoder_inpred=True, proprio_encoder_inpred=True, proprio_tokens=0, proprio_dim=None`; `use_silu=None` -> GELU);
  `--with-encoder` asserts state-dict identity against the real `init_video_model` build (passed for token and feature).
- `build_vendor_predictor` dispatches to `build_ac_predictor` when `cfg.pred_type != "AdaLN"` (AdaLN branch untouched).
- `forward_pred` dispatches on the class name to `forward_pred_ac` = `VideoWM.forward_pred`'s `vjepa2_ac` branch:
  `predictor(feats.flatten(1,4), actions [B,T,A], None) -> view(B,T,1,G,G,D)`; the vendor class strips its own
  conditioning tokens (`cond_tokens=1`, action token only) / action slice before `predictor_proj`. `jepa_wm_loss`,
  `unroll_eval`, `evaluate`, optimiser, schedules, split, checkpoint format are unchanged and shared.
- `eval_config()` / `train_config()`: only `action_conditioning`, `proprio_encoding`, `action_encoder`,
  `proprio_encoder`, `predictor.pred_type` vary (token: `token/token/{tokens 1, emb 0, inpred}/{tokens 0, emb 0,
  inpred true}/vjepa2_ac`; feature: `feature/feature/{tokens 0, emb 64, inpred}/…/vjepa2_ac`), mirroring the vendor
  DROID `vj2ac … noprop` yamls; a `cgs_pilot.panel_b` block is added for non-AdaLN only. Meta gains
  `model.pred_type/action_emb_dim` (+ `block_width`, `state_token`, `vendor.panel_b_templates` for Panel B).
- New `scripts/cgs_pilot/check_panelb_loader.py`: loads `<dir>/eval_config.yaml` + checkpoint through the UNCHANGED
  `model_action_sensitivity.load_model` (the call latent_cache.py / action_jacobian_sonar.py make), asserts class +
  every state-dict tensor equal (vendor loads strict=False), reproduces `probe.npz` via `EncPredWM.unroll`, and compares
  `VideoWM.forward_pred` / `EncPredWM.unroll` with the trainer's `forward_pred` / `unroll_eval` on random inputs.
- New `scripts/cgs_pilot/run_panelb_token_arms.sh` (see 4); `_panelb_smoke_box.sh` = the smoke driver used below.
- No frozen file was edited (run_wave_drive.sh, run_drive_arms.sh, latent_cache.py, … untouched; vendor untouched).

## 2. AdaLN path byte-for-byte proof
CPU fp32 run (synthetic 24,4,2, grid 4, dim 64, depth 2, 2 epochs), original file vs new: predictor tensors
`torch.equal` on every key, per-epoch history equal, probe.npz equal, `eval_config.yaml` and `train_config.yaml`
byte-equal modulo the folder line. GPU bf16 runs of old vs new differ only at 1e-7 (run-to-run SDPA/bf16 noise; the
old file is not bitwise reproducible with itself either), yamls byte-equal. The box's
`test_train_feature_predictor.py` (4 tests) passes against the new file. Loader check on the AdaLN smoke: PASS, 0.0.

## 3. Smoke tests on the box GPU (RTX 3090 shared with the generators: load 23-34, 2-17 GB VRAM used by others)
`--synthetic 64,4,2 --epochs 2 --batch-size 16 --pred-depth 6 --pred-embed-dim 512` (8 heads), bf16 autocast:
| model | class | params | peak VRAM | ms/iter (bench, batch 16, fwd+bwd+step) | loader check |
| AdaLN | VisionTransformerAdaLN | 29.42 M | 7.16 GB | 485 (orig. real run: 222) | PASS 0.0 |
| token | VisionTransformerPredictorAC (block 512) | 19.97 M | 4.19 GB | 345 | PASS 0.0 (also from the on-disk store) |
| feature (emb 64) | VisionTransformerPredictorAC (block 576) | 24.98 M | 5.07 GB | 272 | PASS 0.0 |
Loader check = class equal, 80/80 (92/92) tensors equal, no missing/unexpected keys, `EncPredWM.unroll` vs stored probe
0.0, `VideoWM.forward_pred`/`unroll` vs trainer 0.0 on random inputs. FeatureStore directory path
(`--write-synthetic` -> `--features <dir>` memmap) trains identically to the in-RAM store (same losses, 1.1811).
Losses fall below epoch 1 for all three (token 1.18 train / 1.24 val-unroll after 2 epochs vs AdaLN 1.30 / 1.38).
First feature attempt used emb 16 -> head dim 66 (not a multiple of 8) -> SDPA math kernel: 10.2 GB, 560 ms/iter,
and it OOM'd on the shared GPU; default is now 64 (head dim 72) and a warning fires for non-multiple-of-8 head dims.
Timing relative to AdaLN's bench on the same loaded box: token 0.71x, feature 0.56x -> projected real run (1433 train
clips, 90 iter/epoch, 30 epochs) ~ 8-10 min/model incl. validation, well under the 20-min cap; VRAM 4.2 / 5.1 GB.
Artefacts kept locally: `panelB_smoke/` (smoke.log, cpu_equiv.txt, loader_check_*.json, meta_*.json, eval_config_*.yaml);
`/root/cgs-pilot/artifacts/_panelB_dev` deleted (box disk now 6.2 GB free).

## 4. Exact commands (v2 clips) — packaged as `run_panelb_token_arms.sh` (local + box code dir, bash -n OK, NOT launched)
State: `/root/cgs-pilot/artifacts/drive_train_v2` is the generator's RAW output (train_summary.json present, 1598
clips/arm, `armA/shard_0200000_0200099.{jsonl,npz}`, `shared/`); v1's `drive_train_shards` was deleted for disk, so
conversion to precompute-format shards is required first (as `relaunch_train.sh` did for v1):
```
DPY=/opt/conda/envs/metadrive/bin/python; MPY=/opt/conda/bin/python; B=/root/cgs-pilot; REPO=$B/vendor/jepa-wms; cd $B/code/cgs_pilot
$DPY metadrive_train_to_shards.py --root $B/artifacts/drive_train_v2 --out $B/artifacts/drive_train_v2_shards --clips-per-shard 100   # + --delete-source-frames if disk demands
export JEPAWM_HOME=$B/vendor JEPAWM_OSSCKPT=$B/checkpoints JEPAWM_LOGS=$B/logs PYTHONPATH=$REPO:$B/code/cgs_pilot JEPAWM_DRIVING_ACTION_DIM=2
for ARM in A B; do   # (a) stage 1b, identical flags; then the stage-1c split heredoc VERBATIM (hash byte 230, split seed 0)
  $MPY precompute_dinov3_features.py --shards "$B/artifacts/drive_train_v2_shards/arm$ARM/shard_*.npz" --repo $REPO \
    --config $REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml \
    --checkpoint $B/checkpoints/jepa_wm_droid.pth.tar --out $B/artifacts/drive_features_v2/arm$ARM --pool none --dtype float16 \
    --batch-frames 32 --split-seed 0 --split-names train,val --verify-clips 2 --self-check 2 --resume
  $MPY - $B/artifacts/drive_features_v2/arm$ARM $B/artifacts/drive_train_v2_shards/arm$ARM 0 230 <<'EOF' ... (run_drive_arms.sh lines 135-171, verbatim in the script)
  for s in 0 1 2; do   # (b) == drive_models/armA_seed0 hyperparameters (train_config/jepa-latest.meta.json) + --pred-type token
    MD=$B/artifacts/drive_models_token/arm${ARM}_seed$s; mkdir -p $MD
    $MPY train_feature_predictor.py --repo $REPO --features $B/artifacts/drive_features_v2/arm$ARM --pred-type token --pred-depth 6 \
      --pred-embed-dim 512 --batch-size 16 --epochs 30 --max-minutes 20 --seed $s --arm ${ARM}_seed$s --out $MD --no-save-opt
    $MPY - $REPO $MD $B/artifacts/drive_features_v2/arm$ARM <<'EOF' ... (stage-2 post-train check, run_drive_arms.sh lines 236-268 VERBATIM -> post_train_check.json)
    $MPY check_panelb_loader.py --repo $REPO --model-dir $MD          # -> loader_check.json (must print LOADER_CHECK PASS)
  done; rm -f $B/artifacts/drive_features_v2/arm$ARM/features.npy      # arm-outer, LOW_DISK: 3.35 GB per arm, box has ~6 GB free
done; <equivalence table heredoc VERBATIM> -> $B/artifacts/drive_models_token/equivalence.json
```
Defaults implied by the AdaLN run and reproduced: 8 heads (`default_heads(512)`), ref-lr 5e-4, start 1e-6, final 1e-5,
warm-up 1 epoch, wd 1e-7 -> 1e-6, betas 0.9/0.995, eps 1e-8, clip 1.0, rollout_steps 2 (random prefix, stop-grad,
ctxt 3), eval ctxt 2, val_fraction 0.1 (overridden by `meta.split`), bf16. Launch:
`ssh -n -p 20566 root@192.220.55.116 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_panelb_token_arms.sh > /root/cgs-pilot/logs/panelb_token_driver.log 2>&1 < /dev/null &'`
(`PANELB_PRED_TYPE=feature` -> drive_models_feature; `PANELB_PRED_TYPE=AdaLN` -> drive_models_v2 for the matched Panel A
retrain on the same v2 features). Logs: `logs/panelb_token_arms.log`, `panelb_features_v2.log`, `panelb_train_token_arms.log`.

## 5. How the eval stack loads token models (no change to any frozen file)
`eval_config.yaml` of a token/feature model is a valid input to the existing path:
`latent_cache.py` -> `model_action_sensitivity.load_model(repo, eval_config.yaml, jepa-latest.pth.tar, "jepa_wm_driving", dev)`
-> `hubconf._load_model_with_config` -> `init_module` -> `init_video_model(pred_type="vjepa2_ac", action_conditioning=token|feature,
proprio_encoding=token|feature, proprio_tokens=0, use_proprio=False)` -> `VideoWM(pred_type="vjepa2_ac")` -> `EncPredWM`.
`model.encode(frames)` and `model.unroll(z, act_suffix=[T,B,2])` work unchanged (verified: 0.0 vs trainer). So for
run_wave_drive.sh just point `DRIVE_CFG=$MD/eval_config.yaml DRIVE_CKPT=$MD/jepa-latest.pth.tar DRIVE_MODEL_NAME=jepa_wm_driving`
at a drive_models_token dir, as stage 4 of run_drive_arms.sh does for AdaLN. No wrapper module was needed.
Hook-stack caveat: `predictor_hooks.py` / token_groups target AdaLN block names; the token predictor's blocks are
`predictor_blocks.{i}` (`attn`, `mlp`, no `adaLN_modulation`) and the residual stream per frame is `[action_token, 256 patches]`
(token index 0 = action token; strip/offset by `cond_tokens=1` when grouping) — the "token adapter" of the design doc is still to be written.

## 6. Caveats
- State token: NONE (`proprio_tokens: 0`) — the vendor's own DROID vj2ac "noprop" recipe. The design doc's "one action
  token + one state token" is not used because the feature store has no proprio, a zero-filled state token would be a
  learned constant token, and the frozen eval path (`EncPredWM.unroll` with a Tensor context) passes `proprio=None`
  (a `proprio_tokens: 1` model would crash there). Documented in the trainer docstring and `eval_config.cgs_pilot.panel_b`.
- Attention window: the AC predictor uses the vendor full block-causal mask over all 4 frames (`local_window_time=T`),
  AdaLN uses `local_window_time 3`; the only difference is teacher-forced frame 3 seeing frame 0 (eval ctxt windows <= 3
  are unaffected). Vendor architecture as-is; changing it would break the loader equivalence.
- Loss-equivalence (P2) risk: the token model has fewer params (20.0 M vs 29.4 M) and a different conditioning path;
  the 5 % hazard-free val-loss margin is between arms A/B within an architecture, which is what the script reports —
  cross-architecture loss levels will differ and must not be used for selection. Feature (block 576, 25.0 M) is a third
  variant, not matched in width to AdaLN.
- GPU timings above are inflated by the box's CPU/GPU load; AdaLN measured 485 ms/iter now vs 222 in its real run.
- The v2 shard conversion (~2.5 GB) + one arm's features (3.35 GB) barely fit in the 6.2 GB free; use
  `PANELB_DELETE_SOURCE_FRAMES=1` or free space first. The script reuses completed models/features on relaunch.
- Not tested on real features (being regenerated); the FeatureStore memmap path was exercised with the on-disk synthetic store.

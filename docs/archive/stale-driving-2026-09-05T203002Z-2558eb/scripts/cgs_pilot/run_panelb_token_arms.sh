#!/bin/bash
# run_panelb_token_arms.sh — Panel B (cross model design jepa.md): train the action-as-token predictor
# (train_feature_predictor.py --pred-type token) on the REGENERATED v2 clips, arms A/B x seeds 0 1 2, with the
# SAME split rule, hyper-parameters, post-train check and equivalence table as run_drive_arms.sh stages 1b/1c/2
# (those blocks are copied VERBATIM from run_drive_arms.sh; run_drive_arms.sh itself is frozen and untouched).
#
#   0 SHARDS    metadrive_train_to_shards.py on the generator output $GEN (raw jsonl/npz per seed range) -> $SHARDS/arm{A,B}/
#               shard_NNNN.npz + .meta.json (precompute format), exactly as relaunch_train.sh did for v1.  Skipped when
#               $SHARDS/shards.meta.json exists.  With PANELB_DELETE_SOURCE_FRAMES=1 the raw frames are removed after
#               verified writes (disk: the box has ~6 GB free; shards ~2.5 GB, full-grid fp16 features 3.35 GB per arm).
#   1 FEATURES  per arm: precompute_dinov3_features.py (identical flags to stage 1b) -> $FEAT/arm$ARM, then the stage 1c
#               split (sha256(f"{split_seed}:{seed}")[0] < 230 -> train else val, by clip seed, identical across arms),
#               training_reference.npz, sample_frames.npz, identity_sample.npz.
#   2 TRAIN     per arm, seeds 0 1 2: train_feature_predictor.py --pred-type token, depth 6 / width 512 / 8 heads /
#               batch 16 / 30 epochs / --max-minutes 20 / --no-save-opt (== drive_models/armA_seed0 hyperparameters),
#               then the stage-2 post-train check VERBATIM (encoder identity + hazard-free val loss; it rebuilds the
#               predictor through tfp.build_vendor_predictor, which dispatches on meta.model.pred_type) and
#               check_panelb_loader.py (reload equivalence through the unchanged eval loader).
#               Arm-outer with features.npy deleted after the arm's three trainings (DRIVE_LOW_DISK=1 behaviour).
#   3 TABLE     paired equivalence table (P2, reported only) VERBATIM -> $MODELS/equivalence.json
#
# Usage (remote): ssh -n -p PORT root@HOST 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_panelb_token_arms.sh > /root/cgs-pilot/logs/panelb_token_driver.log 2>&1 < /dev/null &'
# Env overrides: PANELB_PRED_TYPE (token|feature|AdaLN; default token), PANELB_MODELS (default $OUT/drive_models_$PRED_TYPE,
# drive_models_v2 for AdaLN), PANELB_FEAT (default $OUT/drive_features_v2), PANELB_GEN, PANELB_SHARDS, PANELB_SEEDS, PANELB_EPOCHS.
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}
CODE=${DRIVE_CODE:-$BASE/code/cgs_pilot}
REPO=${DRIVE_REPO:-$BASE/vendor/jepa-wms}
MPY=${DRIVE_MPY:-/opt/conda/bin/python}
DPY=${DRIVE_DPY:-/opt/conda/envs/metadrive/bin/python}
ENC_CFG=${DRIVE_ENC_CFG:-$REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml}
ENC_CKPT=${DRIVE_ENC_CKPT:-$BASE/checkpoints/jepa_wm_droid.pth.tar}
PRED_TYPE=${PANELB_PRED_TYPE:-token}
PRED_DEPTH=6; PRED_WIDTH=512; BATCH=16; SEEDS=${PANELB_SEEDS:-"0 1 2"}; SPLIT_SEED=0; VAL_HASH_BYTE=230
EPOCHS=${PANELB_EPOCHS:-30}; TRAIN_MAX_MIN=${PANELB_TRAIN_MAX_MIN:-20}
OUT=${DRIVE_OUT_ROOT:-$BASE/artifacts}; LOGDIR=${DRIVE_LOGDIR:-$BASE/logs}
GEN=${PANELB_GEN:-$OUT/drive_train_v2}                       # generator output (metadrive_hazard_pilot.py train --out ...)
SHARDS=${PANELB_SHARDS:-$OUT/drive_train_v2_shards}          # precompute-format shards (written by stage 0, READ-ONLY after)
FEAT=${PANELB_FEAT:-$OUT/drive_features_v2}
if [ "$PRED_TYPE" = AdaLN ]; then MODELS=${PANELB_MODELS:-$OUT/drive_models_v2}; else MODELS=${PANELB_MODELS:-$OUT/drive_models_$PRED_TYPE}; fi
mkdir -p $OUT $LOGDIR $FEAT $MODELS
DRIVER=$LOGDIR/panelb_${PRED_TYPE}_arms.log
export JEPAWM_HOME=$BASE/vendor JEPAWM_OSSCKPT=$BASE/checkpoints JEPAWM_LOGS=$BASE/logs PYTHONPATH=$REPO:$CODE
export JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12}
FILTER='^INFO:|^🔧|^🧠|^🔮|^📉|Encoder: |Predictor: |^Loading|^Loaded'
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*" | tee -a $DRIVER; }
disk() { df -h $BASE | tail -1 | awk '{print "disk used "$3" of "$2", free "$4}'; }
cd $CODE
say "run_panelb_token_arms start pred_type=$PRED_TYPE gen=$GEN shards=$SHARDS feat=$FEAT models=$MODELS epochs=$EPOCHS max_min=$TRAIN_MAX_MIN seeds='$SEEDS'; $(disk)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a $DRIVER

# ============================================================================ 0. SHARDS (generator raw -> precompute format)
if [ ! -f $SHARDS/shards.meta.json ]; then
  [ -f $GEN/train_summary.json ] || { say "shards: $GEN/train_summary.json missing (generator not finished)"; exit 1; }
  say "shards: converting $GEN -> $SHARDS"
  $DPY metadrive_train_to_shards.py --root $GEN --out $SHARDS --clips-per-shard 100 ${PANELB_DELETE_SOURCE_FRAMES:+--delete-source-frames} >> $DRIVER 2>&1 || { say "shards: conversion FAILED"; exit 1; }
  $DPY - $SHARDS >> $DRIVER 2>&1 <<'EOF'
import json, sys
r = json.load(open(sys.argv[1] + '/shards.meta.json')); a = r['arms']['A']; b = r['arms']['B']
print('SHARDS_DONE clips/arm A=%d B=%d hazard A=%d B=%d colliding A=%d B=%d shared=%d shared_identical_across_arms=%s clip_ids_equal_across_arms=%s shards/arm=%d bytes_total=%d' % (
  a['n_clips'], b['n_clips'], a['n_hazard'], b['n_hazard'], a['n_contact'], b['n_contact'], a['n_shared'], r['shared_identical_across_arms'], r['clip_ids_equal_across_arms'], len(a['shards']), r['total_bytes']))
EOF
  say "shards done: $(grep SHARDS_DONE $DRIVER | tail -1); $(disk)"
else say "shards: $SHARDS/shards.meta.json present, reused"; fi

# ============================================================================ 1. FEATURES (stage 1b + 1c of run_drive_arms.sh, verbatim flags)
FLOG=$LOGDIR/panelb_features_v2.log; touch $FLOG
features_arm() {   # features_arm ARM : precompute + split + training reference + samples for one arm
  local ARM=$1 FD=$FEAT/arm$1; mkdir -p $FD
  if [ -f $FD/training_reference.npz ] && [ -f $FD/features.npy ]; then echo "[$(ts)] features arm $ARM complete, reused" >> $FLOG; return 0; fi
    if [ ! -f $FD/features.npy ]; then rm -f $FD/features.meta.json $FD/progress.json
      echo "[$(ts)] precompute arm $ARM" >> $FLOG
      $MPY precompute_dinov3_features.py --shards "$SHARDS/arm$ARM/shard_*.npz" --repo $REPO --config $ENC_CFG --checkpoint $ENC_CKPT \
        --out $FD --pool none --dtype float16 --batch-frames 32 --split-seed $SPLIT_SEED --split-names train,val --verify-clips 2 --self-check 2 --resume 2>&1 | grep -Ev "$FILTER" >> $FLOG
      [ -f $FD/features.meta.json ] || { say "features: precompute arm $ARM FAILED (no features.meta.json)"; return 1; }
    else echo "[$(ts)] features arm $ARM present, reused" >> $FLOG; fi
    # 1c stable 90/10 split by seed + training reference + sample frames (VERBATIM from run_drive_arms.sh)
    $MPY - $FD $SHARDS/arm$ARM $SPLIT_SEED $VAL_HASH_BYTE >> $FLOG 2>&1 <<'EOF' || { say "features: split/reference arm $ARM FAILED"; return 1; }
import hashlib, json, sys, numpy as np
from pathlib import Path
fd, sd, split_seed, thr = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
clips = json.loads((fd / "clips.json").read_text())
orig = fd / "clips.hash128.json"
if not orig.exists():
    orig.write_text(json.dumps(clips, indent=1) + "\n")  # precompute's own 50/50 hash split, kept for provenance
for c in clips:
    h = hashlib.sha256(f"{split_seed}:{c['seed']}".encode()).digest()[0]
    c["split"] = "train" if h < thr else "val"
rule = {"split_rule": f"sha256(f'{split_seed}:{{seed}}').digest()[0] < {thr} -> train else val (by clip seed; identical across arms)",
        "split_seed": split_seed, "threshold_byte": thr, "n_train": sum(c["split"] == "train" for c in clips), "n_val": sum(c["split"] == "val" for c in clips),
        "n_val_hazard_free": sum(c["split"] == "val" and not c["hazard"] for c in clips), "n_train_hazard": sum(c["split"] == "train" and c["hazard"] for c in clips)}
(fd / "clips.json").write_text(json.dumps(clips, indent=1) + "\n")
(fd / "split_rule.json").write_text(json.dumps(rule, indent=1) + "\n")
feats = np.load(fd / "features.npy", mmap_mode="r")
acts = np.load(fd / "actions.npy").astype(np.float32)
N, T = feats.shape[:2]
assert N == len(clips) and acts.shape[0] == N, (N, len(clips), acts.shape)
train_idx = [i for i, c in enumerate(clips) if c["split"] == "train"]
pooled = np.empty((len(train_idx) * T, feats.shape[-1]), dtype=np.float32)
for k, i in enumerate(train_idx):
    pooled[k * T:(k + 1) * T] = np.asarray(feats[i], dtype=np.float32).mean(axis=(1, 2))
cid = np.array([clips[i]["clip_id"] for i in train_idx])
np.savez(fd / "training_reference.npz", pooled=pooled, clip_id=np.repeat(cid, T), frame_index=np.tile(np.arange(T), len(train_idx)),
         action_chunks=acts[train_idx], chunk_clip=cid, note=np.asarray("pooled = mean over the 16x16 grid of every frame of every TRAIN-split clip; action_chunks [M, T-1, 2] raw (steer, throttle)"))
# sample frames (2 clips) read from the shards for the post-training encoder-identity check
pick = [0, N // 2]
fr, idx = [], []
for i in pick:
    c = clips[i]
    with np.load(sd / c["shard"]) as z:
        fr.append(np.asarray(z["frames"][c["index_in_shard"]]))
    idx.append(i)
np.savez(fd / "sample_frames.npz", frames=np.stack(fr), index=np.asarray(idx))
# identity sample (P1 check across arms without both feature files on disk): 8 shared hazard-free + 4 hazard clips, deterministic
rng = np.random.default_rng(0)
sh = [i for i, c in enumerate(clips) if not c["hazard"]]; hz = [i for i, c in enumerate(clips) if c["hazard"]]
sel = sorted(rng.choice(sh, min(8, len(sh)), replace=False).tolist() + rng.choice(hz, min(4, len(hz)), replace=False).tolist())
np.savez(fd / "identity_sample.npz", index=np.asarray(sel), clip_id=np.array([clips[i]["clip_id"] for i in sel]), hazard=np.asarray([clips[i]["hazard"] for i in sel]),
         features=np.stack([np.asarray(feats[i]) for i in sel]))
print(json.dumps({"arm": fd.name, **rule, "reference_rows": int(len(pooled)), "reference_clips": int(len(cid)), "features_shape": list(feats.shape), "actions_shape": list(acts.shape)}))
EOF
  return 0
}

# ============================================================================ 2. TRAIN (stage 2 of run_drive_arms.sh + --pred-type)
TLOG=$LOGDIR/panelb_train_${PRED_TYPE}_arms.log
echo "[$(ts)] TRAIN start pred_type=$PRED_TYPE depth=$PRED_DEPTH width=$PRED_WIDTH batch=$BATCH epochs=$EPOCHS max_min=$TRAIN_MAX_MIN seeds='$SEEDS'; $(disk)" >> $TLOG
train_model() {   # train_model ARM SEED
  local ARM=$1 s=$2 MD=$MODELS/arm$1_seed$2 FD=$FEAT/arm$1
  if [ -f $MD/jepa-latest.meta.json ] && [ -f $MD/post_train_check.json ] && [ -f $MD/loader_check.json ]; then echo "[$(ts)] model arm$ARM seed$s present, reused" >> $TLOG; return 0; fi
  rm -rf $MD; mkdir -p $MD
  echo "[$(ts)] === train $PRED_TYPE arm $ARM seed $s -> $MD" >> $TLOG
  $MPY train_feature_predictor.py --repo $REPO --features $FD --pred-type $PRED_TYPE --pred-depth $PRED_DEPTH --pred-embed-dim $PRED_WIDTH --batch-size $BATCH \
    --epochs $EPOCHS --max-minutes $TRAIN_MAX_MIN --seed $s --arm ${ARM}_seed$s --out $MD --no-save-opt 2>&1 | grep -Ev "$FILTER" >> $TLOG
  [ -f $MD/jepa-latest.pth.tar ] || { say "train: arm $ARM seed $s FAILED"; return 1; }
  grep -q "stopping: --max-minutes" $TLOG && echo "[$(ts)] WARNING arm $ARM seed $s hit the --max-minutes cap (pair no longer has identical update counts)" >> $TLOG
  # post-train check VERBATIM from run_drive_arms.sh (encoder identity + hazard-free val loss, latest AND best)
    $MPY - $REPO $MD $FD >> $TLOG 2>&1 <<'EOF' || { say "train: post-check arm $ARM seed $s FAILED"; return 1; }
import json, sys, numpy as np, torch
from pathlib import Path
repo, md, fd = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo))
import train_feature_predictor as tfp
from model_action_sensitivity import encode_frames, load_model
dev = torch.device("cuda:0")
meta = json.loads((md / "jepa-latest.meta.json").read_text())
rep = {"arm": meta["arm"], "epochs_run": len(meta["history"]), "epochs_requested": meta["hyperparameters"]["epochs"], "best": meta["best"],
       "split_rule": meta["data"]["split_rule"], "n_train": meta["data"]["n_train"], "n_val": meta["data"]["n_val"],
       "hazard_fraction_train": meta["data"]["hazard_fraction_train"], "elapsed_s": meta["timing"]["elapsed_s"], "ms_per_iter": meta["timing"]["ms_per_iter_mean"],
       "steps_per_epoch": meta["schedule"]["steps_per_epoch"], "final_epoch": meta["history"][-1]}
# (a) encoder identity: the driving model's own encoder on stored frames == stored fp16 features (half-ulp + 1e-4)
model, _ = load_model(repo, md / "eval_config.yaml", md / "jepa-latest.pth.tar", "jepa_wm_driving", "cuda:0")
feats = np.load(fd / "features.npy", mmap_mode="r"); smp = np.load(fd / "sample_frames.npz")
worst = 0.0; ok = True
for k, i in enumerate(smp["index"].tolist()):
    for t in range(feats.shape[1]):
        with torch.inference_mode():
            ref = encode_frames(model, np.ascontiguousarray(smp["frames"][k, t:t + 1]), dev)[0, 0, 0].float().cpu().numpy()
        got = np.asarray(feats[i, t], dtype=np.float32)
        err = np.abs(got - ref); bound = 2.0 ** -11 * np.abs(ref) + 1e-4
        worst = max(worst, float(err.max())); ok = ok and bool(np.all(err <= bound))
rep["encoder_identity"] = {"ok": ok, "max_abs_err": worst, "clips_checked": int(len(smp["index"])), "rule": "fp16 half-ulp + 1e-4 (as run_precompute_features_smoke.sh)"}
# (b) hazard-free validation loss for latest and best through the trainer's own evaluate()
store = tfp.load_store(fd, None)
tr, va, rule = tfp.split_indices(store, 0.1, 0)
hf = np.asarray([i for i in va if not store.clips[i]["hazard"]]); hz = np.asarray([i for i in va if store.clips[i]["hazard"]])
mcfg = {k: v for k, v in meta["model"].items() if k in tfp.ModelCfg.__dataclass_fields__}
cfg = tfp.ModelCfg(**mcfg)
pred, _ = tfp.build_vendor_predictor(cfg, dev)
ac = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
rep["val"] = {"rule": rule, "n_val": int(len(va)), "n_val_hazard_free": int(len(hf)), "n_val_hazard": int(len(hz))}
for name in ("jepa-latest", "jepa-best"):
    sd = torch.load(md / f"{name}.pth.tar", map_location="cpu", weights_only=True)
    pred.load_state_dict(sd["predictor"]); pred.eval()
    with torch.no_grad():
        rep["val"][name] = {"epoch": int(sd["epoch"]), "hazard_free": tfp.evaluate(pred, store, hf, 16, dev, 2, ac) if len(hf) else None,
                            "hazard": tfp.evaluate(pred, store, hz, 16, dev, 2, ac) if len(hz) else None}
(md / "post_train_check.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
print("POST", json.dumps({k: rep[k] for k in ("arm", "epochs_run", "elapsed_s", "ms_per_iter", "encoder_identity")}),
      "hazard_free_unroll_l2 latest/best", rep["val"]["jepa-latest"]["hazard_free"]["unroll_l2"] if len(hf) else None, rep["val"]["jepa-best"]["hazard_free"]["unroll_l2"] if len(hf) else None)
EOF
  # Panel B reload equivalence through the unchanged eval loader (class, state dict, probe, forward_pred, unroll)
  $MPY check_panelb_loader.py --repo $REPO --model-dir $MD >> $TLOG 2>&1 || { say "train: loader check arm $ARM seed $s FAILED"; return 1; }
  ls -la $MD | grep -v "^total" >> $TLOG
}
for ARM in A B; do
  features_arm $ARM || exit 1
  for s in $SEEDS; do train_model $ARM $s || exit 1; done
  rm -f $FEAT/arm$ARM/features.npy; echo "[$(ts)] arm $ARM features.npy deleted after its trainings (training_reference/samples kept); $(disk)" >> $TLOG
done
# paired equivalence table VERBATIM (P2: hazard-free val loss equal within 5 % relative; reported only)
  $MPY - $MODELS "$SEEDS" >> $TLOG 2>&1 <<'EOF'
import json, sys
from pathlib import Path
models = Path(sys.argv[1]); seeds = sys.argv[2].split()
rows = []
for s in seeds:
    r = {}
    for arm in "AB":
        p = models / f"arm{arm}_seed{s}" / "post_train_check.json"
        if p.exists():
            d = json.loads(p.read_text()); v = d["val"]["jepa-latest"]["hazard_free"] or {}
            r[arm] = {"unroll_l2": v.get("unroll_l2"), "tf_l2": v.get("tf_l2"), "copy_baseline_l2": v.get("copy_baseline_l2"), "epochs_run": d["epochs_run"], "elapsed_s": d["elapsed_s"],
                      "encoder_identity_ok": d["encoder_identity"]["ok"], "best_epoch": (d["best"] or {}).get("epoch")}
    if "A" in r and "B" in r and r["A"]["unroll_l2"] is not None:
        a, b = r["A"]["unroll_l2"], r["B"]["unroll_l2"]
        rel = abs(a - b) / (0.5 * (a + b)); r["rel_diff_unroll_l2"] = rel; r["within_5pct"] = rel <= 0.05
        r["same_update_count"] = r["A"]["epochs_run"] == r["B"]["epochs_run"]
    rows.append({"seed": s, **r})
out = {"margin": 0.05, "metric": "hazard-free val unroll_l2 of jepa-latest (final checkpoint)", "pairs": rows,
       "all_within_5pct": all(r.get("within_5pct", False) for r in rows), "all_same_update_count": all(r.get("same_update_count", False) for r in rows)}
(models / "equivalence.json").write_text(json.dumps(out, indent=1) + "\n"); print("EQUIVALENCE", json.dumps(out))
EOF
echo "[$(ts)] PANELB_${PRED_TYPE}_TRAIN_DONE; $(disk)" >> $TLOG; say "train done: $(grep EQUIVALENCE $TLOG | tail -1 | cut -c1-400)"

#!/bin/bash
# run_drive_arms.sh — the whole driving cell (protocol v0.7 reversed-assignment arms A/B, v0.8 factorial) with FROZEN
# parameters, end to end on the remote box. Every stage writes a marker and is skipped when its marker is present, so
# the script can be relaunched after a crash (use --from STAGE to force a stage and everything after it).
#
#   Disk (DRIVE_LOW_DISK=1, default): the box has < 10 GB free and full-grid fp16 features cost 2.1 GB per 1k clips per arm,
#   so features are computed, used for the arm's three trainings and then features.npy is deleted (clips.json,
#   training_reference.npz, sample_frames.npz, identity_sample.npz, features.meta.json stay); the training loop is then
#   arm-outer (A seeds 0 1 2, then B) — identical seeds/initialisation per pair, so the order carries no information.
#   DRIVE_LOW_DISK=0 keeps both arms' features and trains seed-outer.
#   1 FEATURES  precompute-format shards $SHARDS_ROOT/arm{A,B}/shard_NNNN.npz + .meta.json (written by the generator agent
#               with metadrive_train_to_shards.py; marker SHARDS_DONE in drive_train_full.log; READ-ONLY here)
#               -> precompute_dinov3_features.py per arm
#               (frozen DINOv3 ViT-L/16 through the pilot loader; full 16x16 grid, fp16) -> stable 90/10 train/val
#               split by seed (sha256(f"{split_seed}:{seed}")[0] < 230; identical across arms because the arms share
#               seeds) -> training_reference.npz (pooled per-frame latents + 2-D action chunks of TRAIN clips, for the
#               nearest-training-clip audit) -> shared-clip feature identity across arms (P1 check).
#               log $LOGDIR/drive_features.log, marker FEATURES_DONE
#   2 TRAIN     train_feature_predictor.py depth 6 / width 512 / batch 16 / EPOCHS epochs, identical hyper-parameters
#               for seed in 0 1 2, arm in A B (same --seed => identical initialisation and clip order per pair).
#               Final-checkpoint-only rule: jepa-latest.pth.tar is the evaluated model; jepa-best is recorded, never used.
#               Per model: encoder-identity check (stored feature == jepa_wm_driving encoder), hazard-free val loss.
#               Paired equivalence table (P2 margin 5 % relative, REPORTED, never selects).
#               log $LOGDIR/drive_train_arms.log, marker TRAIN_ARMS_DONE
#   3 MERGE     merge_heldout.py --domain driving --protocol v0.8 per arm on $FACT_ROOT/arm{A,B} with the combined
#               validator JSON $FACT_ROOT/_validation/hazard_label_validation.json (same admitted seeds + same hash
#               discovery/confirmation split in both arms); masks copied. -> $OUT/drive_factorial_merged/arm{A,B}
#               log $LOGDIR/drive_merge.log, marker MERGE_DONE
#   4 EVAL      per model (6): eval dir = the arm's merge restricted to DISCOVERY scenes (confirmation sealed; the latent
#               cache is ~4.8 MB per cell, so the cache is discovery-only) -> run_wave_drive.sh (CFG=eval_config.yaml,
#               CKPT=jepa-latest.pth.tar, MODEL_NAME=jepa_wm_driving, --primary-hazard-level A->1 / B->3,
#               --training-reference audit, B-gate; its "cf_gate_all" therefore equals the discovery set) + cross-identity
#               gate at the OTHER level (reported) -> $OUT/drive_eval/arm{A,B}_seed{s}/. latent_cache.npz is deleted right
#               after the JSONs are written unless the B-gate PASSED (kept for MECH, deleted after the pair's stage 5/7).
#               log $LOGDIR/drive_wave_arm{A,B}_seed{s}.log: 'B-GATE PASS|FAIL' line + marker DRIVE_WAVE_DONE
#   5+7 PAIRS   one seed pair at a time: MECH arm A -> MECH arm B -> CROSS -> delete both activation dumps.
#     MECH      run_wave_mech_drive.sh only for B-gate PASS models (discovery seeds only): localization dump (hooks
#               resid_post/attn_out/mlp_out, dump-groups hazard+corridor = egg+corridor aliases, no background tokens),
#               step-0 donor patching with controls (identity donor), Jacobian sonar, geometry tournament
#               -> $OUT/drive_mech/arm{A,B}_seed{s}/; log $LOGDIR/drive_mech_arm{A,B}_seed{s}.log, marker DRIVE_MECH_DONE.
#               A FAIL arm gets the same restricted localization DUMP only (logged "dump for cross-arm geometry only",
#               no mechanism claim). Every dump is then slimmed to imagined step 0 (activations/*.npz [:, :1],
#               index.json n_steps=1; the mech-stage geometry tournament has already consumed the full dump).
#     CROSS     geometry_cross_arm.py --steps 0 (registered cross-arm geometry: appearance subspace shared, relational
#               subspace attached to the solid identity) with --stimulus $MERGED/armA (same scenes/masks in both arms),
#               --discovery-seeds = the COMMON discovery list -> $OUT/drive_geometry/seed{s}/; then activations/ of both
#               dumps and the pair's latent caches are deleted (interaction_map.json / token_interaction_maps.npz kept).
#               log $LOGDIR/drive_cross_arm.log, 'PAIR_DONE seed s' per pair, marker CROSS_ARM_DONE.
#               Dry run: minimal dumps (--max-layers 1, attn_out/mlp_out) for seed 0 only, reduced permutation counts.
#
# Usage:  run_drive_arms.sh [--dry-run] [--skip-mech] [--from features|train|merge|eval|mech|cross]
#   --dry-run  : 20-seed pilot factorial + pilot train set, 2-minute training cap, --skip-mech, everything under
#                $BASE/artifacts/_dry_drive (delete that directory when done).
# Launch (remote): ssh -n -p PORT root@HOST 'setsid nohup bash /root/cgs-pilot/run_drive_arms.sh > /root/cgs-pilot/logs/drive_arms_driver.log 2>&1 < /dev/null &'
#
# Frozen parameters (do not tune on results): predictor depth 6 / width 512 / 8 heads, batch 16, AdamW (trainer defaults:
# ref-lr 5e-4, wd 1e-7 -> 1e-6, betas 0.9/0.995, warm-up 1 epoch, cosine), rollout_steps 2, bf16 autocast, EPOCHS below,
# seeds 0 1 2, split seed 0, train/val hash byte 230 (~90/10). EPOCHS is chosen from the smoke/dry-run timing so that one
# run stays <= 20 min at ~1.6k clips per arm (see the report in the dry-run log); DRIVE_TRAIN_MAX_MIN is a SAFETY cap only:
# if it ever triggers the run is flagged (update count no longer identical across the pair) and must be repeated.
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}
CODE=${DRIVE_CODE:-$BASE/code/cgs_pilot}
REPO=${DRIVE_REPO:-$BASE/vendor/jepa-wms}
MPY=${DRIVE_MPY:-/opt/conda/bin/python}                       # model env
DPY=${DRIVE_DPY:-/opt/conda/envs/metadrive/bin/python}        # MetaDrive env (merge_heldout, as in run_wave_drive.sh)
ENC_CFG=${DRIVE_ENC_CFG:-$REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml}
ENC_CKPT=${DRIVE_ENC_CKPT:-$BASE/checkpoints/jepa_wm_droid.pth.tar}   # frozen DINOv3 encoder comes from here (pilot loader)

PRED_DEPTH=6; PRED_WIDTH=512; BATCH=16; SEEDS=${DRIVE_SEEDS:-"0 1 2"}; SPLIT_SEED=0; VAL_HASH_BYTE=230
EPOCHS=${DRIVE_EPOCHS:-30}                 # dry-run measured 245-257 ms/iter (batch 16, GPU shared with the generators): ~1.6k clips
                                           # -> 90 iters/epoch ~ 27 s/epoch incl. val -> 30 epochs ~ 13.5 min (< 20 min)
LOW_DISK=${DRIVE_LOW_DISK:-1}
TRAIN_MAX_MIN=${DRIVE_TRAIN_MAX_MIN:-20}

DRY=0; SKIP_MECH=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1;; --skip-mech) SKIP_MECH=1;; --from) FROM=$2; shift;;
    *) echo "unknown arg $1"; exit 2;;
  esac; shift
done
if [ $DRY = 1 ]; then
  OUT=${DRIVE_OUT_ROOT:-$BASE/artifacts/_dry_drive}
  LOGDIR=${DRIVE_LOGDIR:-$OUT/logs}
  SHARDS=${DRIVE_SHARDS_ROOT:-$BASE/artifacts/drive_pilot/train_shards}      # READ-ONLY pilot train shards (398 clips/arm)
  FACT_ROOT=${DRIVE_FACT_ROOT:-$BASE/artifacts/drive_pilot/factorial}        # READ-ONLY 20-seed pilot factorial
  TRAIN_MAX_MIN=2; SKIP_MECH=1; WAIT_GEN=0
  CROSS_SEEDS=${DRIVE_CROSS_SEEDS:-0}; LOC_ARGS="--max-layers 1 --hooks attn_out mlp_out --dump-hooks attn_out mlp_out"; CROSS_ARGS="--n-boot 200 --n-perm 200 --n-perm-refit 19"
else
  OUT=${DRIVE_OUT_ROOT:-$BASE/artifacts}
  LOGDIR=${DRIVE_LOGDIR:-$BASE/logs}
  SHARDS=${DRIVE_SHARDS_ROOT:-$BASE/artifacts/drive_train_shards}            # written by the generator agent (SHARDS_DONE)
  FACT_ROOT=${DRIVE_FACT_ROOT:-$BASE/artifacts/drive_factorial}
  WAIT_GEN=${DRIVE_WAIT_GEN:-1}
  CROSS_SEEDS=${DRIVE_CROSS_SEEDS:-$SEEDS}; LOC_ARGS="--dump-hooks resid_post attn_out mlp_out"; CROSS_ARGS=""   # full: geometry_cross_arm.py defaults (n_boot 2000, n_perm 2000, refit 999)
fi
TRAIN_GEN_LOG=${DRIVE_TRAIN_GEN_LOG:-$BASE/logs/drive_train_full.log}
FACT_GEN_LOG=${DRIVE_FACT_GEN_LOG:-$BASE/logs/drive_factorial_full.log}
FEAT=$OUT/drive_features; MODELS=$OUT/drive_models; MERGED=$OUT/drive_factorial_merged
EVAL=$OUT/drive_eval; MECH=$OUT/drive_mech; GEO=$OUT/drive_geometry
mkdir -p $OUT $LOGDIR
DRIVER=$LOGDIR/drive_arms.log
export JEPAWM_HOME=$BASE/vendor JEPAWM_OSSCKPT=$BASE/checkpoints JEPAWM_LOGS=$BASE/logs PYTHONPATH=$REPO:$CODE
export JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12}
FILTER='^INFO:|^🔧|^🧠|^🔮|^📉|Encoder: |Predictor: |^Loading|^Loaded'
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*" | tee -a $DRIVER; }
stage_order() { case "$1" in features) echo 1;; train) echo 2;; merge) echo 3;; eval) echo 4;; mech|cross) echo 5;; *) echo 9;; esac; }
forced() { [ -n "$FROM" ] && [ "$(stage_order $1)" -ge "$(stage_order $FROM)" ]; }
done_marker() { [ -f "$2" ] && grep -q "$1" "$2"; }   # done_marker MARKER LOG
disk() { df -h $BASE | tail -1 | awk '{print "disk used "$3" of "$2", free "$4}'; }
# generator markers: TRAIN_GEN_DONE / FACTORIAL_DONE must appear AFTER the last RELAUNCH line (a failed first attempt wrote one)
gen_done() { local log=$1 marker=$2; [ -f $log ] || return 1; local n; n=$(grep -n RELAUNCH $log | tail -1 | cut -d: -f1); tail -n +${n:-1} $log | grep -q $marker; }
wait_gen() { local log=$1 marker=$2; if [ $WAIT_GEN = 1 ]; then until gen_done $log $marker; do sleep 120; done; fi; }
cd $CODE
say "run_drive_arms start dry=$DRY skip_mech=$SKIP_MECH from='$FROM' out=$OUT shards=$SHARDS fact_root=$FACT_ROOT epochs=$EPOCHS max_min=$TRAIN_MAX_MIN seeds='$SEEDS'; $(disk)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a $DRIVER

features_arm() {   # features_arm ARM : precompute + split + training reference + samples for one arm
  local ARM=$1 FD=$FEAT/arm$1; mkdir -p $FD
  if [ -f $FD/training_reference.npz ] && [ -f $FD/features.npy ]; then echo "[$(ts)] features arm $ARM complete, reused" >> $FLOG; return 0; fi
    # 1b frozen-encoder features through the pilot loader (DROID yaml + checkpoint = the same encoder the driving eval yaml names)
    if [ ! -f $FD/features.npy ]; then rm -f $FD/features.meta.json $FD/progress.json
      echo "[$(ts)] precompute arm $ARM" >> $FLOG
      $MPY precompute_dinov3_features.py --shards "$SHARDS/arm$ARM/shard_*.npz" --repo $REPO --config $ENC_CFG --checkpoint $ENC_CKPT \
        --out $FD --pool none --dtype float16 --batch-frames 32 --split-seed $SPLIT_SEED --split-names train,val --verify-clips 2 --self-check 2 --resume 2>&1 | grep -Ev "$FILTER" >> $FLOG
      [ -f $FD/features.meta.json ] || { say "features: precompute arm $ARM FAILED (no features.meta.json)"; return 1; }
    else echo "[$(ts)] features arm $ARM present, reused" >> $FLOG; fi
    # 1c stable 90/10 split by seed + training reference + sample frames for the encoder-identity check
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

p1_check() {   # P1: shared (hazard-free) clips must have identical features in both arms; hazard clips share the context frame
  $MPY - $FEAT >> $FLOG 2>&1 <<'EOF'
import json, sys, numpy as np
from pathlib import Path
fe = Path(sys.argv[1])
ca = json.loads((fe / "armA/clips.json").read_text()); cb = json.loads((fe / "armB/clips.json").read_text())
sa = np.load(fe / "armA/identity_sample.npz"); sb = np.load(fe / "armB/identity_sample.npz")
ida = {c["clip_id"]: i for i, c in enumerate(ca)}; idb = {c["clip_id"]: i for i, c in enumerate(cb)}
def lookup(sample, fdir, cid):
    hit = np.nonzero(sample["clip_id"] == cid)[0]
    if len(hit): return np.asarray(sample["features"][hit[0]], dtype=np.float32)
    if (fdir / "features.npy").exists():
        ids = ida if fdir.name == "armA" else idb
        return np.asarray(np.load(fdir / "features.npy", mmap_mode="r")[ids[cid]], dtype=np.float32)
    return None
worst = hz = 0.0; n_sh = n_hz = 0
for sample, other, odir in ((sa, sb, fe / "armB"), (sb, sa, fe / "armA")):
    for k, cid in enumerate(sample["clip_id"].tolist()):
        o = lookup(other, odir, cid)
        if o is None: continue
        mine = np.asarray(sample["features"][k], dtype=np.float32)
        if bool(sample["hazard"][k]): hz = max(hz, float(np.abs(mine[0] - o[0]).max())); n_hz += 1
        else: worst = max(worst, float(np.abs(mine - o).max())); n_sh += 1
out = {"n_clips": [len(ca), len(cb)], "clip_ids_equal": set(ida) == set(idb), "n_shared_hazard_free_total": sum(not c["hazard"] for c in ca), "n_hazard_total": sum(c["hazard"] for c in ca),
       "shared_clip_feature_max_abs_diff": worst, "shared_clips_compared": n_sh, "hazard_clip_context_frame_max_abs_diff": hz, "hazard_clips_compared": n_hz,
       "split_equal_across_arms": all(ca[ida[k]]["split"] == cb[idb[k]]["split"] for k in ida if k in idb),
       "hazard_fraction": [float(np.mean([c["hazard"] for c in ca])), float(np.mean([c["hazard"] for c in cb]))], "contact_count": [sum(c["contact"] for c in ca), sum(c["contact"] for c in cb)]}
(fe / "arm_identity_check.json").write_text(json.dumps(out, indent=1) + "\n"); print("P1", json.dumps(out))
EOF
}

# ============================================================================ 1. FEATURES
FLOG=$LOGDIR/drive_features.log
if done_marker FEATURES_DONE $FLOG && ! forced features; then say "features: marker present, skipping"; else
  say "features: waiting for the training shards ($TRAIN_GEN_LOG SHARDS_DONE)"; wait_gen $TRAIN_GEN_LOG SHARDS_DONE
  echo "[$(ts)] FEATURES start shards=$SHARDS; $(disk)" > $FLOG
  # 1a shards are produced by the generator agent (metadrive_train_to_shards.py) and are never modified or deleted here
  for ARM in A B; do n=$(ls $SHARDS/arm$ARM/shard_*.npz 2>/dev/null | wc -l); [ "$n" -gt 0 ] || { say "features: no shards under $SHARDS/arm$ARM"; exit 1; }; echo "[$(ts)] arm $ARM: $n shards ($(du -sh $SHARDS/arm$ARM | cut -f1))" >> $FLOG; done
  [ -f $SHARDS/shards.meta.json ] && $MPY -c "import json;m=json.load(open('$SHARDS/shards.meta.json'));print('shards.meta', {a:{k:v[k] for k in ('n_clips','n_hazard','n_contact','n_shared','bytes')} for a,v in m['arms'].items()}, 'shared_identical', m.get('shared_identical_across_arms'), 'ids_equal', m.get('clip_ids_equal_across_arms'))" >> $FLOG 2>&1
  for ARM in A B; do
    if [ $LOW_DISK = 1 ] && [ $ARM = B ]; then echo "[$(ts)] arm B features deferred to the TRAIN stage (DRIVE_LOW_DISK=1)" >> $FLOG; continue; fi
    features_arm $ARM || exit 1
  done
  [ $LOW_DISK = 1 ] && echo "[$(ts)] P1 shared-clip identity check deferred to the TRAIN stage (arm B features not yet computed)" >> $FLOG || p1_check
  du -sh $FEAT/arm* >> $FLOG 2>/dev/null; echo "[$(ts)] FEATURES_DONE; $(disk)" >> $FLOG; say "features done: $(tail -1 $FLOG)"
fi

# ============================================================================ 2. TRAIN
TLOG=$LOGDIR/drive_train_arms.log
if done_marker TRAIN_ARMS_DONE $TLOG && ! forced train; then say "train: marker present, skipping"; else
  echo "[$(ts)] TRAIN start depth=$PRED_DEPTH width=$PRED_WIDTH batch=$BATCH epochs=$EPOCHS max_min=$TRAIN_MAX_MIN seeds='$SEEDS'; $(disk)" > $TLOG
  train_model() {   # train_model ARM SEED
    local ARM=$1 s=$2 MD=$MODELS/arm$1_seed$2 FD=$FEAT/arm$1
    if [ -f $MD/jepa-latest.meta.json ] && [ -f $MD/post_train_check.json ] && ! forced train; then echo "[$(ts)] model arm$ARM seed$s present, reused" >> $TLOG; return 0; fi
    rm -rf $MD; mkdir -p $MD
    echo "[$(ts)] === train arm $ARM seed $s -> $MD" >> $TLOG
    $MPY train_feature_predictor.py --repo $REPO --features $FD --pred-depth $PRED_DEPTH --pred-embed-dim $PRED_WIDTH --batch-size $BATCH \
      --epochs $EPOCHS --max-minutes $TRAIN_MAX_MIN --seed $s --arm ${ARM}_seed$s --out $MD --no-save-opt 2>&1 | grep -Ev "$FILTER" >> $TLOG
    [ -f $MD/jepa-latest.pth.tar ] || { say "train: arm $ARM seed $s FAILED"; return 1; }
    grep -q "stopping: --max-minutes" $TLOG && echo "[$(ts)] WARNING arm $ARM seed $s hit the --max-minutes cap (dry-run: expected; full run: pair no longer has identical update counts)" >> $TLOG
    # encoder identity + hazard-free val loss (latest AND best; latest is the evaluated model)
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
    ls -la $MD | grep -v "^total" >> $TLOG
  }
  if [ $LOW_DISK = 1 ]; then
    for ARM in A B; do
      [ -f $FEAT/arm$ARM/features.npy ] || features_arm $ARM || exit 1
      for s in $SEEDS; do train_model $ARM $s || exit 1; done
      if [ $ARM = B ] && [ ! -f $FEAT/arm_identity_check.json ]; then p1_check; fi
      rm -f $FEAT/arm$ARM/features.npy; echo "[$(ts)] arm $ARM features.npy deleted after its trainings (DRIVE_LOW_DISK=1; training_reference/samples kept); $(disk)" >> $TLOG
    done
  else
    for s in $SEEDS; do for ARM in A B; do train_model $ARM $s || exit 1; done; done
  fi
  # paired equivalence table (P2: hazard-free val loss equal within 5 % relative; reported only)
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
  echo "[$(ts)] TRAIN_ARMS_DONE; $(disk)" >> $TLOG; say "train done: $(grep EQUIVALENCE $TLOG | tail -1 | cut -c1-400)"
fi

# ============================================================================ 3. MERGE
MLOG=$LOGDIR/drive_merge.log
LABELS=$FACT_ROOT/_validation/hazard_label_validation.json
if done_marker MERGE_DONE $MLOG && ! forced merge; then say "merge: marker present, skipping"; else
  say "merge: waiting for the factorial generator + validator ($FACT_GEN_LOG FACTORIAL_DONE)"; wait_gen $FACT_GEN_LOG FACTORIAL_DONE
  echo "[$(ts)] MERGE start fact_root=$FACT_ROOT labels=$LABELS" > $MLOG
  [ -f $LABELS ] || { say "merge: validator JSON missing: $LABELS"; exit 1; }
  for ARM in A B; do
    MO=$MERGED/arm$ARM; rm -rf $MO; mkdir -p $MO
    $DPY merge_heldout.py --root $FACT_ROOT/arm$ARM --output $MO --protocol cgs-metadrive-pilot-v0.8 --domain driving \
      --label-validation $LABELS --split-seed $SPLIT_SEED --exclude-seeds >> $MLOG 2>&1 || { say "merge arm $ARM FAILED"; exit 1; }
    mkdir -p $MO/masks; for d in $FACT_ROOT/arm$ARM/seed_*/masks; do [ -d $d ] && cp -n $d/*.npz $MO/masks/ 2>/dev/null; done
    echo "[$(ts)] arm $ARM merged: $(grep -c . $MO/manifest.jsonl) cells ($(($(grep -c . $MO/manifest.jsonl) / 8)) octets), masks $(ls $MO/masks | wc -l); discovery [$(tr '\n' ' ' < $MO/discovery_seeds.txt)] confirmation [$(tr '\n' ' ' < $MO/confirmation_seeds.txt)]" >> $MLOG
  done
  cmp -s $MERGED/armA/discovery_seeds.txt $MERGED/armB/discovery_seeds.txt && echo "[$(ts)] discovery split identical across arms" >> $MLOG || echo "[$(ts)] WARNING discovery split differs across arms (per-arm replay/label gates); common lists written as *_seeds_common.txt (informational, gates use the per-arm lists)" >> $MLOG
  for k in discovery confirmation; do comm -12 <(sort $MERGED/armA/${k}_seeds.txt) <(sort $MERGED/armB/${k}_seeds.txt) | sort -n > $MERGED/${k}_seeds_common.txt; cp $MERGED/${k}_seeds_common.txt $MERGED/armA/; cp $MERGED/${k}_seeds_common.txt $MERGED/armB/; done
  echo "[$(ts)] common discovery [$(tr '\n' ' ' < $MERGED/discovery_seeds_common.txt)] common confirmation [$(tr '\n' ' ' < $MERGED/confirmation_seeds_common.txt)]" >> $MLOG
  echo "[$(ts)] MERGE_DONE" >> $MLOG; say "merge done: $(grep 'arm A merged' $MLOG | cut -c1-300)"
fi

# ============================================================================ 4. EVAL (per model)
for s in $SEEDS; do for ARM in A B; do
  MD=$MODELS/arm${ARM}_seed$s; EV=$EVAL/arm${ARM}_seed$s; WLOG=$LOGDIR/drive_wave_arm${ARM}_seed$s.log
  LEVEL=$([ "$ARM" = "B" ] && echo 3 || echo 1); OTHER=$([ "$ARM" = "B" ] && echo 1 || echo 3)
  if done_marker DRIVE_WAVE_DONE $WLOG && ! forced eval; then say "eval arm$ARM seed$s: marker present, skipping"; continue; fi
  say "eval arm $ARM seed $s (primary level $LEVEL, cross-identity level $OTHER)"
  rm -rf $EV; mkdir -p $EV
  # per-model copy of the arm's merge RESTRICTED TO DISCOVERY SCENES (confirmation sealed; keeps the latent cache small):
  # manifest with ABSOLUTE artifact paths (cells stay in the factorial dir), seed lists, summary, masks symlink
  $MPY - $MERGED/arm$ARM $EV <<'EOF' || { say "eval: prep FAILED"; exit 1; }
import json, os, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
disc = {int(x) for x in (src / "discovery_seeds.txt").read_text().split()}
n = 0; missing = 0; dropped = 0
with (src / "manifest.jsonl").open() as f, (dst / "manifest.jsonl").open("w") as g:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        if int(r["seed"]) not in disc: dropped += 1; continue
        p = (src / r["artifact"]).resolve(); r["artifact"] = str(p); missing += not p.exists(); n += 1
        g.write(json.dumps(r) + "\n")
for p in src.iterdir():
    if p.is_file() and p.name != "manifest.jsonl" and p.suffix in (".json", ".txt"):
        shutil.copy2(p, dst / p.name)
os.symlink(str((src / "masks").resolve()), str(dst / "masks"))
(dst / "EVAL_SCOPE.json").write_text(json.dumps({"scope": "discovery scenes only (confirmation sealed)", "n_cells": n, "n_cells_confirmation_dropped": dropped, "discovery_seeds": sorted(disc), "merge_dir": str(src)}, indent=1) + "\n")
assert missing == 0, f"{missing} cell files missing"
print(f"eval dir {dst}: {n} discovery cells ({dropped} confirmation cells sealed), masks -> {os.readlink(dst / 'masks')}")
EOF
  DRIVE_SKIP_WAIT=1 DRIVE_SKIP_MERGE=1 DRIVE_ROOT=$FACT_ROOT/arm$ARM DRIVE_OUT=$EV DRIVE_CODE=$CODE DRIVE_REPO=$REPO DRIVE_MPY=$MPY DRIVE_DPY=$DPY \
    DRIVE_CFG=$MD/eval_config.yaml DRIVE_CKPT=$MD/jepa-latest.pth.tar DRIVE_MODEL_NAME=jepa_wm_driving DRIVE_MODEL_LABEL="JEPA-WM driving arm $ARM seed $s (jepa-latest)" \
    DRIVE_TRAIN_REF=$FEAT/arm$ARM/training_reference.npz DRIVE_PRIMARY_LEVEL=$LEVEL DRIVE_LOG=$WLOG \
    bash $CODE/run_wave_drive.sh $ARM seed$s
  grep -q WAVE_DONE $WLOG || { say "eval arm $ARM seed $s: wave did not finish"; exit 1; }
  [ -f $EV/cf_gate_discovery.json ] || { say "eval arm $ARM seed $s: no discovery gate output (see $WLOG)"; exit 1; }
  # cross-identity gate: the OTHER in-lane identity scored as primary (arm A: cone; arm B: pedestrian) — reported, double-dissociation readout
  $MPY counterfactual_validity_gate.py --artifacts $EV --cache $EV/latent_cache.npz --output $EV/cf_gate_discovery_level$OTHER.json --seeds-file $EV/discovery_seeds.txt --domain driving --primary-hazard-level $OTHER --model-label "JEPA-WM driving arm $ARM seed $s (cross-identity level $OTHER)" >> $WLOG 2>&1
  $MPY counterfactual_validity_gate.py --artifacts $EV --cache $EV/latent_cache.npz --output $EV/cf_gate_all_level$OTHER.json --domain driving --primary-hazard-level $OTHER --model-label "JEPA-WM driving arm $ARM seed $s (cross-identity level $OTHER)" >> $WLOG 2>&1
  $MPY behavior_gate.py --gate $EV/cf_gate_discovery_level$OTHER.json --output $EV/behavior_gate_discovery_level$OTHER.json >> $WLOG 2>&1 || true
  # primary B-gate re-evaluated WITH the cross-identity gate (T1c' identity contrast; experiment_design.md amendment 2026-09-02 ~19:50 UTC)
  $MPY behavior_gate.py --gate $EV/cf_gate_discovery.json --cross-gate $EV/cf_gate_discovery_level$OTHER.json --output $EV/behavior_gate_discovery.json >> $WLOG 2>&1 || true
  V=$(grep -o '"verdict": "[A-Z_]*"' $EV/behavior_gate_discovery.json 2>/dev/null | head -1 | cut -d'"' -f4); V=${V:-FAIL}
  VX=$(grep -o '"verdict": "[A-Z_]*"' $EV/behavior_gate_discovery_level$OTHER.json 2>/dev/null | head -1 | cut -d'"' -f4); VX=${VX:-FAIL}
  $MPY - $EV $LEVEL $OTHER >> $WLOG 2>&1 <<'EOF'
import json, sys
from pathlib import Path
ev, lvl, oth = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def g(p):
    try: return json.loads((ev / p).read_text())
    except Exception as e: return {"error": str(e)}
def summ(gate):
    if "pooled" not in gate: return gate
    p = gate["pooled"]; bg = gate.get("planner_currency", {}).get("token_groups", {}); prim = gate.get("planner_currency", {}).get("primary_group")
    pc = (bg.get(prim) or {}).get("pooled", {}) if prim else {}
    return {"n_scenes": gate.get("n_scenes"), "interaction_nmse_median": p.get("interaction_nmse", {}).get("median"),
            "rec_h1_minus_h0": (pc.get("rec_h1_minus_h0") or {}).get("mean"), "rec_h1_minus_h0_p": (pc.get("rec_h1_minus_h0") or {}).get("sign_flip_p"),
            "flip_rate_h1_vs_h0": (pc.get("ranking") or pc.get("planner_ranking") or {}).get("flip_rate_h1_vs_h0")}
out = {"primary_level": lvl, "cross_level": oth, "gate_discovery": summ(g("cf_gate_discovery.json")), "gate_all": summ(g("cf_gate_all.json")),
       "gate_discovery_cross": summ(g(f"cf_gate_discovery_level{oth}.json")), "b_gate": {k: g("behavior_gate_discovery.json").get(k) for k in ("verdict", "n_scenes", "T1a_enough_scenes", "T1b_interaction_nmse_median", "T1c_rec_h1_minus_h0", "T1c_p", "T2a_flip_rate_h1_vs_h0")},
       "b_gate_cross": {k: g(f"behavior_gate_discovery_level{oth}.json").get(k) for k in ("verdict", "T1b_interaction_nmse_median", "T1c_rec_h1_minus_h0", "T1c_p")},
       "contamination": {k: g("contamination.json").get(k) for k in ("summary", "verdict") if k in g("contamination.json")},
       "retrieval": {k: g("retrieval_baseline.json").get(k) for k in ("summary", "verdict") if k in g("retrieval_baseline.json")}}
(ev / "eval_summary.json").write_text(json.dumps(out, indent=1, default=str) + "\n"); print("EVAL_SUMMARY", json.dumps(out, default=str)[:1500])
EOF
  echo "[$(ts)] B-GATE $([ "$V" = FAIL ] && echo FAIL || echo PASS) (verdict $V at primary level $LEVEL; cross-identity level $OTHER verdict $VX)" >> $WLOG
  if [ "$V" = FAIL ]; then rm -f $EV/latent_cache.npz; echo "[$(ts)] latent_cache.npz deleted (B-gate FAIL; gate/audit/retrieval JSONs kept); $(disk)" >> $WLOG
  else echo "[$(ts)] latent_cache.npz kept for the mechanism stage ($(du -sh $EV/latent_cache.npz | cut -f1))" >> $WLOG; fi
  echo "[$(ts)] DRIVE_WAVE_DONE" >> $WLOG
  say "eval arm $ARM seed $s done: $(grep 'B-GATE' $WLOG | tail -1)"
done; done

# ============================================================================ 5+7. PAIRS: MECH A -> MECH B -> CROSS -> delete dumps
CLOG=$LOGDIR/drive_cross_arm.log
slim_dump() {   # slim_dump DUMP : keep imagined step 0 only in activations/*.npz (+ index.json n_steps=1); idempotent
  $MPY - $1 >> $CLOG 2>&1 <<'EOF'
import json, sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1]) / "activations"; idx = json.loads((d / "index.json").read_text())
if idx.get("slimmed_to_step0"): print("slim: already step-0 only"); sys.exit(0)
n_steps = int(idx.get("n_steps") or 0); before = after = 0
for p in sorted(d.glob("*.npz")):
    before += p.stat().st_size
    with np.load(p) as z: arrs = {k: z[k] for k in z.files}
    out = {k: (v[:, :1] if (v.ndim >= 2 and n_steps > 1 and v.shape[1] == n_steps) else v) for k, v in arrs.items()}
    np.savez(p, **out); after += p.stat().st_size
for p in d.glob("*.token_index.npy"):  # memmap leftovers of the writer, if any
    p.unlink()
idx.update({"n_steps_original": n_steps, "n_steps": 1 if n_steps else n_steps, "slimmed_to_step0": True, "slim_note": "run_drive_arms.sh: activations restricted to imagined step 0 for the cross-arm stage (disk)"})
(d / "index.json").write_text(json.dumps(idx, indent=1) + "\n")
print(f"slim: {d.parent}: {before/1e6:.0f} MB -> {after/1e6:.0f} MB, n_steps {n_steps} -> 1")
EOF
}
if done_marker CROSS_ARM_DONE $CLOG && ! forced cross; then say "pairs: marker present, skipping"; else
  echo "[$(ts)] PAIRS start seeds='$CROSS_SEEDS' skip_mech=$SKIP_MECH loc_args='$LOC_ARGS' cross_args='$CROSS_ARGS'" > $CLOG
  for s in $CROSS_SEEDS; do
    if grep -q "PAIR_DONE seed $s\b" $CLOG 2>/dev/null && ! forced cross; then continue; fi
    for ARM in A B; do
      MD=$MODELS/arm${ARM}_seed$s; EV=$EVAL/arm${ARM}_seed$s; WLOG=$LOGDIR/drive_wave_arm${ARM}_seed$s.log; KLOG=$LOGDIR/drive_mech_arm${ARM}_seed$s.log
      MO=$MECH/arm${ARM}_seed$s; DUMP=$MO/localize; mkdir -p $MO
      PASS=0; grep -q "B-GATE PASS" $WLOG 2>/dev/null && PASS=1
      if [ $PASS = 1 ] && [ $SKIP_MECH = 0 ]; then
        if done_marker DRIVE_MECH_DONE $KLOG && ! forced mech; then say "mech arm$ARM seed$s: marker present, skipping"; else
          say "mech arm $ARM seed $s (B-gate PASS)"
          [ -f $EV/latent_cache.npz ] || echo "[$(ts)] note: latent cache absent for PASS model arm $ARM seed $s (mech arms recompute what they need)" >> $CLOG
          DRIVE_OUT=$EV DRIVE_MECH=$MO DRIVE_CODE=$CODE DRIVE_REPO=$REPO DRIVE_MPY=$MPY DRIVE_CFG=$MD/eval_config.yaml DRIVE_CKPT=$MD/jepa-latest.pth.tar DRIVE_MODEL_NAME=jepa_wm_driving \
            DRIVE_LOG=$KLOG DRIVE_WAVE_LOG=$WLOG bash $CODE/run_wave_mech_drive.sh $ARM seed$s
          grep -q MECH_DONE $KLOG || { say "mech arm $ARM seed $s: did not finish"; exit 1; }
          echo "[$(ts)] DRIVE_MECH_DONE" >> $KLOG; say "mech arm $ARM seed $s done; $(du -sh $MO | cut -f1)"
        fi
      else
        [ $PASS = 1 ] || { echo "[$(ts)] B-gate FAIL: mechanism arms not licensed for arm $ARM seed $s; DRIVE_MECH_DONE (behaviour-only)" > $KLOG; say "mech arm $ARM seed $s: B-gate not PASS, mechanism arms not licensed"; }
      fi
      if [ ! -f $DUMP/activations/index.json ]; then
        [ -f $EV/discovery_seeds.txt ] || { say "pairs: no eval dir for arm $ARM seed $s"; exit 1; }
        echo "[$(ts)] arm $ARM seed $s: restricted localization DUMP for cross-arm geometry only (B-gate $([ $PASS = 1 ] && echo PASS || echo FAIL); no mechanism claim)" >> $CLOG
        $MPY localize_interaction.py --repo $REPO --config $MD/eval_config.yaml --checkpoint $MD/jepa-latest.pth.tar --model-name jepa_wm_driving --artifacts $EV --output $DUMP --domain driving \
          --seeds-file $EV/discovery_seeds.txt $LOC_ARGS --dump-groups egg corridor >> $CLOG 2>&1
        [ -f $DUMP/activations/index.json ] || { say "pairs: localization dump arm $ARM seed $s FAILED (see $CLOG)"; exit 1; }
        echo "[$(ts)] arm $ARM seed $s: dump done ($(du -sh $DUMP | cut -f1), $(ls $DUMP/activations/*.npz | wc -l) sites)" >> $CLOG
      fi
      slim_dump $DUMP
    done
    # common discovery/confirmation lists (normally written by MERGE; derived here when that stage was skipped by marker)
    for k in discovery confirmation; do [ -f $MERGED/${k}_seeds_common.txt ] || comm -12 <(sort $MERGED/armA/${k}_seeds.txt) <(sort $MERGED/armB/${k}_seeds.txt) | sort -n > $MERGED/${k}_seeds_common.txt; done
    GO=$GEO/seed$s; mkdir -p $GO
    $MPY geometry_cross_arm.py --dump-a $MECH/armA_seed$s/localize --dump-b $MECH/armB_seed$s/localize --stimulus $MERGED/armA --out $GO \
      --discovery-seeds $MERGED/discovery_seeds_common.txt --steps 0 $CROSS_ARGS >> $CLOG 2>&1 || { say "pairs: geometry_cross_arm seed $s FAILED (see $CLOG)"; exit 1; }
    echo "[$(ts)] CROSS_ARM seed $s done -> $GO ($(ls $GO | tr '\n' ' '))" >> $CLOG
    for ARM in A B; do rm -rf $MECH/arm${ARM}_seed$s/localize/activations; rm -f $EVAL/arm${ARM}_seed$s/latent_cache.npz; done
    echo "[$(ts)] PAIR_DONE seed $s: activation dumps + latent caches deleted (interaction maps kept); $(disk)" >> $CLOG; say "pair seed $s done; $(disk)"
  done
  echo "[$(ts)] CROSS_ARM_DONE" >> $CLOG
fi
say "ALL_DONE; $(disk)"

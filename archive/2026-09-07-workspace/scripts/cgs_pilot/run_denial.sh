#!/bin/bash
# run_denial.sh — experiment 2 (CAFT-inspired denial retraining, denial_finetune.py) on box 2 (cgs-pilot-2), arm A seed 0.
# ENGINEERING / NECESSITY follow-up (design doc "Registered additions" (e)): not C3 evidence on its own.
#
#   0 self-test   denial_finetune.py --self-test (tiny predictor, CPU); aborts on failure.
#   1 subspaces   waits for DUMPS_READY in logs/drive_arm_diff.log (run_arm_diff.sh's copies of the seed-0 dumps), then
#                 denial_finetune.py fit-subspaces at $SITE, $WRONG_SITE and $SITE2 (corridor group, step 0):
#                 relational conceptor (geometry_cross_arm rel_ped fit) + transported arm-diff PCs -> $ROOT/subspaces/.
#   2 features    waits for PULL_SHARDS_DONE (drive_train_v2_shards) and the GPU rule, then precompute_dinov3_features.py
#                 with the flags of run_drive_arms.sh stage 1b + the stage-1c split rule (verbatim) -> $ROOT/features_armA
#                 (3.35 GB fp16; the v1 store's features.npy no longer exists and the Panel B job deletes its transient
#                 copy).  Identity check against drive_features/armA/identity_sample.npz (12 stored clips) -> features_check.json.
#   3 eval prep   waits for PULL_DONE (raw factorial cells referenced by drive_eval/armA_seed0/manifest.jsonl); eval template.
#   4 parent      gate stack on the parent WITHOUT projection and WITH the projection (inference-time ablation):
#                 conceptor k 1/4/16, armdiff k 4, random k 4.
#   5 variants    one model at a time, GPU rule re-checked before each: finetune (7 epochs = 23 % of 30, ref-lr 2e-4, same
#                 seed / split / clip order, hook at the site in forward+backward) -> eval WITHOUT projection (test-time
#                 removal) -> eval WITH projection.  Order: conceptor_k4, continued (no projection), random_k4,
#                 wrongsite_conceptor_k4 (L05.mlp_out), conceptor_k1, conceptor_k16, random_k1, random_k16, armdiff_k4,
#                 armdiff_k1, armdiff_k16, resid_conceptor_k4 (L03.resid_post).
#   6 closed loop (optional, DENIAL_CLOSED_LOOP=1) when logs/drive_closed_loop.log has CLOSED_LOOP_PILOT12_DONE: the native
#                 MetaDrive label (Label-first principle) on the first 12 discovery scenes for the parent (no / with
#                 projection) and the main variants, through closed_loop_rollout.py (unchanged) / run-with-hook.
#   7 summary     denial_finetune.py summarize -> $ROOT/denial_summary.{json,md}; marker DENIAL_DONE in $LOG.
# Gate stack per evaluation = latent_cache.py (driving token mask) -> counterfactual_validity_gate.py at the primary level
# (1) and the cross-identity level (3) -> behavior_gate.py with --cross-gate (T1c'), exactly the EVAL stage of
# run_drive_arms.sh minus the model-independent contamination / retrieval audits (already passed by the parent).
# GPU rule (user): run only after G2_DONE in logs/box2_orchestrator.log or while nvidia-smi shows < 8 GB used; one model at
# a time; OMP 4.  Writes only under artifacts/drive_denial/ and logs/; never deletes anything outside it.
# Usage: run_denial.sh [--skip-self-test] [--variants "name ..."]   (env: DENIAL_EPOCHS 7, DENIAL_REF_LR 2e-4, DENIAL_CLOSED_LOOP 1, DENIAL_FORCE_GPU 0)
# Launch: ssh -n -p 45460 root@70.27.250.55 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_denial.sh > /root/cgs-pilot/logs/drive_denial_driver.out 2>&1 < /dev/null &'
set -u
B=${DRIVE_BASE:-/root/cgs-pilot}; C=${DRIVE_CODE:-$B/code/cgs_pilot}; REPO=$B/vendor/jepa-wms
MPY=${DRIVE_MPY:-/opt/conda/bin/python}; DPY=${DRIVE_DPY:-/opt/conda/envs/metadrive/bin/python}
ROOT=${DENIAL_ROOT:-$B/artifacts/drive_denial}; ADIFF=$B/artifacts/drive_arm_diff; MODELS=$B/artifacts/drive_models; EVAL=$B/artifacts/drive_eval
MERGED=$B/artifacts/drive_factorial_merged; FEAT_V1=$B/artifacts/drive_features/armA; SHARDS=$B/artifacts/drive_train_v2_shards; FEAT=$ROOT/features_armA
LOG=${DENIAL_LOG:-$B/logs/drive_denial.log}; ORCH=$B/logs/box2_orchestrator.log; PL=/root/pull_box1.log; CLLOG=$B/logs/drive_closed_loop.log
ENC_CFG=$REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
ENC_CKPT=$B/checkpoints/jepa_wm_droid.pth.tar
PARENT=$MODELS/armA_seed0; SITE=${DENIAL_SITE:-L03.mlp_out}; WRONG_SITE=${DENIAL_WRONG_SITE:-L05.mlp_out}; SITE2=${DENIAL_SITE2:-L03.resid_post}; GROUP=corridor; STEP=0
EPOCHS=${DENIAL_EPOCHS:-7}; REF_LR=${DENIAL_REF_LR:-2e-4}; WARMUP=${DENIAL_WARMUP:-1.0}; FINAL_LR=${DENIAL_FINAL_LR:-1e-5}; SEED=0; BATCH=16
CLOSED_LOOP=${DENIAL_CLOSED_LOOP:-1}; CL_SCENES=${DENIAL_CL_SCENES:-12}; FORCE_GPU=${DENIAL_FORCE_GPU:-0}
VARIANTS=${DENIAL_VARIANTS:-"conceptor_k4 continued random_k4 wrongsite_conceptor_k4 conceptor_k1 conceptor_k16 random_k1 random_k16 armdiff_k4 armdiff_k1 armdiff_k16 resid_conceptor_k4"}
export JEPAWM_HOME=$B/vendor JEPAWM_OSSCKPT=$B/checkpoints JEPAWM_LOGS=$B/logs PYTHONPATH=$REPO:$C JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
FILTER='^INFO:|^🔧|^🧠|^🔮|^📉|Encoder: |Predictor: |^Loading|^Loaded'
SKIP_ST=0; while [ $# -gt 0 ]; do case "$1" in --skip-self-test) SKIP_ST=1;; --variants) VARIANTS=$2; shift;; *) echo "unknown arg $1"; exit 2;; esac; shift; done
ts(){ date -u +%FT%TZ; }; say(){ echo "[$(ts)] $*" | tee -a $LOG; }
disk(){ df -h $B | tail -1 | awk '{print "disk used "$3" of "$2", free "$4}'; }
gpu_ok(){ [ "$FORCE_GPU" = 1 ] && return 0; grep -q G2_DONE $ORCH 2>/dev/null && return 0; local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ -n "$u" ] && [ "$u" -lt 8000 ]; }
wait_gpu(){ local n=0; until gpu_ok; do n=$((n + 1)); [ $((n % 30)) = 0 ] && say "waiting for the GPU rule (G2_DONE or < 8 GB used; now $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1))"; sleep 60; done; }
mkdir -p $ROOT/subspaces $ROOT/parent/eval $ROOT/variants $B/logs; cd $C
say "DENIAL start root=$ROOT site=$SITE wrong=$WRONG_SITE site2=$SITE2 epochs=$EPOCHS ref_lr=$REF_LR variants='$VARIANTS' closed_loop=$CLOSED_LOOP; $(disk)"

# ---- 0. self-test (CPU)
if [ $SKIP_ST = 0 ] && [ ! -f $ROOT/self_test/self_test_verdict.json ]; then
  CUDA_VISIBLE_DEVICES="" $MPY denial_finetune.py --self-test --out $ROOT/self_test --seed 0 > $ROOT/self_test.log 2>&1
  grep -q SELF_TEST_PASSED $ROOT/self_test.log || { say "self-test FAILED (see $ROOT/self_test.log); aborting"; exit 1; }
  say "self-test passed: $(grep -o '"passed": [a-z]*' $ROOT/self_test/self_test_verdict.json | head -1)"
fi

# ---- 1. subspaces from the seed-0 dumps
SUB=$ROOT/subspaces/subspace_${SITE}__s${STEP}__${GROUP}.npz; SUBW=$ROOT/subspaces/subspace_${WRONG_SITE}__s${STEP}__${GROUP}.npz; SUB2=$ROOT/subspaces/subspace_${SITE2}__s${STEP}__${GROUP}.npz
if [ ! -f $SUB ] || [ ! -f $SUBW ] || [ ! -f $SUB2 ]; then
  say "waiting for DUMPS_READY in $B/logs/drive_arm_diff.log"
  until grep -q DUMPS_READY $B/logs/drive_arm_diff.log 2>/dev/null; do sleep 60; done
  [ -f $MERGED/discovery_seeds_common.txt ] || comm -12 <(sort $MERGED/armA/discovery_seeds.txt) <(sort $MERGED/armB/discovery_seeds.txt) | sort -n > $MERGED/discovery_seeds_common.txt
  $MPY denial_finetune.py --seed $SEED fit-subspaces --dump-a $ADIFF/dumps/dump_armA --dump-b $ADIFF/dumps/dump_armB --stimulus $MERGED/armA --discovery-seeds $MERGED/discovery_seeds_common.txt \
    --sites $SITE $WRONG_SITE $SITE2 --group $GROUP --step $STEP --out-dir $ROOT/subspaces --seed $SEED >> $LOG 2>&1 || { say "fit-subspaces FAILED"; exit 1; }
  say "subspaces: $(ls $ROOT/subspaces/*.npz | tr '\n' ' ')"
fi

# ---- 2. training features for arm A (verbatim flags of run_drive_arms.sh stage 1b/1c) + identity check vs the v1 store
if [ ! -f $FEAT/features.npy ] || [ ! -f $FEAT/split_rule.json ]; then
  say "waiting for PULL_SHARDS_DONE ($SHARDS)"
  until grep -q PULL_SHARDS_DONE $PL 2>/dev/null; do sleep 60; done
  n=$(ls $SHARDS/armA/shard_*.npz 2>/dev/null | wc -l); [ "$n" -gt 0 ] || { say "no shards under $SHARDS/armA"; exit 1; }
  wait_gpu; mkdir -p $FEAT
  say "features: precompute arm A from $n shards -> $FEAT; $(disk)"
  $MPY precompute_dinov3_features.py --shards "$SHARDS/armA/shard_*.npz" --repo $REPO --config $ENC_CFG --checkpoint $ENC_CKPT \
    --out $FEAT --pool none --dtype float16 --batch-frames 32 --split-seed 0 --split-names train,val --verify-clips 2 --self-check 2 --resume 2>&1 | grep -Ev "$FILTER" >> $LOG
  [ -f $FEAT/features.meta.json ] || { say "features: precompute FAILED"; exit 1; }
  $MPY - $FEAT $SHARDS/armA 0 230 >> $LOG 2>&1 <<'EOF' || { say "features: split FAILED"; exit 1; }
import hashlib, json, sys, numpy as np
from pathlib import Path
fd, sd, split_seed, thr = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
clips = json.loads((fd / "clips.json").read_text())
orig = fd / "clips.hash128.json"
if not orig.exists():
    orig.write_text(json.dumps(clips, indent=1) + "\n")
for c in clips:
    h = hashlib.sha256(f"{split_seed}:{c['seed']}".encode()).digest()[0]
    c["split"] = "train" if h < thr else "val"
rule = {"split_rule": f"sha256(f'{split_seed}:{{seed}}').digest()[0] < {thr} -> train else val (by clip seed; identical across arms)",
        "split_seed": split_seed, "threshold_byte": thr, "n_train": sum(c["split"] == "train" for c in clips), "n_val": sum(c["split"] == "val" for c in clips),
        "n_val_hazard_free": sum(c["split"] == "val" and not c["hazard"] for c in clips), "n_train_hazard": sum(c["split"] == "train" and c["hazard"] for c in clips)}
(fd / "clips.json").write_text(json.dumps(clips, indent=1) + "\n")
(fd / "split_rule.json").write_text(json.dumps(rule, indent=1) + "\n")
print(json.dumps(rule))
EOF
fi
if [ ! -f $ROOT/features_check.json ]; then
  $MPY - $FEAT $FEAT_V1 $ROOT/features_check.json >> $LOG 2>&1 <<'EOF'
import json, sys, numpy as np
from pathlib import Path
new, old, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cn = json.loads((new / "clips.json").read_text()); co = json.loads((old / "clips.json").read_text())
idn = {c["clip_id"]: i for i, c in enumerate(cn)}; ido = {c["clip_id"]: c for c in co}
rep = {"n_new": len(cn), "n_old": len(co), "clip_ids_equal": set(idn) == set(ido), "hazard_flags_equal": all(bool(cn[idn[k]]["hazard"]) == bool(ido[k]["hazard"]) for k in idn if k in ido),
       "split_equal": all(cn[idn[k]].get("split") == ido[k].get("split") for k in idn if k in ido), "order_equal": [c["clip_id"] for c in cn] == [c["clip_id"] for c in co]}
feats = np.load(new / "features.npy", mmap_mode="r")
acts_new = np.load(new / "actions.npy");
try:
    acts_old = np.load(old / "actions.npy"); rep["actions_max_abs_diff"] = float(np.abs(acts_new - acts_old).max()) if acts_new.shape == acts_old.shape else None
except Exception as e: rep["actions_max_abs_diff"] = str(e)
s = np.load(old / "identity_sample.npz")
worst, n = 0.0, 0
for k, cid in enumerate(s["clip_id"].tolist()):
    if cid in idn:
        a = np.asarray(feats[idn[cid]], dtype=np.float32); b = np.asarray(s["features"][k], dtype=np.float32)
        worst = max(worst, float(np.abs(a - b).max())); n += 1
rep.update({"identity_sample_clips_compared": n, "identity_sample_max_abs_diff": worst, "features_identical_to_v1": bool(n > 0 and worst == 0.0), "features_within_fp16_half_ulp": bool(n > 0 and worst <= 2.0 ** -11 * 8 + 1e-4),
            "features_shape": list(feats.shape), "note": "the regenerated v2 clips must reproduce the v1 clips bit-exactly for the fine-tuning to run on the parent's own training features; otherwise the deviation is reported"})
outp.write_text(json.dumps(rep, indent=1) + "\n"); print("FEATURES_CHECK", json.dumps(rep))
EOF
  say "features check: $(grep FEATURES_CHECK $LOG | tail -1 | cut -c1-300)"
fi

# ---- 3. eval template (discovery scenes of arm A; confirmation sealed) -- needs the raw factorial cells
if [ ! -f $ROOT/eval_template/manifest.jsonl ]; then
  say "waiting for PULL_DONE (raw factorial cells referenced by $EVAL/armA_seed0/manifest.jsonl)"
  until grep -q PULL_DONE $PL 2>/dev/null; do sleep 60; done
  mkdir -p $ROOT/eval_template
  cp $EVAL/armA_seed0/manifest.jsonl $ROOT/eval_template/; for f in $EVAL/armA_seed0/*.txt $EVAL/armA_seed0/summary.json $EVAL/armA_seed0/EVAL_SCOPE.json; do [ -f $f ] && cp $f $ROOT/eval_template/; done
  $MPY - $ROOT/eval_template >> $LOG 2>&1 <<'EOF' || { say "eval template: cells missing"; exit 1; }
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l.strip()]
missing = [r["artifact"] for r in rows if not Path(r["artifact"]).exists()]
assert not missing, f"{len(missing)} cells missing, e.g. {missing[:2]}"
print("eval template:", len(rows), "cells, all present")
EOF
  say "eval template ready ($(grep -c . $ROOT/eval_template/manifest.jsonl) cells)"
fi
MASKS=$MERGED/armA/masks

# ---- helpers
prep_ev(){ local EV=$1; mkdir -p $EV; cp $ROOT/eval_template/manifest.jsonl $EV/; for f in $ROOT/eval_template/*.txt $ROOT/eval_template/summary.json $ROOT/eval_template/EVAL_SCOPE.json; do [ -f $f ] && cp $f $EV/; done; [ -e $EV/masks ] || ln -s $MASKS $EV/masks; }
gates(){  # gates EV LABEL
  local EV=$1 LBL=$2
  $MPY counterfactual_validity_gate.py --artifacts $EV --cache $EV/latent_cache.npz --output $EV/cf_gate_discovery.json --seeds-file $EV/discovery_seeds.txt --domain driving --primary-hazard-level 1 --model-label "$LBL" >> $LOG 2>&1
  $MPY counterfactual_validity_gate.py --artifacts $EV --cache $EV/latent_cache.npz --output $EV/cf_gate_discovery_level3.json --seeds-file $EV/discovery_seeds.txt --domain driving --primary-hazard-level 3 --model-label "$LBL (cross-identity level 3)" >> $LOG 2>&1
  $MPY behavior_gate.py --gate $EV/cf_gate_discovery_level3.json --output $EV/behavior_gate_discovery_level3.json >> $LOG 2>&1 || true
  $MPY behavior_gate.py --gate $EV/cf_gate_discovery.json --cross-gate $EV/cf_gate_discovery_level3.json --output $EV/behavior_gate_discovery.json >> $LOG 2>&1 || true
  rm -f $EV/latent_cache.npz
  say "  gates $LBL: $(grep -o '"verdict": "[A-Z_]*"' $EV/behavior_gate_discovery.json | head -1) captured $(grep -o '"model_captured_interaction_fraction": [0-9.]*' $EV/behavior_gate_discovery.json | head -1 | cut -d' ' -f2) ghost $(grep -o '"model_captured_interaction_fraction": [0-9.]*' $EV/behavior_gate_discovery_level3.json | head -1 | cut -d' ' -f2)"
}
eval_nohook(){  # eval_nohook EV CFG CKPT LABEL
  local EV=$1 CFG=$2 CKPT=$3 LBL=$4
  [ -f $EV/behavior_gate_discovery.json ] && { say "  eval present: $EV"; return 0; }
  prep_ev $EV; wait_gpu
  $MPY -c "import sys, latent_cache as lc, token_groups as tg; lc.load_token_mask = tg.cache_token_mask_driving; sys.argv = ['latent_cache.py', '--artifacts', '$EV', '--cache', '$EV/latent_cache.npz', '--repo', '$REPO', '--config', '$CFG', '--checkpoint', '$CKPT', '--model-name', 'jepa_wm_driving']; lc.main()" >> $LOG 2>&1 || { say "  latent cache FAILED ($EV)"; return 1; }
  gates $EV "$LBL"
}
eval_hook(){  # eval_hook EV CFG CKPT LABEL SITE SUBSPACE KEY K RSEED
  local EV=$1 CFG=$2 CKPT=$3 LBL=$4 S=$5 SB=$6 KEY=$7 K=$8 RS=${9:-0}
  [ -f $EV/behavior_gate_discovery.json ] && { say "  eval present: $EV"; return 0; }
  prep_ev $EV; wait_gpu
  $MPY denial_finetune.py run-with-hook --script latent_cache --site $S --subspace $SB --key $KEY --k $K --random-seed $RS -- \
    --artifacts $EV --cache $EV/latent_cache.npz --repo $REPO --config $CFG --checkpoint $CKPT --model-name jepa_wm_driving >> $LOG 2>&1 || { say "  hooked latent cache FAILED ($EV)"; return 1; }
  gates $EV "$LBL"
}
# variant name -> site / subspace / key / k
spec(){ case "$1" in
  conceptor_k*)       echo "$SITE $SUB conceptor ${1#conceptor_k} 0";;
  armdiff_k*)         echo "$SITE $SUB armdiff ${1#armdiff_k} 0";;
  random_k*)          echo "$SITE $SUB random ${1#random_k} 1";;
  wrongsite_conceptor_k*) echo "$WRONG_SITE $SUBW conceptor ${1#wrongsite_conceptor_k} 0";;
  resid_conceptor_k*) echo "$SITE2 $SUB2 conceptor ${1#resid_conceptor_k} 0";;
  continued)          echo "$SITE $SUB none 0 0";;
  *) echo "";; esac; }

# ---- 4. parent: no projection, and inference-time ablations
say "parent evaluations"
eval_nohook $ROOT/parent/eval/nohook $PARENT/eval_config.yaml $PARENT/jepa-latest.pth.tar "armA seed0 parent (no projection)"
for ab in conceptor_k4 conceptor_k1 conceptor_k16 armdiff_k4 random_k4; do
  read -r S SB KEY K RS <<< "$(spec $ab)"
  eval_hook $ROOT/parent/eval/hook_$ab $PARENT/eval_config.yaml $PARENT/jepa-latest.pth.tar "armA seed0 parent WITH projection $ab" $S $SB $KEY $K $RS
done

# ---- 5. variants
for v in $VARIANTS; do
  read -r S SB KEY K RS <<< "$(spec $v)"; [ -n "$S" ] || { say "unknown variant $v"; continue; }
  VD=$ROOT/variants/$v
  if [ ! -f $VD/jepa-latest.pth.tar ] || [ ! -f $VD/denial_meta.json ]; then
    wait_gpu; rm -rf $VD; mkdir -p $VD
    say "variant $v: finetune site=$S key=$KEY k=$K epochs=$EPOCHS ref_lr=$REF_LR"
    $MPY denial_finetune.py --seed $SEED finetune --model-dir $PARENT --repo $REPO --features $FEAT --out $VD --site $S --subspace $SB --key $KEY --k $K --random-seed $RS \
      --epochs $EPOCHS --ref-lr $REF_LR --warmup-epochs $WARMUP --final-lr $FINAL_LR --batch-size $BATCH --label "A_seed0_denial_$v" 2>&1 | grep -Ev "$FILTER" >> $LOG
    [ -f $VD/jepa-latest.pth.tar ] && [ -f $VD/denial_meta.json ] || { say "variant $v: finetune FAILED"; continue; }
    say "variant $v: trained ($(grep -o '"rel_diff": [0-9.e-]*' $VD/denial_meta.json | tail -1 | cut -d' ' -f2) hazard-free rel diff, no projection)"
  fi
  eval_nohook $VD/eval/nohook $VD/eval_config.yaml $VD/jepa-latest.pth.tar "denial $v (test-time removal)"
  if [ "$KEY" != none ]; then eval_hook $VD/eval/hook $VD/eval_config.yaml $VD/jepa-latest.pth.tar "denial $v (WITH projection)" $S $SB $KEY $K $RS; fi
  $MPY denial_finetune.py summarize --root $ROOT > /dev/null 2>&1 || true
done

# ---- 6. closed-loop native label (optional; Label-first principle)
if [ "$CLOSED_LOOP" = 1 ] && grep -q CLOSED_LOOP_PILOT12_DONE $CLLOG 2>/dev/null; then
  SEEDS=$MERGED/discovery_seeds_common.txt
  cl_run(){  # cl_run OUT MODEL_DIR [hook args: SITE SUBSPACE KEY K RSEED]
    local O=$1 MD=$2; shift 2
    [ -f $O/summary.json ] && return 0
    wait_gpu; mkdir -p $O
    if [ $# -gt 0 ]; then
      $MPY denial_finetune.py run-with-hook --script closed_loop_rollout --site $1 --subspace $2 --key $3 --k $4 --random-seed ${5:-0} -- run --arm A --model-dir $MD --repo $REPO --stimulus $MERGED/armA \
        --seeds-file $SEEDS --max-scenes $CL_SCENES --episode-seeds 2 --n-replans 8 --out $O --bridge-python $DPY >> $LOG 2>&1
    else
      $MPY closed_loop_rollout.py run --arm A --model-dir $MD --repo $REPO --stimulus $MERGED/armA --seeds-file $SEEDS --max-scenes $CL_SCENES --episode-seeds 2 --n-replans 8 --out $O --bridge-python $DPY >> $LOG 2>&1
    fi
    say "  closed loop $O: $(grep -o '"verdict": "[A-Z_]*"' $O/summary.json 2>/dev/null | head -1)"
  }
  say "closed-loop native label on $CL_SCENES discovery scenes"
  cl_run $ROOT/parent/closed_loop/nohook $PARENT
  read -r S SB KEY K RS <<< "$(spec conceptor_k4)"; cl_run $ROOT/parent/closed_loop/hook_conceptor_k4 $PARENT $S $SB $KEY $K $RS
  for v in conceptor_k4 continued random_k4; do
    VD=$ROOT/variants/$v; [ -f $VD/jepa-latest.pth.tar ] || continue
    read -r S SB KEY K RS <<< "$(spec $v)"
    cl_run $VD/closed_loop/nohook $VD
    [ "$KEY" != none ] && cl_run $VD/closed_loop/hook $VD $S $SB $KEY $K $RS
  done
else
  say "closed-loop stage skipped (DENIAL_CLOSED_LOOP=$CLOSED_LOOP, pilot marker $(grep -c CLOSED_LOOP_PILOT12_DONE $CLLOG 2>/dev/null))"
fi

# ---- 7. summary
$MPY denial_finetune.py summarize --root $ROOT >> $LOG 2>&1 && say "summary -> $ROOT/denial_summary.md"
echo "[$(ts)] DENIAL_DONE" >> $LOG
say "DENIAL_DONE; $(disk)"

#!/bin/bash
# run_panelb_eval.sh — Panel B (action-as-token predictors, drive_models_token) gate stage, mirroring run_drive_arms.sh
# stage 4 and run_v08_eval.sh: (1) v0.7 factorial (drive_factorial_merged, 54 discovery scenes) -> drive_eval_token/,
# (2) cross-truth via a path-parameterised copy of the frozen run_crosstruth.sh -> drive_eval_crosstruth_token/,
# (3) v0.8 static set (drive_v08_merged/static) -> drive_v08_eval_token/. Discovery scenes only; caches deleted per model.
# Waits for PANELB_TOKEN_DONE (logs/panelb_gated.log) unless PB_SKIP_WAIT=1. Marker PANELB_EVAL_DONE in logs/panelb_eval.log.
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}; CODE=$BASE/code/cgs_pilot; REPO=$BASE/vendor/jepa-wms
MPY=/opt/conda/bin/python; DPY=/opt/conda/envs/metadrive/bin/python
MODELS=${PB_MODELS:-$BASE/artifacts/drive_models_token}; FEAT=${PB_FEAT:-$BASE/artifacts/drive_features_v2}
LOGDIR=${PB_LOGDIR:-$BASE/logs/panelb}; LOG=$BASE/logs/panelb_eval.log; GLOG=$BASE/logs/panelb_gated.log
SEEDS=${PB_SEEDS:-"0 1 2"}; TAG=${PB_TAG:-token}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
ts(){ date -u +%FT%TZ; }; say(){ echo "[$(ts)] $*" >> $LOG; }; disk(){ df -h $BASE | tail -1 | awk '{print "disk free " $4}'; }
freegb(){ df -BG $BASE | tail -1 | awk '{gsub("G","",$4); print $4}'; }
mkdir -p $LOGDIR
if [ "${PB_SKIP_WAIT:-0}" != "1" ]; then say "waiting for PANELB_TOKEN_DONE in $GLOG"; until grep -q PANELB_TOKEN_DONE $GLOG 2>/dev/null; do sleep 120; done; fi
say "Panel B eval start tag=$TAG models=$MODELS seeds='$SEEDS'; $(disk)"
# training reference for the audit: v2 features if present, else the v1 reference (same clips by construction; noted in the log)
train_ref() { local ARM=$1; if [ -f $FEAT/arm$ARM/training_reference.npz ]; then echo $FEAT/arm$ARM/training_reference.npz; else echo $BASE/artifacts/drive_features/arm$ARM/training_reference.npz; fi; }

eval_model() { # eval_model SET SRC_MERGED_DIR EVALROOT ARM SEED
  local SET=$1 SRC=$2 EVROOT=$3 ARM=$4 s=$5; local MD=$MODELS/arm${ARM}_seed$s EV=$EVROOT/arm${ARM}_seed$s WLOG=$LOGDIR/drive_wave_arm${ARM}_seed$s.log
  [ "$SET" = v08 ] && WLOG=$LOGDIR/drive_v08_wave_static_arm${ARM}_seed$s.log
  [ -f $MD/jepa-latest.pth.tar ] || { say "eval $SET arm$ARM seed$s: no model at $MD"; return 0; }
  [ -f $SRC/manifest.jsonl ] || { say "eval $SET arm$ARM seed$s: no merged set at $SRC"; return 0; }
  grep -q DRIVE_WAVE_DONE $WLOG 2>/dev/null && { say "eval $SET arm$ARM seed$s: marker present, skipping"; return 0; }
  until [ "$(freegb)" -ge 4 ]; do say "eval $SET arm$ARM seed$s: waiting for 4 GB free ($(freegb) GB)"; sleep 120; done
  local LEVEL=$([ "$ARM" = "B" ] && echo 3 || echo 1) OTHER=$([ "$ARM" = "B" ] && echo 1 || echo 3)
  say "eval $SET arm $ARM seed $s (primary level $LEVEL, cross-identity level $OTHER); $(disk)"
  rm -rf $EV; mkdir -p $EV
  $MPY - $SRC $EV <<'EOF' >> $LOG 2>&1 || { say "eval prep FAILED"; return 1; }
import json, os, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
disc = {int(x) for x in (src / "discovery_seeds.txt").read_text().split()}
n = missing = dropped = 0
with (src / "manifest.jsonl").open() as f, (dst / "manifest.jsonl").open("w") as g:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        if int(r["seed"]) not in disc: dropped += 1; continue
        p = Path(r["artifact"]); p = p if p.is_absolute() else (src / p).resolve(); r["artifact"] = str(p); missing += not p.exists(); n += 1; g.write(json.dumps(r) + "\n")
for p in src.iterdir():
    if p.is_file() and p.name != "manifest.jsonl" and p.suffix in (".json", ".txt"): shutil.copy2(p, dst / p.name)
os.symlink(str((src / "masks").resolve()), str(dst / "masks"))
(dst / "EVAL_SCOPE.json").write_text(json.dumps({"scope": "discovery scenes only (confirmation sealed); Panel B token predictor", "n_cells": n, "n_cells_confirmation_dropped": dropped, "discovery_seeds": sorted(disc), "merge_dir": str(src)}, indent=1) + "\n")
assert missing == 0, f"{missing} cell files missing"
print(f"eval dir {dst}: {n} discovery cells ({dropped} confirmation cells sealed)")
EOF
  DRIVE_SKIP_WAIT=1 DRIVE_SKIP_MERGE=1 DRIVE_ROOT=$BASE/artifacts/drive_factorial/arm$ARM DRIVE_OUT=$EV DRIVE_CODE=$CODE DRIVE_REPO=$REPO DRIVE_MPY=$MPY DRIVE_DPY=$DPY \
    DRIVE_CFG=$MD/eval_config.yaml DRIVE_CKPT=$MD/jepa-latest.pth.tar DRIVE_MODEL_NAME=jepa_wm_driving DRIVE_MODEL_LABEL="JEPA $TAG-predictor driving arm $ARM seed $s (jepa-latest) $SET" \
    DRIVE_TRAIN_REF=$(train_ref $ARM) DRIVE_PRIMARY_LEVEL=$LEVEL DRIVE_LOG=$WLOG bash $CODE/run_wave_drive.sh $ARM ${TAG}_$SET
  grep -q WAVE_DONE $WLOG || { say "eval $SET arm$ARM seed$s: wave did not finish"; return 1; }
  [ -f $EV/cf_gate_discovery.json ] || { say "eval $SET arm$ARM seed$s: no discovery gate output"; rm -f $EV/latent_cache.npz; return 1; }
  $MPY $CODE/counterfactual_validity_gate.py --artifacts $EV --cache $EV/latent_cache.npz --output $EV/cf_gate_discovery_level$OTHER.json --seeds-file $EV/discovery_seeds.txt --domain driving --primary-hazard-level $OTHER --model-label "JEPA $TAG-predictor arm $ARM seed $s (cross-identity level $OTHER) $SET" >> $WLOG 2>&1
  $MPY $CODE/behavior_gate.py --gate $EV/cf_gate_discovery_level$OTHER.json --output $EV/behavior_gate_discovery_level$OTHER.json >> $WLOG 2>&1 || true
  $MPY $CODE/behavior_gate.py --gate $EV/cf_gate_discovery.json --cross-gate $EV/cf_gate_discovery_level$OTHER.json --output $EV/behavior_gate_discovery.json >> $WLOG 2>&1 || true
  [ -f $CODE/training_copy_baseline.py ] && $MPY $CODE/training_copy_baseline.py --artifacts $EV --cache $EV/latent_cache.npz --training-reference $(train_ref $ARM) --output $EV/training_copy_baseline.json --seeds-file $EV/discovery_seeds.txt --model-label "$TAG arm $ARM seed $s $SET" >> $WLOG 2>&1 || say "training-copy baseline failed ($SET arm$ARM seed$s)"
  local V; V=$(grep -o '"verdict": "[A-Z_]*"' $EV/behavior_gate_discovery.json 2>/dev/null | head -1 | cut -d'"' -f4); V=${V:-FAIL}
  local VX; VX=$(grep -o '"verdict": "[A-Z_]*"' $EV/behavior_gate_discovery_level$OTHER.json 2>/dev/null | head -1 | cut -d'"' -f4); VX=${VX:-FAIL}
  echo "[$(ts)] B-GATE $([ "$V" = FAIL ] && echo FAIL || echo PASS) (verdict $V at primary level $LEVEL; cross-identity level $OTHER verdict $VX)" >> $WLOG
  rm -f $EV/latent_cache.npz
  echo "[$(ts)] DRIVE_WAVE_DONE" >> $WLOG
  say "eval $SET arm$ARM seed$s done: $(grep 'B-GATE' $WLOG | tail -1); $(disk)"
}

# 1. v0.7 factorial (54 discovery scenes) -> drive_eval_token
for s in $SEEDS; do for ARM in A B; do eval_model v07 $BASE/artifacts/drive_factorial_merged/arm$ARM $BASE/artifacts/drive_eval_$TAG $ARM $s; done; done
# 2. cross-truth (path-parameterised copy of the frozen script; markers read from $LOGDIR)
CT=$BASE/run_crosstruth_$TAG.sh
sed -e "s#^MODELS=.*#MODELS=$MODELS#" -e "s#^EVAL=.*#EVAL=$BASE/artifacts/drive_eval_$TAG#" $CODE/run_crosstruth.sh > $CT && chmod +x $CT
grep -q "drive_models_$TAG" $CT && say "cross-truth copy written: $CT" || say "cross-truth copy: substitution failed"
DRIVE_LOGDIR=$LOGDIR CROSSTRUTH_OUT=$BASE/artifacts/drive_eval_crosstruth_$TAG CROSSTRUTH_SEEDS="$SEEDS" CROSSTRUTH_MIN_FREE_GB=3.5 bash $CT
say "cross-truth done: $(tail -1 $LOGDIR/drive_crosstruth.log 2>/dev/null | cut -c1-200)"
# 3. v0.8 static set -> drive_v08_eval_token
for s in $SEEDS; do for ARM in A B; do eval_model v08 $BASE/artifacts/drive_v08_merged/static/arm$ARM $BASE/artifacts/drive_v08_eval_$TAG $ARM $s; done; done
say "PANELB_EVAL_DONE; $(disk)"

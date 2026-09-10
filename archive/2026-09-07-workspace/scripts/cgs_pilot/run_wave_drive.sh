#!/bin/bash
# Driving (protocol v0.8, MetaDrive) wave pipeline, arm-parametrised copy of run_wave.sh:
#   validator wait -> merge v0.8 (--domain driving, 8-cell scenes) -> masks -> latent cache (hazard|corridor token mask)
#   -> counterfactual gate (all + discovery, --domain driving) -> training-clip audit -> retrieval control
#   -> B-GATE STOP (behavior_gate.py on the discovery gate; exit 3 = mechanism arms NOT licensed)
# Mechanism arms run separately in run_wave_mech_drive.sh, which re-checks the B-gate before anything else.
# Usage: run_wave_drive.sh <ARM: A|B> [WAVE tag]; every path/model knob can be overridden by environment variables.
set -u
ARM=${1:-A}; WAVE=${2:-d1}
ROOT=${DRIVE_ROOT:-/root/cgs-pilot/artifacts/drive_factorial}                  # seed_<s>/ dirs from metadrive_hazard_pilot.py
OUT=${DRIVE_OUT:-/root/cgs-pilot/artifacts/drive_merged_${WAVE}_arm$ARM}
CODE=${DRIVE_CODE:-/root/cgs-pilot/code/cgs_pilot}
DPY=${DRIVE_DPY:-/opt/conda/envs/metadrive/bin/python}                        # MetaDrive env: generator + validator
MPY=${DRIVE_MPY:-/opt/conda/bin/python}                                       # model env
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:$CODE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12}
REPO=${DRIVE_REPO:-/root/cgs-pilot/vendor/jepa-wms}
CFG=${DRIVE_CFG:-/root/cgs-pilot/configs/drive_predictor_arm$ARM.yaml}
CKPT=${DRIVE_CKPT:-/root/cgs-pilot/checkpoints/drive_predictor_arm$ARM.pth.tar}
MODEL_NAME=${DRIVE_MODEL_NAME:-jepa_wm_drive}
MODEL_LABEL=${DRIVE_MODEL_LABEL:-"JEPA-WM drive arm $ARM"}
TRAIN_REF=${DRIVE_TRAIN_REF:-/root/cgs-pilot/reference/drive_train_arm$ARM/training_reference.npz}   # pooled [N,D], clip_id [N] (+ action_chunks, chunk_clip)
VLOG=${DRIVE_VALIDATE_LOG:-/root/cgs-pilot/logs/metadrive_validate_labels.log}
LABELS=${DRIVE_LABELS:-$ROOT/_validation/hazard_label_validation.json}
JITTER=${DRIVE_JITTER:-"$ROOT/_validation/render_jitter_seed_*.json"}
# arm A: pedestrian carries the consequence (primary hazard level 1); arm B: the cone does (level 3 scored as H1)
PRIMARY_LEVEL=${DRIVE_PRIMARY_LEVEL:-$([ "$ARM" = "B" ] && echo 3 || echo 1)}
# run_drive_arms.sh compatibility (2026-09-03): DRIVE_LOG overrides the log path; DRIVE_SKIP_MERGE=1 skips the merge + mask
# copy (the runner merges ONCE per arm and points DRIVE_OUT at a per-model copy of that merge). Nothing else changes.
LOG=${DRIVE_LOG:-/root/cgs-pilot/logs/wave_drive_${WAVE}_arm$ARM.log}
mkdir -p $OUT
echo "[$(date -u +%FT%TZ)] drive wave $WAVE arm $ARM start (primary hazard level $PRIMARY_LEVEL; ckpt $CKPT)" > $LOG
if [ "${DRIVE_SKIP_WAIT:-0}" != "1" ]; then until grep -q VALIDATE_DONE $VLOG 2>/dev/null; do sleep 60; done; fi
echo "[$(date -u +%FT%TZ)] validation done" >> $LOG
cd $CODE
# 1. merge under v0.8 (8-cell completeness, partial scenes reported), label validation, jitter counts, stable hash split; no calibration seeds in driving
if [ "${DRIVE_SKIP_MERGE:-0}" != "1" ]; then
$DPY merge_heldout.py --root $ROOT --output $OUT --log-dir /root/cgs-pilot/logs --protocol cgs-metadrive-pilot-v0.8 --domain driving \
  --jitter-json "$JITTER" --label-validation $LABELS --split-seed 0 --exclude-seeds >> $LOG 2>&1
mkdir -p $OUT/masks; for d in $ROOT/seed_*/masks; do [ -d $d ] && cp -n $d/*.npz $OUT/masks/ 2>/dev/null; done
else echo "[$(date -u +%FT%TZ)] merge skipped (DRIVE_SKIP_MERGE=1; using pre-merged $OUT)" >> $LOG; fi
echo "[$(date -u +%FT%TZ)] merged: $(grep -c . $OUT/manifest.jsonl) cells ($(($(grep -c . $OUT/manifest.jsonl) / 8)) octets), masks $(ls $OUT/masks | wc -l); discovery $(tr '\n' ' ' < $OUT/discovery_seeds.txt)" >> $LOG
# 2. latent cache (GPU). latent_cache.py is frozen and reads egg_mask|robot_mask; in the driving domain its token mask is
#    hazard|corridor via token_groups.cache_token_mask_driving (documented shim; no DROID reference in driving).
$MPY -c "import sys, latent_cache as lc, token_groups as tg; lc.load_token_mask = tg.cache_token_mask_driving; sys.argv = ['latent_cache.py', '--artifacts', '$OUT', '--cache', '$OUT/latent_cache.npz', '--repo', '$REPO', '--config', '$CFG', '--checkpoint', '$CKPT', '--model-name', '$MODEL_NAME']; lc.main()" >> $LOG 2>&1
# 3. counterfactual-validity gate (driving currency: hazard_corridor group, approach DiD, identity contrast): all admitted, and discovery-only
$MPY counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_all.json --domain driving --primary-hazard-level $PRIMARY_LEVEL --model-label "$MODEL_LABEL" >> $LOG 2>&1
$MPY counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_discovery.json --seeds-file $OUT/discovery_seeds.txt --domain driving --primary-hazard-level $PRIMARY_LEVEL --model-label "$MODEL_LABEL" >> $LOG 2>&1
# 4. nearest-training-clip audit (frames + 2-D actions) and retrieval/copying baseline (identity block included)
$MPY audit_contamination.py --artifacts $OUT --output $OUT/contamination.json --cache $OUT/latent_cache.npz --domain driving --training-reference $TRAIN_REF >> $LOG 2>&1
$MPY retrieval_baseline.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/retrieval_baseline.json --domain driving --model-label "$MODEL_LABEL" >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] gate done" >> $LOG
# 5. B-GATE (preregistered; thresholds fixed in behavior_gate.py): FAIL => mechanism arms are not licensed for this arm.
$MPY behavior_gate.py --gate $OUT/cf_gate_discovery.json --output $OUT/behavior_gate_discovery.json >> $LOG 2>&1 \
  || { echo "[$(date -u +%FT%TZ)] B_GATE_FAIL: no unambiguous behavioural target on arm $ARM; mechanism arms NOT licensed" >> $LOG; echo "[$(date -u +%FT%TZ)] WAVE_DONE" >> $LOG; exit 0; }
echo "[$(date -u +%FT%TZ)] B_GATE_PASS ($(grep -o '"verdict": "[A-Z_]*"' $OUT/behavior_gate_discovery.json | head -1))" >> $LOG
echo "[$(date -u +%FT%TZ)] WAVE_DONE" >> $LOG

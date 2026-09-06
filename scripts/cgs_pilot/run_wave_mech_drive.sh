#!/bin/bash
# Driving (protocol v0.8) mechanism stage on DISCOVERY seeds only, arm-parametrised copy of run_wave_mech.sh.
# Waits for WAVE_DONE from run_wave_drive.sh, then RE-CHECKS the B-gate (as in run_loop2_cup_v3.sh) and stops
# before any mechanism arm on FAIL. Token groups use the egg names as aliases (egg=hazard, gripper_corridor=corridor).
# Usage: run_wave_mech_drive.sh <ARM: A|B> [WAVE tag]
set -u
ARM=${1:-A}; WAVE=${2:-d1}
OUT=${DRIVE_OUT:-/root/cgs-pilot/artifacts/drive_merged_${WAVE}_arm$ARM}
MECH=${DRIVE_MECH:-/root/cgs-pilot/artifacts/mech_drive_${WAVE}_arm$ARM}
CODE=${DRIVE_CODE:-/root/cgs-pilot/code/cgs_pilot}
MPY=${DRIVE_MPY:-/opt/conda/bin/python}
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:$CODE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12}
REPO=${DRIVE_REPO:-/root/cgs-pilot/vendor/jepa-wms}
CFG=${DRIVE_CFG:-/root/cgs-pilot/configs/drive_predictor_arm$ARM.yaml}
CKPT=${DRIVE_CKPT:-/root/cgs-pilot/checkpoints/drive_predictor_arm$ARM.pth.tar}
MODEL_NAME=${DRIVE_MODEL_NAME:-jepa_wm_drive}
# run_drive_arms.sh compatibility (2026-09-03): DRIVE_LOG / DRIVE_WAVE_LOG override the mech log and the wave log it waits on.
LOG=${DRIVE_LOG:-/root/cgs-pilot/logs/mech_drive_${WAVE}_arm$ARM.log}
WAVE_LOG=${DRIVE_WAVE_LOG:-/root/cgs-pilot/logs/wave_drive_${WAVE}_arm$ARM.log}
mkdir -p $MECH
until grep -q WAVE_DONE $WAVE_LOG 2>/dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] mech drive $WAVE arm $ARM start; discovery seeds: $(tr '\n' ' ' < $OUT/discovery_seeds.txt)" > $LOG
cd $CODE
# B-gate stop (inserted after the discovery gate, as in run_loop2_cup_v3.sh): exit 3 = FAIL => no mechanism arms.
$MPY behavior_gate.py --gate $OUT/cf_gate_discovery.json --output $OUT/behavior_gate_discovery.json >> $LOG 2>&1 \
  || { echo "[$(date -u +%FT%TZ)] B-GATE FAIL: mechanism arms not licensed on arm $ARM; MECH_DONE (behaviour-only)" >> $LOG; exit 0; }
echo "[$(date -u +%FT%TZ)] B-GATE PASS" >> $LOG
# 10. predictor-wide interaction localization (levels 0/1/2/3 dumped; identity contrast; hazard/corridor groups via aliases)
$MPY localize_interaction.py --repo $REPO --config $CFG --checkpoint $CKPT --model-name $MODEL_NAME --artifacts $OUT --output $MECH/localize --domain driving \
  --seeds-file $OUT/discovery_seeds.txt --dump-hooks resid_post attn_out mlp_out --dump-groups egg corridor >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] localization done" >> $LOG
# time bound (2026-09-03 04:00 UTC, before any mechanism result): full patch sweep + Jacobian only for DRIVE_MECH_FULL_SEEDS; Jacobian on the first DRIVE_JAC_N discovery seeds, step 0
FULL_SEEDS=${DRIVE_MECH_FULL_SEEDS:-0}; JAC_N=${DRIVE_JAC_N:-8}; SEED_TAG=${WAVE#seed}
if echo " $FULL_SEEDS " | grep -q " $SEED_TAG "; then
# 11. patching sweep at step 0 (primary group gripper_corridor = corridor alias; egg = hazard alias; level-3 identity donor reported)
$MPY patch_site.py --repo $REPO --config $CFG --checkpoint $CKPT --model-name $MODEL_NAME --artifacts $OUT --output $MECH/patch_step0 --domain driving \
  --seeds-file $OUT/discovery_seeds.txt --hooks attn_out mlp_out --steps 0 --groups gripper_corridor egg --mode one_shot >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] patch step0 done" >> $LOG
head -n $JAC_N $OUT/discovery_seeds.txt > $MECH/jacobian_seeds.txt
# 12. cross-token action-Jacobian sonar (route = corridor tokens; identity contrast dJ_hazard - dJ_object)
$MPY action_jacobian_sonar.py --repo $REPO --config $CFG --checkpoint $CKPT --model-name $MODEL_NAME --artifacts $OUT --output $MECH/jacobian_sonar --domain driving \
  --seeds-file $MECH/jacobian_seeds.txt --hooks attn_out mlp_out --groups gripper_corridor egg all --steps 0 --mediation-steps 0 >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] jacobian done (subset $JAC_N seeds, step 0)" >> $LOG
else echo "[$(date -u +%FT%TZ)] patch/jacobian SKIPPED for seed $SEED_TAG (time bound; full seeds: $FULL_SEEDS)" >> $LOG; fi
# 13. geometry tournament on the discovery dump (confirmation withheld)
$MPY geometry_tournament.py --dump $MECH/localize --stimulus $OUT --out $MECH/geometry --discovery-seeds $OUT/discovery_seeds.txt --export-edits >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] MECH_DONE" >> $LOG

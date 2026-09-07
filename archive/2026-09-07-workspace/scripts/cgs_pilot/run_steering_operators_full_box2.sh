#!/bin/bash
# run_steering_operators_full_box2.sh -- follow-up to run_steering_operators_box2.sh (coordinator request 2026-09-03 ~23:00 UTC):
#   F  min_distortion steering of arms A and B on the FULL common discovery set (53 scenes) with the operators ALREADY fitted
#      (zero refitting) -> $SO/arm{A,B}_seed0/steer_min_distortion_full/{discovery,calibration.json}
#   V  Operator 1 on the v0.9 randomised-factor stimulus: fit on the CCGP job's dumps ($OUT/drive_ccgp_v09/dump_arm{A,B}, 66 common
#      discovery seeds, manifest factors hazard_dist_m / prefix_throttle / hazard_lateral_offset_m -> factors_used = true expected)
#      -> $SO/v09/arm{A,B}_seed0/min_distortion/ ; then the steering table on the v0.9 rand discovery seeds at the band
#      -> $SO/v09/arm{A,B}_seed0/steer_min_distortion/discovery
# Runs in parallel with the first driver (which keeps the modulation steering stages); shares the GPU; never kills anything.
# Markers in $LOG: SOF_FULL_DONE, SOF_V09_FIT_DONE, SOF_V09_STEER_DONE, STEERING_OPERATORS_FULL_DONE.
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}
CODE=$BASE/code/cgs_pilot; REPO=$BASE/vendor/jepa-wms; MPY=/opt/conda/bin/python
OUT=$BASE/artifacts; LOGDIR=$BASE/logs
MODELS=$OUT/drive_models; MERGED=$OUT/drive_factorial_merged
SO=${SO_OUT:-$OUT/steering_operators}
V09_DUMPS=$OUT/drive_ccgp_v09; V09_STIM=$OUT/drive_v09_merged/rand; V09_SEEDS=$OUT/drive_ccgp_v09/discovery_seeds_common.txt
LOG=$LOGDIR/steering_operators_full.log
SEED=0
BETAS="0.25 0.5 1.0"; MODES="strengthen suppress"; GOALS="progress brake"
export JEPAWM_HOME=$BASE/vendor JEPAWM_OSSCKPT=$BASE/checkpoints JEPAWM_LOGS=$BASE/logs PYTHONPATH=$REPO:$CODE
export JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
FILTER='^INFO:|^🔧|^🧠|^🔮|^📉|Encoder: |Predictor: |^Loading|^Loaded'
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*" | tee -a $LOG; }
mkdir -p $SO/v09 $LOGDIR
cd $CODE || exit 1
say "STEERING_OPERATORS_FULL start -> $SO"
common() {   # common ARM
  local ARM=$1 LEVEL MD
  LEVEL=$([ "$ARM" = B ] && echo 3 || echo 1); MD=$MODELS/arm${ARM}_seed$SEED
  echo "--repo $REPO --config $MD/eval_config.yaml --checkpoint $MD/jepa-latest.pth.tar --model-name jepa_wm_driving --domain driving --primary-hazard-level $LEVEL --arm $ARM --model-seed $SEED --beta $BETAS --conceptor-mode $MODES --goal $GOALS --step 0 --n-boot 2000 --no-singles --group hazard_corridor"
}
band() { [ "$1" = B ] && echo "L02.mlp_out L03.mlp_out" || echo "L03.attn_out L03.mlp_out"; }

# ---- F. full common discovery set, operators already fitted (zero refitting)
for ARM in A B; do
  RD=$SO/arm${ARM}_seed$SEED/steer_min_distortion_full
  if [ -f $RD/discovery/steered_ranking.json ]; then say "arm $ARM: full-discovery steering present"; continue; fi
  say "arm $ARM: min_distortion steering on $(grep -c . $MERGED/discovery_seeds_common.txt) common discovery scenes at band [$(band $ARM)]"
  $MPY steered_planner_ranking.py $(common $ARM) --artifacts $MERGED/arm$ARM --seeds-file $MERGED/discovery_seeds_common.txt --band-sites $(band $ARM) \
    --min-distortion-dir $SO/arm${ARM}_seed$SEED/min_distortion --model-label "JEPA-WM driving arm $ARM seed $SEED (jepa-latest)" \
    --out $RD/discovery --calibrate-out $RD/calibration.json 2>&1 | grep -Ev "$FILTER" >> $LOG
  [ -f $RD/discovery/steered_ranking.json ] || { say "arm $ARM: full-discovery steering FAILED"; exit 1; }
  say "arm $ARM: full-discovery steering done ($(($(wc -l < $RD/discovery/coast_table.md) - 2)) rows)"
done
say "SOF_FULL_DONE"

# ---- V. v0.9 randomised-factor stimulus: fit (CPU) + steering (GPU)
for ARM in A B; do
  LEVEL=$([ "$ARM" = B ] && echo 3 || echo 1)
  OD=$SO/v09/arm${ARM}_seed$SEED/min_distortion
  [ -f $V09_DUMPS/dump_arm$ARM/activations/index.json ] || { say "v0.9 dump for arm $ARM missing ($V09_DUMPS/dump_arm$ARM); skipping v0.9"; exit 0; }
  if [ -f $OD/causal_metric_fit.json ]; then say "v0.9 arm $ARM: fit present"; continue; fi
  say "v0.9 arm $ARM: causal_metric_steer.py fit on $(grep -c . $V09_SEEDS) discovery seeds (manifest factors on)"
  $MPY causal_metric_steer.py fit --dump $V09_DUMPS/dump_arm$ARM --stimulus $V09_STIM/arm$ARM --discovery-seeds $V09_SEEDS --solid-level $LEVEL \
    --steps 0 --groups hazard_corridor hazard --rank 4 --k-action 4 --out $OD --seed $SEED >> $LOG 2>&1 || { say "v0.9 arm $ARM: fit FAILED"; exit 1; }
  say "v0.9 arm $ARM: fit done ($(ls $OD/*.npz | wc -l) files); factors_used=$(grep -o '"used": [a-z]*' $OD/L03.mlp_out__s0__hazard_corridor.json | head -1)"
done
say "SOF_V09_FIT_DONE"
for ARM in A B; do
  RD=$SO/v09/arm${ARM}_seed$SEED/steer_min_distortion
  if [ -f $RD/discovery/steered_ranking.json ]; then say "v0.9 arm $ARM: steering present"; continue; fi
  say "v0.9 arm $ARM: min_distortion steering on $(grep -c . $V09_SEEDS) v0.9 discovery scenes at band [$(band $ARM)]"
  $MPY steered_planner_ranking.py $(common $ARM) --artifacts $V09_STIM/arm$ARM --seeds-file $V09_SEEDS --band-sites $(band $ARM) \
    --min-distortion-dir $SO/v09/arm${ARM}_seed$SEED/min_distortion --model-label "JEPA-WM driving arm $ARM seed $SEED (jepa-latest) on v0.9 rand" \
    --out $RD/discovery --calibrate-out $RD/calibration.json 2>&1 | grep -Ev "$FILTER" >> $LOG
  [ -f $RD/discovery/steered_ranking.json ] || { say "v0.9 arm $ARM: steering FAILED"; exit 1; }
  say "v0.9 arm $ARM: steering done ($(($(wc -l < $RD/discovery/coast_table.md) - 2)) rows)"
done
say "SOF_V09_STEER_DONE"
echo "[$(ts)] STEERING_OPERATORS_FULL_DONE" >> $LOG
say "STEERING_OPERATORS_FULL_DONE -> $SO"

#!/bin/bash
# run_steering_operators_box2.sh -- real-data DEVELOPMENT check of the two donor-free steering operators (design doc
# "Registered additions" (a) causal-metric minimum-distortion edit, (b) AdaLN modulation operator) on box 2, seed-0 models.
# Open-loop only: under the Label-first principle the steered-planner table is a CALIBRATION instrument, not an endpoint.
#
#   0 wait     until run_identity_geometry.sh has dumped AND slimmed the seed-0 activations of each arm
#              ("arm X seed 0: slimmed" in logs/drive_identity_geometry.log); the dump is then COPIED to $SO/dump_armX at once
#              (run_identity_geometry.sh deletes the activations in its cleanup stage).  Aborts if the cleanup already ran.
#   1 fit-md   causal_metric_steer.py fit  (numpy, CPU) per arm: every non-adaln site, step 0, groups hazard_corridor + hazard,
#              discovery seeds only -> $SO/arm{A,B}_seed0/min_distortion/{<site>__s0__<group>.npz, causal_metric_fit.json}
#   2 fit-mod  modulation_operator.py fit (GPU, forward-AD JVPs) per arm on <= $MOD_SCENES discovery scenes, all blocks,
#              components attn/mlp/all, ranks 1 2 4 -> $SO/arm{A,B}_seed0/modulation/
#   3 steer    steered_planner_ranking.py on a $STEER_SEEDS-seed subset of the common discovery seeds (development smoke of the
#              calibration run; frozen beta grid; both modes; progress + brake goals; the operator's own control set):
#              (a) --min-distortion-dir at the arm's band (A: L03.attn_out+L03.mlp_out; B: L02.mlp_out+L03.mlp_out),
#              (b) --modulation-operator-dir at L02.adaln / L03.adaln (component all, ranks 1 2 4).
#   Markers in $LOG: SO_DUMPS_COPIED, SO_FIT_MD_DONE, SO_FIT_MOD_DONE, SO_STEER_DONE, STEERING_OPERATORS_DONE.
# Never kills anything; shares the GPU.  Relaunch-safe (stages skipped when their output exists).
# Launch:  ssh -n -p PORT root@HOST 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_steering_operators_box2.sh > /root/cgs-pilot/logs/steering_operators_driver.log 2>&1 < /dev/null &'
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}
CODE=$BASE/code/cgs_pilot; REPO=$BASE/vendor/jepa-wms; MPY=/opt/conda/bin/python
OUT=$BASE/artifacts; LOGDIR=$BASE/logs
MODELS=$OUT/drive_models; MERGED=$OUT/drive_factorial_merged; IG=$OUT/drive_identity_geometry
SO=${SO_OUT:-$OUT/steering_operators}
IGLOG=$LOGDIR/drive_identity_geometry.log
LOG=$LOGDIR/steering_operators.log
SEED=0
STEER_SEEDS=${SO_STEER_SEEDS:-12}
MOD_SCENES=${SO_MOD_SCENES:-16}
BETAS="0.25 0.5 1.0"; MODES="strengthen suppress"; GOALS="progress brake"
export JEPAWM_HOME=$BASE/vendor JEPAWM_OSSCKPT=$BASE/checkpoints JEPAWM_LOGS=$BASE/logs PYTHONPATH=$REPO:$CODE
export JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
FILTER='^INFO:|^🔧|^🧠|^🔮|^📉|Encoder: |Predictor: |^Loading|^Loaded'
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*" | tee -a $LOG; }
mkdir -p $SO $LOGDIR
cd $CODE || exit 1
say "STEERING_OPERATORS start (seed $SEED, steer seeds $STEER_SEEDS, mod scenes $MOD_SCENES) -> $SO"

# ---- 0. wait for the slimmed seed-0 dumps and copy them at once
for ARM in A B; do
  DST=$SO/dump_arm$ARM
  if [ -f $DST/activations/index.json ]; then say "arm $ARM: dump copy present"; continue; fi
  say "arm $ARM: waiting for 'arm $ARM seed $SEED: slimmed' in $IGLOG"
  n=0
  until grep -q "arm $ARM seed $SEED: slimmed" $IGLOG 2>/dev/null; do
    if grep -q "dump activations deleted" $IGLOG 2>/dev/null && ! grep -q "arm $ARM seed $SEED: slimmed" $IGLOG 2>/dev/null; then say "arm $ARM: identity-geometry cleanup already ran without a slim message; aborting"; exit 1; fi
    n=$((n + 1)); [ $((n % 20)) = 0 ] && say "still waiting for arm $ARM dump ($(tail -1 $IGLOG 2>/dev/null | cut -c1-100))"
    sleep 30
  done
  [ -d $IG/dump_arm$ARM/activations ] || { say "arm $ARM: $IG/dump_arm$ARM/activations missing after the slim message (cleanup raced us); aborting"; exit 1; }
  rm -rf $DST; mkdir -p $DST
  cp -r $IG/dump_arm$ARM/activations $DST/activations
  [ -f $IG/dump_arm$ARM/interaction_map.json ] && cp $IG/dump_arm$ARM/interaction_map.json $DST/
  say "arm $ARM: dump copied ($(du -sh $DST | cut -f1), $(ls $DST/activations/*.npz | wc -l) sites)"
done
say "SO_DUMPS_COPIED"

# ---- 1. causal-metric minimum-distortion operators (CPU)
for ARM in A B; do
  LEVEL=$([ "$ARM" = B ] && echo 3 || echo 1)
  OD=$SO/arm${ARM}_seed$SEED/min_distortion
  if [ -f $OD/causal_metric_fit.json ]; then say "arm $ARM: min-distortion fit present"; continue; fi
  say "arm $ARM: causal_metric_steer.py fit (solid level $LEVEL)"
  $MPY causal_metric_steer.py fit --dump $SO/dump_arm$ARM --stimulus $MERGED/arm$ARM --discovery-seeds $MERGED/discovery_seeds_common.txt --solid-level $LEVEL \
    --steps 0 --groups hazard_corridor hazard --rank 4 --k-action 4 --out $OD --seed $SEED >> $LOG 2>&1 || { say "arm $ARM: min-distortion fit FAILED"; exit 1; }
  say "arm $ARM: min-distortion fit done ($(ls $OD/*.npz | wc -l) operator files)"
done
say "SO_FIT_MD_DONE"

# ---- 2. modulation operators (GPU)
for ARM in A B; do
  LEVEL=$([ "$ARM" = B ] && echo 3 || echo 1); MD=$MODELS/arm${ARM}_seed$SEED
  OD=$SO/arm${ARM}_seed$SEED/modulation
  if [ -f $OD/modulation_operator_fit.json ]; then say "arm $ARM: modulation fit present"; continue; fi
  say "arm $ARM: modulation_operator.py fit (<= $MOD_SCENES scenes); GPU $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1)"
  $MPY modulation_operator.py fit --repo $REPO --config $MD/eval_config.yaml --checkpoint $MD/jepa-latest.pth.tar --model-name jepa_wm_driving --artifacts $MERGED/arm$ARM \
    --seeds-file $MERGED/discovery_seeds_common.txt --solid-level $LEVEL --components attn mlp all --ranks 1 2 4 --step 0 --max-scenes $MOD_SCENES \
    --n-boot 500 --n-perm 500 --out $OD --seed $SEED 2>&1 | grep -Ev "$FILTER" >> $LOG || { say "arm $ARM: modulation fit FAILED"; exit 1; }
  [ -f $OD/modulation_operator_fit.json ] || { say "arm $ARM: modulation fit produced no JSON"; exit 1; }
  say "arm $ARM: modulation fit done ($(ls $OD/*.npz | wc -l) operator files)"
done
say "SO_FIT_MOD_DONE"

# ---- 3. steered-planner calibration smoke (development; open-loop = calibration instrument only)
head -n $STEER_SEEDS $MERGED/discovery_seeds_common.txt > $SO/steer_seeds_${STEER_SEEDS}.txt
for ARM in A B; do
  LEVEL=$([ "$ARM" = B ] && echo 3 || echo 1); MD=$MODELS/arm${ARM}_seed$SEED
  BAND=$([ "$ARM" = B ] && echo "L02.mlp_out L03.mlp_out" || echo "L03.attn_out L03.mlp_out")
  COMMON="--repo $REPO --config $MD/eval_config.yaml --checkpoint $MD/jepa-latest.pth.tar --model-name jepa_wm_driving --artifacts $MERGED/arm$ARM --domain driving \
          --primary-hazard-level $LEVEL --arm $ARM --model-seed $SEED --beta $BETAS --conceptor-mode $MODES --goal $GOALS --step 0 --n-boot 500 --no-singles \
          --seeds-file $SO/steer_seeds_${STEER_SEEDS}.txt --model-label 'JEPA-WM driving arm $ARM seed $SEED (jepa-latest)'"
  RD=$SO/arm${ARM}_seed$SEED/steer_min_distortion
  if [ ! -f $RD/discovery/steered_ranking.json ]; then
    say "arm $ARM: steered planner (min_distortion) at band [$BAND]"
    eval $MPY steered_planner_ranking.py $COMMON --band-sites $BAND --group hazard_corridor --min-distortion-dir $SO/arm${ARM}_seed$SEED/min_distortion \
      --out $RD/discovery --calibrate-out $RD/calibration.json 2>&1 | grep -Ev "$FILTER" >> $LOG
    [ -f $RD/discovery/steered_ranking.json ] || { say "arm $ARM: min_distortion steering FAILED"; exit 1; }
    say "arm $ARM: min_distortion steering done ($(($(wc -l < $RD/discovery/coast_table.md) - 2)) rows)"
  fi
  RD=$SO/arm${ARM}_seed$SEED/steer_modulation
  if [ ! -f $RD/discovery/steered_ranking.json ]; then
    say "arm $ARM: steered planner (modulation) at L02.adaln L03.adaln"
    eval $MPY steered_planner_ranking.py $COMMON --sites L02.adaln L03.adaln --modulation-operator-dir $SO/arm${ARM}_seed$SEED/modulation --modulation-component all --modulation-rank 1 2 4 \
      --out $RD/discovery --calibrate-out $RD/calibration.json 2>&1 | grep -Ev "$FILTER" >> $LOG
    [ -f $RD/discovery/steered_ranking.json ] || { say "arm $ARM: modulation steering FAILED"; exit 1; }
    say "arm $ARM: modulation steering done ($(($(wc -l < $RD/discovery/coast_table.md) - 2)) rows)"
  fi
done
say "SO_STEER_DONE"
echo "[$(ts)] STEERING_OPERATORS_DONE" >> $LOG
say "STEERING_OPERATORS_DONE -> $SO"

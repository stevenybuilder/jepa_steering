#!/bin/bash
# Run MetaWorld component-extension slots on one GPU: for each "task:slot" arg,
# run device-bound engineering once per task, then native, visual_only,
# action_condition_only, joint for that logical rank. Verified DONE shards are
# skipped; an unverified/partial shard dir is removed and the whole 12-episode
# stream rerun from its seed (never resumed mid-stream).
# Usage: mw_slot_driver.sh reach:4 reach-wall:4 ...
set -uo pipefail
ROOT=/workspace/metaworld-components-20260911-v1
PY=/workspace/component-python/bin/python
MOD="-u -m offline_study.metaworld_component_behavior"
VENDOR=$ROOT/code/vendor/jepa-wms
ORIGINAL=$ROOT/code/artifacts/offline_study/primary-durable-20260907
STIMULI=$ROOT/code/artifacts/offline_study/restored-behavioral-inputs-20260908-v1
FREEZE=$ROOT/code/artifacts/offline_study/table-completion-20260911-v1/component-extension-v1/freeze
CHECKPOINT=$ROOT/checkpoints/jepa_wm_metaworld.pth.tar
GPU_IDX=${CUDA_VISIBLE_DEVICES:-0}
UUID=$(nvidia-smi -i "$GPU_IDX" --query-gpu=uuid --format=csv,noheader)
ENG_BASE=$ROOT/engineering/$UUID
RESULTS=$ROOT/results
LOG=$ROOT/slot-driver-gpu$GPU_IDX.log
export PYTHONPATH=$ROOT/code/src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd $ROOT/code

verified () { local d=$1; [ -f "$d/DONE.json" ] && [ -f "$d/report.json" ] && [ ! -f "$d/FAILED.json" ] && \
  [ "$($PY -c "import json,hashlib;print(json.load(open('$d/DONE.json'))['report_sha256']==hashlib.sha256(open('$d/report.json','rb').read()).hexdigest())")" = "True" ]; }

run () { local label=$1; shift
  echo "$(date -u) START $label" >> $LOG
  if "$@" >> $ROOT/slot-driver-$label.log 2>&1; then echo "$(date -u) OK $label" >> $LOG; else echo "$(date -u) FAILED $label" >> $LOG; exit 1; fi; }

for spec in "$@"; do
  task=${spec%%:*}; slot=${spec##*:}; shard=$(printf 'shard-%02d' "$slot")
  if [ ! -f "$ENG_BASE/$task/report.json" ]; then
    run "engineering-$task" $PY $MOD engineering --vendor $VENDOR --original $ORIGINAL --stimuli $STIMULI \
      --output $ENG_BASE/$task --freeze $FREEZE --checkpoint $CHECKPOINT --task $task
  fi
  for arm in ${ARMS:-native visual_only action_condition_only joint}; do
    out=$RESULTS/$task/$arm/$shard
    if verified "$out"; then echo "$(date -u) SKIP verified $task/$arm/$shard" >> $LOG; continue; fi
    [ -d "$out" ] && { rm -rf "$out"; echo "$(date -u) REPLACED partial $task/$arm/$shard" >> $LOG; }
    run "$task-$arm-$slot" $PY $MOD run --vendor $VENDOR --original $ORIGINAL --stimuli $STIMULI \
      --output $out --freeze $FREEZE --checkpoint $CHECKPOINT --task $task \
      --engineering $ENG_BASE/$task --arm $arm --logical-ranks $slot
  done
done
echo "$(date -u) ALL_DONE $*" >> $LOG

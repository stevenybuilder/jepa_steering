#!/bin/bash
# MetaWorld component-extension slot driver, v2. Same science as v1 (device-bound
# engineering once per task per GPU; each (task, arm, logical slot) is a whole
# 12-episode RNG stream run on one device; verified shards skipped; partial shards
# wiped and rerun whole). Additions:
#   * spec form task:slot[:arm1+arm2+...]  (default all four arms)
#   * waits for any orphaned `run` process on this box to finish before starting
#     (lets a v1 driver be replaced without losing its in-flight arm)
#   * skips a unit whose marker $RESULTS/<task>/<arm>/<shard>.MOVED exists
#     (unit reassigned to another host — prevents cross-host duplication)
# Usage: mw_slot_driver_v2.sh reach:4 reach-wall:4:native+visual_only ...
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
echo "$(date -u) V2 START specs=[$*]" >> $LOG
while pgrep -f "metaworld_component_behavior run" >/dev/null; do sleep 30; done

verified () { local d=$1; [ -f "$d/DONE.json" ] && [ -f "$d/report.json" ] && [ ! -f "$d/FAILED.json" ] && \
  [ "$($PY -c "import json,hashlib;print(json.load(open('$d/DONE.json'))['report_sha256']==hashlib.sha256(open('$d/report.json','rb').read()).hexdigest())")" = "True" ]; }

run () { local label=$1; shift
  echo "$(date -u) START $label" >> $LOG
  if "$@" >> $ROOT/slot-driver-$label.log 2>&1; then echo "$(date -u) OK $label" >> $LOG; else echo "$(date -u) FAILED $label" >> $LOG; exit 1; fi; }

for spec in "$@"; do
  IFS=: read -r task slot arms <<< "$spec"; shard=$(printf 'shard-%02d' "$slot"); arms=${arms:-native+visual_only+action_condition_only+joint}
  todo=; for arm in ${arms//+/ }; do out=$RESULTS/$task/$arm/$shard
    if [ -f "$out.MOVED" ]; then echo "$(date -u) SKIP moved $task/$arm/$shard" >> $LOG; continue; fi
    if verified "$out"; then echo "$(date -u) SKIP verified $task/$arm/$shard" >> $LOG; continue; fi
    todo="$todo $arm"; done
  [ -z "$todo" ] && continue
  if [ ! -f "$ENG_BASE/$task/report.json" ]; then
    run "engineering-$task" $PY $MOD engineering --vendor $VENDOR --original $ORIGINAL --stimuli $STIMULI \
      --output $ENG_BASE/$task --freeze $FREEZE --checkpoint $CHECKPOINT --task $task
  fi
  for arm in $todo; do out=$RESULTS/$task/$arm/$shard
    [ -f "$out.MOVED" ] && { echo "$(date -u) SKIP moved $task/$arm/$shard" >> $LOG; continue; }
    verified "$out" && { echo "$(date -u) SKIP verified $task/$arm/$shard" >> $LOG; continue; }
    [ -d "$out" ] && { rm -rf "$out"; echo "$(date -u) REPLACED partial $task/$arm/$shard" >> $LOG; }
    run "$task-$arm-$slot" $PY $MOD run --vendor $VENDOR --original $ORIGINAL --stimuli $STIMULI \
      --output $out --freeze $FREEZE --checkpoint $CHECKPOINT --task $task \
      --engineering $ENG_BASE/$task --arm $arm --logical-ranks $slot
  done
done
echo "$(date -u) ALL_DONE $*" >> $LOG

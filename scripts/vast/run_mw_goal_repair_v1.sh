#!/usr/bin/env bash
# Main GPU4..7: finish the owned original scope, then repair only input-mismatched streams.
set -eu
gpu="$1"
case "$gpu" in
  4) task=reach; arm=coupling_only ;;
  5) task=reach; arm=matched_random_coupling ;;
  6) task=reach-wall; arm=coupling_only ;;
  7) task=reach-wall; arm=matched_random_coupling ;;
  *) exit 2 ;;
esac
runtime=/workspace/jepa-runtime
original="$runtime/behavioral-development-20260907-v1/$task/$arm/shard-0"
while test ! -f "$original/DONE.json"; do
  test ! -f "$original/FAILED.json" || exit 3
  sleep 15
done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"; do sleep 10; done
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl
export JEPA_VERIFIED_LOCAL_DINO=1 SDL_VIDEODRIVER=dummy
export PYTHONPATH="$runtime/coupling-repair-code-v1/src:/workspace/jepa-python/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/opt/conda/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
plan="$runtime/behavioral-input-repairs-20260907-v1/$task/$arm/shard-0"
common=(--freeze "$runtime/behavioral-development-freeze-20260907-v1"
  --goal-bank "$runtime/metaworld-native-goal-bank-20260907-v1"
  --goal-delivery-proof "$runtime/metaworld-goal-delivery-check-20260907-v2"
  --task "$task" --arm "$arm")
/workspace/jepa-planning-python/bin/python -u -m offline_study.behavioral_repair_plan \
  "${common[@]}" --reference-root "$runtime/behavioral-development-20260907-v1" \
  --candidate-shard "$original" --output "$plan"
rank_list=$(/workspace/jepa-planning-python/bin/python -c \
  'import json,sys; print(" ".join(map(str,json.load(open(sys.argv[1]))["logical_ranks"])))' "$plan/plan.json")
test -n "$rank_list" || exit 0
read -r -a ranks <<< "$rank_list"
exec /workspace/jepa-planning-python/bin/python -u -m offline_study.behavioral_candidate \
  "${common[@]}" --vendor /workspace/jepa_steering/vendor/jepa-wms \
  --baseline-code "$runtime/author-correction-20260907/code-v33" \
  --original-root "$runtime/author-correction-20260907" \
  --checkpoint "$runtime/checkpoints/jepa_wm_metaworld.pth.tar" \
  --engineering "$runtime/planning-panel-coupling-engineering-20260907-v1/$task-$arm" \
  --logical-ranks "${ranks[@]}" --repair-plan "$plan" \
  --output "$runtime/behavioral-development-canonical-20260907-v1/$task/$arm/shard-0-repair"

#!/usr/bin/env bash
# Independent frozen logical streams, same RTX5090 class as paired baseline.
set -eu
gpu="$1"
case "$gpu" in
  2) task=reach; arm=matched_random_combined; engineering=reach-combined-matched_random_rank4; logical=0 ;;
  3) task=reach; arm=combined; engineering=reach-combined-rank4; logical=1 ;;
  *) exit 2 ;;
esac
runtime=/workspace/jepa-runtime
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl
export JEPA_VERIFIED_LOCAL_DINO=1 SDL_VIDEODRIVER=dummy
export PYTHONPATH="$runtime/canonical-panel-code-v1/src:/workspace/jepa-python/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/opt/conda/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arguments=(--vendor /workspace/jepa_steering/vendor/jepa-wms
  --freeze "$runtime/behavioral-development-freeze-20260907-v1"
  --baseline-code "$runtime/author-correction-20260907/code-v33"
  --engineering-code "$runtime/author-correction-20260907/code-v28"
  --original-root "$runtime/author-correction-20260907"
  --checkpoint "$runtime/checkpoints/jepa_wm_metaworld.pth.tar"
  --engineering "$runtime/planning-selected-full-cem-smoke-20260907-v1/$engineering"
  --goal-bank "$runtime/metaworld-native-goal-bank-20260907-v1"
  --goal-delivery-proof "$runtime/metaworld-goal-delivery-check-20260907-v2"
  --task "$task" --arm "$arm" --logical-ranks "$logical"
  --output "$runtime/behavioral-development-canonical-20260907-v1/$task/$arm/rank-$logical")
/workspace/jepa-planning-python/bin/python -u -m offline_study.behavioral_rank "${arguments[@]}" --verify-only
test "${2:-run}" != verify-only || exit 0
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)" || exit 4
exec /workspace/jepa-planning-python/bin/python -u -m offline_study.behavioral_rank "${arguments[@]}"

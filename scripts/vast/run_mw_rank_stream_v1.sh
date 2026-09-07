#!/usr/bin/env bash
# Owner rep_geometry_transcoder/root, exclusive 50125440 GPU0/2/3 assignments.
set -eu
gpu="$1"
arm="$2"
rank="$3"
case "$gpu:$arm:$rank" in
  0:combined:0) task=reach; engineering=reach-combined-rank4 ;;
  2:rank4_only:0) task=reach-wall; engineering=reach-wall-rank4 ;;
  3:matched_random_rank4:0) task=reach-wall; engineering=reach-wall-matched_random_rank4 ;;
  *) exit 2 ;;
esac
runtime=/workspace/jepa-runtime
test -z "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)" || exit 4
export CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" MUJOCO_GL=egl
export JEPA_VERIFIED_LOCAL_DINO=1 SDL_VIDEODRIVER=dummy
export PYTHONPATH="$runtime/rank-panel-code-v2/src:/workspace/jepa-python/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/opt/conda/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
arguments=(--vendor /workspace/jepa_steering/vendor/jepa-wms
  --freeze "$runtime/behavioral-development-freeze-20260907-v1"
  --baseline-code "$runtime/author-correction-20260907/code-v33"
  --engineering-code "$runtime/author-correction-20260907/code-v28"
  --original-root "$runtime/author-correction-20260907"
  --checkpoint "$runtime/checkpoints/jepa_wm_metaworld.pth.tar"
  --engineering "$runtime/planning-selected-full-cem-smoke-20260907-v1/$engineering"
  --task "$task" --arm "$arm" --logical-ranks "$rank"
  --output "$runtime/behavioral-development-20260907-v1/$task/$arm/rank-$rank")
/workspace/jepa-planning-python/bin/python -u -m offline_study.behavioral_rank "${arguments[@]}" --verify-only
exec /workspace/jepa-planning-python/bin/python -u -m offline_study.behavioral_rank "${arguments[@]}"

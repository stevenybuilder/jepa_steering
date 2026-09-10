#!/usr/bin/env bash
# Exclusive owned50205763 GPU0. Every stage has a new output and fails closed.
set -eu
runtime=/workspace/jepa-runtime
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 4
export CUDA_VISIBLE_DEVICES=0 JEPA_VERIFIED_LOCAL_DINO=1
export PYTHONPATH="$runtime/code-v14/src:/workspace/jepa-python/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/opt/conda/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for precision in bfloat16 float32; do
  for task in wall pointmaze; do
    folder=wall_single
    test "$task" != pointmaze || folder=point_maze
    for stage in baseline fit; do
      output="$runtime/navigation-offline-20260907-v1/$task/$precision"
      test "$stage" != fit || output="$runtime/navigation-fits-20260907-v1/$precision/$task"
      /workspace/jepa-planning-python/bin/python -u -m offline_study.navigation_offline "$stage" \
        --vendor /workspace/jepa_steering/vendor/jepa-wms \
        --checkpoint "$runtime/navigation-assets-20260907-v1/downloads/model/jepa_wm_$task.pth.tar" \
        --cohort "$runtime/navigation-offline-cohorts-20260907-v1/$task/cohort.json" \
        --data-root "$runtime/navigation-assets-20260907-v1/extracted/$task/$folder" \
        --precision "$precision" --output "$output"
    done
  done
done

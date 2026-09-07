#!/usr/bin/env bash
# Next exclusive 50205763 GPU0 workload, after the complete baseline/fit queue.
set -eu
runtime=/workspace/jepa-runtime
while test ! -f "$runtime/navigation-fits-20260907-v1/float32/pointmaze/DONE.json"; do
  if find "$runtime/navigation-offline-20260907-v1" "$runtime/navigation-fits-20260907-v1" -name FAILED.json -print -quit | grep -q .; then exit 3; fi
  sleep 10
done
while test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; do sleep 10; done
export CUDA_VISIBLE_DEVICES=0 JEPA_VERIFIED_LOCAL_DINO=1
export PYTHONPATH="$runtime/navigation-evaluation-code-v1/src:/workspace/jepa-python/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/opt/conda/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for category in vision_action_coupling action_response_geometry operator_rank distribution_layer distribution_spatial; do
  shards=8
  case "$category" in operator_rank|distribution_layer|distribution_spatial) shards=32 ;; esac
  for precision in bfloat16 float32; do
    for task in wall pointmaze; do
      folder=wall_single
      test "$task" != pointmaze || folder=point_maze
      for ((index=0; index<shards; index++)); do
        output="$runtime/navigation-comparisons-20260907-v1/$precision/$task/$category/shard-$index"
        /workspace/jepa-planning-python/bin/python -u -m offline_study.navigation_evaluate \
          --vendor /workspace/jepa_steering/vendor/jepa-wms \
          --checkpoint "$runtime/navigation-assets-20260907-v1/downloads/model/jepa_wm_$task.pth.tar" \
          --cohort "$runtime/navigation-offline-cohorts-20260907-v1/$task/cohort.json" \
          --data-root "$runtime/navigation-assets-20260907-v1/extracted/$task/$folder" \
          --baseline "$runtime/navigation-offline-20260907-v1/$task/$precision" \
          --fit "$runtime/navigation-fits-20260907-v1/$precision/$task/$category" \
          --precision "$precision" --shard-count "$shards" --shard-index "$index" --output "$output"
      done
    done
  done
done

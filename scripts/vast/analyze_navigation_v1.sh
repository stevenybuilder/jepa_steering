#!/usr/bin/env bash
# CPU-only completed-scope verification alongside the exclusive Indiana GPU queue.
set -eu
runtime=/workspace/jepa-runtime
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$runtime/navigation-analysis-code-v1/src:/workspace/jepa-python/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/opt/conda/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for category in vision_action_coupling action_response_geometry operator_rank distribution_layer distribution_spatial; do
  shards=8
  case "$category" in operator_rank|distribution_layer|distribution_spatial) shards=32 ;; esac
  for precision in bfloat16 float32; do
    for task in wall pointmaze; do
      root="$runtime/navigation-comparisons-20260907-v1/$precision/$task/$category"
      for ((index=0; index<shards; index++)); do
        while test ! -f "$root/shard-$index/DONE.json"; do
          test ! -f "$root/shard-$index/FAILED.json" || exit 3
          sleep 30
        done
      done
      /workspace/jepa-planning-python/bin/python -u -m offline_study.author_analyze \
        --cohort "$runtime/navigation-offline-cohorts-20260907-v1/$task/cohort.json" \
        --fit "$runtime/navigation-fits-20260907-v1/$precision/$task/$category" \
        --shards "$root" --shard-count "$shards" \
        --output "$runtime/navigation-analysis-20260907-v1/$precision/$task/$category"
    done
  done
done

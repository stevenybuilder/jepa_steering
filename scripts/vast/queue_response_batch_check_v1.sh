#!/usr/bin/env bash
# Owned50189244 GPU0, exclusively after the already reserved navigation job.
set -eu
runtime=/workspace/jepa-runtime
until test -f "$runtime/navigation-native-smoke-20260907-v2/pointmaze/DONE.json"; do
  test ! -f "$runtime/navigation-native-smoke-20260907-v2/pointmaze/FAILED.json" || exit 3
  sleep 15
done
while pgrep -f '^/workspace/jepa-maze-python/bin/python -u -m offline_study.navigation_smoke' >/dev/null; do
  sleep 1
done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || exit 4
exec env CUDA_VISIBLE_DEVICES=0 JEPA_VERIFIED_LOCAL_DINO=1 \
  PYTHONPATH="$runtime/code-v12/src" TORCH_HOME="$runtime/rank-prefix-inputs-v1/.cache/torch" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/jepa-droid-python/bin/python -u -m offline_study.support_batch_check \
  --vendor /workspace/jepa_steering/vendor/jepa-wms \
  --inputs "$runtime/rank-prefix-inputs-v1" \
  --output "$runtime/response-batch-check-20260907-v1"

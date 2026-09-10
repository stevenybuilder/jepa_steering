#!/usr/bin/env bash
# Instance50189244 exclusive GPU0 next jobs; no concurrent experiment sharing.
set -u
runtime=/workspace/jepa-runtime
until test -f "$runtime/droid-native-replication-20260907-v1/shard-all/DONE.json"; do
  test ! -f "$runtime/droid-native-replication-20260907-v1/shard-all/FAILED.json" || exit 3
  sleep 15
done
while pgrep -f '^/workspace/jepa-droid-python/bin/python -u -m offline_study.droid_replication' >/dev/null; do
  sleep 1
done
gpu_free() {
  test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
}
gpu_free || exit 4
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$runtime/code-v11/src" \
  TORCH_HOME="$runtime/rank-prefix-inputs-v1/.cache/torch" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/jepa-droid-python/bin/python -u -m offline_study.training_pilot \
  --vendor /workspace/jepa_steering/vendor/jepa-wms \
  --assets "$runtime/navigation-assets-20260907-v1" \
  --input-check "$runtime/navigation-input-check-20260907-v1" \
  --output "$runtime/wall-training-accumulation-pilot-20260907-v2" --task wall
pilot_status=$?
echo "Training engineering exit status: $pilot_status. Next task is an independent native navigation check."
# A training engineering failure is preserved; it does not veto an independently
# validated simulator/model check. Its own CPU gate and exclusive GPU are required.
until test -f "$runtime/pointmaze-env-check-50189244-v2/DONE.json"; do
  test ! -f "$runtime/pointmaze-env-check-50189244-v2/FAILED.json" || exit 5
  sleep 15
done
gpu_free || exit 6
exec env CUDA_VISIBLE_DEVICES=0 MUJOCO_PY_FORCE_CPU=1 JEPA_VERIFIED_LOCAL_DINO=1 \
  PYTHONPATH="$runtime/code-v37/src:/workspace/jepa-maze-python/lib/python3.10/site-packages:/workspace/jepa-planning-python/lib/python3.10/site-packages:/workspace/jepa-python/lib/python3.10/site-packages" \
  LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/opt/conda/lib \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_VIDEODRIVER=dummy \
  /workspace/jepa-maze-python/bin/python -u -m offline_study.navigation_smoke \
  --vendor /workspace/jepa_steering/vendor/jepa-wms --task pointmaze \
  --checkpoint "$runtime/navigation-assets-20260907-v1/downloads/model/jepa_wm_pointmaze.pth.tar" \
  --output "$runtime/navigation-native-smoke-20260907-v2/pointmaze"

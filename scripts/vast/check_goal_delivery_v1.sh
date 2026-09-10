#!/usr/bin/env bash
set -eu
runtime=/workspace/jepa-runtime
test -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" || exit 4
export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl
export PYTHONPATH="$runtime/goal-delivery-code-v1/src:/workspace/jepa-python/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/opt/conda/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /workspace/jepa-planning-python/bin/python -u -m offline_study.verify_planning_goal_bank \
  --vendor /workspace/jepa_steering/vendor/jepa-wms \
  --freeze "$runtime/behavioral-development-freeze-20260907-v1" \
  --reference-root "$runtime/behavioral-development-20260907-v1" \
  --goal-bank "$runtime/metaworld-native-goal-bank-20260907-v1" \
  --output "$runtime/metaworld-goal-delivery-check-20260907-v1"

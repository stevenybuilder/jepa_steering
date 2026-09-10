#!/usr/bin/env bash
set -euo pipefail
cd /workspace/decision
decision_output="${1:-artifacts/offline_study/decision-diagnostic-20260910-v2}"
export PYTHONPATH=/workspace/decision/src
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JEPA_VERIFIED_LOCAL_DINO=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export JEPAWM_DSET=/workspace/decision-runtime/data
export JEPAWM_LOGS=/workspace/decision-runtime/logs
export JEPAWM_HOME=/workspace/decision/vendor
export JEPAWM_CKPT=/workspace/decision-runtime/checkpoints
mkdir -p /workspace/decision-runtime/logs
/workspace/decision-python/bin/python vendor/jepa-wms/setup_macros.py
/workspace/decision-python/bin/python -m pip freeze > /workspace/decision-packages.txt 2>&1 || true
timeout --signal=TERM 10800 /workspace/decision-python/bin/python -u -m offline_study.decision_runtime run \
  --vendor vendor/jepa-wms \
  --fits artifacts/offline_study/fixed-response-20260908-v1/fits \
  --stimuli artifacts/offline_study/restored-behavioral-inputs-20260908-v1 \
  --original artifacts/offline_study/primary-durable-20260907 \
  --output "${decision_output}" \
  --checkpoint /workspace/decision-runtime/checkpoints/jepa_wm_metaworld.pth.tar \
  --max-seconds 10500
/workspace/decision-python/bin/python -m offline_study.decision_diagnostic \
  "${decision_output}"

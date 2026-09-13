#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
export MUJOCO_PY_FORCE_CPU=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PREP_RUNTIME=/workspace/preparation-runtime
git clone --no-checkout https://github.com/Farama-Foundation/D4RL.git "$PREP_RUNTIME/D4RL"
git -C "$PREP_RUNTIME/D4RL" checkout --detach 89141a689b0353b0dac3da5cba60da4b1b16254d
uv pip install --python /workspace/preparation-python/bin/python --no-deps "$PREP_RUNTIME/D4RL"
curl --fail --location --retry 2 --connect-timeout 20 --max-time 180 \
  https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
  -o "$PREP_RUNTIME/mujoco210-linux-x86_64.tar.gz"
tar -xzf "$PREP_RUNTIME/mujoco210-linux-x86_64.tar.gz" -C "$PREP_RUNTIME"
export MUJOCO_PY_MUJOCO_PATH="$PREP_RUNTIME/mujoco210"
export LD_LIBRARY_PATH="$PREP_RUNTIME/mujoco210/bin:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
/workspace/preparation-python/bin/python -c \
  'import mujoco_py; from d4rl import offline_env; print("LEGACY_POINTMAZE_IMPORTS_OK")'
touch "$PREP_RUNTIME/POINTMAZE_READY"

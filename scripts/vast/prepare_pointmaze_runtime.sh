#!/usr/bin/env bash
# Isolated, CPU-only restoration of the upstream legacy simulator dependency.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
export MUJOCO_PY_FORCE_CPU=1
export D4RL_SUPPRESS_IMPORT_ERROR=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DEBIAN_FRONTEND=noninteractive
POINTMAZE_REPAIR=/workspace/table-completion-20260911-v1/pointmaze-runtime-repair-v1
test -d "$POINTMAZE_REPAIR"
test ! -e "$POINTMAZE_REPAIR/READY.json"
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=
# Add the missing compiler/OSMesa packages, without upgrading installed packages.
apt-get -o Acquire::ForceIPv4=true -o Acquire::https::Timeout=25 -o Acquire::Retries=1 \
  install -y --no-upgrade --no-install-recommends build-essential libosmesa6-dev patchelf
/opt/conda/bin/uv venv --system-site-packages --python /workspace/table-python-inherited/bin/python \
  "$POINTMAZE_REPAIR/python"
/opt/conda/bin/uv pip install --python "$POINTMAZE_REPAIR/python/bin/python" \
  --index-url https://pypi.org/simple 'cython==0.29.37' 'mujoco-py==2.1.2.14' \
  'gym==0.23.1' 'numpy==1.26.4' 'fasteners==0.20' 'patchelf==0.17.2.4' h5py
# A nested venv inherits the base interpreter, not the parent venv packages.
# Keep this shim after the repair's own pinned packages in sys.path.
install -m 644 "$POINTMAZE_REPAIR/pointmaze_parent.pth" \
  "$POINTMAZE_REPAIR/python/lib/python3.11/site-packages/pointmaze_parent.pth"
git clone --no-checkout https://github.com/Farama-Foundation/D4RL.git "$POINTMAZE_REPAIR/D4RL"
git -C "$POINTMAZE_REPAIR/D4RL" checkout --detach 89141a689b0353b0dac3da5cba60da4b1b16254d
/opt/conda/bin/uv pip install --python "$POINTMAZE_REPAIR/python/bin/python" \
  --no-deps --index-url https://pypi.org/simple "$POINTMAZE_REPAIR/D4RL"
curl --fail --location --retry 2 --connect-timeout 30 --max-time 300 \
  https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
  -o "$POINTMAZE_REPAIR/mujoco210-linux-x86_64.tar.gz"
tar -xzf "$POINTMAZE_REPAIR/mujoco210-linux-x86_64.tar.gz" -C "$POINTMAZE_REPAIR"
export MUJOCO_PY_MUJOCO_PATH="$POINTMAZE_REPAIR/mujoco210"
export LD_LIBRARY_PATH="$POINTMAZE_REPAIR/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/opt/conda/lib"
"$POINTMAZE_REPAIR/python/bin/python" -c 'import mujoco_py; from d4rl import offline_env; print("LEGACY_SIMULATOR_IMPORT_PASSED")'

#!/usr/bin/env bash
# New preparation-only runtime. No weights, intervention fitting or policy calls.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
PREP_ROOT=/workspace/protected-prep
PREP_RUNTIME=/workspace/preparation-runtime
mkdir -p "$PREP_RUNTIME"
apt-get update -qq
apt-get install -y --no-install-recommends build-essential libosmesa6-dev libgl1-mesa-dev \
  libglfw3 libegl1 libglib2.0-0 patchelf ffmpeg git rsync curl
python -m pip install 'uv==0.8.15'
uv venv --system-site-packages --python "$(command -v python)" /workspace/preparation-python
uv pip install --python /workspace/preparation-python/bin/python \
  'torch==2.7.1' 'torchvision==0.22.1' \
  'numpy==1.26.4' 'gym==0.23.1' 'gymnasium==1.3.0' 'metaworld==3.1.1' 'mujoco==3.3.0' \
  'pygame==2.6.1' 'pymunk==6.8.0' 'tensordict==0.9.1' 'torchrl==0.9.2' 'timm==1.0.19' \
  'decord==0.6.0' 'cython==0.29.37' 'mujoco-py==2.1.2.14' 'fasteners==0.20' \
  scipy pandas h5py einops datasets opencv-python-headless pillow imageio imageio-ffmpeg \
  moviepy mediapy lpips matplotlib seaborn plotly termcolor hydra-core \
  hydra-submitit-launcher omegaconf wandb tqdm submitit clusterscope ruamel.yaml nevergrad \
  shapely huggingface-hub pyyaml scikit-image
export PYTHONPATH="$PREP_ROOT/src:$PREP_ROOT/vendor/jepa-wms"
export JEPAWM_DSET="$PREP_RUNTIME/data"
export JEPAWM_LOGS="$PREP_RUNTIME/logs"
export JEPAWM_HOME="$PREP_ROOT/vendor"
export JEPAWM_CKPT="$PREP_RUNTIME/checkpoints-not-downloaded"
mkdir -p "$JEPAWM_DSET" "$JEPAWM_LOGS"
/workspace/preparation-python/bin/python "$PREP_ROOT/vendor/jepa-wms/setup_macros.py"
/workspace/preparation-python/bin/python -c \
  'import decord, torch, mujoco, metaworld, h5py; print("COMMON_PREPARATION_IMPORTS_OK", torch.__version__)'
touch "$PREP_RUNTIME/COMMON_READY"
# The legacy compile proceeds independently from other input-preparation jobs.
bash "$PREP_ROOT/scripts/vast/bootstrap_protected_pointmaze.sh"

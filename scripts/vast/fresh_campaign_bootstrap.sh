#!/usr/bin/env bash
# Run only on an owned Linux receiving worker, after archive checksum verification.
set -euo pipefail
campaign_root=${1:?Pass absolute extracted campaign root}
export FRESH_EXPECTED_GPUS=${2:-8}
cd "$campaign_root"
python scripts/vast/fresh_campaign_stage.py --verify "$campaign_root"
export DEBIAN_FRONTEND=noninteractive
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
apt-get update -qq
apt-get install -y --no-install-recommends build-essential libosmesa6-dev libgl1-mesa-dev libglfw3 libegl1 libglib2.0-0 patchelf ffmpeg git rsync
python -m pip install 'uv==0.8.15'
if [ ! -x "$campaign_root/python/bin/python" ]; then
 uv venv --system-site-packages --python "$(command -v python)" "$campaign_root/python"
fi
uv pip install --python "$campaign_root/python/bin/python" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
 'numpy==1.26.4' 'gym==0.23.1' 'gymnasium==1.3.0' 'metaworld==3.1.1' 'mujoco==3.3.0' \
 'pygame==2.6.1' 'pymunk==6.8.0' 'tensordict==0.9.1' 'torchrl==0.9.2' 'timm==1.0.19' \
 'decord==0.6.0' 'cython==0.29.37' 'mujoco-py==2.1.2.14' 'fasteners==0.20' \
 scipy pandas h5py einops datasets 'opencv-python-headless<4.12' pillow imageio imageio-ffmpeg \
 moviepy mediapy lpips matplotlib seaborn plotly termcolor hydra-core hydra-submitit-launcher \
 omegaconf wandb tqdm submitit clusterscope ruamel.yaml nevergrad shapely huggingface-hub pyyaml scikit-image
tar --keep-old-files --no-same-owner -xzf runtime-cache.tgz
uv pip install --python "$campaign_root/python/bin/python" --no-deps "$campaign_root/preparation-runtime/D4RL"
# Reuse the already compiled Python-3.11 CPU MuJoCo extension, not a new build.
"$campaign_root/python/bin/python" -c 'import importlib.metadata, pathlib, shutil; p=pathlib.Path(importlib.metadata.distribution("mujoco-py").locate_file("mujoco_py")); s=pathlib.Path("mujoco_py"); shutil.copytree(s, p, dirs_exist_ok=True) if s.resolve()!=p.resolve() else None'
export PYTHONPATH="$campaign_root/src:$campaign_root/vendor/jepa-wms"
export JEPAWM_DSET="$campaign_root/data" JEPAWM_LOGS="$campaign_root/logs"
export JEPAWM_HOME="$campaign_root/vendor" JEPAWM_CKPT="$campaign_root/checkpoints"
export MUJOCO_PY_MUJOCO_PATH="$campaign_root/preparation-runtime/mujoco210"
export MUJOCO_PY_FORCE_CPU=1 D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
source "$campaign_root/scripts/vast/fresh_campaign_env.sh" "$campaign_root"
mkdir -p "$JEPAWM_DSET" "$JEPAWM_LOGS" "$JEPAWM_CKPT"
"$campaign_root/python/bin/python" vendor/jepa-wms/setup_macros.py
"$campaign_root/python/bin/python" -c 'import json, os, torch, mujoco, mujoco_py, metaworld; assert torch.__version__ == "2.7.1+cu128"; assert torch.cuda.device_count() == int(os.environ["FRESH_EXPECTED_GPUS"]); json.dump({"runtime_imports_ready":True,"scientific_launch_ready":False},open("RUNTIME_READY.json","x")); print("RUNTIME_IMPORTS_READY_NOT_SCIENTIFIC_CLEARANCE")'
uv pip freeze --python "$campaign_root/python/bin/python" > "$campaign_root/receiving-packages.txt"
"$campaign_root/python/bin/python" scripts/vast/fresh_campaign_remote_assets.py "$campaign_root"

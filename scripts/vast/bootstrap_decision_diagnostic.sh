#!/usr/bin/env bash
set -euo pipefail
# Isolated, newly rented worker ONLY. No old queues, full datasets or refits.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq libegl1 libgl1 libgl1-mesa-dri libglib2.0-0 ffmpeg git rsync
python3 -m pip install --no-cache-dir 'uv==0.8.15'
uv venv --python 3.10 /workspace/decision-python
uv pip install --python /workspace/decision-python/bin/python --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.1' 'torchvision==0.22.1'
uv pip install --python /workspace/decision-python/bin/python \
  'numpy==2.2.6' 'gym==0.23.1' 'gymnasium==1.3.0' 'metaworld==3.1.1' 'mujoco==3.3.0' \
  'pygame==2.6.1' 'pymunk==6.8.0' 'tensordict==0.9.1' 'torchrl==0.9.2' 'timm==1.0.19' \
  scipy pandas h5py einops datasets opencv-python-headless pillow decord imageio imageio-ffmpeg \
  moviepy mediapy lpips 'torchcodec==0.5' matplotlib seaborn plotly termcolor hydra-core \
  hydra-submitit-launcher omegaconf wandb tqdm submitit clusterscope ruamel.yaml nevergrad \
  shapely huggingface-hub pyyaml scikit-image
/workspace/decision-python/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
# Checkpoints are fetched from the immutable release revision, without training data.
mkdir -p /workspace/decision-runtime/checkpoints
/workspace/decision-python/bin/python -c 'from huggingface_hub import hf_hub_download; hf_hub_download("facebook/jepa-wms", "jepa_wm_metaworld.pth.tar", revision="9b9c41ef249466630dbf1a20e78391865d07b3b9", local_dir="/workspace/decision-runtime/checkpoints")'
/workspace/decision-python/bin/python -c 'import torch; torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)'
# Loading is permitted only if this public cache exactly matches the old audit.
PYTHONPATH=/workspace/decision/src /workspace/decision-python/bin/python -c 'from offline_study.model_loader import verified_local_dino_cache; ctx=verified_local_dino_cache(); print(ctx.__enter__()); ctx.__exit__(None,None,None)'

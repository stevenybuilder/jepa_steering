#!/usr/bin/env bash
# Bootstrap a temporary Panel-P worker from the same public PyTorch image as the
# reference box. Data/code are copied separately and verified before any run.
set -euo pipefail

PANELP_BOOTSTRAP_LOG=/root/panelp_bootstrap.log
exec > >(tee -a "$PANELP_BOOTSTRAP_LOG") 2>&1

date -u '+bootstrap_start=%Y-%m-%dT%H:%M:%SZ'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

apt-get update -qq
apt-get install -y -qq \
  libegl1 libgl1 libgl1-mesa-dri libglapi-mesa libegl-mesa0 \
  libxrender1 libxext6 libsm6 ffmpeg rsync git jq >/dev/null

/opt/conda/bin/pip install -q \
  timm==1.0.29 \
  einops==0.8.2 \
  h5py==3.16.0 \
  opencv-python==5.0.0.93 \
  pillow \
  imageio==2.37.4 \
  imageio-ffmpeg \
  lpips==0.1.4 \
  matplotlib==3.11.1 \
  scipy==1.17.1 \
  scikit-learn==1.9.0 \
  scikit-image==0.26.0 \
  seaborn==0.13.2 \
  pyyaml==6.0.2 \
  hydra-core==1.3.6 \
  omegaconf==2.3.1 \
  huggingface_hub==1.30.0 \
  datasets==5.0.1 \
  clusterscope==0.0.32 \
  nevergrad==1.0.12 \
  termcolor==3.3.0 \
  tensordict==0.13.0 \
  torchrl==0.13.3 \
  torchcodec==0.4.0 \
  decord==0.6.0 \
  pandas==3.0.5 \
  numpy==2.2.6 \
  gym==0.23.1 \
  gymnasium==1.3.0 \
  metaworld==3.1.1 \
  mujoco==3.3.0 \
  pygame==2.6.1 \
  pymunk==6.8.0 \
  shapely==2.1.2 \
  pytest

mkdir -p \
  /root/cgs-pilot/code/cgs_pilot \
  /root/cgs-pilot/code/tests \
  /root/cgs-pilot/vendor \
  /root/cgs-pilot/checkpoints/public \
  /root/cgs-pilot/datasets \
  /root/cgs-pilot/artifacts/public_panel \
  /root/.cache/torch/hub/checkpoints

/opt/conda/bin/python - <<'PY'
import json
import platform
from pathlib import Path

import gymnasium
import gym
import metaworld
import matplotlib
import mujoco
import numpy
import torch
import torchvision

record = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchvision": torchvision.__version__,
    "numpy": numpy.__version__,
    "gymnasium": gymnasium.__version__,
    "gym": gym.__version__,
    "metaworld": getattr(metaworld, "__version__", "3.1.1-package"),
    "matplotlib": matplotlib.__version__,
    "mujoco": mujoco.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
Path("/root/PANELP_ENV_READY.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record))
if not record["cuda_available"] or record["gpu"] != "NVIDIA GeForce RTX 3090":
    raise SystemExit("worker does not expose the required RTX 3090")
PY

date -u '+bootstrap_done=%Y-%m-%dT%H:%M:%SZ'

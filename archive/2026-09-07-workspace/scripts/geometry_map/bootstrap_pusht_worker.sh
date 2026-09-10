#!/usr/bin/env bash
# Pinned base image: pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel.
# Preparation only: experiment collectors are dispatched to disjoint roots later.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export MUJOCO_GL=egl SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
task_root=/root/geometry-map-jepawm-pusht-v1
mkdir -p "$task_root"
exec >"$task_root/bootstrap.log" 2>&1
date -u
apt-get update -qq
apt-get install -y -qq git rsync libegl1 libgl1 libglvnd0 libopengl0 libglib2.0-0 ffmpeg
python -m pip install --no-cache-dir 'numpy==2.2.6' 'einops==0.8.2' 'omegaconf==2.3.1' 'hydra-core==1.3.6' 'hydra-submitit-launcher==1.2.0' 'timm==1.0.29' 'torch==2.7.1' 'torchvision==0.22.1' 'tensordict==0.13.0' 'torchrl==0.13.3' 'mujoco==3.3.0' 'metaworld==3.1.1' 'gym==0.23.1' 'gymnasium==1.3.0' 'pygame==2.6.1' 'pymunk==6.8.0' 'shapely==2.1.2' 'opencv-python-headless==5.0.0.93' 'huggingface_hub==1.30.0' 'decord==0.6.0' 'lpips==0.1.4' 'seaborn==0.13.2' 'scikit-image==0.26.0' 'scikit-learn==1.9.0' 'datasets==5.0.1' 'wandb==0.29.0' 'h5py==3.16.0' 'nevergrad==1.0.12' 'clusterscope==0.0.32' 'submitit==1.5.4' 'termcolor==3.3.0' 'imageio-ffmpeg==0.6.0' 'ruamel.yaml==0.18.12'
mkdir -p "$task_root/vendor"
git clone https://github.com/facebookresearch/jepa-wms.git "$task_root/vendor/jepa-wms"
git -C "$task_root/vendor/jepa-wms" checkout --detach 13cf1d9c7e476f53c17714d2e0f1dc239a883ce0
python -m pip install --no-deps --ignore-requires-python -e "$task_root/vendor/jepa-wms"
cd "$task_root/vendor/jepa-wms"
python - <<'PY'
import json
from pathlib import Path
import torch
import mujoco.egl
from huggingface_hub import hf_hub_download
from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.eval import init_module
assert torch.cuda.is_available(), 'CUDA unavailable: do not dispatch experiments'
assert (torch.ones(2, device='cuda') + 1).sum().item() == 4
p = hf_hub_download('facebook/jepa-wms', 'jepa_wm_pusht.pth.tar', revision='9b9c41ef249466630dbf1a20e78391865d07b3b9')
Path('/root/geometry-map-jepawm-pusht-v1/RUNTIME_READY.json').write_text(json.dumps({'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'checkpoint':p,'note':'imports and CUDA passed; full collector canary still required'}))
print('RUNTIME_READY', flush=True)
PY

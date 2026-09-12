#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export UV_HTTP_TIMEOUT=180
export UV_HTTP_RETRIES=3
if ! dpkg-query -W libegl1 libgl1 libgl1-mesa-dri libglib2.0-0 ffmpeg git rsync >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq libegl1 libgl1 libgl1-mesa-dri libglib2.0-0 ffmpeg git rsync
fi
python3 -m pip install --no-cache-dir 'uv==0.8.15'
# The rented image already contains the exact Torch/CUDA build. Reuse that
# stack instead of redownloading gigabytes of identical libraries. Python 3.11
# differs from the historical 3.10 worker; full receiving-native parity remains
# mandatory before any scientific continuation, and must not be bypassed.
if [ ! -x /workspace/table-python-inherited/bin/python ]; then
  uv venv --system-site-packages --python /opt/conda/bin/python3 /workspace/table-python-inherited
fi
/workspace/table-python-inherited/bin/python -c 'import torch; assert torch.__version__ == "2.7.1+cu128"; assert torch.cuda.is_available()'
# pip honors inherited packages here; uv's resolver tried to replace the
# inherited CUDA build with PyPI's other build. Explicit local-build pins make
# that accidental replacement fail instead of silently changing CUDA versions.
/workspace/table-python-inherited/bin/python -m pip install --timeout 180 \
  'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' \
  'numpy==2.2.6' 'gym==0.23.1' 'gymnasium==1.3.0' 'metaworld==3.1.1' 'mujoco==3.3.0' \
  'pygame==2.6.1' 'pymunk==6.8.0' 'tensordict==0.9.1' 'torchrl==0.9.2' 'timm==1.0.19' \
  scipy pandas h5py einops datasets opencv-python-headless pillow decord imageio imageio-ffmpeg \
  moviepy mediapy lpips 'torchcodec==0.5' matplotlib seaborn plotly termcolor hydra-core \
  hydra-submitit-launcher omegaconf wandb tqdm submitit clusterscope ruamel.yaml nevergrad \
  shapely huggingface-hub pyyaml scikit-image
export PYTHONPATH=/workspace/table-completion-20260911-v1/code/src
export JEPAWM_DSET=/workspace/table-completion-20260911-v1/data
export JEPAWM_LOGS=/workspace/table-completion-20260911-v1/logs
export JEPAWM_HOME=/workspace/table-completion-20260911-v1/code/vendor
export JEPAWM_CKPT=/workspace/table-completion-20260911-v1/checkpoints
mkdir -p "$JEPAWM_DSET" "$JEPAWM_LOGS" "$JEPAWM_CKPT"
/workspace/table-python-inherited/bin/python /workspace/table-completion-20260911-v1/code/vendor/jepa-wms/setup_macros.py
/workspace/table-python-inherited/bin/python -c 'import torch; assert torch.__version__ == "2.7.1+cu128"; assert torch.cuda.is_available(); print("GPU_RUNTIME",torch.__version__,torch.cuda.device_count())'
uv pip freeze --python /workspace/table-python-inherited/bin/python > /workspace/table-completion-20260911-v1/packages.txt
/workspace/table-python-inherited/bin/python - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
import os
import torch
def fetch(task):
    return hf_hub_download('facebook/jepa-wms', 'jepa_wm_'+task+'.pth.tar',
        revision='9b9c41ef249466630dbf1a20e78391865d07b3b9', local_dir=os.environ['JEPAWM_CKPT'])
with ThreadPoolExecutor(max_workers=2) as pool:
    for name in pool.map(fetch, ('wall', 'pointmaze', 'pusht')):
        print('CHECKPOINT_DOWNLOADED',name,flush=True)
torch.hub.load('facebookresearch/dinov2','dinov2_vits14',trust_repo=True)
from offline_study.model_loader import verified_local_dino_cache
with verified_local_dino_cache() as proof:
    print('DINO_VERIFIED',proof,flush=True)
PY

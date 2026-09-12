#!/usr/bin/env bash
set -euo pipefail
task_root=/workspace/metaworld-components-20260911-v1
export DEBIAN_FRONTEND=noninteractive
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=
if [ -f /etc/apt/sources.list ]; then
  sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list
fi
apt-get -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=25 -o Acquire::Retries=1 update -qq
apt-get -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=25 -o Acquire::Retries=1 install -y -qq libegl1 libgl1 libgl1-mesa-dri libglib2.0-0 ffmpeg git
python3 -m pip install --no-cache-dir 'uv==0.8.15'
uv venv --system-site-packages --python /opt/conda/bin/python /workspace/component-python
/workspace/component-python/bin/python -m pip install --timeout 60 --retries 2 \
 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' 'numpy==2.2.6' \
 'gym==0.23.1' 'gymnasium==1.3.0' 'metaworld==3.1.1' 'mujoco==3.3.0' \
 'pygame==2.6.1' 'pymunk==6.8.0' 'tensordict==0.9.1' 'torchrl==0.9.2' 'timm==1.0.19' \
 scipy pandas h5py einops datasets opencv-python-headless pillow decord imageio imageio-ffmpeg \
 moviepy mediapy lpips 'torchcodec==0.5' matplotlib seaborn plotly termcolor hydra-core \
 hydra-submitit-launcher omegaconf wandb tqdm submitit clusterscope ruamel.yaml nevergrad \
 shapely huggingface-hub pyyaml scikit-image
/workspace/component-python/bin/python -m pip freeze > "$task_root/packages.txt"

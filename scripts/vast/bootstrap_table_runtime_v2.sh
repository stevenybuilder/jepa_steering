#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_EXTRA_INDEX_URL=
# Mechanical mirror-transport correction inside this exclusively owned container.
# Ubuntu signature verification and package identities are unchanged.
sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list
dpkg --configure -a
apt-get -o Acquire::ForceIPv4=true -o Acquire::https::Timeout=25 -o Acquire::Retries=1 install -y -qq libegl1 libgl1 libgl1-mesa-dri libglib2.0-0 ffmpeg git rsync
python3 -m pip install --index-url https://pypi.org/simple 'uv==0.8.15'
if [ ! -x /workspace/table-python-inherited/bin/python ]; then
  uv venv --system-site-packages --python /opt/conda/bin/python3 /workspace/table-python-inherited
fi
/workspace/table-python-inherited/bin/python -m pip install --index-url https://pypi.org/simple --timeout 45 \
 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' 'numpy==2.2.6' \
 'gym==0.23.1' 'gymnasium==1.3.0' 'metaworld==3.1.1' 'mujoco==3.3.0' \
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
/workspace/table-python-inherited/bin/python -c 'import torch; assert torch.__version__=="2.7.1+cu128"; assert torch.cuda.is_available()'
uv pip freeze --python /workspace/table-python-inherited/bin/python > /workspace/table-completion-20260911-v1/packages-recovery-v2.txt

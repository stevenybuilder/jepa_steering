#!/usr/bin/env bash
# OS libraries only. Never resolves, installs, upgrades, or removes Python packages.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
/opt/conda/bin/python -c 'import sys,torch,torchvision; assert sys.version_info[:2]==(3,11); assert torch.__version__=="2.7.1+cu128"; assert torchvision.__version__=="0.22.1+cu128"; print("BASE_IMAGE_EXACT")'
apt-get -o DPkg::Lock::Timeout=120 -o Acquire::Retries=1 update -qq
apt-get -o DPkg::Lock::Timeout=120 -o Acquire::Retries=1 install -y --no-install-recommends libosmesa6 libgl1 libglfw3 libegl1 libglib2.0-0 ffmpeg patchelf
echo RESTORE_OS_LIBRARIES_READY_NO_PYTHON_PACKAGE_CHANGES

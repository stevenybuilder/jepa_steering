#!/usr/bin/env bash
# Host dependencies only; safe to overlap with the input archive upload.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get -o DPkg::Lock::Timeout=300 update -qq
apt-get -o DPkg::Lock::Timeout=300 install -y --no-install-recommends build-essential libosmesa6-dev libgl1-mesa-dev libglfw3 libegl1 libglib2.0-0 patchelf ffmpeg git rsync
python -m pip install 'uv==0.8.15'
echo DEPENDENCY_PREPARATION_COMPLETE_NO_MODEL_CALLS

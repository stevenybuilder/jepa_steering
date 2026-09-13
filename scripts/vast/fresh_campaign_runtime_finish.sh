#!/usr/bin/env bash
set -euo pipefail
campaign_root=${1:?Pass campaign root}
export FRESH_EXPECTED_GPUS=${2:?Pass expected GPU count}
cd "$campaign_root"
source scripts/vast/fresh_campaign_env.sh "$campaign_root"
python -c 'import os,torch,torchvision; assert torch.__version__=="2.7.1+cu128"; assert torchvision.__version__=="0.22.1+cu128"; assert torch.cuda.device_count()==int(os.environ["FRESH_EXPECTED_GPUS"]);print("FROZEN_TORCH_CONFIRMED",torch.__file__)'
mkdir -p "$JEPAWM_DSET" "$JEPAWM_LOGS" "$JEPAWM_CKPT"
python vendor/jepa-wms/setup_macros.py
python -c 'import json,mujoco,metaworld;json.dump({"runtime_imports_ready":True,"tasks":["reach","reach-wall","wall"],"scientific_launch_ready":False},open("RUNTIME_READY.json","x"));print("METAWORLD_RUNTIME_READY")'
python -c 'import importlib.metadata,pathlib,shutil;p=pathlib.Path(importlib.metadata.distribution("mujoco-py").locate_file("mujoco_py"));s=pathlib.Path("mujoco_py");shutil.copytree(s,p,dirs_exist_ok=True) if s.resolve()!=p.resolve() else None'
python -c 'import json,mujoco_py;from d4rl import offline_env;json.dump({"runtime_imports_ready":True,"scientific_launch_ready":False},open("POINTMAZE_RUNTIME_READY.json","x"));print("POINTMAZE_RUNTIME_READY")' > pointmaze-runtime.log 2>&1
uv pip freeze --python "$campaign_root/python/bin/python" > receiving-packages.txt

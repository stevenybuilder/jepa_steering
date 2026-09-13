#!/usr/bin/env bash
# Restore an already hash-verified archive; no scientific execution or pip changes.
set -euo pipefail
campaign_root=/workspace/fresh-four-20260912-v1
archive=/workspace/prepared-runtime-inputs-v2.tgz
expected_gpus=${1:?Pass the smoke-verified GPU count}
case "$expected_gpus" in 4|8) ;; *) exit 2 ;; esac
test ! -e "$campaign_root"
echo 'd3712d0a01c7eb19e3f2cf115d33516e00e4ff229d0c6bad8bf1edd5db7d78cf  /workspace/prepared-runtime-inputs-v2.tgz' | sha256sum -c -
mkdir "$campaign_root"
tar --keep-old-files --no-same-owner -xzf "$archive" -C "$campaign_root"
cd "$campaign_root"
# Provider copies can omit this absolute link; tar normally preserves it.
if [ ! -e python/bin/python ] && [ ! -L python/bin/python ]; then
  test -x /opt/conda/bin/python3.11
  ln -s /opt/conda/bin/python3.11 python/bin/python
fi
test -x python/bin/python
mkdir -p /root/.cache/torch
if [ ! -e /root/.cache/torch/hub ] && [ ! -L /root/.cache/torch/hub ]; then
  ln -s "$campaign_root/torch-hub" /root/.cache/torch/hub
fi
source scripts/vast/fresh_campaign_env.sh "$campaign_root"
python -c 'import sys,torch,torchvision; assert sys.version_info[:2]==(3,11); assert torch.__version__=="2.7.1+cu128"; assert torchvision.__version__=="0.22.1+cu128"; assert torch.cuda.device_count()==int(sys.argv[1]); print("RESTORED_BASE_EXACT",torch.__file__)' "$expected_gpus"
python -c 'import hashlib,json; p="fresh-freeze-v2/protocol.json"; f=json.load(open("fresh-freeze-v2/FROZEN.json")); expected="7d5def0122f0dcf80e07e43ddc4f28ef6532fb04e6eb9acc4bc428427165ba90"; assert f["protocol_sha256"]==expected==hashlib.sha256(open(p,"rb").read()).hexdigest(); assert json.load(open(p))["source_sha256"]=="cde8274dad2bbc91efea3c5baba2e2266217949f9993aa77ce695d3bbabd4d82"; print("V2_HASH_BINDINGS_EXACT")'
python vendor/jepa-wms/setup_macros.py
python scripts/run_fresh_confirmation.py validate --project "$campaign_root" --freeze "$campaign_root/fresh-freeze-v2"
python scripts/vast/fresh_campaign_remote_assets.py "$campaign_root"
python -c 'import json,mujoco,metaworld; json.dump({"runtime_imports_ready":True,"scientific_launch_ready":False,"restored_without_pip_changes":True},open("RUNTIME_READY.json","x")); print("METAWORLD_RUNTIME_READY")'
python -c 'import json,mujoco_py; from d4rl import offline_env; json.dump({"runtime_imports_ready":True,"scientific_launch_ready":False,"restored_without_pip_changes":True},open("POINTMAZE_RUNTIME_READY.json","x")); print("POINTMAZE_RUNTIME_READY")'
python -m pip freeze > restored-packages.txt
echo RESTORE_COMPLETE_RECEIVING_ENGINEERING_STILL_REQUIRED

#!/bin/bash
# One-shot MetaWorld worker setup on a fresh pytorch/pytorch:2.7.1-cuda12.8 box.
# Expects /workspace/bootstrap.sh, /workspace/mw-code.tgz, /workspace/mw_slot_driver.sh
# already copied in, and HF_TOKEN in the environment. Args: task:slot ...
set -euo pipefail
ROOT=/workspace/metaworld-components-20260911-v1
LOG=/workspace/setup.log
echo "$(date -u) setup start: $*" >> $LOG
mkdir -p $ROOT/checkpoints
# 1. runtime (10 min) and code unpack in parallel
bash /workspace/bootstrap.sh > /workspace/bootstrap.log 2>&1 &
BOOT=$!
tar xzf /workspace/mw-code.tgz -C $ROOT
# 2. pinned MetaWorld checkpoint
curl -sSL -H "Authorization: Bearer $HF_TOKEN" -o $ROOT/checkpoints/jepa_wm_metaworld.pth.tar \
  "https://huggingface.co/facebook/jepa-wms/resolve/9b9c41ef249466630dbf1a20e78391865d07b3b9/jepa_wm_metaworld.pth.tar?download=1"
echo "c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8  $ROOT/checkpoints/jepa_wm_metaworld.pth.tar" | sha256sum -c >> $LOG
wait $BOOT
/workspace/component-python/bin/python -c 'import torch,mujoco,metaworld; assert torch.cuda.is_available()' >> $LOG 2>&1
echo "$(date -u) runtime ready" >> $LOG
# 3. frozen source hash must match the fleet snapshot
cd $ROOT/code && SRC=$(/workspace/component-python/bin/python -c "import sys; sys.path.insert(0,'src'); from offline_study.author_fit import source_hash; print(source_hash())")
echo "$(date -u) source_hash=$SRC" >> $LOG
mkdir -p $ROOT/ops && cp /workspace/mw_slot_driver.sh $ROOT/ops/ && chmod +x $ROOT/ops/mw_slot_driver.sh
nohup $ROOT/ops/mw_slot_driver.sh "$@" > /workspace/mw-driver.nohup 2>&1 &
echo "$(date -u) DRIVER_LAUNCHED pid=$! slots=$*" >> $LOG

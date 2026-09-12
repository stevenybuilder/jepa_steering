#!/bin/bash
# Multi-GPU host setup for refined panels (Push-T, Wall) plus MetaWorld slots.
# Shared read-only inputs live once per task; each GPU gets its own behavior
# root (engineering + conditions) via symlinks, and one driver per GPU with
# CUDA_VISIBLE_DEVICES. Distinct shards per GPU -> no duplication.
#
# Expects in /workspace: bootstrap.sh, mw-code.tgz, mw_slot_driver.sh,
#   pusht-src.tgz pusht-behavior-inputs.tgz prereq.tar.gz SOURCE.json (Push-T)
#   wall-prereq.tgz wall-behavior.tgz wall-fit-v1.tgz reset_check.py (Wall)
#   refined_shard_driver.py; HF_TOKEN in env.
# Layout args: PUSHT="0:6 1:7"  WALL="2:4 3:5 4:6 5:7"  MW="6:reach-wall:0,reach:0 7:reach-wall:1"
set -euo pipefail
LOG=/workspace/multi-setup.log; M=/workspace/metaworld-components-20260911-v1
PY=/workspace/component-python/bin/python
echo "$(date -u) start PUSHT=[$PUSHT] WALL=[$WALL] MW=[$MW]" >> $LOG
mkdir -p $M/checkpoints && tar xzf /workspace/mw-code.tgz -C $M
if [ -f $M/packages.txt ]; then BOOT=; else bash /workspace/bootstrap.sh > /workspace/bootstrap.log 2>&1 & BOOT=$!; fi
MODEL="https://huggingface.co/facebook/jepa-wms/resolve/9b9c41ef249466630dbf1a20e78391865d07b3b9"
DSET="https://huggingface.co/datasets/facebook/jepa-wms/resolve/main"
# resumable download: retry with -C - until the pinned sha matches ($3 = sha256)
hf () { local url=$1 out=$2 sha=$3 i
  for i in 1 2 3 4 5 6; do
    curl -sSL -C - -H "Authorization: Bearer $HF_TOKEN" -o "$out" "$url?download=1" || true
    if [ -z "$sha" ] || echo "$sha  $out" | sha256sum -c --quiet 2>/dev/null; then echo "$(date -u) verified $out" >> $LOG; return 0; fi
    echo "$(date -u) retry $i for $out ($(stat -c %s "$out" 2>/dev/null) bytes)" >> $LOG; sleep 5
  done; echo "$(date -u) DOWNLOAD_FAILED $out" >> $LOG; return 1; }

# ---------- Push-T shared inputs ----------
if [ -n "${PUSHT:-}" ]; then
  P=/workspace/refined-pusht-prerequisite-20260911-v1; B0=/workspace/pusht-shared
  cd /workspace && tar xzf prereq.tar.gz && mkdir -p $B0 && tar xzf pusht-behavior-inputs.tgz -C /tmp && cp -R /tmp/behavior/* $B0/
  mkdir -p $B0/code/src && tar xzf pusht-src.tgz -C $B0/code/src && ln -sfn $M/code/vendor $B0/code/vendor
  find $B0 $P -name '._*' -delete
  mkdir -p $P/checkpoints; hf $MODEL/jepa_wm_pusht.pth.tar $P/checkpoints/jepa_wm_pusht.pth.tar 9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb
  hf $DSET/pusht/pusht_noise.zip /workspace/pusht_noise.zip 442f5dee246edf670964ed7bdecd248683cd6d00580fa0e4d458abb53f92da08
  rm -rf $P/data/val
  python3 - <<'PY'
import zipfile, pathlib
z=zipfile.ZipFile('/workspace/pusht_noise.zip'); names=z.namelist()
prefix=next(n for n in names if n.endswith('states.pth') and ('/val/' in n or n.startswith('val/'))).rsplit('val/',1)[0]+'val/'
req=['states.pth','rel_actions.pth','velocities.pth','seq_lengths.pkl']+[f'obses/episode_{i:03d}.mp4' for i in range(21)]
if prefix+'shapes.pkl' in names: req.append('shapes.pkl')
dst=pathlib.Path('/workspace/refined-pusht-prerequisite-20260911-v1/data/val')
for r in req:
    o=dst/r; o.parent.mkdir(parents=True,exist_ok=True)
    with z.open(prefix+r) as s, open(o,'wb') as f:
        for c in iter(lambda: s.read(8<<20), b''): f.write(c)
PY
fi
# ---------- Wall shared inputs ----------
if [ -n "${WALL:-}" ]; then
  W=/workspace/refined-nav-wall-prerequisite-20260911-v2; WB0=/workspace/wall-shared
  mkdir -p $W && tar xzf /workspace/wall-prereq.tgz -C /workspace && cp -a /workspace/wall-prereq/. $W/ && rm -rf /workspace/wall-prereq
  rm -rf $W/code/vendor && ln -sfn $M/code/vendor $W/code/vendor
  tar xzf /workspace/wall-fit-v1.tgz -C $W
  mkdir -p $WB0 && tar xzf /workspace/wall-behavior.tgz -C /workspace && cp -a /workspace/wall-behavior/. $WB0/ && rm -rf /workspace/wall-behavior
  mkdir -p $WB0/code/src && cp -a $W/code/src/offline_study $WB0/code/src/ && ln -sfn $M/code/vendor $WB0/code/vendor
  cp -a $W/cohort $WB0/cohort; cp -a $W/fit-v1 $WB0/fit
  find $W $WB0 -name '._*' -delete
fi
[ -n "$BOOT" ] && wait $BOOT; $PY -c 'import torch,mujoco,metaworld; assert torch.cuda.is_available()' >> $LOG 2>&1; echo "$(date -u) runtime ready" >> $LOG

if [ -n "${WALL:-}" ]; then
  W=/workspace/refined-nav-wall-prerequisite-20260911-v2; WB0=/workspace/wall-shared; rm -rf $W/data $W/downloads
  # wall data via HF (checkpoint + cohort subset), verified against pins, using the prereq's own frozen code
  cd $W/code && PYTHONPATH=$W/code/src HF_HUB_DISABLE_PROGRESS_BARS=1 $PY - "$W" <<'PY'
import json, sys, zipfile, shutil, os
from pathlib import Path
root=Path(sys.argv[1]); sys.path.insert(0, str(root/'code/src'))
from offline_study.protocol import sha256
from offline_study.navigation_assets import safe_members
from offline_study.navigation_cohort import verify_inputs
from huggingface_hub import hf_hub_download
spec=json.loads((root/'code/configs/navigation_assets.json').read_text())
for item in [a for a in spec['assets'] if a['task']=='wall']:
    kind=item['repo_type']
    path=Path(hf_hub_download(item['repo_id'], item['filename'], repo_type=None if kind=='model' else kind, revision=item['revision'], token=os.environ['HF_TOKEN'], local_dir=root/'downloads'/kind))
    assert path.stat().st_size==item['size'] and sha256(path)==item['sha256']
    if item['kind']=='checkpoint': path.rename(root/'checkpoint.pth.tar')
    else:
        files=json.loads((root/'cohort/input_files.json').read_text())
        with zipfile.ZipFile(path) as z:
            safe_members(z)
            for name in files:
                t=root/'data'/name; t.parent.mkdir(parents=True, exist_ok=True)
                with z.open('wall_single/'+name) as s, t.open('xb') as o: shutil.copyfileobj(s,o,4<<20)
                assert sha256(t)==files[name]
verify_inputs(root/'cohort/cohort.json', root/'data'); print('wall inputs verified')
PY
fi

launch_refined () {  # task gpu shard
  local task=$1 gpu=$2 shard=$3
  local shared prereq; if [ "$task" = pusht ]; then shared=/workspace/pusht-shared; prereq=/workspace/refined-pusht-prerequisite-20260911-v1; else shared=/workspace/wall-shared; prereq=/workspace/refined-nav-wall-prerequisite-20260911-v2; fi
  local B=/workspace/refined-$task-gpu$gpu; mkdir -p $B
  for d in fit cohort reference reference-source code; do ln -sfn $shared/$d $B/$d; done
  cp /workspace/refined_shard_driver.py $B/ops_driver.py
  local ck; if [ "$task" = pusht ]; then ck=$prereq/checkpoints/jepa_wm_pusht.pth.tar; else ck=$prereq/checkpoint.pth.tar; fi
  local uuid; uuid=$(nvidia-smi -i $gpu --query-gpu=uuid --format=csv,noheader)
  cat > $B/PANEL.json <<JSON
{"task": "$task", "logical_ranks": [0,1,2,3,4,5,6,7], "vendor": "$M/code/vendor/jepa-wms", "checkpoint": "$ck",
 "fit": "$B/fit", "cohort": "$B/cohort/cohort.json", "reference": "$B/reference", "reference-source": "$B/reference-source",
 "data-root": "$prereq/data", "old_solver": false, "physical_gpu_uuid": "$uuid"}
JSON
  cd $B/code
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$B/code/src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $PY -u -m offline_study.refined_task_behavior freeze --task $task \
    --vendor $M/code/vendor/jepa-wms --checkpoint $ck --fit $B/fit --cohort $B/cohort/cohort.json --reference $B/reference \
    --reference-source $B/reference-source --data-root $prereq/data --freeze $B/freeze --output $B/freeze > $B/freeze.log 2>&1 || { echo "$(date -u) FREEZE_FAILED $task gpu$gpu" >> $LOG; tail -5 $B/freeze.log >> $LOG; return 1; }
  if [ "$task" = wall ] && [ ! -f /workspace/wall-reset-parity.ok ]; then
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=$B/code/src $PY /workspace/reset_check.py $B > $B/reset-check.log 2>&1 && touch /workspace/wall-reset-parity.ok && echo "$(date -u) wall RESET_PARITY_OK" >> $LOG || { echo "$(date -u) wall RESET_PARITY_FAILED" >> $LOG; tail -5 $B/reset-check.log >> $LOG; return 1; }
  fi
  nohup env CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$B/code/src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $PY -u $B/ops_driver.py --root $B --shards $shard > $B/driver.log 2>&1 &
  echo "$(date -u) DRIVER $task gpu$gpu shard$shard pid=$!" >> $LOG
}
for spec in ${PUSHT:-}; do launch_refined pusht ${spec%%:*} ${spec##*:} || true; done
for spec in ${WALL:-};  do launch_refined wall  ${spec%%:*} ${spec##*:} || true; done

# ---------- MetaWorld slots ----------
if [ -n "${MW:-}" ]; then
  hf $MODEL/jepa_wm_metaworld.pth.tar $M/checkpoints/jepa_wm_metaworld.pth.tar c3297772c7af4e84c28bd0c0937e22398a888848305014bb3ab3f1508a310bd8
  mkdir -p $M/ops && cp /workspace/mw_slot_driver.sh $M/ops/ && chmod +x $M/ops/mw_slot_driver.sh
  for spec in $MW; do gpu=${spec%%:*}; slots=${spec#*:}; slots=${slots//,/ }
    nohup env CUDA_VISIBLE_DEVICES=$gpu $M/ops/mw_slot_driver.sh $slots > /workspace/mw-driver-gpu$gpu.nohup 2>&1 &
    echo "$(date -u) MW DRIVER gpu$gpu slots=[$slots] pid=$!" >> $LOG
    sleep 90; done   # stagger: concurrent first-time torch.hub DINOv2 loads race on the shared cache dir (seen on 2xL40S)
fi
echo "$(date -u) ALL_LAUNCHED" >> $LOG

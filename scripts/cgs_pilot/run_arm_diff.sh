#!/bin/bash
# run_arm_diff.sh — experiment 1 (CAFT-inspired arm diffing, arm_diff_pca.py) on box 2 (cgs-pilot-2). CPU only (numpy).
#
#   0 self-test   arm_diff_pca.py --self-test (planted + null synthetic dump pairs); aborts on failure.
#   1 dumps       waits for the seed-0 localization dumps re-created by run_identity_geometry.sh (re-queued by
#                 /root/requeue_identity_geometry.sh): $IG/dump_arm{A,B}/activations/index.json with "slimmed_to_step0": true
#                 (the slimming rewrites the npz files first and index.json last), then COPIES each dump into
#                 $OUT/dumps/dump_arm{A,B} at once — run_identity_geometry.sh deletes its activations in its cleanup stage,
#                 so the copy must land inside the window of its analysis stage (5-15 min).  Marker DUMPS_READY.
#                 Fallback (only if IDENTITY_GEOMETRY_DONE arrives without a copy having been possible, i.e. the window was
#                 missed): after PULL_DONE and with the GPU free (< 8 GB used or G2_DONE) the same localize_interaction.py
#                 command as run_identity_geometry.sh re-dumps both arms into $OUT/dumps (logged DUMPS_FALLBACK).
#   2 analysis    arm_diff_pca.py on the common discovery scenes at step 0 (18 sites x 3 groups x 4 subsets x 2 conventions;
#                 n_boot / n_perm 200 transported, 50 raw) -> $OUT/arm_diff_map.{json,md}, $OUT/arm_diff_subspaces.npz.
#                 Marker ARM_DIFF_DONE in $LOG.
# Never touches anything under drive_* except reading; writes only under artifacts/drive_arm_diff/ and logs/.
# Usage: run_arm_diff.sh [--skip-self-test]
# Launch: ssh -n -p 45460 root@70.27.250.55 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_arm_diff.sh > /root/cgs-pilot/logs/drive_arm_diff_driver.out 2>&1 < /dev/null &'
set -u
B=${DRIVE_BASE:-/root/cgs-pilot}; C=${DRIVE_CODE:-$B/code/cgs_pilot}; REPO=$B/vendor/jepa-wms; MPY=${DRIVE_MPY:-/opt/conda/bin/python}
OUT=${ARM_DIFF_OUT:-$B/artifacts/drive_arm_diff}; IG=${DRIVE_IG_OUT:-$B/artifacts/drive_identity_geometry}
MERGED=$B/artifacts/drive_factorial_merged; EVAL=$B/artifacts/drive_eval; MODELS=$B/artifacts/drive_models
LOG=${ARM_DIFF_LOG:-$B/logs/drive_arm_diff.log}; IGLOG=$B/logs/drive_identity_geometry.log; ORCH=$B/logs/box2_orchestrator.log; PL=/root/pull_box1.log
N_BOOT=${ARM_DIFF_N_BOOT:-200}; N_PERM=${ARM_DIFF_N_PERM:-200}
export JEPAWM_HOME=$B/vendor JEPAWM_OSSCKPT=$B/checkpoints JEPAWM_LOGS=$B/logs PYTHONPATH=$REPO:$C JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
SKIP_ST=0; while [ $# -gt 0 ]; do case "$1" in --skip-self-test) SKIP_ST=1;; *) echo "unknown arg $1"; exit 2;; esac; shift; done
ts(){ date -u +%FT%TZ; }; say(){ echo "[$(ts)] $*" | tee -a $LOG; }
gpu_ok(){ grep -q G2_DONE $ORCH 2>/dev/null && return 0; local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); [ -n "$u" ] && [ "$u" -lt 8000 ]; }
mkdir -p $OUT/dumps $B/logs; cd $C
say "ARM_DIFF start out=$OUT ig=$IG n_boot=$N_BOOT n_perm=$N_PERM"

# ---- 0. self-test
if [ $SKIP_ST = 0 ] && [ ! -f $OUT/self_test/self_test_verdict.json ]; then
  $MPY arm_diff_pca.py --self-test --out $OUT/self_test --n-boot 40 --n-perm 40 --seed 0 > $OUT/self_test.log 2>&1
  grep -q SELF_TEST_PASSED $OUT/self_test.log || { say "self-test FAILED (see $OUT/self_test.log); aborting"; exit 1; }
  say "self-test passed: $(grep -o '"passed": [a-z]*' $OUT/self_test/self_test_verdict.json | head -1)"
fi

# ---- 1. dumps: copy the identity-geometry dumps inside their window
slimmed(){ [ -f $1/activations/index.json ] && grep -q '"slimmed_to_step0": true' $1/activations/index.json; }
copied(){ [ -f $1/activations/index.json ] && [ "$(ls $1/activations/*.npz 2>/dev/null | wc -l)" -ge 18 ] && [ -f $1/COPY_OK ]; }
copy_dump(){  # copy_dump SRC DST
  rm -rf $2; mkdir -p $2
  cp -r $1/activations $2/ && [ -f $1/interaction_map.json ] && cp $1/interaction_map.json $2/ 2>/dev/null
  $MPY - $2 <<'EOF' && touch $2/COPY_OK
import json, sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1]) / "activations"; idx = json.loads((d / "index.json").read_text())
n = 0
for p in sorted(d.glob("*.npz")):
    with np.load(p) as z: assert z["acts"].shape[0] == len(idx["cells"]), (p, z["acts"].shape)
    n += 1
assert n >= 18 and idx.get("slimmed_to_step0"), (n, idx.get("slimmed_to_step0"))
print("copy verified:", d.parent, n, "sites,", len(idx["cells"]), "cells, n_steps", idx.get("n_steps"))
EOF
}
if ! (copied $OUT/dumps/dump_armA && copied $OUT/dumps/dump_armB); then
  say "waiting for $IG/dump_arm{A,B} (slimmed) from run_identity_geometry.sh; polling every 30 s"
  n=0; missed=0
  while :; do
    for ARM in A B; do
      if ! copied $OUT/dumps/dump_arm$ARM && slimmed $IG/dump_arm$ARM; then
        say "arm $ARM: slimmed dump present; copying"
        copy_dump $IG/dump_arm$ARM $OUT/dumps/dump_arm$ARM >> $LOG 2>&1 && say "arm $ARM: copied ($(du -sh $OUT/dumps/dump_arm$ARM | cut -f1))" || say "arm $ARM: copy FAILED"
      fi
    done
    copied $OUT/dumps/dump_armA && copied $OUT/dumps/dump_armB && break
    if grep -q IDENTITY_GEOMETRY_DONE $IGLOG 2>/dev/null && ! [ -d $IG/dump_armA/activations ] && ! (copied $OUT/dumps/dump_armA && copied $OUT/dumps/dump_armB); then missed=1; break; fi
    n=$((n + 1)); [ $((n % 60)) = 0 ] && say "still waiting (ig: $(tail -1 $IGLOG 2>/dev/null | cut -c1-120))"
    sleep 30
  done
  if [ $missed = 1 ]; then
    say "DUMPS_MISSED: identity geometry finished and deleted its dumps before a copy was possible; fallback re-dump after PULL_DONE with the GPU free"
    until grep -q PULL_DONE $PL 2>/dev/null; do sleep 60; done
    for ARM in A B; do
      copied $OUT/dumps/dump_arm$ARM && continue
      until gpu_ok; do sleep 60; done
      MD=$MODELS/arm${ARM}_seed0; EV=$EVAL/arm${ARM}_seed0; DUMP=$OUT/dumps/dump_arm$ARM; rm -rf $DUMP; mkdir -p $DUMP
      say "DUMPS_FALLBACK arm $ARM: localize_interaction.py (identical to run_identity_geometry.sh stage 2)"
      $MPY localize_interaction.py --repo $REPO --config $MD/eval_config.yaml --checkpoint $MD/jepa-latest.pth.tar --model-name jepa_wm_driving --artifacts $EV --output $DUMP --domain driving \
        --seeds-file $EV/discovery_seeds.txt --dump-hooks resid_post attn_out mlp_out --dump-groups egg corridor >> $LOG 2>&1
      [ -f $DUMP/activations/index.json ] || { say "fallback dump arm $ARM FAILED"; exit 1; }
      $MPY - $DUMP >> $LOG 2>&1 <<'EOF'
import json, sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1]) / "activations"; idx = json.loads((d / "index.json").read_text())
n_steps = int(idx.get("n_steps") or 0)
for p in sorted(d.glob("*.npz")):
    with np.load(p) as z: arrs = {k: z[k] for k in z.files}
    np.savez(p, **{k: (v[:, :1] if (v.ndim >= 2 and n_steps > 1 and v.shape[1] == n_steps) else v) for k, v in arrs.items()})
for p in d.glob("*.token_index.npy"): p.unlink()
idx.update({"n_steps_original": n_steps, "n_steps": 1 if n_steps else n_steps, "slimmed_to_step0": True, "slim_note": "run_arm_diff.sh fallback: step 0 only (same slimming as run_drive_arms.sh)"})
(d / "index.json").write_text(json.dumps(idx, indent=1) + "\n"); print("slimmed", d.parent)
EOF
      touch $DUMP/COPY_OK
    done
  fi
fi
say "DUMPS_READY $OUT/dumps ($(du -sh $OUT/dumps | cut -f1))"

# ---- 2. analysis
if [ ! -f $OUT/arm_diff_map.json ]; then
  [ -f $MERGED/discovery_seeds_common.txt ] || comm -12 <(sort $MERGED/armA/discovery_seeds.txt) <(sort $MERGED/armB/discovery_seeds.txt) | sort -n > $MERGED/discovery_seeds_common.txt
  say "analysis: arm_diff_pca.py on $(grep -c . $MERGED/discovery_seeds_common.txt) common discovery seeds"
  $MPY arm_diff_pca.py --dump-a $OUT/dumps/dump_armA --dump-b $OUT/dumps/dump_armB --stimulus $MERGED/armA --discovery-seeds $MERGED/discovery_seeds_common.txt \
    --out $OUT --export $OUT/arm_diff_subspaces.npz --steps 0 --n-boot $N_BOOT --n-perm $N_PERM --seed 0 > $OUT/arm_diff.log 2>&1 || { say "arm_diff_pca FAILED (see $OUT/arm_diff.log)"; exit 1; }
  say "analysis done -> $OUT/arm_diff_map.json ($(du -h $OUT/arm_diff_map.json | cut -f1)), $OUT/arm_diff_map.md"
fi
$MPY - $OUT/arm_diff_map.json >> $LOG 2>&1 <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
for key, rows in d["ranked_tables"].items():
    top = rows[:3]
    print("ARM_DIFF", key, [(r["site_id"], round(r.get("transported.all.pc1") or 0, 3), round(r.get("transported.all.conceptor_mass") or 0, 3), r.get("transported.all.conceptor_mass_p_perm")) for r in top])
EOF
echo "[$(ts)] ARM_DIFF_DONE" >> $LOG
say "ARM_DIFF_DONE -> $OUT"

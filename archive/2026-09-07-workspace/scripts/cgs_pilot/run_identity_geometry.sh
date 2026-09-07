#!/bin/bash
# run_identity_geometry.sh — identity-contrast geometry under the shared consequence template (METHOD fix for the seed-0
# cross-arm null, design doc F17 / Step 4 "discovery-field mean template ... applied before any subspace fit").
# Chains AFTER the driving cell and the Panel B token stage; touches nothing under drive_* except reading it.
#
#   0 wait      until logs/drive_cross_arm.log contains CROSS_ARM_DONE AND logs/panelb_gated.log contains
#               PANELB_TOKEN_DONE or PANELB_TOKEN_INCOMPLETE AND >= $MIN_FREE_GB GB are free (df); re-checked every 2 min.
#   1 preflight identity_contrast_geometry.py --self-test (numpy only, ~15 s): planted case detected at every site, null
#               case at none; aborts the run otherwise.
#   2 re-dump   run_drive_arms.sh deletes the pair dumps after the cross-arm stage, so the seed-0 localization dumps of
#               both arms are regenerated EXACTLY as its PAIRS stage does (same localize_interaction.py invocation:
#               --artifacts $EVAL/arm{X}_seed0, --seeds-file $EVAL/arm{X}_seed0/discovery_seeds.txt, hooks resid_post
#               attn_out mlp_out, --dump-groups egg corridor) and slimmed to imagined step 0 with the same slim_dump code
#               -> $IG/dump_arm{A,B}  (2.5 GB per arm before slimming, 0.84 GB after; ~5-10 min per arm on the box GPU).
#   3 analysis  identity_contrast_geometry.py --dump-a $IG/dump_armA --dump-b $IG/dump_armB --stimulus $MERGED/armA
#               --discovery-seeds $MERGED/discovery_seeds_common.txt --out $IG   (18 sites x 3 groups, step 0; ~5-15 min)
#               -> $IG/identity_geometry.json + identity_geometry.md
#   4 cleanup   rm -rf $IG/dump_arm{A,B}/activations (index.json + interaction_map.json kept); marker IDENTITY_GEOMETRY_DONE
#               in $LOG.
# Every stage is skipped when its output is present (relaunch-safe); --force redoes 2-3; --skip-wait skips stage 0.
# Usage:   run_identity_geometry.sh [--skip-wait] [--force]
# Launch (remote; do NOT launch while other GPU jobs must not be disturbed -- the wait gate handles the ordering):
#   ssh -n -p PORT root@HOST 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_identity_geometry.sh > /root/cgs-pilot/logs/identity_geometry_driver.log 2>&1 < /dev/null &'
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}
CODE=${DRIVE_CODE:-$BASE/code/cgs_pilot}
REPO=${DRIVE_REPO:-$BASE/vendor/jepa-wms}
MPY=${DRIVE_MPY:-/opt/conda/bin/python}
OUT=${DRIVE_OUT_ROOT:-$BASE/artifacts}
LOGDIR=${DRIVE_LOGDIR:-$BASE/logs}
MODELS=$OUT/drive_models; MERGED=$OUT/drive_factorial_merged; EVAL=$OUT/drive_eval
IG=${DRIVE_IG_OUT:-$OUT/drive_identity_geometry}
CLOG=$LOGDIR/drive_cross_arm.log
PLOG=$LOGDIR/panelb_gated.log
LOG=${DRIVE_IG_LOG:-$LOGDIR/drive_identity_geometry.log}
MIN_FREE_GB=${DRIVE_IG_MIN_FREE_GB:-8}
SEED=${DRIVE_IG_SEED:-0}
LOC_ARGS="--dump-hooks resid_post attn_out mlp_out"             # identical to run_drive_arms.sh full-mode LOC_ARGS
IG_ARGS=${DRIVE_IG_ARGS:-""}                                     # identity_contrast_geometry.py defaults (n_boot 2000, n_perm 2000)
SKIP_WAIT=0; FORCE=0
while [ $# -gt 0 ]; do case "$1" in --skip-wait) SKIP_WAIT=1;; --force) FORCE=1;; *) echo "unknown arg $1"; exit 2;; esac; shift; done
export JEPAWM_HOME=$BASE/vendor JEPAWM_OSSCKPT=$BASE/checkpoints JEPAWM_LOGS=$BASE/logs PYTHONPATH=$REPO:$CODE
export JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12}
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*" | tee -a $LOG; }
disk() { df -h $BASE | tail -1 | awk '{print "disk used "$3" of "$2", free "$4}'; }
freegb() { df -BG $BASE | tail -1 | awk '{gsub("G","",$4); print $4}'; }
mkdir -p $IG $LOGDIR
cd $CODE || { echo "no code dir $CODE"; exit 1; }
echo "[$(ts)] IDENTITY_GEOMETRY start seed=$SEED out=$IG min_free_gb=$MIN_FREE_GB skip_wait=$SKIP_WAIT force=$FORCE ig_args='$IG_ARGS'; $(disk)" >> $LOG

# ---- 0. wait gate: cross-arm stage done AND Panel B token stage done/incomplete AND disk
gate_ok() {
  grep -q CROSS_ARM_DONE $CLOG 2>/dev/null || return 1
  grep -qE "PANELB_TOKEN_DONE|PANELB_TOKEN_INCOMPLETE" $PLOG 2>/dev/null || return 1
  [ "$(freegb)" -ge "$MIN_FREE_GB" ] || return 1
  return 0
}
if [ $SKIP_WAIT != 1 ]; then
  say "waiting: CROSS_ARM_DONE in $CLOG, PANELB_TOKEN_DONE|PANELB_TOKEN_INCOMPLETE in $PLOG, >= $MIN_FREE_GB GB free"
  n=0
  until gate_ok; do
    n=$((n + 1)); [ $((n % 30)) = 0 ] && say "still waiting (cross_arm=$(grep -c CROSS_ARM_DONE $CLOG 2>/dev/null) panelb=$(grep -cE 'PANELB_TOKEN_DONE|PANELB_TOKEN_INCOMPLETE' $PLOG 2>/dev/null) free=$(freegb)G)"
    sleep 120
  done
fi
say "gate open: $(grep -m1 CROSS_ARM_DONE $CLOG 2>/dev/null | cut -c1-40) | $(grep -m1 -E 'PANELB_TOKEN_DONE|PANELB_TOKEN_INCOMPLETE' $PLOG 2>/dev/null | cut -c1-60) | $(disk)"
say "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)"

# ---- 1. preflight self-test (numpy only)
if [ ! -f $IG/self_test/verdict.json ] || [ $FORCE = 1 ]; then
  $MPY identity_contrast_geometry.py --self-test --out $IG --n-boot 300 --n-perm 1000 --seed $SEED > $IG/self_test.log 2>&1
  grep -q SELF_TEST_PASSED $IG/self_test.log || { say "preflight self-test FAILED (see $IG/self_test.log); aborting"; exit 1; }
  say "preflight self-test passed: $(grep -o '"passed": [a-z]*' $IG/self_test/verdict.json | head -1)"
fi

# ---- 2. re-dump both arms at seed $SEED exactly as run_drive_arms.sh PAIRS does, then slim to step 0
slim_dump() {   # slim_dump DUMP : keep imagined step 0 only in activations/*.npz (+ index.json n_steps=1); idempotent (verbatim from run_drive_arms.sh)
  $MPY - $1 >> $LOG 2>&1 <<'EOF'
import json, sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1]) / "activations"; idx = json.loads((d / "index.json").read_text())
if idx.get("slimmed_to_step0"): print("slim: already step-0 only"); sys.exit(0)
n_steps = int(idx.get("n_steps") or 0); before = after = 0
for p in sorted(d.glob("*.npz")):
    before += p.stat().st_size
    with np.load(p) as z: arrs = {k: z[k] for k in z.files}
    out = {k: (v[:, :1] if (v.ndim >= 2 and n_steps > 1 and v.shape[1] == n_steps) else v) for k, v in arrs.items()}
    np.savez(p, **out); after += p.stat().st_size
for p in d.glob("*.token_index.npy"):  # memmap leftovers of the writer, if any
    p.unlink()
idx.update({"n_steps_original": n_steps, "n_steps": 1 if n_steps else n_steps, "slimmed_to_step0": True, "slim_note": "run_identity_geometry.sh: activations restricted to imagined step 0 (disk), same slimming as run_drive_arms.sh"})
(d / "index.json").write_text(json.dumps(idx, indent=1) + "\n")
print(f"slim: {d.parent}: {before/1e6:.0f} MB -> {after/1e6:.0f} MB, n_steps {n_steps} -> 1")
EOF
}
if [ ! -f $IG/identity_geometry.json ] || [ $FORCE = 1 ]; then
  for ARM in A B; do
    MD=$MODELS/arm${ARM}_seed$SEED; EV=$EVAL/arm${ARM}_seed$SEED; DUMP=$IG/dump_arm$ARM
    for f in $MD/eval_config.yaml $MD/jepa-latest.pth.tar $EV/discovery_seeds.txt; do [ -f $f ] || { say "missing $f"; exit 1; }; done
    if [ -f $DUMP/activations/index.json ] && [ $FORCE = 0 ]; then say "arm $ARM seed $SEED: dump present, reused"; slim_dump $DUMP; continue; fi
    [ "$(freegb)" -ge "$MIN_FREE_GB" ] || { say "arm $ARM: only $(freegb) GB free before the dump (< $MIN_FREE_GB); aborting"; exit 1; }
    rm -rf $DUMP; mkdir -p $DUMP
    say "arm $ARM seed $SEED: localization dump (identity-contrast geometry only; no mechanism claim) -> $DUMP"
    $MPY localize_interaction.py --repo $REPO --config $MD/eval_config.yaml --checkpoint $MD/jepa-latest.pth.tar --model-name jepa_wm_driving --artifacts $EV --output $DUMP --domain driving \
      --seeds-file $EV/discovery_seeds.txt $LOC_ARGS --dump-groups egg corridor >> $LOG 2>&1
    [ -f $DUMP/activations/index.json ] || { say "arm $ARM seed $SEED: localization dump FAILED (see $LOG)"; exit 1; }
    say "arm $ARM seed $SEED: dump done ($(du -sh $DUMP | cut -f1), $(ls $DUMP/activations/*.npz | wc -l) sites); $(disk)"
    slim_dump $DUMP
    say "arm $ARM seed $SEED: slimmed ($(du -sh $DUMP | cut -f1)); $(disk)"
  done
  # ---- 3. analysis
  [ -f $MERGED/discovery_seeds_common.txt ] || comm -12 <(sort $MERGED/armA/discovery_seeds.txt) <(sort $MERGED/armB/discovery_seeds.txt) | sort -n > $MERGED/discovery_seeds_common.txt
  say "analysis: identity_contrast_geometry.py on $(grep -c . $MERGED/discovery_seeds_common.txt) common discovery seeds"
  $MPY identity_contrast_geometry.py --dump-a $IG/dump_armA --dump-b $IG/dump_armB --stimulus $MERGED/armA --discovery-seeds $MERGED/discovery_seeds_common.txt \
    --out $IG --step 0 --seed $SEED $IG_ARGS >> $LOG 2>&1 || { say "identity_contrast_geometry FAILED (see $LOG)"; exit 1; }
  [ -f $IG/identity_geometry.json ] || { say "no identity_geometry.json written"; exit 1; }
  say "analysis done -> $IG/identity_geometry.json, $IG/identity_geometry.md ($(du -h $IG/identity_geometry.json | cut -f1))"
else
  say "identity_geometry.json present, analysis skipped (use --force to redo)"
fi

# ---- 4. cleanup + marker
for ARM in A B; do rm -rf $IG/dump_arm$ARM/activations; done
say "dump activations deleted (index/interaction maps kept); $(disk)"
$MPY - $IG/identity_geometry.json >> $LOG 2>&1 <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("IDENTITY_GEOMETRY summary:", json.dumps({g: {v: {k: s["variants"][v][k] for k in ("n_anti_alignment_detected", "n_within_A_detected", "n_within_B_detected")} for v in s["variants"]} for g, s in d["summary"].items()}))
print("template fraction ranges:", json.dumps({g: s["template_insample_mean_dir_fraction_range"] for g, s in d["summary"].items()}), "runtime_s", round(d["runtime_s"]))
EOF
echo "[$(ts)] IDENTITY_GEOMETRY_DONE" >> $LOG
say "IDENTITY_GEOMETRY_DONE -> $IG"

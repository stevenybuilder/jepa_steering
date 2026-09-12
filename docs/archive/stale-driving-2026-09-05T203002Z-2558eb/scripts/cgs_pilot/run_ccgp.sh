#!/bin/bash
# run_ccgp.sh — abstraction / ontology gate (Bernardi et al. 2020 CCGP, parallelism score, shattering dimensionality;
# cross model design jepa.md Step 5 "ontology first") on the seed-0 activation dumps of both arms. Optional wrapper around
# geometry_ccgp.py; touches nothing under drive_* except reading it.
#
#   0 keep      run_identity_geometry.sh deletes its dump activations (stage 4) right after its own analysis, so this stage
#               polls $IG/dump_arm{A,B}/activations/index.json and copies each arm's dump to $CC/dump_arm$X as soon as the
#               index carries "slimmed_to_step0": true (the slimming rewrites every npz and writes the index last).  If the
#               identity-geometry job never produces a dump (e.g. it failed) and CCGP_OWN_DUMP=1, the same
#               localize_interaction.py invocation as run_identity_geometry.sh / run_drive_arms.sh PAIRS is run here
#               (all imagined steps kept, ~2.5 GB per arm).  Skipped when $CC/dump_arm$X/activations/index.json exists.
#   1 self-test geometry_ccgp.py --self-test (numpy only): planted abstract variables -> CCGP ~ 1, PS ~ 1, gate PASS;
#               planted XOR-like code -> SD high, CCGP ~ chance, gate FAIL.  Aborts otherwise.
#   2 analysis  geometry_ccgp.py --dump-a $CC/dump_armA --dump-b $CC/dump_armB --stimulus $STIM --discovery-seeds $SEEDS
#               --out $CC  -> $CC/ccgp.json + ccgp.md ; marker CCGP_DONE in $LOG.
# Every stage is relaunch-safe (skipped when its output exists; --force redoes 2).
# Env overrides (v0.9 dumps later, unchanged script):  CCGP_DUMP_A / CCGP_DUMP_B (existing dumps to analyse instead of the
# identity-geometry copies), CCGP_STIM (stimulus dir with masks/ + manifest.jsonl), CCGP_SEEDS (discovery seeds file),
# CCGP_OUT (output dir), CCGP_ARGS (extra geometry_ccgp.py args), CCGP_OWN_DUMP=1 (stage 0 dump fallback, model
# eval dirs $CCGP_EVAL/arm{A,B}_seed$SEED with manifest.jsonl + masks/ + discovery_seeds.txt; default drive_eval),
# CCGP_KEEP_TIMEOUT_MIN=0 (do not wait for the identity-geometry dumps).  A missing CCGP_SEEDS file is built as the
# intersection of $STIM/discovery_seeds.txt with its armB sibling's (as run_identity_geometry.sh does).
# v0.9 example:  CCGP_OUT=$B/artifacts/drive_ccgp_v09 CCGP_EVAL=$B/artifacts/drive_v09_eval/static CCGP_STIM=$B/artifacts/drive_v09_merged/static/armA \
#   CCGP_SEEDS=$B/artifacts/drive_ccgp_v09/discovery_seeds_common.txt CCGP_KEEP_TIMEOUT_MIN=0 CCGP_OWN_DUMP=1 CCGP_LOG=$B/logs/drive_ccgp_v09.log run_ccgp.sh
# Usage:  run_ccgp.sh [--force] [--skip-selftest]
# Launch (remote):
#   ssh -n -p PORT root@HOST 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_ccgp.sh > /root/cgs-pilot/logs/ccgp_driver.log 2>&1 < /dev/null &'
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}
CODE=${DRIVE_CODE:-$BASE/code/cgs_pilot}
REPO=${DRIVE_REPO:-$BASE/vendor/jepa-wms}
MPY=${DRIVE_MPY:-/opt/conda/bin/python}
OUT=${DRIVE_OUT_ROOT:-$BASE/artifacts}
LOGDIR=${DRIVE_LOGDIR:-$BASE/logs}
MODELS=$OUT/drive_models; MERGED=$OUT/drive_factorial_merged; EVAL=${CCGP_EVAL:-$OUT/drive_eval}
IG=${DRIVE_IG_OUT:-$OUT/drive_identity_geometry}
CC=${CCGP_OUT:-$OUT/drive_ccgp}
STIM=${CCGP_STIM:-$MERGED/armA}
SEEDS=${CCGP_SEEDS:-$MERGED/discovery_seeds_common.txt}
LOG=${CCGP_LOG:-$LOGDIR/drive_ccgp.log}
SEED=${DRIVE_IG_SEED:-0}
LOC_ARGS="--dump-hooks resid_post attn_out mlp_out"             # identical to run_drive_arms.sh / run_identity_geometry.sh
CCGP_ARGS=${CCGP_ARGS:-""}
KEEP_TIMEOUT_MIN=${CCGP_KEEP_TIMEOUT_MIN:-240}                    # stage 0: give up waiting for the identity-geometry dumps after this
FORCE=0; SKIP_ST=0
while [ $# -gt 0 ]; do case "$1" in --force) FORCE=1;; --skip-selftest) SKIP_ST=1;; *) echo "unknown arg $1"; exit 2;; esac; shift; done
export JEPAWM_HOME=$BASE/vendor JEPAWM_OSSCKPT=$BASE/checkpoints JEPAWM_LOGS=$BASE/logs PYTHONPATH=$REPO:$CODE
export JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*" | tee -a $LOG; }
disk() { df -h $BASE | tail -1 | awk '{print "disk used "$3" of "$2", free "$4}'; }
mkdir -p $CC $LOGDIR
cd $CODE || { echo "no code dir $CODE"; exit 1; }
echo "[$(ts)] CCGP start seed=$SEED out=$CC stim=$STIM seeds=$SEEDS force=$FORCE ccgp_args='$CCGP_ARGS'; $(disk)" >> $LOG

# ---- 0. dumps: explicit (CCGP_DUMP_X), else keep copies of the identity-geometry dumps, else (opt-in) own dump
slimmed() { grep -q '"slimmed_to_step0": true' "$1/activations/index.json" 2>/dev/null; }
DUMPS=()
for ARM in A B; do
  VAR="CCGP_DUMP_$ARM"; EXPL=${!VAR:-""}
  if [ -n "$EXPL" ]; then
    [ -f $EXPL/activations/index.json ] || { say "arm $ARM: explicit dump $EXPL has no activations/index.json"; exit 1; }
    say "arm $ARM: explicit dump $EXPL"; DUMPS+=("$EXPL"); continue
  fi
  DUMP=$CC/dump_arm$ARM
  if [ -f $DUMP/activations/index.json ]; then say "arm $ARM: dump present at $DUMP, reused ($(ls $DUMP/activations/*.npz | wc -l) sites)"; DUMPS+=("$DUMP"); continue; fi
  SRC=$IG/dump_arm$ARM
  say "arm $ARM: waiting for the identity-geometry dump $SRC (slimmed index) up to $KEEP_TIMEOUT_MIN min"
  n=0; got=0
  while [ $n -lt $((KEEP_TIMEOUT_MIN * 3)) ]; do
    if slimmed $SRC; then
      sleep 5   # let the slimming's last np.savez settle
      rm -rf $DUMP.partial; mkdir -p $DUMP.partial
      if cp -r $SRC/. $DUMP.partial/ 2>>$LOG && [ -f $DUMP.partial/activations/index.json ] && [ "$(ls $DUMP.partial/activations/*.npz 2>/dev/null | wc -l)" -ge 1 ]; then
        mv $DUMP.partial $DUMP; got=1; say "arm $ARM: kept a copy of $SRC -> $DUMP ($(du -sh $DUMP | cut -f1), $(ls $DUMP/activations/*.npz | wc -l) sites); $(disk)"; break
      fi
      say "arm $ARM: copy of $SRC failed (deleted under us?); retrying"; rm -rf $DUMP.partial
    fi
    n=$((n + 1)); [ $((n % 30)) = 0 ] && say "arm $ARM: still waiting for $SRC ($(tail -1 $LOGDIR/drive_identity_geometry.log 2>/dev/null | cut -c1-120))"
    sleep 20
  done
  if [ $got = 0 ]; then
    if [ "${CCGP_OWN_DUMP:-0}" = "1" ]; then
      MD=$MODELS/arm${ARM}_seed$SEED; EV=$EVAL/arm${ARM}_seed$SEED
      for f in $MD/eval_config.yaml $MD/jepa-latest.pth.tar $EV/discovery_seeds.txt; do [ -f $f ] || { say "missing $f"; exit 1; }; done
      rm -rf $DUMP; mkdir -p $DUMP
      say "arm $ARM seed $SEED: own localization dump (all imagined steps; same invocation as run_identity_geometry.sh) -> $DUMP"
      $MPY localize_interaction.py --repo $REPO --config $MD/eval_config.yaml --checkpoint $MD/jepa-latest.pth.tar --model-name jepa_wm_driving --artifacts $EV --output $DUMP --domain driving \
        --seeds-file $EV/discovery_seeds.txt $LOC_ARGS --dump-groups egg corridor >> $LOG 2>&1
      [ -f $DUMP/activations/index.json ] || { say "arm $ARM seed $SEED: own dump FAILED (see $LOG)"; exit 1; }
      say "arm $ARM seed $SEED: own dump done ($(du -sh $DUMP | cut -f1)); $(disk)"
    else
      say "arm $ARM: no dump after $KEEP_TIMEOUT_MIN min and CCGP_OWN_DUMP != 1; aborting"; exit 1
    fi
  fi
  DUMPS+=("$DUMP")
done

# ---- 1. preflight self-test (numpy only)
until [ -f $CODE/geometry_ccgp.py ]; do say "waiting for $CODE/geometry_ccgp.py"; sleep 60; done
if [ $SKIP_ST = 0 ] && { [ ! -f $CC/self_test/verdict.json ] || [ $FORCE = 1 ]; }; then
  $MPY geometry_ccgp.py --self-test --out $CC --seed $SEED > $CC/self_test.log 2>&1
  grep -q SELF_TEST_PASSED $CC/self_test.log || { say "preflight self-test FAILED (see $CC/self_test.log); aborting"; exit 1; }
  say "preflight self-test passed: $(grep -o '"passed": [a-z]*' $CC/self_test/verdict.json | head -1)"
fi

# ---- 2. analysis
if [ ! -f $CC/ccgp.json ] || [ $FORCE = 1 ]; then
  if [ ! -f $SEEDS ]; then
    SIB=$(dirname $STIM)/armB
    [ -f $STIM/discovery_seeds.txt ] && [ -f $SIB/discovery_seeds.txt ] || { say "no discovery seeds file $SEEDS and no armA/armB discovery_seeds.txt to intersect"; exit 1; }
    comm -12 <(sort $STIM/discovery_seeds.txt) <(sort $SIB/discovery_seeds.txt) | sort -n > $SEEDS
    say "built $SEEDS = $(grep -c . $SEEDS) seeds common to $STIM and $SIB"
  fi
  say "analysis: geometry_ccgp.py on $(grep -c . $SEEDS) discovery seeds; dumps ${DUMPS[0]} ${DUMPS[1]}; stimulus $STIM"
  $MPY geometry_ccgp.py --dump-a ${DUMPS[0]} --dump-b ${DUMPS[1]} --stimulus $STIM --discovery-seeds $SEEDS --out $CC --seed $SEED $CCGP_ARGS >> $LOG 2>&1 || { say "geometry_ccgp FAILED (see $LOG)"; exit 1; }
  [ -f $CC/ccgp.json ] || { say "no ccgp.json written"; exit 1; }
  say "analysis done -> $CC/ccgp.json, $CC/ccgp.md ($(du -h $CC/ccgp.json | cut -f1))"
else
  say "ccgp.json present, analysis skipped (use --force to redo)"
fi
$MPY - $CC/ccgp.json >> $LOG 2>&1 <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("CCGP gate verdicts:", json.dumps(d.get("gate", {}).get("summary", {})))
print("runtime_s", round(d.get("runtime_s", 0)))
EOF
echo "[$(ts)] CCGP_DONE" >> $LOG
say "CCGP_DONE -> $CC"

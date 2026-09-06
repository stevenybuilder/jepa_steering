#!/bin/bash
# run_planner_diag_rangek.sh — HAZARD-FREE-ONLY range-k diagnostic queue (coordinator 18:35 UTC), armA_seed0 + armB_seed0 in parallel,
# discovery seeds 9 10 19. Order (each pass resumes; finished (seed, config) pairs are skipped):
#   1. fixed_k10, fixed_k20   2. fixed_k6, fixed_k40, released_k10   3. the remaining earlier configs (compact_rolling6/10, released_cem_dest)
# Marker PLANNER_DIAG_RANGEK_DONE in logs/drive_closed_loop.log. Outputs: artifacts/drive_closed_loop/planner_diagnostics/<model>/diag.jsonl (same file)
set -u
B=/root/cgs-pilot; C=$B/code/cgs_pilot; L=$B/logs/drive_closed_loop.log; MPY=/opt/conda/bin/python; DPY=/opt/conda/envs/metadrive/bin/python; REPO=$B/vendor/jepa-wms
OUT=$B/artifacts/drive_closed_loop/planner_diagnostics; MERGED=$B/artifacts/drive_factorial_merged; MODELS=$B/artifacts/drive_models
export JEPAWM_HOME=$B/vendor JEPAWM_OSSCKPT=$B/checkpoints JEPAWM_LOGS=$B/logs PYTHONPATH=$REPO:$C JEPAWM_DRIVING_ACTION_DIM=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
SEEDS=${DIAG_SEEDS:-"9 10 19"}
ts(){ date -u +%FT%TZ; }; say(){ echo "[$(ts)] $*" >> $L; }
run(){ local ARM=$1 CFGS=$2 M=arm${1}_seed0; (cd $C && $MPY closed_loop_planner_diag.py --arm $ARM --model-dir $MODELS/$M --repo $REPO --stimulus $MERGED/arm$ARM --seeds $SEEDS --configs $CFGS --out $OUT/$M --bridge-python $DPY >> $B/logs/drive_planner_diag_$M.log 2>&1); }
for CFGS in "fixed_k10,fixed_k20" "fixed_k6,fixed_k40,released_k10" "compact_rolling6,compact_rolling10,released_cem_dest"; do
  say "range-k diagnostics pass: $CFGS (seeds $SEEDS; hazard-free only)"
  run A "$CFGS" & PA=$!; run B "$CFGS" & PB=$!; wait $PA; rcA=$?; wait $PB; rcB=$?
  say "range-k pass done ($CFGS) rcA=$rcA rcB=$rcB"
done
say "PLANNER_DIAG_RANGEK_DONE"

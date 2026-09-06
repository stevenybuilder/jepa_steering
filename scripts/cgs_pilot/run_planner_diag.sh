#!/bin/bash
# run_planner_diag.sh — HAZARD-FREE-ONLY planner diagnostics (closed_loop_planner_diag.py) for armA_seed0 + armB_seed0 on
# discovery scenes 9, 10, 19 (SSS maps: reachable by always-throttle). No hazard level is rolled out (coordinator instruction
# 2026-09-03 ~17:55 UTC; the planner configuration must be confirmed as an amendment to the Label-first principle first).
# Marker PLANNER_DIAG_DONE in logs/drive_closed_loop.log. Outputs artifacts/drive_closed_loop/planner_diagnostics/<model>/{diag.jsonl,summary.json,driver.log}
set -u
B=/root/cgs-pilot; C=$B/code/cgs_pilot; L=$B/logs/drive_closed_loop.log; MPY=/opt/conda/bin/python; DPY=/opt/conda/envs/metadrive/bin/python; REPO=$B/vendor/jepa-wms
OUT=$B/artifacts/drive_closed_loop/planner_diagnostics; MERGED=$B/artifacts/drive_factorial_merged; MODELS=$B/artifacts/drive_models
export JEPAWM_HOME=$B/vendor JEPAWM_OSSCKPT=$B/checkpoints JEPAWM_LOGS=$B/logs PYTHONPATH=$REPO:$C JEPAWM_DRIVING_ACTION_DIM=2 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
SEEDS=${DIAG_SEEDS:-"9 10 19"}; CONFIGS=${DIAG_CONFIGS:-compact_rolling3,compact_dest,compact_rolling6,compact_rolling10,released_cem_dest}
ts(){ date -u +%FT%TZ; }; say(){ echo "[$(ts)] $*" >> $L; }
mkdir -p $OUT
say "planner diagnostics start (hazard-free only; seeds $SEEDS; configs $CONFIGS; armA_seed0 + armB_seed0 in parallel)"
run(){ local ARM=$1 M=arm${1}_seed0; (cd $C && $MPY closed_loop_planner_diag.py --arm $ARM --model-dir $MODELS/$M --repo $REPO --stimulus $MERGED/arm$ARM --seeds $SEEDS --configs $CONFIGS --out $OUT/$M --bridge-python $DPY >> $B/logs/drive_planner_diag_$M.log 2>&1); }
run A & PA=$!; run B & PB=$!
wait $PA; rcA=$?; wait $PB; rcB=$?
say "PLANNER_DIAG_DONE rcA=$rcA rcB=$rcB"

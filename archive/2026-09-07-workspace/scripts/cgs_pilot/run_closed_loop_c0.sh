#!/bin/bash
# run_closed_loop_c0.sh — closed-loop CEM adapter (closed_loop_rollout.py + closed_loop_bridge.py) on box 2 (cgs-pilot-2), FULL mode.
# Label-first principle, amendment 1 (cross model design jepa.md; COAST arXiv 2605.17144 s3.2 / s4.1 / App. A.8): the label is
# MetaDrive's NATIVE outcome (success = arrive_dest at the env's own done with no crash predicate; failure subtype = the native
# reason: crash_human / crash_object / crash_vehicle / crash_sidewalk / out_of_road / max_step). MetaDrive default termination
# flags; horizon guard 1000 env steps (engineering guard -> native max_step). Our progress/goal quantities are descriptive_* only.
# Discovery scenes ONLY (the driver refuses sealed confirmation seeds); a scene whose hazard-free always-throttle reference does
# not reach arrive_dest is refused and recorded (skipped_scenes.jsonl), never re-goaled. No planner tuning on outcomes.
#
# Stages (markers in logs/drive_closed_loop.log):
#   S. native SMOKE: 1 scene (seed 9, SSS map, freshly generated with the frozen v0.7 generator) x armA_seed0 -> CLOSED_LOOP_NATIVE_SMOKE_DONE | _FAIL
#   1. v0.7 common discovery scenes (drive_factorial_merged), armA_seed0 + armB_seed0            -> CLOSED_LOOP_C0_SEED0_DONE
#   2. the remaining four models (seeds 1-2), same scenes, two models at a time                     -> CLOSED_LOOP_C0_V07_DONE
#   3. v0.9 randomised set discovery scenes (drive_v09_merged/rand/arm{A,B}), all six models        -> CLOSED_LOOP_C0_V09_DONE
#   c0_table.{json,md} under artifacts/drive_closed_loop is rewritten after every stage (closed_loop_rollout.py table).
# Outputs: artifacts/drive_closed_loop/<set>/<arm>_seed<s>/{rollouts.jsonl,summary.json,run_config.json,driver.log,bridge.log,latents/,skipped_scenes.jsonl}
# Process budget: 2 drivers + 2 bridges at a time (OMP 4). Waits for the legacy progress-rule pilot (CLOSED_LOOP_PILOT12_DONE)
# before stage 1; the legacy runner's later "all" stage is neutralised by the driver's label-mode guard (its out dirs hold
# progress_rule records), so no process is killed.
# Resumable: relaunch the same command; finished episodes / stages are skipped.
# Launch: ssh -n -p 45460 root@70.27.250.55 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_closed_loop_c0.sh > /root/cgs-pilot/logs/drive_closed_loop_driver.out 2>&1 < /dev/null &'
# Knobs (env): CL_SMOKE_SEED (9), CL_EPISODE_SEEDS (2), CL_EXTRA_ARGS (passed to closed_loop_rollout.py), CL_SKIP_SMOKE=1, CL_SKIP_LEGACY_WAIT=1
set -u
B=${CL_BASE:-/root/cgs-pilot}; C=$B/code/cgs_pilot; L=$B/logs/drive_closed_loop.log
MPY=${CL_MPY:-/opt/conda/bin/python}; DPY=${CL_DPY:-/opt/conda/envs/metadrive/bin/python}; REPO=$B/vendor/jepa-wms
OUT=${CL_OUT:-$B/artifacts/drive_closed_loop}; MERGED=${CL_MERGED:-$B/artifacts/drive_factorial_merged}; MODELS=${CL_MODELS:-$B/artifacts/drive_models}
V09=${CL_V09:-$B/artifacts/drive_v09_merged/rand}
export JEPAWM_HOME=$B/vendor JEPAWM_OSSCKPT=$B/checkpoints JEPAWM_LOGS=$B/logs PYTHONPATH=$REPO:$C JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
SMOKE_SEED=${CL_SMOKE_SEED:-9}; EP_SEEDS=${CL_EPISODE_SEEDS:-2}; EXTRA=${CL_EXTRA_ARGS:-}
ts(){ date -u +%FT%TZ; }; say(){ echo "[$(ts)] $*" >> $L; }
mkdir -p $OUT $B/logs
say "closed-loop FULL runner start (native label, amendment 1; smoke seed $SMOKE_SEED, episode seeds $EP_SEEDS, extra '$EXTRA')"
# ---- 0. GPU
until grep -q FIX_DONE $B/logs/fix_box2_gpu.log 2>/dev/null; do sleep 60; done
$MPY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || { say "CUDA_NOT_AVAILABLE; abort"; exit 1; }
wait_model(){ until [ -f $MODELS/$1/jepa-latest.pth.tar ] && [ -f $MODELS/$1/eval_config.yaml ] && ! ls $MODELS/$1/.jepa-latest.pth.tar.* >/dev/null 2>&1; do sleep 60; done; }
run_model(){ # SET ARM SEED STIMULUS SEEDSFILE [extra]
  local SET=$1 ARM=$2 S=$3 STIM=$4 SEEDS=$5; shift 5; local M=arm${ARM}_seed$S
  mkdir -p $OUT/$SET
  (cd $C && $MPY closed_loop_rollout.py run --label native --arm $ARM --model-dir $MODELS/$M --repo $REPO --stimulus $STIM --seeds-file $SEEDS \
      --episode-seeds $EP_SEEDS --out $OUT/$SET/$M --bridge-python $DPY $EXTRA "$@" >> $B/logs/drive_closed_loop_${SET}_$M.log 2>&1)
}
verdict(){ grep -o '"verdict": "[A-Z]*"' $OUT/$1/summary.json 2>/dev/null | head -1; }
table(){ (cd $C && $MPY closed_loop_rollout.py table --root $OUT --sets v07,v09 --out $OUT/c0_table >> $L 2>&1); say "c0_table rewritten: $OUT/c0_table.{json,md}"; }
# ---- S. native smoke (SSS seed; full cell npz for the <= 1e-6 replay check)
SMOKE_STIM=$OUT/_smoke_factorial/armA
if [ ! -f $SMOKE_STIM/seed_$SMOKE_SEED/manifest.jsonl ]; then
  say "generating smoke scene seed $SMOKE_SEED (frozen v0.7 generator defaults: hazard 9 m, static body, arms AB, fresh-process replay check)"
  (cd $C && $DPY metadrive_hazard_pilot.py factorial --out $OUT/_smoke_factorial --seeds $SMOKE_SEED --arms AB --hazard-dist 9.0 --procs 1 > $OUT/_smoke_factorial_seed$SMOKE_SEED.gen.log 2>&1); say "smoke gen rc=$?"
fi
if [ "${CL_SKIP_SMOKE:-0}" != "1" ] && ! grep -q CLOSED_LOOP_NATIVE_SMOKE_DONE $L; then
  wait_model armA_seed0
  say "native smoke: armA_seed0 x seed $SMOKE_SEED x {hazard-free, 4 levels} x {always_throttle, always_brake, cem}, 1 episode seed, stop-on-unreachable"
  (cd $C && $MPY closed_loop_rollout.py run --label native --arm A --model-dir $MODELS/armA_seed0 --repo $REPO --stimulus $SMOKE_STIM --seeds $SMOKE_SEED \
      --episode-seeds 1 --stop-on-unreachable --out $OUT/smoke_native/armA_seed0 --bridge-python $DPY $EXTRA >> $L 2>&1); rc=$?
  ok=$($MPY -c "import json; s=json.load(open('$OUT/smoke_native/armA_seed0/summary.json')); r=s['replay']; h=s['hazard_free_reference']; print(int(s['label_mode']=='native' and r['n_failed']==0 and r['n_ok']>0 and s['n_rollouts']>=15 and h['n_reachable']>=1))" 2>/dev/null)
  if [ "$rc" = "0" ] && [ "$ok" = "1" ]; then say "CLOSED_LOOP_NATIVE_SMOKE_DONE $(verdict smoke_native/armA_seed0)"; else say "CLOSED_LOOP_NATIVE_SMOKE_FAIL rc=$rc ok=${ok:-none}"; exit 1; fi
fi
# ---- stage prerequisites
until grep -q PULL_MODELS_DONE /root/pull_box1.log 2>/dev/null; do sleep 60; done
until [ -f $MERGED/armA/manifest.jsonl ] && [ -f $MERGED/armB/manifest.jsonl ]; do sleep 60; done
SEEDS07=""; for f in $MERGED/discovery_seeds_common.txt $B/artifacts/drive_eval/armA_seed0/discovery_seeds_common.txt $MERGED/armA/discovery_seeds.txt; do [ -z "$SEEDS07" ] && [ -f $f ] && SEEDS07=$f; done
[ -n "$SEEDS07" ] || { say "no v0.7 discovery seed list; abort"; exit 1; }
if [ "${CL_SKIP_LEGACY_WAIT:-0}" != "1" ]; then
  until grep -q CLOSED_LOOP_PILOT12_DONE $L 2>/dev/null || ! pgrep -f "closed_loop_rollout.py run .*--out $OUT/arm[AB]_seed0 " >/dev/null; do sleep 60; done
  say "legacy progress-rule pilot finished or absent; process budget free"
fi
# ---- 1. seed-0 pair on v0.7 discovery
if ! grep -q CLOSED_LOOP_C0_SEED0_DONE $L; then
  wait_model armA_seed0; wait_model armB_seed0
  say "stage 1 start: v0.7 common discovery ($(wc -l < $SEEDS07) seeds from $SEEDS07), armA_seed0 + armB_seed0; cells present: $([ -f "$MERGED/armA/$(head -1 $MERGED/armA/manifest.jsonl | $MPY -c 'import json,sys; print(json.loads(sys.stdin.read())["artifact"])')" ] && echo yes || echo no-manifest-only-replay-check)"
  run_model v07 A 0 $MERGED/armA $SEEDS07 & PA=$!; run_model v07 B 0 $MERGED/armB $SEEDS07 & PB=$!
  wait $PA; rcA=$?; wait $PB; rcB=$?
  table
  say "stage 1 done rcA=$rcA rcB=$rcB :: armA_seed0 $(verdict v07/armA_seed0) armB_seed0 $(verdict v07/armB_seed0)"
  [ "$rcA" = "0" ] && [ "$rcB" = "0" ] && say "CLOSED_LOOP_C0_SEED0_DONE" || say "stage 1 had a non-zero rc; markers withheld (relaunch to resume)"
fi
# ---- 2. seeds 1-2 on v0.7 discovery
if grep -q CLOSED_LOOP_C0_SEED0_DONE $L && ! grep -q CLOSED_LOOP_C0_V07_DONE $L; then
  ok=1
  for S in 1 2; do
    wait_model armA_seed$S; wait_model armB_seed$S
    say "stage 2: v0.7 discovery, armA_seed$S + armB_seed$S"
    run_model v07 A $S $MERGED/armA $SEEDS07 & PA=$!; run_model v07 B $S $MERGED/armB $SEEDS07 & PB=$!
    wait $PA; rcA=$?; wait $PB; rcB=$?
    table
    say "stage 2 seed $S done rcA=$rcA rcB=$rcB :: $(verdict v07/armA_seed$S) $(verdict v07/armB_seed$S)"
    [ "$rcA" = "0" ] && [ "$rcB" = "0" ] || ok=0
  done
  [ "$ok" = "1" ] && say "CLOSED_LOOP_C0_V07_DONE" || say "stage 2 had a non-zero rc; marker withheld (relaunch to resume)"
fi
# ---- 3. v0.9 randomised set (discovery), all six models
if grep -q CLOSED_LOOP_C0_V07_DONE $L && ! grep -q CLOSED_LOOP_C0_V09_DONE $L; then
  until grep -q "merged rand armB" $B/logs/drive_v09_eval.log 2>/dev/null && [ -f $V09/armA/discovery_seeds.txt ] && [ -f $V09/armB/discovery_seeds.txt ]; do sleep 120; done
  say "stage 3 start: v0.9 rand discovery (armA $(wc -l < $V09/armA/discovery_seeds.txt) / armB $(wc -l < $V09/armB/discovery_seeds.txt) seeds)"
  ok=1
  for S in 0 1 2; do
    run_model v09 A $S $V09/armA $V09/armA/discovery_seeds.txt & PA=$!; run_model v09 B $S $V09/armB $V09/armB/discovery_seeds.txt & PB=$!
    wait $PA; rcA=$?; wait $PB; rcB=$?
    table
    say "stage 3 seed $S done rcA=$rcA rcB=$rcB :: $(verdict v09/armA_seed$S) $(verdict v09/armB_seed$S)"
    [ "$rcA" = "0" ] && [ "$rcB" = "0" ] || ok=0
  done
  [ "$ok" = "1" ] && say "CLOSED_LOOP_C0_V09_DONE" || say "stage 3 had a non-zero rc; marker withheld (relaunch to resume)"
fi
say "FULL runner exit"

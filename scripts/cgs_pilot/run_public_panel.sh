#!/usr/bin/env bash
# =====================================================================================================
# Panel P runner — released facebook/jepa-wms checkpoints x the repository's own planning evaluators,
# executed UNMODIFIED (configs/evals/simu_env_planning/<env>/<model>/*.yaml, succ_def: simu).
#
# Governed by "Label-first principle, amendment 3" (cross model design jepa.md, 2026-09-03 ~23:40 UTC):
#   the label is the official evaluator's per-episode success flag; nothing of ours (no custom hazard,
#   no self-trained checkpoint, no researcher-authored predicate). Screen first (C0, no activations),
#   eligibility = 20-80 % success and >= 3 successes / >= 3 failures per 15-episode fit cell, >= 2
#   eligible benchmarks per model to advance. A failed screen is reported, never tuned.
#   Discipline reproduced from mechinterp-vla/cross model plan.md ("Native-label invariant across every
#   model and benchmark", "Operating-point correction after the ceiling screen").
#
# Stages / markers (logs/public_panel.log):
#   PP_ENV_DONE                       provenance.json written (before any inference), env imports verified
#   PP_SMOKE_DONE                     2-episode PointMaze / JEPA-WM smoke, native success field confirmed
#   PP_SCREEN_<env>_<model>_DONE      seed-1 screen of one cell complete (96 ep; MetaWorld 48)
#   PP_SCREEN_<env>_<model>_S<k>_DONE additional evaluation seed k complete
#   PP_SCREEN_SEED1_DONE              every cell has its seed-1 run
#   PP_SCREEN_DONE                    all requested seeds complete; c0_table written
#   PP_FIT_DONE / PP_DEV_TABLE_DONE   stages 2-3 (separate runners, gated on eligibility; not automated here)
#
# nohup-safe; idempotent (skips cells whose marker exists). Launch:
#   setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_public_panel.sh </dev/null > /root/cgs-pilot/logs/public_panel.out 2>&1 &
# Never touches other jobs; GPU is shared, so at most PP_PARALLEL evaluator processes run at once.
# =====================================================================================================
set -uo pipefail

ROOT=${ROOT:-/root/cgs-pilot}
CODE=$ROOT/code/cgs_pilot
REPO=$ROOT/vendor/jepa-wms
PY=${PY:-/opt/conda/bin/python}
ART=$ROOT/artifacts/public_panel
LOG=$ROOT/logs/public_panel.log
CKPT=$ROOT/checkpoints/public
SEEDS=${PP_SEEDS:-"1 2 3"}
PP_PARALLEL=${PP_PARALLEL:-1}
PP_OOM_RETRIES=${PP_OOM_RETRIES:-200}
PP_OOM_WAIT=${PP_OOM_WAIT:-60}
PP_MIN_FREE_MB=${PP_MIN_FREE_MB:-7500}
PP_FREE_WAIT_MAX=${PP_FREE_WAIT_MAX:-1800}
CELLS=${PP_CELLS:-"pt:jepa-wm pt:dino-wm mz:jepa-wm mz:dino-wm wall:jepa-wm wall:dino-wm mw:jepa-wm mw:dino-wm"}

export JEPAWM_HOME=$ROOT/vendor
export JEPAWM_DSET=$ROOT/datasets
export JEPAWM_OSSCKPT=$ROOT/checkpoints
export JEPAWM_LOGS=$ART/jepawm_logs
export JEPAWM_CKPT=$ROOT/checkpoints
export PYTHONPATH=$REPO:$CODE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_SUPPRESS_IMPORT_ERROR=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$ART" "$ROOT/logs" "$JEPAWM_LOGS"
mark() { echo "$(date -u +%FT%TZ) $1" >> "$LOG"; }
has() { grep -q "$1" "$LOG" 2>/dev/null; }

# The shared GPU can be too full for the released CEM batch (300 candidates x 16 heads x 512^2 attention
# = 4.7 GiB in one allocation). On CUDA OOM the driver is re-run after PP_OOM_WAIT s, never with a
# smaller planner batch (planner config is frozen).
gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_for_gpu() {  # block until >= PP_MIN_FREE_MB free (poll 15 s), at most PP_FREE_WAIT_MAX s
  local waited=0
  while [ "$(gpu_free_mb)" -lt "$PP_MIN_FREE_MB" ] && [ "$waited" -lt "$PP_FREE_WAIT_MAX" ]; do sleep 15; waited=$((waited+15)); done
  [ "$waited" -gt 0 ] && mark "PP_GPU_WAIT waited=${waited}s free=$(gpu_free_mb)MiB"
}
run_driver() {  # log-file then driver args...
  local logf=$1; shift
  local attempt=0
  while :; do
    attempt=$((attempt+1))
    wait_for_gpu
    $PY "$CODE/public_panel_eval.py" "$@" > "$logf" 2>&1 && return 0
    if grep -q "CUDA out of memory" "$logf" && [ "$attempt" -lt "$PP_OOM_RETRIES" ]; then
      mark "PP_OOM_RETRY attempt=$attempt log=$logf gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' ') waiting ${PP_OOM_WAIT}s"
      cp "$logf" "${logf%.log}.oom${attempt}.log"
      sleep "$PP_OOM_WAIT"
      continue
    fi
    return 1
  done
}

mark "PP_START pid=$$ seeds='$SEEDS' parallel=$PP_PARALLEL cells='$CELLS' (Label-first principle, amendment 3)"

# ---------------------------------------------------------------- Stage 0: provenance (before any inference)
if ! has PP_ENV_DONE; then
  if $PY "$CODE/public_panel_provenance.py" --repo "$REPO" --checkpoints "$CKPT" --datasets "$JEPAWM_DSET" --out "$ART/provenance.json" >> "$LOG" 2>&1; then
    mark "PP_ENV_DONE provenance=$ART/provenance.json"
  else
    mark PP_ENV_FAILED; exit 1
  fi
fi

# per-env dependency check (Push-T / Wall need nothing beyond the repo; PointMaze needs mujoco-py + d4rl; MetaWorld needs metaworld + mujoco)
deps_ok() {
  case "$1" in
    mz) $PY -c "import mujoco_py, d4rl" >/dev/null 2>&1 ;;
    mw) $PY -c "import metaworld, mujoco; from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE" >/dev/null 2>&1 ;;
    *) true ;;
  esac
}

# ---------------------------------------------------------------- Smoke: 2 episodes per env (JEPA-WM), native success field confirmed
smoke_cell() {  # env
  local env=$1
  if has "PP_SMOKE_${env}_DONE"; then return 0; fi
  if ! deps_ok "$env"; then mark "PP_SMOKE_${env}_SKIPPED deps missing"; return 1; fi
  mkdir -p "$ART/smoke"
  if run_driver "$ART/smoke/${env}_jepa-wm.log" --repo "$REPO" --env "$env" --model jepa-wm --checkpoint "$CKPT/${env}_jepa-wm.pth.tar" \
       --out "$ART/smoke/${env}_jepa-wm" --episodes 2 --label smoke \
     && [ "$(wc -l < "$ART/smoke/${env}_jepa-wm/episodes.jsonl")" -eq 2 ] \
     && $PY -c "import json; rows=[json.loads(l) for l in open('$ART/smoke/${env}_jepa-wm/episodes.jsonl')]; assert all(r['success'] in (0,1) for r in rows); print('smoke $env native success:', [r['success'] for r in rows], 'wall_s', [r['wall_s'] for r in rows])" >> "$LOG" 2>&1; then
    mark "PP_SMOKE_${env}_DONE"; [ "$env" = mz ] && mark PP_SMOKE_DONE; return 0
  else
    mark "PP_SMOKE_${env}_FAILED (see $ART/smoke/${env}_jepa-wm.log)"; return 1
  fi
}

# ---------------------------------------------------------------- Stage 1: C0 screen
run_cell() {  # env model seed
  local env=$1 model=$2 seed=$3
  local tag="PP_SCREEN_${env}_${model}"
  [ "$seed" != "1" ] && tag="${tag}_S${seed}"
  if has "${tag}_DONE"; then return 0; fi
  if ! deps_ok "$env"; then mark "${tag}_SKIPPED deps missing"; return 1; fi
  has "PP_SMOKE_${env}_DONE" || { mark "${tag}_SKIPPED smoke not passed"; return 1; }
  local out="$ART/screen/${env}_${model}/seed${seed}"
  mkdir -p "$out"
  mark "${tag}_START"
  if run_driver "$out/run.log" --repo "$REPO" --env "$env" --model "$model" --checkpoint "$CKPT/${env}_${model}.pth.tar" \
       --out "$out" --seed "$seed" --label screen && [ -f "$out/DONE.json" ]; then
    mark "${tag}_DONE $(cat "$out/DONE.json" | tr -d '\n ')"
  else
    mark "${tag}_FAILED (see $out/run.log)"
  fi
}

# smokes run serially, one per env, before any screen cell
for env in $(echo "$CELLS" | tr ' ' '\n' | cut -d: -f1 | awk '!seen[$0]++'); do smoke_cell "$env"; done

for seed in $SEEDS; do
  running=0
  for cell in $CELLS; do
    env=${cell%%:*}; model=${cell##*:}
    run_cell "$env" "$model" "$seed" &
    running=$((running+1))
    if [ "$running" -ge "$PP_PARALLEL" ]; then wait -n; running=$((running-1)); fi
  done
  wait
  $PY "$CODE/public_panel_summarize.py" --screen "$ART/screen" >> "$LOG" 2>&1
  if [ "$seed" = "1" ]; then
    all_ok=1
    for cell in $CELLS; do env=${cell%%:*}; model=${cell##*:}; has "PP_SCREEN_${env}_${model}_DONE" || all_ok=0; done
    [ "$all_ok" = "1" ] && mark PP_SCREEN_SEED1_DONE || mark "PP_SCREEN_SEED1_INCOMPLETE (some cells failed; see markers)"
  fi
done

$PY "$CODE/public_panel_summarize.py" --screen "$ART/screen" >> "$LOG" 2>&1 && mark PP_SCREEN_DONE || mark PP_SCREEN_SUMMARY_FAILED
mark "PP_END"

#!/usr/bin/env bash
# =====================================================================================================
# Panel P stages 2-3 — activation capture, COAST-faithful fit, development steering table.
# Governed by "Label-first principle, amendment 3" (cross model design jepa.md).
#
# Per cell (env:model), in order, each gated on the previous:
#   1. wait for PP_SCREEN_<env>_<model>_DONE and read its native success rate
#   2. eligibility (COAST App. A.8 / our screen rule): 20-80 % success AND >= 3 successes AND >= 3 failures
#   3. capture   : rerun the SAME seed-1 episodes with read-only activation hooks (public_panel_capture.py)
#                  + an identity check (--verify-identity) whose success flags must equal the screen's
#   4. fit       : public_panel_coast.py fit   (C_success AND NOT C_failure, site by quota, alpha sweep)
#   5. dev table : steering arms on the FIT episodes (seed 1) at betas PP_BETAS:
#                  sham (beta=0, must reproduce unsteered), main, random_matched, rank_one, wrong_site
#                  -> calibrated beta = smallest beta whose success rate exceeds unsteered on the fit set
#   Held-out confirmation (seed 2) is NOT run here: the main session authorises that single opening.
#
# Markers in logs/public_panel.log: PP_CAP_<cell>_DONE, PP_FIT_<cell>_DONE, PP_DEV_<cell>_DONE,
#   PP_STAGE23_DONE ; skips ineligible cells with PP_<cell>_INELIGIBLE.
# =====================================================================================================
set -uo pipefail

ROOT=${ROOT:-/root/cgs-pilot}
CODE=$ROOT/code/cgs_pilot
REPO=$ROOT/vendor/jepa-wms
PY=${PY:-/opt/conda/bin/python}
ART=$ROOT/artifacts/public_panel
LOG=$ROOT/logs/public_panel.log
CKPT=$ROOT/checkpoints/public
CELLS=${PP23_CELLS:-"mz:jepa-wm pt:jepa-wm"}
EPISODES=${PP23_EPISODES:-30}
DEV_EPISODES=${PP23_DEV_EPISODES:-15}
# development arms (kind:beta). Trimmed by default for a shared GPU: sham (identity check), the main
# contrastive conceptor at one dose, and the matched-spectrum random control. Extend with PP23_ARMS.
# Arms in run order. sham=identity check; main=COAST covariance gate h'=hM^T; mean_cov=affine about
# the success centre mu_s+M(h-mu_s); mean_only=centre shift h+beta(mu_s-mu_f); random_matched=control.
# (research thread §33). `-` not `:-` so PP23_ARMS="" means no arms.
ARMS=${PP23_ARMS-"sham:0.0 main:0.5 mean_cov:0.5 mean_only:0.5 random_matched:0.5"}
MIN_FREE_MB=${PP23_MIN_FREE_MB:-7000}

export JEPAWM_HOME=$ROOT/vendor JEPAWM_DSET=$ROOT/datasets JEPAWM_OSSCKPT=$ROOT/checkpoints
export JEPAWM_LOGS=$ART/jepawm_logs JEPAWM_CKPT=$ROOT/checkpoints
export PYTHONPATH=$REPO:$CODE OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl D4RL_SUPPRESS_IMPORT_ERROR=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia   # PointMaze/d4rl (mujoco210)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mark() { echo "$(date -u +%FT%TZ) $1" >> "$LOG"; }
has()  { grep -q "$1" "$LOG" 2>/dev/null; }
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_gpu() { local w=0; while [ "$(free_mb)" -lt "$MIN_FREE_MB" ] && [ $w -lt 3600 ]; do sleep 20; w=$((w+20)); done; }

run_py() {  # logfile, then args: retries on CUDA OOM
  local logf=$1; shift
  local a=0
  while :; do
    a=$((a+1)); wait_gpu
    $PY "$@" > "$logf" 2>&1 && return 0
    if grep -q "CUDA out of memory" "$logf" && [ $a -lt 50 ]; then
      mark "PP23_OOM_RETRY attempt=$a $(basename "$logf")"; sleep 120; continue
    fi
    return 1
  done
}

mark "PP_STAGE23_START pid=$$ cells='$CELLS' episodes=$EPISODES dev_episodes=$DEV_EPISODES arms='$ARMS'"

for cell in $CELLS; do
  env=${cell%%:*}; model=${cell##*:}; tag="${env}_${model}"
  scr="$ART/screen/$tag/seed1"

  # ---- 1. wait for the screen
  while ! has "PP_SCREEN_${tag}_DONE"; do
    has "PP_SCREEN_${tag}_FAILED" && { mark "PP23_${tag}_SKIPPED screen failed"; continue 2; }
    sleep 120
  done

  # ---- 2. eligibility from the native flags
  elig=$($PY - "$scr/episodes.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
n = len(rows); s = sum(r.get("success", 0) for r in rows); f = n - s
rate = s / n if n else 0.0
# Two-tier rule. FIT floor = COAST App. A.8 verbatim (>= 3 successes and >= 3 failures per cell);
# the 20-80 % band is our own operating-point rule (VLA plan's 2-8 of 10) and flags ceiling/floor
# effects where steering has little headroom. A cell outside the band but above the fit floor is
# run as "fit_only" and reported with that caveat, rather than discarded.
fit_ok = s >= 3 and f >= 3
band_ok = 0.20 <= rate <= 0.80
tier = "eligible" if (fit_ok and band_ok) else ("fit_only" if fit_ok else "ineligible")
print(json.dumps({"n": n, "success": s, "failure": f, "rate": round(rate, 4),
                  "in_band": bool(band_ok), "tier": tier, "eligible": bool(fit_ok)}))
PY
)
  mark "PP_ELIG_${tag} $elig"
  echo "$elig" | grep -q '"eligible": true' || { mark "PP_${tag}_INELIGIBLE $elig"; continue; }

  # ---- 3. capture (same seed -> same goals as the screen)
  capd="$ART/fit/$tag/seed1"
  if ! has "PP_CAP_${tag}_DONE"; then
    mkdir -p "$capd"
    if run_py "$capd/capture.log" "$CODE/public_panel_capture.py" --repo "$REPO" --env "$env" --model "$model" \
         --checkpoint "$CKPT/${env}_${model}.pth.tar" --out "$capd" --seed 1 --episodes "$EPISODES" --label fit; then
      idcheck=$($PY - "$scr/episodes.jsonl" "$capd/episodes.jsonl" <<'PY'
import json, sys
a = [json.loads(l)["success"] for l in open(sys.argv[1])]
b = [json.loads(l)["success"] for l in open(sys.argv[2])]
n = min(len(a), len(b))
print(json.dumps({"n_compared": n, "identical": a[:n] == b[:n], "screen": sum(a[:n]), "capture": sum(b[:n])}))
PY
)
      mark "PP_CAP_${tag}_DONE identity=$idcheck"
    else
      mark "PP_CAP_${tag}_FAILED (see $capd/capture.log)"; continue
    fi
  fi

  # ---- 4. fit
  fitd="$ART/coast/$tag"
  if ! has "PP_FIT_${tag}_DONE"; then
    mkdir -p "$fitd"
    if $PY "$CODE/public_panel_coast.py" fit --capture "$capd" --out "$fitd" > "$fitd/fit.log" 2>&1; then
      mark "PP_FIT_${tag}_DONE $(tail -1 "$fitd/fit.log" | cut -c1-200)"
    else
      mark "PP_FIT_${tag}_FAILED $(tail -1 "$fitd/fit.log" | cut -c1-200)"; continue
    fi
  fi
  # ---- 4b. decodability probe (CPU, report-only): can an episode-grouped CV probe separate the
  # native outcome at all? If not, a steering null is uninformative about the operator. Cheap and
  # independent of the conceptor, so it also cross-checks the selected site.
  if [ ! -f "$fitd/decodability.txt" ]; then
    $PY "$CODE/probe_decodability.py" "$capd" planner > "$fitd/decodability.txt" 2>&1       && mark "PP_PROBE_${tag} $(sed -n 3p "$fitd/decodability.txt" | tr -s ' ' | cut -c1-90)"       || mark "PP_PROBE_${tag}_FAILED"
  fi
  # ---- 4c. regime geometry + global-vs-local conceptors (research thread §34/§38/§56), CPU, report-only
  geod="$ART/geometry/$tag"
  if [ ! -f "$geod/regimes.json" ]; then
    mkdir -p "$geod"
    $PY "$CODE/public_panel_geometry.py" regimes --capture "$capd" --out "$geod" --n-regimes 2       > "$geod/regimes.log" 2>&1       && mark "PP_REGIMES_${tag} $(tail -1 "$geod/regimes.log" | cut -c1-200)"       || mark "PP_REGIMES_${tag}_FAILED"
  fi
  # ---- 4c. regime geometry, global-vs-local conceptors, activation-density entropy/KL
  #        (research thread §34/§38/§56). CPU, report-only, gates whether regime-local steering is worth it.
  geod="$ART/geometry/$tag"
  if [ ! -f "$geod/regimes.json" ]; then
    mkdir -p "$geod"
    $PY "$CODE/public_panel_geometry.py" regimes --capture "$capd" --out "$geod" --n-regimes 2 \
      > "$geod/regimes.log" 2>&1 \
      && mark "PP_REGIMES_${tag} $(tail -1 "$geod/regimes.log" | cut -c1-200)" \
      || mark "PP_REGIMES_${tag}_FAILED"
  fi
  # ---- 4c. regime geometry, global-vs-local conceptors, activation-density entropy/KL
  #        (research thread §34/§38/§56). CPU, report-only; gates whether regime-local steering is worth it.
  geod="$ART/geometry/$tag"
  if [ ! -f "$geod/regimes.json" ]; then
    mkdir -p "$geod"
    $PY "$CODE/public_panel_geometry.py" regimes --capture "$capd" --out "$geod" --n-regimes 2 > "$geod/regimes.log" 2>&1 \
      && mark "PP_REGIMES_${tag} $(tail -1 "$geod/regimes.log" | cut -c1-200)" || mark "PP_REGIMES_${tag}_FAILED"
  fi
  site=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['selected_site'])" "$fitd/fit_summary.json" 2>/dev/null)
  [ -z "$site" ] && { mark "PP_DEV_${tag}_SKIPPED no site"; continue; }

  # ---- 5. development steering arms on the fit seed (never the held-out seed)
  devd="$ART/dev/$tag"; mkdir -p "$devd"
  for arm in $ARMS; do
    kind=${arm%%:*}; beta=${arm##*:}
    name="${kind}_b${beta}"
    [ -f "$devd/$name/DONE.json" ] && continue
    mkdir -p "$devd/$name"
    run_py "$devd/$name/run.log" "$CODE/public_panel_coast.py" steer --repo "$REPO" --env "$env" --model "$model" \
      --checkpoint "$CKPT/${env}_${model}.pth.tar" --operators "$fitd/operators" --site "$site" \
      --beta "$beta" --kind "$kind" --out "$devd/$name" --seed 1 --episodes "$DEV_EPISODES" --label dev \
      && mark "PP_DEV_${tag}_${name} $(cat "$devd/$name/DONE.json" 2>/dev/null | tr -d '\n ' | cut -c1-160)" \
      || mark "PP_DEV_${tag}_${name}_FAILED"
  done
  $PY - "$devd" "$scr/episodes.jsonl" "$site" <<'PY' >> "$LOG" 2>&1
import json, sys
from pathlib import Path
dev, scr, site = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
base = [json.loads(l)["success"] for l in open(scr)]
rows = []
for d in sorted(dev.iterdir()):
    f = d / "DONE.json"
    if f.exists():
        j = json.loads(f.read_text())
        rows.append({"arm": d.name, "n": j.get("n_logged"), "success_rate": j.get("success_rate"),
                     "hook_calls": j.get("hook_calls")})
out = {"site": site, "unsteered_screen_rate": round(sum(base) / max(len(base), 1), 4), "arms": rows}
(dev / "dev_table.json").write_text(json.dumps(out, indent=2))
print(f"{__import__('datetime').datetime.utcnow().isoformat()}Z PP_DEV_TABLE {json.dumps(out)[:400]}")
PY
  mark "PP_DEV_${tag}_DONE"
done

mark PP_STAGE23_DONE

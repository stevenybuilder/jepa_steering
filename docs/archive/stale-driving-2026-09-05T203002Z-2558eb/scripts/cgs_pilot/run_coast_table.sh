#!/bin/bash
# run_coast_table.sh — COAST-style before/after table (arXiv:2605.17144) in the planner's own currency for every
# B-gate-PASS driving model (experiment_design.md, "COAST-style before/after endpoint", preregistered 2026-09-02 22:50 UTC).
# Chain AFTER run_drive_arms.sh has written CROSS_ARM_DONE (logs/drive_cross_arm.log); nothing under drive_* is modified.
#
# Per seed pair s (arms A and B trained on the identical clip multiset):
#   0 verdicts   $EVAL/arm{A,B}_seed$s/behavior_gate_discovery.json -> PASS / PASS_PLANNER = evaluable; FAIL = row
#                "failed behaviour gate, not evaluable" (kept in coast_table.json, no steering).
#   1 band       frozen rule, per PASS arm: $MECH/arm${ARM}_seed$s/patch_step0/band.json (patch band, discovery-frozen)
#                if present, else the top-3 step-0 sites of the primary group by relational t in
#                $MECH/arm${ARM}_seed$s/localize/interaction_map.json (steered_planner_ranking.top_relational_sites).
#   2 re-dump    run_drive_arms.sh deletes the activation dumps after the cross-arm stage, so both arms are re-dumped at
#                the union of the pair's band sites only (imagined step 0, COMMON discovery seeds, hazard+corridor tokens):
#                localize_interaction.py --sites ... -> $COAST/seed$s/dump_arm{A,B}
#   3 transport  geometry_cross_arm.py --sites <band> --steps 0 --export-transport $COAST/seed$s/transport.npz: the
#                hazard-free Procrustes map W per site and both arms' relational conceptors (rel_ped, rel_cone) + means,
#                fitted on the common discovery scenes (the registered cross-arm tests already live in drive_geometry/;
#                this run is export-only with reduced permutation counts).
#   4 steering   per PASS arm and steering source; planner goals --goal progress brake: PRIMARY = progress (goal latent =
#                the scene's hazard-free throttle true future h0a1, "make progress down the lane"; brake is preferred only
#                when the model predicts that throttle leads far from progress, so H0 ~ 0 = the unsafe baseline), SECONDARY =
#                brake (goal = same-level brake future, the gate's a1_ranks_worse currency; H0 ~ 1 by construction).
#                Calibration and selectivity use the primary goal; brake rows are descriptive.
#                  own          = the arm's own relational conceptor from the export (--transport-to == --transport-from)
#                  transported  = the OTHER arm's relational conceptor moved into this arm's coordinates through W
#                  conceptor_dir (only if $MECH/arm${ARM}_seed$s/geometry/conceptors exists: geometry_conceptor.py export)
#                CALIBRATION on the arm's DISCOVERY seeds over the frozen grid beta in {0.25, 0.5, 1.0} x {strengthen,
#                suppress} at the band (single-site rows only with DRIVE_COAST_SINGLES=1) with all controls (sham,
#                matched-spectrum random, rank-one, wrong-site, wrong-group) -> calibration.json (frozen rule: smallest
#                beta whose steered-unsteered safe-choice DiD CI excludes 0), then the sealed CONFIRMATION run on the
#                arm's confirmation seeds with --calibration-file (only the calibrated beta per family; zero refitting).
#   5 table      all confirmation rows (+ discovery rows flagged phase=discovery) -> $COAST/coast_table.json + coast_table.md
#
# Budget (smoke on armA_seed0, 2 scenes): ~1 s per config x scene, peak VRAM 1.5 GB; per PASS model and source: calibration
#   16 scenes x 33 configs ~ 9 min + confirmation ~ 4 min; re-dump + transport export per pair ~ 5 min.
# Usage:  run_coast_table.sh [--skip-wait] [--dry-run]     (env: DRIVE_BASE, DRIVE_OUT_ROOT, DRIVE_SEEDS, DRIVE_COAST_BETAS, DRIVE_COAST_SINGLES, ...)
#   --dry-run: reduced bootstrap / permutation counts and the first PASS model only (smoke of the chain; delete the output).
# Launch (remote, after CROSS_ARM_DONE):
#   ssh -n -p PORT root@HOST 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_coast_table.sh > /root/cgs-pilot/logs/coast_table_driver.log 2>&1 < /dev/null &'
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}
CODE=${DRIVE_CODE:-$BASE/code/cgs_pilot}
REPO=${DRIVE_REPO:-$BASE/vendor/jepa-wms}
MPY=${DRIVE_MPY:-/opt/conda/bin/python}
OUT=${DRIVE_OUT_ROOT:-$BASE/artifacts}
LOGDIR=${DRIVE_LOGDIR:-$BASE/logs}
MODELS=$OUT/drive_models; MERGED=$OUT/drive_factorial_merged; EVAL=$OUT/drive_eval; MECH=$OUT/drive_mech
COAST=${DRIVE_COAST_OUT:-$OUT/drive_coast_table}
CLOG=$LOGDIR/drive_cross_arm.log
LOG=${DRIVE_COAST_LOG:-$LOGDIR/drive_coast_table.log}
SEEDS=${DRIVE_SEEDS:-"0 1 2"}
BETAS=${DRIVE_COAST_BETAS:-"0.25 0.5 1.0"}          # frozen calibration grid
MODES=${DRIVE_COAST_MODES:-"strengthen suppress"}
GROUP=${DRIVE_COAST_GROUP:-hazard_corridor}
GOALS=${DRIVE_COAST_GOALS:-"progress brake"}          # first = PRIMARY currency (progress: goal = hazard-free throttle future); brake = secondary rows
SINGLES=${DRIVE_COAST_SINGLES:-0}                     # 1 = also steer each band site alone (secondary rows; ~4x the runtime)
N_TOP=3
SKIP_WAIT=${DRIVE_SKIP_WAIT:-0}; DRY=0
while [ $# -gt 0 ]; do case "$1" in --skip-wait) SKIP_WAIT=1;; --dry-run) DRY=1;; *) echo "unknown arg $1"; exit 2;; esac; shift; done
if [ $DRY = 1 ]; then STAT_ARGS="--n-boot 200"; CROSS_ARGS="--n-boot 100 --n-perm 100 --n-perm-refit 19"; else STAT_ARGS="--n-boot 2000"; CROSS_ARGS="--n-boot 200 --n-perm 200 --n-perm-refit 19"; fi
export JEPAWM_HOME=$BASE/vendor JEPAWM_OSSCKPT=$BASE/checkpoints JEPAWM_LOGS=$BASE/logs PYTHONPATH=$REPO:$CODE
export JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12}
FILTER='^INFO:|^🔧|^🧠|^🔮|^📉|Encoder: |Predictor: |^Loading|^Loaded'
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*" | tee -a $LOG; }
mkdir -p $COAST $LOGDIR
cd $CODE
echo "[$(ts)] COAST table start seeds='$SEEDS' betas='$BETAS' modes='$MODES' group=$GROUP goals='$GOALS' (first = primary) dry=$DRY out=$COAST" > $LOG
if [ $SKIP_WAIT != 1 ]; then say "waiting for CROSS_ARM_DONE in $CLOG"; until grep -q CROSS_ARM_DONE $CLOG 2>/dev/null; do sleep 120; done; fi
say "cross-arm stage done; GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"

verdict() { grep -o '"verdict": "[A-Z_]*"' $EVAL/arm$1_seed$2/behavior_gate_discovery.json 2>/dev/null | head -1 | cut -d'"' -f4; }
is_pass() { local v; v=$(verdict $1 $2); [ "$v" = PASS ] || [ "$v" = PASS_PLANNER ]; }
band_sites() {   # band_sites ARM SEED -> site ids (space separated) by the frozen rule
  local BF=$MECH/arm$1_seed$2/patch_step0/band.json MAP=$MECH/arm$1_seed$2/localize/interaction_map.json
  $MPY - $BF $MAP $N_TOP <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from steered_planner_ranking import top_relational_sites
bf, mp, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
if bf.exists():
    print(" ".join(b["site_id"] for b in json.loads(bf.read_text())["band"]))
elif mp.exists():
    print(" ".join(top_relational_sites(json.loads(mp.read_text()), n=n, step=0)))
EOF
}
band_source() { [ -f $MECH/arm$1_seed$2/patch_step0/band.json ] && echo "patch band.json" || echo "top-$N_TOP step-0 relational sites (interaction_map.json)"; }

NOT_EVAL=$COAST/not_evaluable.jsonl; : > $NOT_EVAL
DONE_ONE=0
for s in $SEEDS; do
  PASS_ARMS=""
  for ARM in A B; do
    V=$(verdict $ARM $s); V=${V:-MISSING}
    if is_pass $ARM $s; then PASS_ARMS="$PASS_ARMS $ARM"; say "seed $s arm $ARM: B-gate $V -> evaluable"
    else echo "{\"arm\": \"$ARM\", \"seed\": $s, \"verdict\": \"$V\", \"row\": \"failed behaviour gate, not evaluable\"}" >> $NOT_EVAL; say "seed $s arm $ARM: B-gate $V -> not evaluable"; fi
  done
  [ -n "$PASS_ARMS" ] || continue
  if [ $DRY = 1 ] && [ $DONE_ONE = 1 ]; then say "dry-run: one model done, stopping"; break; fi
  SD=$COAST/seed$s; mkdir -p $SD
  # ---- 1. bands (union over the pair's PASS arms for the re-dump; each arm is steered at its own band)
  ALL_SITES=""
  for ARM in $PASS_ARMS; do
    B=$(band_sites $ARM $s); [ -n "$B" ] || { say "seed $s arm $ARM: no band (no band.json / interaction_map.json) -> skipped"; continue; }
    echo "$B" > $SD/band_arm$ARM.txt; echo "{\"arm\": \"$ARM\", \"seed\": $s, \"sites\": \"$B\", \"source\": \"$(band_source $ARM $s)\"}" > $SD/band_arm$ARM.json
    say "seed $s arm $ARM band [$B] ($(band_source $ARM $s))"; ALL_SITES="$ALL_SITES $B"
  done
  SITES=$(echo $ALL_SITES | tr ' ' '\n' | sort -u | tr '\n' ' '); [ -n "$SITES" ] || continue
  # ---- 2. re-dump both arms at the band sites (step 0, common discovery seeds) and 3. export the transport
  for ARM in A B; do
    MD=$MODELS/arm${ARM}_seed$s; DUMP=$SD/dump_arm$ARM
    if [ -f $DUMP/activations/index.json ]; then say "seed $s arm $ARM: dump present, reused"; continue; fi
    say "seed $s arm $ARM: re-dump at [$SITES]"
    $MPY localize_interaction.py --repo $REPO --config $MD/eval_config.yaml --checkpoint $MD/jepa-latest.pth.tar --model-name jepa_wm_driving --artifacts $MERGED/arm$ARM \
      --output $DUMP --domain driving --seeds-file $MERGED/discovery_seeds_common.txt --sites $SITES --dump-hooks resid_post attn_out mlp_out --dump-groups egg corridor \
      --n-boot 200 --n-perm 200 2>&1 | grep -Ev "$FILTER" >> $LOG
    [ -f $DUMP/activations/index.json ] || { say "seed $s arm $ARM: re-dump FAILED"; exit 1; }
  done
  if [ ! -f $SD/transport.npz ]; then
    $MPY geometry_cross_arm.py --dump-a $SD/dump_armA --dump-b $SD/dump_armB --stimulus $MERGED/armA --out $SD/cross_arm --discovery-seeds $MERGED/discovery_seeds_common.txt \
      --steps 0 --sites $SITES --groups hazard corridor hazard_corridor --export-transport $SD/transport.npz $CROSS_ARGS >> $LOG 2>&1 || { say "seed $s: transport export FAILED"; exit 1; }
    say "seed $s: transport export written ($(du -h $SD/transport.npz | cut -f1))"
  fi
  # ---- 4. steering: calibration on discovery, sealed confirmation
  for ARM in $PASS_ARMS; do
    [ -f $SD/band_arm$ARM.txt ] || continue
    BAND=$(cat $SD/band_arm$ARM.txt); MD=$MODELS/arm${ARM}_seed$s; LEVEL=$([ "$ARM" = B ] && echo 3 || echo 1); OTHER=$([ "$ARM" = B ] && echo A || echo B)
    LABEL="JEPA-WM driving arm $ARM seed $s (jepa-latest)"
    COMMON="--repo $REPO --config $MD/eval_config.yaml --checkpoint $MD/jepa-latest.pth.tar --model-name jepa_wm_driving --artifacts $MERGED/arm$ARM --domain driving \
            --primary-hazard-level $LEVEL --arm $ARM --model-seed $s --band-sites $BAND --beta $BETAS --conceptor-mode $MODES --group $GROUP --goal $GOALS --step 0 $STAT_ARGS $([ $SINGLES = 1 ] || echo --no-singles)"
    SOURCES=("own|--transported-conceptor $SD/transport.npz --transport-from $ARM --transport-to $ARM"
             "transported|--transported-conceptor $SD/transport.npz --transport-from $OTHER --transport-to $ARM")
    CDIR=$MECH/arm${ARM}_seed$s/geometry/conceptors
    [ -d $CDIR ] && SOURCES+=("conceptor_dir|--conceptor-dir $CDIR --conceptor-key safety")
    for entry in "${SOURCES[@]}"; do
      SRC=${entry%%|*}; SPEC=${entry#*|}
      RD=$SD/arm$ARM/$SRC; mkdir -p $RD
      if [ ! -f $RD/discovery/steered_ranking.json ]; then
        say "seed $s arm $ARM [$SRC]: calibration on discovery seeds"
        $MPY steered_planner_ranking.py $COMMON $SPEC --model-label "$LABEL" --seeds-file $MERGED/arm$ARM/discovery_seeds.txt --out $RD/discovery --calibrate-out $RD/calibration.json 2>&1 | grep -Ev "$FILTER" >> $LOG
        [ -f $RD/discovery/steered_ranking.json ] || { say "seed $s arm $ARM [$SRC]: calibration run FAILED"; exit 1; }
      fi
      if [ ! -f $RD/confirmation/steered_ranking.json ]; then
        say "seed $s arm $ARM [$SRC]: sealed confirmation ($(tr '\n' ' ' < $MERGED/arm$ARM/confirmation_seeds.txt))"
        $MPY steered_planner_ranking.py $COMMON $SPEC --model-label "$LABEL" --seeds-file $MERGED/arm$ARM/confirmation_seeds.txt --out $RD/confirmation --calibration-file $RD/calibration.json 2>&1 | grep -Ev "$FILTER" >> $LOG
        [ -f $RD/confirmation/steered_ranking.json ] || { say "seed $s arm $ARM [$SRC]: confirmation run FAILED"; exit 1; }
      fi
      say "seed $s arm $ARM [$SRC] done: $(($(wc -l < $RD/confirmation/coast_table.md) - 2)) confirmation rows"
      DONE_ONE=1
    done
  done
  rm -rf $SD/dump_armA/activations $SD/dump_armB/activations; say "seed $s: re-dump activations deleted (transport.npz + interaction maps kept)"
done

# ---- 5. aggregate every model's rows
$MPY - $COAST $NOT_EVAL >> $LOG 2>&1 <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from steered_planner_ranking import INTERPRETATION_SCOPE, coast_markdown
root, not_eval = Path(sys.argv[1]), Path(sys.argv[2])
rows, models, cal = [], [], {}
for p in sorted(root.glob("seed*/arm*/*/confirmation/steered_ranking.json")):
    d = json.loads(p.read_text()); src = p.parents[1].name
    for r in d["coast_table"]["rows"]:
        rows.append({**r, "phase": "confirmation", "source_dir": src})
    disc = json.loads((p.parents[1] / "discovery" / "steered_ranking.json").read_text())
    for r in disc["coast_table"]["rows"]:
        rows.append({**r, "phase": "discovery", "source_dir": src, "confirmatory": False})
    models.append({"arm": d["arm"], "seed": d["model_seed"], "source": src, "checkpoint_sha256": d["checkpoint_sha256"], "n_confirmation_scenes": d["n_scenes"], "n_discovery_scenes": disc["n_scenes"],
                   "band": d["steering"]["sites"], "families_without_calibrated_beta": d["families_without_calibrated_beta"]})
    cal[f"arm{d['arm']}_seed{d['model_seed']}/{src}"] = {k: v.get("beta") for k, v in (d.get("calibration") or {}).get("families", {}).items()}
ne = [json.loads(l) for l in not_eval.read_text().splitlines() if l.strip()] if not_eval.exists() else []
conf = [r for r in rows if r["phase"] == "confirmation"]
primary = next((json.loads(p.read_text()).get("goal_currency", {}).get("primary") for p in root.glob("seed*/arm*/*/confirmation/steered_ranking.json")), "progress")
goal_def = next((json.loads(p.read_text()).get("goal_currency") for p in root.glob("seed*/arm*/*/confirmation/steered_ranking.json")), None)
out = {"protocol": "cgs-coast-table-v0.1", "columns": ["arm", "seed", "goal", "setting", "safe_choice_H1_ped", "safe_choice_H1_obj", "safe_choice_H0", "safe_choice_H0prime", "n_scenes", "CIs", "did_solid_vs_h0", "selective", "confirmatory"],
       "goal_currency": goal_def, "primary_goal": primary,
       "rows": rows, "n_confirmation_rows": len(conf), "models": models, "calibrated_betas": cal, "not_evaluable": ne,
       "calibration_rule": "smallest beta in {0.25, 0.5, 1.0} whose steered-unsteered safe-choice DiD (solid identity in lane vs H0) scene-bootstrap CI excludes 0 on discovery scenes; band = patch band.json if present else top-3 step-0 relational sites; confirmation rows use that beta with zero refitting",
       "interpretation_scope": INTERPRETATION_SCOPE}
(root / "coast_table.json").write_text(json.dumps(out, indent=1) + "\n")
prim_c = [r for r in conf if r.get("goal_primary", True)]; sec_c = [r for r in conf if not r.get("goal_primary", True)]
disc = [r for r in rows if r["phase"] == "discovery"]
md = ["# COAST-style before/after table (planner safe-choice rate, sealed confirmation scenes)", "", "Scope: " + INTERPRETATION_SCOPE, "",
      f"Primary goal currency: {primary} -- " + (goal_def or {}).get("definition", {}).get(primary, ""), "",
      f"## Confirmation rows, primary goal ({primary})", "", coast_markdown(prim_c) if prim_c else "(no evaluable model)", "",
      "## Confirmation rows, secondary goal (brake; descriptive, no unsafe baseline by construction)", "", coast_markdown(sec_c) if sec_c else "(none)", "",
      "## Discovery rows (calibration; not confirmatory)", "", coast_markdown(disc) if disc else "", "", "## Not evaluable", ""] + [f"- arm {r['arm']} seed {r['seed']}: {r['row']} (verdict {r['verdict']})" for r in ne]
(root / "coast_table.md").write_text("\n".join(md) + "\n")
print("COAST_TABLE", json.dumps({"n_rows": len(rows), "n_confirmation_rows": len(conf), "models": len(models), "not_evaluable": len(ne)}))
EOF
say "COAST_TABLE_DONE -> $COAST/coast_table.json, $COAST/coast_table.md"

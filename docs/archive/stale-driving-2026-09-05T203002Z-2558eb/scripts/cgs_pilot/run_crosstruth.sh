#!/bin/bash
# run_crosstruth.sh — CROSS-TRUTH evaluation of the driving models (protocol v0.8 factorial, arms A/B).
#
# Each model arm X in {A,B}, seed s in {0,1,2} is scored against the OTHER arm's merged factorial
# ($MERGED/arm{Y}, Y != X): same seeds, same renders up to contact, but the futures follow arm Y's physics
# (arm A: pedestrian solid / cone ghost; arm B: cone solid / pedestrian ghost). DISCOVERY scenes only, using the
# SAME discovery list as the main run (read from the main eval dir $EVAL/arm{X}_seed{s}/discovery_seeds.txt, falling
# back to its EVAL_SCOPE.json, then to $MERGED/arm{Y}/discovery_seeds.txt).
#
# Prediction (recorded before the first number): a model scored against the other arm's truth fails at BOTH levels —
# at level 1 model A predicts a pedestrian collision that arm-B truth does not contain, at level 3 it predicts pass-through
# where arm-B truth collides — and symmetrically for model B. So the cross-truth interaction NMSE should be high
# (captured fraction low) and Rec_h1-Rec_h0 / flip rate should collapse or reverse, at both levels.
#
# Per model, mirroring run_drive_arms.sh stage 4 (EVAL) minus the own-arm controls (training-clip audit, retrieval
# baseline; both are arm-X training-set controls and are irrelevant to the truth swap):
#   1. eval dir $OUTROOT/arm{X}_seed{s}_on_arm{Y}/: arm Y's merge restricted to the discovery seeds with ABSOLUTE
#      artifact paths, seed lists + summary copied, masks symlink -> $MERGED/arm{Y}/masks, EVAL_SCOPE.json
#   2. latent cache (GPU) with the SAME shim as run_wave_drive.sh (latent_cache.load_token_mask = hazard|corridor),
#      CFG=$MODELS/arm{X}_seed{s}/eval_config.yaml, CKPT=jepa-latest.pth.tar, MODEL_NAME=jepa_wm_driving
#   3. counterfactual_validity_gate.py at BOTH primary levels (1 and 3), discovery seeds file, --domain driving
#      -> cf_gate_discovery_level{1,3}.json  (the manifest is discovery-only, so an "all" gate would be identical)
#   4. behavior_gate.py per level, plain (behavior_gate_discovery_level{L}.json) and, for symmetry with the main run,
#      with --cross-gate = the other level (behavior_gate_discovery_level{L}_crossgate.json)
#   5. latent_cache.npz DELETED, crosstruth_summary.json written (both levels; the main run's own-truth numbers at the
#      same levels are attached as "own_truth_reference" for the contrast)
#   then the aggregate $OUTROOT/summary.json is (re)written: 2x2x2 table model arm x truth arm x level of interaction
#   NMSE median, captured fraction (1 - NMSE median, the main run's definition), Rec_h1-Rec_h0 (+ sign-flip p), flip rate
#   and n, per seed and pooled over seeds (mean, and median for NMSE); own-truth cells come from the main eval dirs.
#
# Scheduling: the model's MAIN eval must have finished first (marker DRIVE_WAVE_DONE in $LOGDIR/drive_wave_arm{X}_seed{s}.log)
# so the GPU is shared sensibly with run_drive_arms.sh; the loop order is the main runner's (seed-outer, A then B).
# Disk: a discovery-only latent cache is ~2.0 GB and the box is < 10 GB free, so the cache is built only when at least
# $MIN_FREE_GB are free and deleted right after its gates. Nothing under $EVAL / $MERGED / $MODELS is modified.
# Log $LOGDIR/drive_crosstruth.log; per-model marker CROSSTRUTH_MODEL_DONE, final marker CROSSTRUTH_DONE.
# Re-runnable: a model with crosstruth_summary.json present is skipped (CROSSTRUTH_FORCE=1 recomputes).
# Usage: run_crosstruth.sh   (env: CROSSTRUTH_SEEDS="0 1 2", CROSSTRUTH_ARMS="A B", CROSSTRUTH_MIN_FREE_GB=3.5, CROSSTRUTH_FORCE=0)
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}
CODE=${DRIVE_CODE:-$BASE/code/cgs_pilot}
REPO=${DRIVE_REPO:-$BASE/vendor/jepa-wms}
MPY=${DRIVE_MPY:-/opt/conda/bin/python}
LOGDIR=${DRIVE_LOGDIR:-$BASE/logs}
MERGED=$BASE/artifacts/drive_factorial_merged
MODELS=$BASE/artifacts/drive_models
EVAL=$BASE/artifacts/drive_eval
OUTROOT=${CROSSTRUTH_OUT:-$BASE/artifacts/drive_eval_crosstruth}
SEEDS=${CROSSTRUTH_SEEDS:-"0 1 2"}; ARMS=${CROSSTRUTH_ARMS:-"A B"}
MIN_FREE_GB=${CROSSTRUTH_MIN_FREE_GB:-3.5}; FORCE=${CROSSTRUTH_FORCE:-0}
LOG=$LOGDIR/drive_crosstruth.log
export JEPAWM_HOME=$BASE/vendor JEPAWM_OSSCKPT=$BASE/checkpoints JEPAWM_LOGS=$BASE/logs PYTHONPATH=$REPO:$CODE
export JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
mkdir -p $OUTROOT $LOGDIR
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
disk() { df -h $BASE | tail -1 | awk '{print "disk used "$3" of "$2", free "$4}'; }
free_gb() { df -BM $BASE | tail -1 | awk '{sub("M","",$4); printf "%.2f", $4/1024}'; }
free_ok() { awk -v f="$(free_gb)" -v m="$MIN_FREE_GB" 'BEGIN{exit !(f >= m)}'; }
other() { [ "$1" = "A" ] && echo B || echo A; }
cd $CODE
say "run_crosstruth start seeds='$SEEDS' arms='$ARMS' out=$OUTROOT min_free_gb=$MIN_FREE_GB force=$FORCE omp=$OMP_NUM_THREADS; $(disk)"
say "PREDICTION (pre-registered here, before any cross-truth number): model X vs truth Y!=X FAILS at level 1 (predicts a collision the truth lacks) AND level 3 (predicts pass-through where the truth collides); own-truth cells (main run) pass."

aggregate() {   # (re)write $OUTROOT/summary.json from the main eval dirs (own truth) + the cross-truth dirs
  $MPY - $EVAL $OUTROOT "$SEEDS" "$ARMS" >> $LOG 2>&1 <<'EOF' || say "aggregate: FAILED (see log)"
import json, statistics, sys
from pathlib import Path
ev_root, out_root, seeds, arms = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(), sys.argv[4].split()
other = {"A": "B", "B": "A"}
def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None
def summ(gate):
    if not gate or "pooled" not in gate: return None
    p = gate["pooled"]; pc = gate.get("planner_currency", {}); tg = pc.get("token_groups", {}); prim = pc.get("primary_group")
    blk = (tg.get(prim) or {}).get("pooled", {}) if prim else {}
    nm = p.get("interaction_nmse", {})
    rec = blk.get("rec_h1_minus_h0") or {}
    rk = blk.get("planner_ranking") or blk.get("ranking") or {}
    med = nm.get("median")
    return {"n": gate.get("n_scenes"), "interaction_nmse_median": med, "captured_fraction": (None if med is None else 1.0 - float(med)),
            "rec_h1_minus_h0": rec.get("mean"), "rec_h1_minus_h0_p": rec.get("sign_flip_p"), "flip_rate_h1_vs_h0": rk.get("flip_rate_h1_vs_h0"),
            "gate_verdict": gate.get("verdict"), "verdict_with_specificity": gate.get("verdict_with_specificity")}
def own_gate(X, s, L):      # main run: primary level file is cf_gate_discovery.json, the other level cf_gate_discovery_level{L}.json
    prim = "1" if X == "A" else "3"
    d = ev_root / f"arm{X}_seed{s}"
    g = load(d / ("cf_gate_discovery.json" if L == prim else f"cf_gate_discovery_level{L}.json"))
    if g is None: return None
    r = summ(g); b = load(d / ("behavior_gate_discovery.json" if L == prim else f"behavior_gate_discovery_level{L}.json"))
    if r is not None: r["b_gate_verdict"] = (b or {}).get("verdict"); r["source"] = str(d)
    return r
def cross_gate(X, s, L):
    d = out_root / f"arm{X}_seed{s}_on_arm{other[X]}"
    g = load(d / f"cf_gate_discovery_level{L}.json")
    if g is None: return None
    r = summ(g); b = load(d / f"behavior_gate_discovery_level{L}.json"); bx = load(d / f"behavior_gate_discovery_level{L}_crossgate.json")
    if r is not None: r["b_gate_verdict"] = (b or {}).get("verdict"); r["b_gate_verdict_crossgate"] = (bx or {}).get("verdict"); r["source"] = str(d)
    return r
def pool(rows):
    rows = [r for r in rows if r]
    if not rows: return {"n_seeds": 0}
    def col(k): return [float(r[k]) for r in rows if r.get(k) is not None]
    def mean(v): return statistics.fmean(v) if v else None
    def med(v): return statistics.median(v) if v else None
    return {"n_seeds": len(rows), "n_scenes": [r["n"] for r in rows],
            "interaction_nmse_median_mean": mean(col("interaction_nmse_median")), "interaction_nmse_median_median": med(col("interaction_nmse_median")),
            "captured_fraction_mean": mean(col("captured_fraction")), "rec_h1_minus_h0_mean": mean(col("rec_h1_minus_h0")),
            "rec_h1_minus_h0_p": [r.get("rec_h1_minus_h0_p") for r in rows], "flip_rate_h1_vs_h0_mean": mean(col("flip_rate_h1_vs_h0")),
            "b_gate_verdicts": [r.get("b_gate_verdict") for r in rows]}
table = {}
for X in arms:
    table[f"model_arm{X}"] = {}
    for Y in arms:
        table[f"model_arm{X}"][f"truth_arm{Y}"] = {}
        for L in ("1", "3"):
            per_seed = {s: (own_gate(X, s, L) if Y == X else cross_gate(X, s, L)) for s in seeds}
            table[f"model_arm{X}"][f"truth_arm{Y}"][f"level{L}"] = {"truth_kind": "own" if Y == X else "cross", "pooled": pool(list(per_seed.values())), "per_seed": per_seed}
n_cross_done = sum(1 for X in arms for s in seeds for L in ("1", "3") if table[f"model_arm{X}"][f"truth_arm{other[X]}"][f"level{L}"]["per_seed"][s])
out = {"design": "model arm x truth arm x primary hazard level; discovery scenes only; own-truth cells from artifacts/drive_eval (main run), cross-truth cells from this script",
       "metrics": {"interaction_nmse_median": "pooled.interaction_nmse.median of the gate", "captured_fraction": "1 - interaction NMSE median (main-run definition)",
                   "rec_h1_minus_h0": "primary token group (hazard_corridor) pooled mean, with scene-level sign-flip p", "flip_rate_h1_vs_h0": "CEM ranking flip rate H1 vs H0", "n": "discovery scenes"},
       "prediction": "cross-truth cells fail at both levels (level 1: predicted collision absent from the truth; level 3: predicted pass-through where the truth collides); own-truth cells pass",
       "seeds": seeds, "arms": arms, "n_cross_cells_done": n_cross_done, "n_cross_cells_total": 2 * len(arms) * len(seeds), "table": table}
(out_root / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
def fmt(c):
    p = c["pooled"]
    if not p.get("n_seeds"): return "n/a"
    return f"nmse={p['interaction_nmse_median_median']:.3f} cap={p['captured_fraction_mean']:.3f} rec={p['rec_h1_minus_h0_mean']:+.3f} flip={p['flip_rate_h1_vs_h0_mean']:.2f} seeds={p['n_seeds']}"
print("AGGREGATE " + " | ".join(f"m{X}/t{Y}/L{L}: {fmt(table[f'model_arm{X}'][f'truth_arm{Y}'][f'level{L}'])}" for X in arms for Y in arms for L in ("1", "3")))
EOF
}

for s in $SEEDS; do for X in $ARMS; do
  Y=$(other $X); MD=$MODELS/arm${X}_seed$s; MEV=$EVAL/arm${X}_seed$s; EV=$OUTROOT/arm${X}_seed${s}_on_arm$Y; WLOG=$LOGDIR/drive_wave_arm${X}_seed$s.log
  LABEL="JEPA-WM driving arm $X seed $s (jepa-latest) scored on arm $Y truth"
  if [ -f $EV/crosstruth_summary.json ] && [ "$FORCE" != 1 ]; then say "model arm$X seed$s on arm$Y: crosstruth_summary.json present, skipping"; continue; fi
  # wait for the model's main eval (GPU sharing + the main eval dir provides the discovery list)
  if ! grep -q DRIVE_WAVE_DONE $WLOG 2>/dev/null; then say "model arm$X seed$s: waiting for the main eval (DRIVE_WAVE_DONE in $WLOG)"; fi
  until grep -q DRIVE_WAVE_DONE $WLOG 2>/dev/null; do sleep 120; done
  for f in $MD/eval_config.yaml $MD/jepa-latest.pth.tar $MERGED/arm$Y/manifest.jsonl; do [ -f $f ] || { say "model arm$X seed$s: missing $f; ABORT"; exit 1; }; done
  # disk guard: the cache is ~2 GB; never start it with less than MIN_FREE_GB free
  free_ok || say "model arm$X seed$s: waiting for disk (free $(free_gb) GB < $MIN_FREE_GB GB)"
  until free_ok; do sleep 120; done
  say "model arm $X seed $s on arm $Y truth: start; $(disk)"
  rm -rf $EV; mkdir -p $EV
  # 1. eval dir: arm Y's merge restricted to the MAIN run's discovery list (absolute artifact paths, masks symlink, scope)
  $MPY - $MERGED/arm$Y $EV $MEV $X $s >> $LOG 2>&1 <<'EOF' || { say "model arm$X seed$s: eval-dir prep FAILED; ABORT"; exit 1; }
import json, os, shutil, sys
from pathlib import Path
src, dst, mev, X, s = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
disc_src = None
for cand in (mev / "discovery_seeds.txt",):
    if cand.exists(): disc = {int(x) for x in cand.read_text().split()}; disc_src = str(cand); break
if disc_src is None and (mev / "EVAL_SCOPE.json").exists():
    disc = {int(x) for x in json.loads((mev / "EVAL_SCOPE.json").read_text())["discovery_seeds"]}; disc_src = str(mev / "EVAL_SCOPE.json")
if disc_src is None:
    disc = {int(x) for x in (src / "discovery_seeds.txt").read_text().split()}; disc_src = str(src / "discovery_seeds.txt") + " (FALLBACK: main eval dir absent)"
truth_disc = {int(x) for x in (src / "discovery_seeds.txt").read_text().split()}
n = 0; missing = 0; dropped = 0; seen = set()
with (src / "manifest.jsonl").open() as f, (dst / "manifest.jsonl").open("w") as g:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        if int(r["seed"]) not in disc: dropped += 1; continue
        p = (src / r["artifact"]).resolve(); r["artifact"] = str(p); missing += not p.exists(); n += 1; seen.add(int(r["seed"]))
        g.write(json.dumps(r) + "\n")
for p in src.iterdir():
    if p.is_file() and p.name != "manifest.jsonl" and p.suffix in (".json", ".txt"):
        shutil.copy2(p, dst / p.name)
(dst / "discovery_seeds.txt").write_text("".join(f"{x}\n" for x in sorted(disc)))     # the MAIN run's list overrides arm Y's copy
os.symlink(str((src / "masks").resolve()), str(dst / "masks"))
scope = {"scope": "CROSS-TRUTH: model arm %s seed %s scored on arm %s truth; discovery scenes only (confirmation sealed)" % (X, s, sys.argv[1].rsplit("arm", 1)[-1]),
         "model_arm": X, "model_seed": int(s), "truth_arm": sys.argv[1].rsplit("arm", 1)[-1], "n_cells": n, "n_cells_confirmation_dropped": dropped,
         "discovery_seeds": sorted(disc), "discovery_seeds_source": disc_src, "discovery_seeds_absent_from_truth_manifest": sorted(disc - seen),
         "truth_arm_discovery_list_identical": truth_disc == disc, "merge_dir": str(src), "main_eval_dir": str(mev)}
(dst / "EVAL_SCOPE.json").write_text(json.dumps(scope, indent=1) + "\n")
assert missing == 0, f"{missing} cell files missing"
print(f"cross-truth eval dir {dst}: {n} discovery cells from arm-{scope['truth_arm']} truth ({dropped} confirmation cells sealed; discovery list from {disc_src}; identical to truth arm's list: {truth_disc == disc}; absent seeds {sorted(disc - seen)}), masks -> {os.readlink(dst / 'masks')}")
EOF
  # 2. latent cache (GPU) — same shim as run_wave_drive.sh (driving token mask = hazard|corridor)
  $MPY -c "import sys, latent_cache as lc, token_groups as tg; lc.load_token_mask = tg.cache_token_mask_driving; sys.argv = ['latent_cache.py', '--artifacts', '$EV', '--cache', '$EV/latent_cache.npz', '--repo', '$REPO', '--config', '$MD/eval_config.yaml', '--checkpoint', '$MD/jepa-latest.pth.tar', '--model-name', 'jepa_wm_driving']; lc.main()" >> $LOG 2>&1
  [ -f $EV/latent_cache.npz ] || { say "model arm$X seed$s: latent cache FAILED (see log); ABORT"; exit 1; }
  say "model arm $X seed $s on arm $Y: latent cache $(du -sh $EV/latent_cache.npz | cut -f1); $(disk)"
  # 3. counterfactual-validity gate at BOTH primary levels (discovery seeds; the manifest is discovery-only)
  for L in 1 3; do
    $MPY counterfactual_validity_gate.py --artifacts $EV --cache $EV/latent_cache.npz --output $EV/cf_gate_discovery_level$L.json --seeds-file $EV/discovery_seeds.txt \
      --domain driving --primary-hazard-level $L --model-label "$LABEL (level $L)" >> $LOG 2>&1
    [ -f $EV/cf_gate_discovery_level$L.json ] || say "model arm$X seed$s: gate level $L produced no output (see log)"
  done
  # 4. B-gate per level: plain, and with --cross-gate = the other level (symmetry with the main run's T1c' route)
  for L in 1 3; do O=$([ $L = 1 ] && echo 3 || echo 1)
    [ -f $EV/cf_gate_discovery_level$L.json ] || continue
    $MPY behavior_gate.py --gate $EV/cf_gate_discovery_level$L.json --output $EV/behavior_gate_discovery_level$L.json >> $LOG 2>&1 || true
    [ -f $EV/cf_gate_discovery_level$O.json ] && { $MPY behavior_gate.py --gate $EV/cf_gate_discovery_level$L.json --cross-gate $EV/cf_gate_discovery_level$O.json --output $EV/behavior_gate_discovery_level${L}_crossgate.json >> $LOG 2>&1 || true; }
  done
  # 5. cache deleted, per-model summary
  rm -f $EV/latent_cache.npz; say "model arm $X seed $s on arm $Y: latent_cache.npz deleted; $(disk)"
  $MPY - $EV $MEV $X $s $Y >> $LOG 2>&1 <<'EOF' || say "model arm$X seed$s: crosstruth_summary FAILED (see log)"
import json, sys
from pathlib import Path
ev, mev, X, s, Y = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception as e: return None
def summ(gate):
    if not gate or "pooled" not in gate: return None
    p = gate["pooled"]; pc = gate.get("planner_currency", {}); tg = pc.get("token_groups", {}); prim = pc.get("primary_group")
    blk = (tg.get(prim) or {}).get("pooled", {}) if prim else {}
    nm = p.get("interaction_nmse", {}); rec = blk.get("rec_h1_minus_h0") or {}; rk = blk.get("planner_ranking") or blk.get("ranking") or {}
    med = nm.get("median")
    return {"n": gate.get("n_scenes"), "interaction_nmse_median": med, "interaction_nmse_ci": [nm.get("ci_low"), nm.get("ci_high")],
            "captured_fraction": (None if med is None else 1.0 - float(med)), "rec_h1_minus_h0": rec.get("mean"), "rec_h1_minus_h0_p": rec.get("sign_flip_p"),
            "flip_rate_h1_vs_h0": rk.get("flip_rate_h1_vs_h0"), "gate_verdict": gate.get("verdict"), "verdict_with_specificity": gate.get("verdict_with_specificity")}
prim = "1" if X == "A" else "3"
out = {"model_arm": X, "model_seed": int(s), "truth_arm": Y, "scope": load(ev / "EVAL_SCOPE.json"), "levels": {}, "own_truth_reference": {}}
for L in ("1", "3"):
    r = summ(load(ev / f"cf_gate_discovery_level{L}.json")); b = load(ev / f"behavior_gate_discovery_level{L}.json") or {}; bx = load(ev / f"behavior_gate_discovery_level{L}_crossgate.json") or {}
    out["levels"][f"level{L}"] = {"gate": r, "b_gate_verdict": b.get("verdict"), "b_gate_route": b.get("t1c_route"), "b_gate_captured_fraction": b.get("model_captured_interaction_fraction"),
                                  "b_gate_verdict_crossgate": bx.get("verdict"), "b_gate_route_crossgate": bx.get("t1c_route")}
    o = summ(load(mev / ("cf_gate_discovery.json" if L == prim else f"cf_gate_discovery_level{L}.json")))
    ob = load(mev / ("behavior_gate_discovery.json" if L == prim else f"behavior_gate_discovery_level{L}.json")) or {}
    out["own_truth_reference"][f"level{L}"] = {"gate": o, "b_gate_verdict": ob.get("verdict"), "source": str(mev)}
def fmt(r):
    if not r: return "n/a"
    return f"nmse={r['interaction_nmse_median']:.3f} cap={r['captured_fraction']:.3f} rec={r['rec_h1_minus_h0']:+.3f}(p={r['rec_h1_minus_h0_p']}) flip={r['flip_rate_h1_vs_h0']} n={r['n']}"
lines = []
for L in ("1", "3"):
    c = out["levels"][f"level{L}"]; o = out["own_truth_reference"][f"level{L}"]
    lines.append(f"L{L} cross(truth {Y}): {fmt(c['gate'])} B-gate {c['b_gate_verdict']}/{c['b_gate_verdict_crossgate']} || own(truth {X}): {fmt(o['gate'])} B-gate {o['b_gate_verdict']}")
out["readout"] = lines
(ev / "crosstruth_summary.json").write_text(json.dumps(out, indent=1) + "\n")
print(f"CROSSTRUTH_SUMMARY model arm{X} seed{s} on arm{Y}: " + " ; ".join(lines))
EOF
  say "CROSSTRUTH_MODEL_DONE arm$X seed$s on arm$Y"
  aggregate
done; done
aggregate
say "CROSSTRUTH_DONE; $(disk)"

#!/bin/bash
# run_v08_eval.sh — protocol v0.8 varied-geometry factorial (cross model design jepa.md, A-2 behaviour gate).
#   Waits for V08_FACT_DONE (run_v08_factorial.sh), merges every batch per arm (merge_heldout.py, hash split, label
#   validation), combines the STATIC distance batches (d06 d07 d08 d10) into one discovery set per arm, then runs the
#   frozen gate stage (run_wave_drive.sh: latent cache -> validity gate -> training audit -> retrieval -> B-gate, plus the
#   cross-identity gate and the T1c' B-gate exactly as run_drive_arms.sh does) on the six existing trained models.
#   Then the same for the knock-over batch (dyn09). Nothing under drive_eval/ drive_mech/ drive_geometry/ is touched.
#   Discovery scenes only; confirmation seeds of the v0.8 split stay sealed. Marker: V08_EVAL_DONE in logs/drive_v08_eval.log.
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}; CODE=$BASE/code/cgs_pilot; REPO=$BASE/vendor/jepa-wms
MPY=/opt/conda/bin/python; DPY=/opt/conda/envs/metadrive/bin/python
FACT=$BASE/artifacts/drive_factorial_v08; MERGED=$BASE/artifacts/drive_v08_merged; EVAL=$BASE/artifacts/drive_v08_eval
MODELS=$BASE/artifacts/drive_models; FEAT=$BASE/artifacts/drive_features
LOGDIR=$BASE/logs; LOG=$LOGDIR/drive_v08_eval.log; FLOG=$LOGDIR/drive_factorial_v08.log
SEEDS=${DRIVE_SEEDS:-"0 1 2"}; STATIC=${V08_STATIC:-"d06 d07 d08 d10"}; DYN=${V08_DYN:-"dyn09"}
KEEP_CACHE_SEEDS=${V08_KEEP_CACHE_SEEDS:-"0"}   # latent caches kept for these model seeds (relational transport); others deleted (disk)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
ts(){ date -u +%FT%TZ; }; say(){ echo "[$(ts)] $*" >> $LOG; }; disk(){ df -h $BASE | tail -1 | awk '{print "disk free " $4}'; }
mkdir -p $MERGED $EVAL
if [ "${V08_SKIP_WAIT:-0}" != "1" ]; then say "waiting for V08_FACT_DONE in $FLOG"; until grep -q V08_FACT_DONE $FLOG 2>/dev/null; do sleep 60; done; fi
say "V08 eval start seeds='$SEEDS' static='$STATIC' dyn='$DYN'; $(disk)"

# ---------------------------------------------------------------- 1. merge every batch per arm (frozen merge, v0.8 driving)
for b in $STATIC $DYN; do for ARM in A B; do
  MO=$MERGED/$b/arm$ARM; [ -f $MO/manifest.jsonl ] && { say "merge $b arm$ARM present, reused"; continue; }
  rm -rf $MO; mkdir -p $MO
  LAB=$FACT/$b/_validation/hazard_label_validation.json
  [ -f $LAB ] || { say "merge $b arm$ARM: validator JSON missing ($LAB) -> batch skipped"; rm -rf $MO; continue; }
  $DPY $CODE/merge_heldout.py --root $FACT/$b/arm$ARM --output $MO --protocol cgs-metadrive-pilot-v0.8 --domain driving \
    --label-validation $LAB --split-seed 0 --exclude-seeds >> $LOGDIR/drive_v08_merge.log 2>&1 || { say "merge $b arm$ARM FAILED (see drive_v08_merge.log)"; rm -rf $MO; continue; }
  mkdir -p $MO/masks; for d in $FACT/$b/arm$ARM/seed_*/masks; do [ -d $d ] && cp -n $d/*.npz $MO/masks/ 2>/dev/null; done
  say "merged $b arm$ARM: $(grep -c . $MO/manifest.jsonl) cells, masks $(ls $MO/masks | wc -l); discovery [$(tr '\n' ' ' < $MO/discovery_seeds.txt)] confirmation [$(tr '\n' ' ' < $MO/confirmation_seeds.txt)]"
done; done

# ---------------------------------------------------------------- 2. combine static batches into one set per arm (absolute artifact paths)
combine() { # combine ARM OUTNAME batch...
  local ARM=$1 NAME=$2; shift 2; local CO=$MERGED/$NAME/arm$ARM; rm -rf $CO; mkdir -p $CO/masks
  $MPY - $CO $MERGED $ARM "$@" >> $LOG 2>&1 <<'EOF' || { say "combine $NAME arm$ARM FAILED"; return 1; }
import json, os, sys
from pathlib import Path
co, merged, arm = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]; batches = sys.argv[4:]
rows, disc, conf, summ, used = [], set(), set(), None, []
for b in batches:
    src = merged / b / f"arm{arm}"
    if not (src / "manifest.jsonl").exists(): print(f"combine: {b} arm{arm} missing, skipped"); continue
    used.append(b)
    for line in (src / "manifest.jsonl").read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line); r["artifact"] = str((src / r["artifact"]).resolve()); r["v08_batch"] = b; rows.append(r)
    disc |= {int(x) for x in (src / "discovery_seeds.txt").read_text().split()}
    conf |= {int(x) for x in (src / "confirmation_seeds.txt").read_text().split()}
    for p in (src / "masks").iterdir():
        dst = co / "masks" / p.name
        if not dst.exists(): os.symlink(str(p.resolve()), str(dst))
    s = json.loads((src / "summary.json").read_text())
    if summ is None: summ = s; summ["v08_combined_from"] = [b]
    else:
        summ["v08_combined_from"].append(b)
        for k, v in s.items():
            if isinstance(v, list) and isinstance(summ.get(k), list): summ[k] = summ[k] + v
            elif isinstance(v, dict) and isinstance(summ.get(k), dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict) and isinstance(summ[k].get(kk), dict): summ[k][kk].update(vv)
                    elif kk not in summ[k]: summ[k][kk] = vv
assert rows, "no rows"
assert not (disc & conf), "discovery/confirmation overlap"
(co / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
(co / "discovery_seeds.txt").write_text("".join(f"{s}\n" for s in sorted(disc)))
(co / "confirmation_seeds.txt").write_text("".join(f"{s}\n" for s in sorted(conf)))
summ["v08_note"] = "combined per-batch merges (lists concatenated, per-seed dicts merged); scalar fields are from the first batch"
(co / "summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
print(f"combine arm{arm}: batches {used}, {len(rows)} cells, discovery {len(disc)}, confirmation {len(conf)}, masks {len(list((co/'masks').iterdir()))}")
EOF
}
for ARM in A B; do combine $ARM static $STATIC; done
for ARM in A B; do combine $ARM dyn $DYN; done

# ---------------------------------------------------------------- 3. gate stage per trained model on each combined set (discovery scenes only)
eval_model() { # eval_model SETNAME ARM SEED
  local SET=$1 ARM=$2 s=$3; local MD=$MODELS/arm${ARM}_seed$s SRC=$MERGED/$SET/arm$ARM EV=$EVAL/$SET/arm${ARM}_seed$s WLOG=$LOGDIR/drive_v08_wave_${SET}_arm${ARM}_seed$s.log
  [ -f $SRC/manifest.jsonl ] || { say "eval $SET arm$ARM seed$s: no combined set"; return 0; }
  grep -q DRIVE_WAVE_DONE $WLOG 2>/dev/null && { say "eval $SET arm$ARM seed$s: marker present, skipping"; return 0; }
  local LEVEL=$([ "$ARM" = "B" ] && echo 3 || echo 1) OTHER=$([ "$ARM" = "B" ] && echo 1 || echo 3)
  say "eval $SET arm $ARM seed $s (primary level $LEVEL, cross-identity level $OTHER); $(disk)"
  rm -rf $EV; mkdir -p $EV
  $MPY - $SRC $EV <<'EOF' >> $LOG 2>&1 || { say "eval prep FAILED"; return 1; }
import json, os, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
disc = {int(x) for x in (src / "discovery_seeds.txt").read_text().split()}
n = missing = dropped = 0
with (src / "manifest.jsonl").open() as f, (dst / "manifest.jsonl").open("w") as g:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        if int(r["seed"]) not in disc: dropped += 1; continue
        p = Path(r["artifact"]); missing += not p.exists(); n += 1; g.write(json.dumps(r) + "\n")
for p in src.iterdir():
    if p.is_file() and p.name != "manifest.jsonl" and p.suffix in (".json", ".txt"): shutil.copy2(p, dst / p.name)
os.symlink(str((src / "masks").resolve()), str(dst / "masks"))
(dst / "EVAL_SCOPE.json").write_text(json.dumps({"scope": "v0.8 discovery scenes only (confirmation sealed)", "n_cells": n, "n_cells_confirmation_dropped": dropped, "discovery_seeds": sorted(disc), "merge_dir": str(src)}, indent=1) + "\n")
assert missing == 0, f"{missing} cell files missing"
print(f"eval dir {dst}: {n} discovery cells ({dropped} confirmation cells sealed)")
EOF
  DRIVE_SKIP_WAIT=1 DRIVE_SKIP_MERGE=1 DRIVE_ROOT=$FACT/d07/arm$ARM DRIVE_OUT=$EV DRIVE_CODE=$CODE DRIVE_REPO=$REPO DRIVE_MPY=$MPY DRIVE_DPY=$DPY \
    DRIVE_CFG=$MD/eval_config.yaml DRIVE_CKPT=$MD/jepa-latest.pth.tar DRIVE_MODEL_NAME=jepa_wm_driving DRIVE_MODEL_LABEL="JEPA-WM driving arm $ARM seed $s (jepa-latest) on v0.8 $SET" \
    DRIVE_TRAIN_REF=$FEAT/arm$ARM/training_reference.npz DRIVE_PRIMARY_LEVEL=$LEVEL DRIVE_LOG=$WLOG bash $CODE/run_wave_drive.sh $ARM v08_$SET
  grep -q WAVE_DONE $WLOG || { say "eval $SET arm$ARM seed$s: wave did not finish"; return 1; }
  [ -f $EV/cf_gate_discovery.json ] || { say "eval $SET arm$ARM seed$s: no discovery gate output"; return 1; }
  $MPY $CODE/counterfactual_validity_gate.py --artifacts $EV --cache $EV/latent_cache.npz --output $EV/cf_gate_discovery_level$OTHER.json --seeds-file $EV/discovery_seeds.txt --domain driving --primary-hazard-level $OTHER --model-label "JEPA-WM driving arm $ARM seed $s (cross-identity level $OTHER) v0.8 $SET" >> $WLOG 2>&1
  $MPY $CODE/behavior_gate.py --gate $EV/cf_gate_discovery_level$OTHER.json --output $EV/behavior_gate_discovery_level$OTHER.json >> $WLOG 2>&1 || true
  $MPY $CODE/behavior_gate.py --gate $EV/cf_gate_discovery.json --cross-gate $EV/cf_gate_discovery_level$OTHER.json --output $EV/behavior_gate_discovery.json >> $WLOG 2>&1 || true
  local V; V=$(grep -o '"verdict": "[A-Z_]*"' $EV/behavior_gate_discovery.json 2>/dev/null | head -1 | cut -d'"' -f4); V=${V:-FAIL}
  local VX; VX=$(grep -o '"verdict": "[A-Z_]*"' $EV/behavior_gate_discovery_level$OTHER.json 2>/dev/null | head -1 | cut -d'"' -f4); VX=${VX:-FAIL}
  echo "[$(ts)] B-GATE $([ "$V" = FAIL ] && echo FAIL || echo PASS) (verdict $V at primary level $LEVEL; cross-identity level $OTHER verdict $VX)" >> $WLOG
  # training-set interaction-copy baseline (pooled memorisation control; cross model design jepa.md F4) while the cache exists
  [ -f $CODE/training_copy_baseline.py ] && $MPY $CODE/training_copy_baseline.py --artifacts $EV --cache $EV/latent_cache.npz --training-reference $FEAT/arm$ARM/training_reference.npz --output $EV/training_copy_baseline.json --seeds-file $EV/discovery_seeds.txt --model-label "arm $ARM seed $s v0.8 $SET" >> $WLOG 2>&1 || say "training-copy baseline failed for $SET arm$ARM seed$s (see $WLOG)"
  local keep=0; for k in $KEEP_CACHE_SEEDS; do [ "$k" = "$s" ] && keep=1; done
  if [ $keep = 0 ] || [ "$SET" != static ]; then rm -f $EV/latent_cache.npz; fi
  echo "[$(ts)] DRIVE_WAVE_DONE" >> $WLOG
  say "eval $SET arm$ARM seed$s done: $(grep 'B-GATE' $WLOG | tail -1); $(disk)"
}
for s in $SEEDS; do for ARM in A B; do eval_model static $ARM $s; done; done
for s in $SEEDS; do for ARM in A B; do eval_model dyn $ARM $s; done; done
say "V08_EVAL_DONE; $(disk)"

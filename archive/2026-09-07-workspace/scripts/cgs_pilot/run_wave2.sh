#!/bin/bash
# Wave 2: after seed driver B finishes (seeds 241-400), augment + validate the new seeds, then wave + mech on the full set.
set -u
ROOT=/root/cgs-pilot/artifacts/heldout_v1
CODE=/root/cgs-pilot/code/cgs_pilot
RPY=/opt/conda/envs/robocasa/bin/python
LOG=/root/cgs-pilot/logs/wave2_chain.log
until grep -q ALL_DONE /root/cgs-pilot/logs/heldout_v1_driver_b.log 2>/dev/null; do sleep 120; done
echo "[$(date -u +%FT%TZ)] driver B done" > $LOG
cd $CODE
$RPY augment_null_factor.py --root $ROOT --only-passing > /root/cgs-pilot/logs/augment_w2.log 2>&1
$RPY augment_null_factor.py --root $ROOT --seeds 201 --force >> /root/cgs-pilot/logs/augment_w2.log 2>&1
echo "[$(date -u +%FT%TZ)] augment done" >> $LOG
# validator caches per-seed results in the output json; seeds re-augmented need a fresh entry -> drop 201 from cache
$RPY - <<PY
import json; p="$ROOT/_validation/hazard_label_validation.json"; d=json.load(open(p)); d["seeds"]=[s for s in d["seeds"] if s["seed"]!=201]; json.dump(d, open(p,"w"), indent=2)
PY
mv -f /root/cgs-pilot/logs/validate_hazard_labels.log /root/cgs-pilot/logs/validate_hazard_labels_w1.log
$RPY validate_hazard_labels.py --root $ROOT --output $ROOT/_validation/hazard_label_validation.json > /root/cgs-pilot/logs/validate_hazard_labels.log 2>&1
echo VALIDATE_DONE >> /root/cgs-pilot/logs/validate_hazard_labels.log
echo "[$(date -u +%FT%TZ)] validate done" >> $LOG
/root/cgs-pilot/run_wave.sh w2
echo "[$(date -u +%FT%TZ)] wave w2 done" >> $LOG
/root/cgs-pilot/run_wave_mech.sh w2
echo "[$(date -u +%FT%TZ)] WAVE2_CHAIN_DONE" >> $LOG

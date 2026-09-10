#!/bin/bash
# Wave pipeline: jitter-verify -> merge v0.3 -> latent cache -> counterfactual gate -> domain-shift audit -> retrieval baseline.
# Localization/patching/geometry run separately (discovery seeds only) once their code is frozen.
set -u
WAVE=${1:-w1}
ROOT=/root/cgs-pilot/artifacts/heldout_v1
OUT=/root/cgs-pilot/artifacts/heldout_v1_merged_$WAVE
CODE=/root/cgs-pilot/code/cgs_pilot
RPY=/opt/conda/envs/robocasa/bin/python
MPY=/opt/conda/bin/python
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:$CODE
REPO=/root/cgs-pilot/vendor/jepa-wms
CFG=$REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
CKPT=/root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar
LOG=/root/cgs-pilot/logs/wave_$WAVE.log
mkdir -p $OUT
echo "[$(date -u +%FT%TZ)] wave $WAVE start" > $LOG
until grep -q VALIDATE_DONE /root/cgs-pilot/logs/validate_hazard_labels.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] validation done" >> $LOG
cd $CODE
# 1. jitter verification for every seed with a non-bit-exact cell (v0.3 needs the differing-pixel count)
for d in $ROOT/seed_*/; do
  s=$(basename $d | cut -d_ -f2)
  [ -f $d/manifest.jsonl ] || continue
  if grep -q '"replay_frames_bit_exact": false' $d/manifest.jsonl && [ ! -f $ROOT/_validation/render_jitter_seed_$s.json ]; then
    $RPY verify_render_jitter.py --stimulus-dir $d --output $ROOT/_validation/render_jitter_seed_$s.json >> /root/cgs-pilot/logs/render_jitter_$WAVE.log 2>&1 && echo "jitter verified $s" >> $LOG
  fi
done
# 2. merge under v0.3 with label validation + jitter counts, stable hash split
$RPY merge_heldout.py --root $ROOT --output $OUT --log-dir /root/cgs-pilot/logs --protocol cgs-robocasa-pilot-v0.3 \
  --jitter-json "$ROOT/_validation/render_jitter_seed_*.json" --label-validation $ROOT/_validation/hazard_label_validation.json --split-seed 0 >> $LOG 2>&1
# masks for the merged dir (consumers resolve <stimulus_dir>/masks/<cell_id>.npz)
mkdir -p $OUT/masks; for d in $ROOT/seed_*/masks; do [ -d $d ] && cp -n $d/*.npz $OUT/masks/ 2>/dev/null; done
echo "[$(date -u +%FT%TZ)] merged: $(grep -c . $OUT/manifest.jsonl) cells, masks $(ls $OUT/masks | wc -l)" >> $LOG
# 3. latent cache (GPU) incl. DROID transitions for the kNN baseline
$MPY latent_cache.py --artifacts $OUT --cache $OUT/latent_cache.npz --repo $REPO --config $CFG --checkpoint $CKPT --droid-reference /root/cgs-pilot/reference/droid_100/droid_100_reference.npz >> $LOG 2>&1
# 4. counterfactual-validity gate: all admitted, and discovery-only
$MPY counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_all.json >> $LOG 2>&1
$MPY counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_discovery.json --seeds-file $OUT/discovery_seeds.txt >> $LOG 2>&1
# 5. domain-shift + action audits, retrieval/copying baseline
$MPY audit_contamination.py --artifacts $OUT --output $OUT/contamination.json --cache $OUT/latent_cache.npz --repo $REPO --config $CFG --checkpoint $CKPT >> $LOG 2>&1
$MPY retrieval_baseline.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/retrieval_baseline.json >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] WAVE_DONE" >> $LOG

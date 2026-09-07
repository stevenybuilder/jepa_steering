#!/bin/bash
# Mechanism stage on DISCOVERY seeds only (confirmation seeds untouched until a band is frozen).
set -u
WAVE=${1:-w1}
OUT=/root/cgs-pilot/artifacts/heldout_v1_merged_$WAVE
MECH=/root/cgs-pilot/artifacts/mech_$WAVE
CODE=/root/cgs-pilot/code/cgs_pilot
MPY=/opt/conda/bin/python
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:$CODE
REPO=/root/cgs-pilot/vendor/jepa-wms
CFG=$REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
CKPT=/root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar
LOG=/root/cgs-pilot/logs/mech_$WAVE.log
mkdir -p $MECH
until grep -q WAVE_DONE /root/cgs-pilot/logs/wave_$WAVE.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%FT%TZ)] mech $WAVE start; discovery seeds: $(tr '\n' ' ' < $OUT/discovery_seeds.txt)" > $LOG
cd $CODE
# (w2: localization v2 flags are defaults now; patch_site supports partial null factor)
# 10. predictor-wide interaction localization (all hooks, all layers, 3 steps, token groups, null factor when present)
$MPY localize_interaction.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --output $MECH/localize \
  --seeds-file $OUT/discovery_seeds.txt --dump-hooks resid_post attn_out mlp_out --dump-groups egg gripper corridor >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] localization done" >> $LOG
# 11. patching sweep at step 0 (primary), component sites, gripper_corridor + egg groups, all controls; then persistent mode on the same sites
$MPY patch_site.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --output $MECH/patch_step0 \
  --seeds-file $OUT/discovery_seeds.txt --hooks attn_out mlp_out --steps 0 --groups gripper_corridor egg --mode one_shot >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] patch step0 done" >> $LOG
$MPY patch_site.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --output $MECH/patch_persistent \
  --seeds-file $OUT/discovery_seeds.txt --sites L07.attn_out L09.resid_post L10.attn_out L06.attn_out L08.attn_out --steps 0 --groups gripper_corridor egg --mode persistent >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] patch persistent done" >> $LOG
# 12. geometry tournament + coordinate-frame triangulation on the discovery dump (confirmation withheld)
$MPY geometry_tournament.py --dump $MECH/localize --stimulus $OUT --out $MECH/geometry --discovery-seeds $OUT/discovery_seeds.txt --export-edits >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] MECH_DONE" >> $LOG

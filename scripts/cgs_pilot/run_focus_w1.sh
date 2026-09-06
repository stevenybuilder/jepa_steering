#!/bin/bash
# Focus battery on the localization-v2 candidates, discovery seeds only, after the wave-1 sweep finishes.
set -u
OUT=/root/cgs-pilot/artifacts/heldout_v1_merged_w1
CODE=/root/cgs-pilot/code/cgs_pilot
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:$CODE
REPO=/root/cgs-pilot/vendor/jepa-wms
CFG=$REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
CKPT=/root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar
LOG=/root/cgs-pilot/logs/focus_w1.log
until grep -q MECH_DONE /root/cgs-pilot/logs/mech_w1.log 2>/dev/null; do sleep 300; done
echo "[$(date -u +%FT%TZ)] focus w1 start" > $LOG
cd $CODE
/opt/conda/bin/python patch_site.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --output /root/cgs-pilot/artifacts/mech_w1/focus \
  --seeds-file $OUT/discovery_seeds.txt --focus-sites L07.attn_out L09.resid_post L10.attn_out --steps 0 1 --groups gripper_corridor egg >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] FOCUS_DONE" >> $LOG

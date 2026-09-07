#!/bin/bash
# E2: action-entry (AdaLN modulation) path patching on discovery seeds, after the Jacobian sonar.
set -u
until grep -q "JACOBIAN_DONE" /root/cgs-pilot/logs/jacobian_w1.log 2>/dev/null; do sleep 300; done
cd /root/cgs-pilot/code/cgs_pilot && export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:/root/cgs-pilot/code/cgs_pilot
CFG=/root/cgs-pilot/vendor/jepa-wms/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
OUT=/root/cgs-pilot/artifacts/heldout_v1_merged_w1
LOG=/root/cgs-pilot/logs/modulation_w1.log; echo "[$(date -u +%FT%TZ)] modulation w1 start" > $LOG
/opt/conda/bin/python modulation_patch.py --repo /root/cgs-pilot/vendor/jepa-wms --config $CFG --checkpoint /root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar \
  --artifacts $OUT --seeds-file $OUT/discovery_seeds.txt --output /root/cgs-pilot/artifacts/mech_w1/modulation_patch --components attn mlp all --step 0 --n-probes 8 >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] MODULATION_DONE" >> $LOG

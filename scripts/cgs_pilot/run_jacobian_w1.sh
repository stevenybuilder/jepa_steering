#!/bin/bash
# Action-Jacobian sonar (JEPA rep geometry.md §2.1) on discovery seeds, after the conceptor intervention finishes.
set -u
until grep -q "CONCEPTOR_DONE" /root/cgs-pilot/logs/conceptor_w1.log 2>/dev/null && ! grep -q "Traceback" /root/cgs-pilot/logs/conceptor_w1.log; do sleep 300; done
cd /root/cgs-pilot/code/cgs_pilot && export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:/root/cgs-pilot/code/cgs_pilot
LOG=/root/cgs-pilot/logs/jacobian_w1.log; echo "[$(date -u +%FT%TZ)] jacobian w1 start" > $LOG
/opt/conda/bin/python action_jacobian_sonar.py --repo /root/cgs-pilot/vendor/jepa-wms \
  --config /root/cgs-pilot/vendor/jepa-wms/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml \
  --checkpoint /root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar \
  --artifacts /root/cgs-pilot/artifacts/heldout_v1_merged_w1 --seeds-file /root/cgs-pilot/artifacts/heldout_v1_merged_w1/discovery_seeds.txt \
  --output /root/cgs-pilot/artifacts/mech_w1/jacobian_sonar --hooks attn_out mlp_out resid_post --groups gripper_corridor egg all --steps 0 1 --mediation-steps 0 --hvp-eps 0.1 >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] JACOBIAN_DONE" >> $LOG

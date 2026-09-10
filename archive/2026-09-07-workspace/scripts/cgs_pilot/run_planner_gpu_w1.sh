#!/bin/bash
set -u
until grep -q "MODULATION_DONE" /root/cgs-pilot/logs/modulation_w1.log 2>/dev/null; do sleep 300; done
cd /root/cgs-pilot/code/cgs_pilot && export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:/root/cgs-pilot/code/cgs_pilot
CFG=/root/cgs-pilot/vendor/jepa-wms/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
OUT=/root/cgs-pilot/artifacts/heldout_v1_merged_w1
LOG=/root/cgs-pilot/logs/planner_gpu_w1.log; echo "[$(date -u +%FT%TZ)] start" > $LOG
/opt/conda/bin/python planner_currency_gpu.py --help > /dev/null 2>&1 && /opt/conda/bin/python planner_currency_gpu.py --repo /root/cgs-pilot/vendor/jepa-wms --config $CFG --checkpoint /root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar --artifacts $OUT --output $OUT/planner_perturbed_costs.npz >> $LOG 2>&1
/opt/conda/bin/python counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_sonar_all.json >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] PLANNER_GPU_DONE" >> $LOG

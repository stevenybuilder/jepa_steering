#!/bin/bash
# COAST-style donor-free conceptor intervention on discovery seeds (pre-committed sites L06/L07/L08.attn_out + band), trimmed budget.
set -u
CONC=/root/cgs-pilot/artifacts/mech_w1/geometry/conceptors
until ls $CONC/L07.attn_out__s0__gripper_corridor.npz >/dev/null 2>&1 && ls $CONC/L06.attn_out__s0__gripper_corridor.npz >/dev/null 2>&1 && ls $CONC/L08.attn_out__s0__gripper_corridor.npz >/dev/null 2>&1; do sleep 300; done
sleep 120
cd /root/cgs-pilot/code/cgs_pilot && export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:/root/cgs-pilot/code/cgs_pilot
LOG=/root/cgs-pilot/logs/conceptor_w1.log
echo "[$(date -u +%FT%TZ)] conceptor w1 start" > $LOG
/opt/conda/bin/python patch_site.py --repo /root/cgs-pilot/vendor/jepa-wms \
  --config /root/cgs-pilot/vendor/jepa-wms/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml \
  --checkpoint /root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar \
  --artifacts /root/cgs-pilot/artifacts/heldout_v1_merged_w1 --seeds-file /root/cgs-pilot/artifacts/heldout_v1_merged_w1/discovery_seeds.txt \
  --output /root/cgs-pilot/artifacts/mech_w1/conceptor --conceptor-dir $CONC \
  --conceptor-mode strengthen suppress --beta 0.5 1.0 --conceptor-key C_safety \
  --sites L06.attn_out L07.attn_out L08.attn_out --band-sites L06.attn_out L07.attn_out L08.attn_out \
  --steps 0 --mode both --groups gripper_corridor --equivalence-margin 0.05 --unrelated-offset 6 >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] CONCEPTOR_DONE" >> $LOG

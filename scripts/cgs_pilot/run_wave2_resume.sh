#!/bin/bash
# Resume wave 2 after the sonar pooling crash (fixed: NaN-padded principal angles, sonar block try/except).
# Thread caps so the concurrent Jacobian sonar is not starved (64 cores; earlier oversubscription = 7x slowdown).
set -u
OUT=/root/cgs-pilot/artifacts/heldout_v1_merged_w2
CODE=/root/cgs-pilot/code/cgs_pilot
MPY=/opt/conda/bin/python
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:$CODE
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
REPO=/root/cgs-pilot/vendor/jepa-wms
CFG=$REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
CKPT=/root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar
LOG=/root/cgs-pilot/logs/wave_w2.log
cd $CODE
echo "[$(date -u +%FT%TZ)] RESUME after sonar pooling crash (patched)" >> $LOG
$MPY counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_discovery.json --seeds-file $OUT/discovery_seeds.txt >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] gate discovery done" >> $LOG
$MPY counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_all.json >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] gate all done" >> $LOG
$MPY audit_contamination.py --artifacts $OUT --output $OUT/contamination.json --cache $OUT/latent_cache.npz --repo $REPO --config $CFG --checkpoint $CKPT >> $LOG 2>&1
$MPY retrieval_baseline.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/retrieval_baseline.json >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] WAVE_DONE" >> $LOG
echo "[$(date -u +%FT%TZ)] wave w2 done (resume)" >> /root/cgs-pilot/logs/wave2_chain.log
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 /root/cgs-pilot/run_wave_mech.sh w2
echo "[$(date -u +%FT%TZ)] WAVE2_CHAIN_DONE" >> /root/cgs-pilot/logs/wave2_chain.log

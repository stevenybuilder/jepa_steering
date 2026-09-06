#!/bin/bash
# Loop-2 cup family, GPU stages. Re-chained on JACOBIAN_DONE (was MECH_DONE of wave 2, which is many hours out).
set -u
G=cup; OUT=/root/cgs-pilot/artifacts/family_${G}_merged; MECH=/root/cgs-pilot/artifacts/mech_$G
CODE=/root/cgs-pilot/code/cgs_pilot; MPY=/opt/conda/bin/python
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:$CODE
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
REPO=/root/cgs-pilot/vendor/jepa-wms
CFG=$REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
CKPT=/root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar
LOG=/root/cgs-pilot/logs/loop2_$G.log
until grep -q JACOBIAN_DONE /root/cgs-pilot/logs/jacobian_w1.log 2>/dev/null; do sleep 300; done
echo "[$(date -u +%FT%TZ)] GPU stages start (after JACOBIAN_DONE)" >> $LOG
cd $CODE; mkdir -p $MECH
$MPY latent_cache.py --artifacts $OUT --cache $OUT/latent_cache.npz --repo $REPO --config $CFG --checkpoint $CKPT --droid-reference /root/cgs-pilot/reference/droid_100/droid_100_reference.npz >> $LOG 2>&1
$MPY counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_all.json >> $LOG 2>&1
$MPY counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_discovery.json --seeds-file $OUT/discovery_seeds.txt >> $LOG 2>&1
$MPY retrieval_baseline.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/retrieval_baseline.json >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] gate done" >> $LOG
$MPY localize_interaction.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --output $MECH/localize --seeds-file $OUT/discovery_seeds.txt --dump-hooks resid_post attn_out mlp_out --dump-groups egg gripper corridor >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] localize done" >> $LOG
$MPY patch_site.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --output $MECH/patch_step0 --seeds-file $OUT/discovery_seeds.txt --hooks attn_out mlp_out --steps 0 --groups gripper_corridor egg --mode one_shot >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] patch done" >> $LOG
$MPY modulation_patch.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --seeds-file $OUT/discovery_seeds.txt --output $MECH/modulation_patch --components attn mlp all --step 0 --n-probes 4 >> $LOG 2>&1
$MPY action_jacobian_sonar.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --seeds-file $OUT/discovery_seeds.txt --output $MECH/jacobian_sonar --hooks attn_out mlp_out --groups gripper_corridor egg all --steps 0 1 --mediation-steps 0 >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] LOOP2_${G}_DONE" >> $LOG

#!/bin/bash
# Loop-2 cup family v2: extend seeds to 541-600 (preregistered extension: reach the >=16 discovery-scene rule under the fixed hash split),
# re-run CPU stages on the superset, then GPU stages once, after JACOBIAN_DONE. Sequential generation, single-threaded BLAS, to leave CPU to the GPU jobs.
set -u
G=cup; C=0.12
ROOT=/root/cgs-pilot/artifacts/family_$G; OUT=/root/cgs-pilot/artifacts/family_${G}_merged; MECH=/root/cgs-pilot/artifacts/mech_$G
CODE=/root/cgs-pilot/code/cgs_pilot; RPY=/opt/conda/envs/robocasa/bin/python; MPY=/opt/conda/bin/python
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:$CODE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
REPO=/root/cgs-pilot/vendor/jepa-wms
CFG=$REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
CKPT=/root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar
LOG=/root/cgs-pilot/logs/loop2_$G.log
cd $CODE
echo "[$(date -u +%FT%TZ)] v2: generating seeds 541-600" >> $LOG
for s in $(seq 541 600); do
  out=$ROOT/seed_$s; [ -f $out/summary.json ] && continue
  if $RPY robocasa_contact_pilot_probe.py --obj-group $G --control-min-m $C --output $out --seeds $s > /root/cgs-pilot/logs/family_${G}_$s.log 2>&1; then echo "OK $G $s" >> /root/cgs-pilot/logs/family_cup_ext.log; else echo "FAIL $G $s" >> /root/cgs-pilot/logs/family_cup_ext.log; fi
done
echo "[$(date -u +%FT%TZ)] v2: generation done: $(grep -c ^OK /root/cgs-pilot/logs/family_cup_ext.log) ok" >> $LOG
$RPY augment_null_factor.py --root $ROOT --only-passing --obj-group $G --control-min-m $C > /root/cgs-pilot/logs/augment_${G}_v2.log 2>&1
$RPY validate_hazard_labels.py --root $ROOT --obj-group $G --control-min-m $C --output $ROOT/_validation/hazard_label_validation.json > /root/cgs-pilot/logs/validate_${G}_v2.log 2>&1
echo "[$(date -u +%FT%TZ)] v2 validate done: $(grep -c passed: true /root/cgs-pilot/logs/validate_${G}_v2.log) pass / $(grep -c ^{ /root/cgs-pilot/logs/validate_${G}_v2.log) (new seeds only; cached seeds kept)" >> $LOG
for d in $ROOT/seed_*/; do s=$(basename $d | cut -d_ -f2); [ -f $d/manifest.jsonl ] || continue
  if grep -q replay_frames_bit_exact: false $d/manifest.jsonl && [ ! -f $ROOT/_validation/render_jitter_seed_$s.json ]; then
    $RPY -c "import robocasa_contact_pilot as g; g.OBJ_GROUP=; g.CONTROL_MIN_M=$C; import runpy,sys; sys.argv=[verify_render_jitter.py,--stimulus-dir,,--output,/_validation/render_jitter_seed_.json]; runpy.run_path(verify_render_jitter.py, run_name=__main__)" >> /root/cgs-pilot/logs/render_jitter_$G.log 2>&1; fi; done
$RPY merge_heldout.py --root $ROOT --output $OUT --log-dir /root/cgs-pilot/logs --protocol cgs-robocasa-pilot-v0.3 --jitter-json "$ROOT/_validation/render_jitter_seed_*.json" --label-validation $ROOT/_validation/hazard_label_validation.json --split-seed 0 --exclude-seeds >> $LOG 2>&1
mkdir -p $OUT/masks; for d in $ROOT/seed_*/masks; do [ -d $d ] && cp -n $d/*.npz $OUT/masks/ 2>/dev/null; done
echo "[$(date -u +%FT%TZ)] v2 merged: $(grep -c . $OUT/manifest.jsonl) cells; discovery $(tr n   < $OUT/discovery_seeds.txt)" >> $LOG
echo CPU_DONE_V2 >> $LOG
until grep -q JACOBIAN_DONE /root/cgs-pilot/logs/jacobian_w1.log 2>/dev/null; do sleep 300; done
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
echo "[$(date -u +%FT%TZ)] GPU stages start" >> $LOG
mkdir -p $MECH; rm -f $OUT/latent_cache.npz
$MPY latent_cache.py --artifacts $OUT --cache $OUT/latent_cache.npz --repo $REPO --config $CFG --checkpoint $CKPT --droid-reference /root/cgs-pilot/reference/droid_100/droid_100_reference.npz >> $LOG 2>&1
$MPY counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_all.json >> $LOG 2>&1
$MPY counterfactual_validity_gate.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/cf_gate_discovery.json --seeds-file $OUT/discovery_seeds.txt >> $LOG 2>&1
$MPY retrieval_baseline.py --artifacts $OUT --cache $OUT/latent_cache.npz --output $OUT/retrieval_baseline.json >> $LOG 2>&1
$MPY audit_contamination.py --artifacts $OUT --output $OUT/contamination.json --cache $OUT/latent_cache.npz --repo $REPO --config $CFG --checkpoint $CKPT >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] gate done" >> $LOG
$MPY localize_interaction.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --output $MECH/localize --seeds-file $OUT/discovery_seeds.txt --dump-hooks resid_post attn_out mlp_out --dump-groups egg gripper corridor >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] localize done" >> $LOG
$MPY patch_site.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --output $MECH/patch_step0 --seeds-file $OUT/discovery_seeds.txt --hooks attn_out mlp_out --steps 0 --groups gripper_corridor egg --mode one_shot >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] patch done" >> $LOG
$MPY modulation_patch.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --seeds-file $OUT/discovery_seeds.txt --output $MECH/modulation_patch --components attn mlp all --step 0 --n-probes 4 >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] modulation done" >> $LOG
$MPY action_jacobian_sonar.py --repo $REPO --config $CFG --checkpoint $CKPT --artifacts $OUT --seeds-file $OUT/discovery_seeds.txt --output $MECH/jacobian_sonar --hooks attn_out mlp_out --groups gripper_corridor egg all --steps 0 1 --mediation-steps 0 >> $LOG 2>&1
echo "[$(date -u +%FT%TZ)] LOOP2_cup_DONE" >> $LOG

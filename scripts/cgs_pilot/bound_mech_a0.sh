#!/bin/bash
# For the already-running armA_seed0 mech instance (old script): kill its full 54-scene Jacobian when it starts, then run the 8-seed subset after MECH_DONE.
L=/root/cgs-pilot/logs/drive_mech_armA_seed0.log
until grep -q "patch step0 done" $L 2>/dev/null; do sleep 60; done
for i in $(seq 1 120); do P=$(pgrep -f "^/opt/conda/bin/python action_jacobian_sonar.py .*armA_seed0" | head -1); [ -n "$P" ] && { kill $P; echo "[$(date -u +%FT%TZ)] coordinator: killed full-set Jacobian (pid $P) for armA_seed0; subset re-run follows MECH_DONE" >> $L; break; }; sleep 10; done
until grep -q "MECH_DONE" $L 2>/dev/null; do sleep 60; done
cd /root/cgs-pilot/code/cgs_pilot; export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:/root/cgs-pilot/code/cgs_pilot OMP_NUM_THREADS=12
MECH=/root/cgs-pilot/artifacts/drive_mech/armA_seed0; OUT=/root/cgs-pilot/artifacts/drive_eval/armA_seed0; MD=/root/cgs-pilot/artifacts/drive_models/armA_seed0
head -n 8 $OUT/discovery_seeds.txt > $MECH/jacobian_seeds.txt
/opt/conda/bin/python action_jacobian_sonar.py --repo /root/cgs-pilot/vendor/jepa-wms --config $MD/eval_config.yaml --checkpoint $MD/jepa-latest.pth.tar --model-name jepa_wm_driving --artifacts $OUT --output $MECH/jacobian_sonar --domain driving --seeds-file $MECH/jacobian_seeds.txt --hooks attn_out mlp_out --groups gripper_corridor egg all --steps 0 --mediation-steps 0 >> /root/cgs-pilot/logs/drive_mech_armA_seed0_jacobian_subset.log 2>&1
echo "[$(date -u +%FT%TZ)] jacobian done (subset 8 seeds, step 0; coordinator re-run) JAC_SUBSET_DONE" >> $L

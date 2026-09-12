#!/bin/bash
# run_relational_transport.sh — regenerate the seed-0 latent caches of arm A and arm B on their DISCOVERY eval dirs
# (artifacts/drive_eval/arm{A,B}_seed0: manifest.jsonl with absolute cell paths, masks/, discovery_seeds.txt — the
# caches themselves were deleted after the gates), then run relational_transport.py (LRH.md section 2, descriptive)
# and delete the caches again.
#
# The cache step is EXACTLY run_wave_drive.sh step 2 as invoked by run_drive_arms.sh stage 4: latent_cache.py through
# the driving mask shim (lc.load_token_mask = token_groups.cache_token_mask_driving), CFG = the model's eval_config.yaml,
# CKPT = jepa-latest.pth.tar, MODEL_NAME = jepa_wm_driving, JEPAWM_DRIVING_ACTION_DIM=2, same JEPAWM_* env.
# Disk: a full cache is ~2.1 GB per arm, so each cache is slimmed to the last predicted frame (--slim, ~0.45 GB) and the
# full cache deleted before the other arm is built. GPU: before every cache step the script WAITS (poll 5 min) until
# no patch_site.py / action_jacobian_sonar.py process is running so the mechanism stage is not slowed.
#
# Launch (remote): ssh -n -p PORT root@HOST 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_relational_transport.sh > /dev/null 2>&1 < /dev/null &'
# Log: logs/drive_relational.log; marker RELATIONAL_DONE. Env: DRIVE_REL_SEED (0), DRIVE_REL_KEEP_SLIM=1 keeps the slim caches.
set -u
BASE=${DRIVE_BASE:-/root/cgs-pilot}
CODE=${DRIVE_CODE:-$BASE/code/cgs_pilot}
REPO=${DRIVE_REPO:-$BASE/vendor/jepa-wms}
MPY=${DRIVE_MPY:-/opt/conda/bin/python}
SEED=${DRIVE_REL_SEED:-0}
OUT=${DRIVE_REL_OUT:-$BASE/artifacts/drive_relational}
LOG=${DRIVE_REL_LOG:-$BASE/logs/drive_relational.log}
EVAL=$BASE/artifacts/drive_eval; MODELS=$BASE/artifacts/drive_models
N_PERM=${DRIVE_REL_N_PERM:-100}; N_DRAWS=${DRIVE_REL_N_DRAWS:-100}
export JEPAWM_HOME=$BASE/vendor JEPAWM_OSSCKPT=$BASE/checkpoints JEPAWM_LOGS=$BASE/logs PYTHONPATH=$REPO:$CODE
export JEPAWM_DRIVING_ACTION_DIM=2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
mkdir -p $OUT $(dirname $LOG)
ts() { date -u +%FT%TZ; }
say() { echo "[$(ts)] $*" >> $LOG; }
disk() { df -h $BASE | tail -1 | awk '{print "disk used "$3" of "$2", free "$4}'; }
gpu_busy() { pgrep -f 'python[^ ]* .*(patch_site|action_jacobian_sonar)\.py' > /dev/null; }
wait_gpu() {
  local n=0
  while gpu_busy; do
    [ $((n % 6)) = 0 ] && say "waiting: mechanism stage on the GPU ($(pgrep -fa 'python[^ ]* .*(patch_site|action_jacobian_sonar)\.py' | sed 's/ --repo.*//' | cut -c1-80 | tr '\n' ';'))"
    n=$((n + 1)); sleep 300
  done
}
say "relational transport start seed=$SEED out=$OUT n_perm=$N_PERM n_draws=$N_DRAWS; $(disk)"
cd $CODE
for ARM in A B; do
  EV=$EVAL/arm${ARM}_seed$SEED; MD=$MODELS/arm${ARM}_seed$SEED; RO=$OUT/arm${ARM}_seed$SEED; mkdir -p $RO
  if [ -f $RO/latent_slim.npz ]; then say "arm $ARM: slim cache present, reused"; continue; fi
  for f in $EV/manifest.jsonl $EV/discovery_seeds.txt $MD/eval_config.yaml $MD/jepa-latest.pth.tar; do [ -e $f ] || { say "arm $ARM: missing $f"; say "RELATIONAL_FAILED"; exit 1; }; done
  [ -e $EV/masks ] || say "arm $ARM: WARNING no masks/ in $EV (token groups fall back to the cache mask; hazard group unavailable)"
  wait_gpu
  say "arm $ARM: latent cache on $(grep -c . $EV/manifest.jsonl) discovery cells (cfg $MD/eval_config.yaml, ckpt $MD/jepa-latest.pth.tar); $(disk)"
  # == run_wave_drive.sh step 2 (frozen latent_cache.py; driving token mask via the documented shim) ==
  $MPY -c "import sys, latent_cache as lc, token_groups as tg; lc.load_token_mask = tg.cache_token_mask_driving; sys.argv = ['latent_cache.py', '--artifacts', '$EV', '--cache', '$RO/latent_cache.npz', '--repo', '$REPO', '--config', '$MD/eval_config.yaml', '--checkpoint', '$MD/jepa-latest.pth.tar', '--model-name', 'jepa_wm_driving']; lc.main()" >> $LOG 2>&1
  [ -f $RO/latent_cache.npz ] || { say "arm $ARM: cache step FAILED"; say "RELATIONAL_FAILED"; exit 1; }
  say "arm $ARM: cache written ($(du -sh $RO/latent_cache.npz | cut -f1)); slimming to the last frame"
  $MPY relational_transport.py --slim $RO/latent_cache.npz --slim-out $RO/latent_slim.npz >> $LOG 2>&1 || { say "arm $ARM: slim FAILED"; say "RELATIONAL_FAILED"; exit 1; }
  rm -f $RO/latent_cache.npz
  say "arm $ARM: full cache deleted, slim kept ($(du -sh $RO/latent_slim.npz | cut -f1)); $(disk)"
done
say "analysis start"
$MPY relational_transport.py --eval-a $EVAL/armA_seed$SEED --eval-b $EVAL/armB_seed$SEED \
  --cache-a $OUT/armA_seed$SEED/latent_slim.npz --cache-b $OUT/armB_seed$SEED/latent_slim.npz --out $OUT \
  --n-perm $N_PERM --n-draws $N_DRAWS --label-a "JEPA-WM driving arm A seed $SEED (jepa-latest)" --label-b "JEPA-WM driving arm B seed $SEED (jepa-latest)" >> $LOG 2>&1
if [ -f $OUT/relational_transport.json ]; then
  if [ "${DRIVE_REL_KEEP_SLIM:-0}" != "1" ]; then rm -f $OUT/arm{A,B}_seed$SEED/latent_slim.npz; say "slim caches deleted (DRIVE_REL_KEEP_SLIM=1 keeps them)"; fi
  say "RELATIONAL_DONE -> $OUT/relational_transport.json (+ .md); $(disk)"
else
  say "analysis FAILED (no relational_transport.json; slim caches kept for a rerun)"; say "RELATIONAL_FAILED"; exit 1
fi

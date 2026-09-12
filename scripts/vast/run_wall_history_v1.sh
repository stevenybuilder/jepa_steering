#!/usr/bin/env bash
# One exclusive GPU per training seed; preserve all epoch checkpoints and monitoring.
set -eu
worker="$1"
gpu="$2"
seed="$3"
runtime=/workspace/jepa-runtime
case "$worker:$gpu:$seed" in
  50189244:0:234) previous="$runtime/navigation-development-20260907-v1/pointmaze/native/shard-all" ;;
  50195621:0:235) previous="$runtime/behavioral-development-20260907-v1/reach/coupling_only/shard-1" ;;
  50195621:1:236) previous="$runtime/behavioral-development-20260907-v1/reach/matched_random_coupling/shard-1" ;;
  *) exit 2 ;;
esac
while test ! -f "$previous/DONE.json"; do
  test ! -f "$previous/FAILED.json" || exit 3
  sleep 10
done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader)"; do sleep 10; done
export CUDA_VISIBLE_DEVICES="$gpu" JEPA_VERIFIED_LOCAL_DINO=1
export PYTHONPATH="$runtime/wall-history-code-v1/src:/workspace/jepa-python/lib/python3.10/site-packages"
export LD_LIBRARY_PATH=/opt/conda/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
common=(--vendor /workspace/jepa_steering/vendor/jepa-wms
  --data-root "$runtime/navigation-assets-20260907-v1/extracted/wall/wall_single"
  --input-check "$runtime/navigation-input-check-20260907-v1"
  --input-receipt "$runtime/wall-training-inputs-20260907-v1"
  --pilot "$runtime/wall-training-accumulation-pilot-20260907-v2" --seed "$seed")
engineering="$runtime/wall-training-history-20260907-v1/seed-$seed/epoch-one-engineering"
timeout 3600 /workspace/jepa-planning-python/bin/python -u -m offline_study.training_history \
  "${common[@]}" --engineering-only --output "$engineering"
exec timeout 172800 /workspace/jepa-planning-python/bin/python -u -m offline_study.training_history \
  "${common[@]}" --resume-from "$engineering/jepa-e0.pth.tar" --engineering-proof "$engineering" \
  --output "$runtime/wall-training-history-20260907-v1/seed-$seed/remaining-epochs"

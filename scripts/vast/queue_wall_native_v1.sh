#!/usr/bin/env bash
# Exclusive next job on owned50125440 GPU1, after its specific Reach shard exits.
set -eu
runtime=/workspace/jepa-runtime
previous="$runtime/behavioral-development-20260907-v1/reach/native/shard-1"
until test -f "$previous/DONE.json"; do
  test ! -f "$previous/FAILED.json" || exit 3
  sleep 15
done
while test -n "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)"; do
  sleep 5
done
exec env CUDA_VISIBLE_DEVICES=1 JEPA_VERIFIED_LOCAL_DINO=1 \
  PYTHONPATH="$runtime/navigation-expansion-code-v1/src:/workspace/jepa-python/lib/python3.10/site-packages" \
  LD_LIBRARY_PATH=/opt/conda/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /workspace/jepa-planning-python/bin/python -u -m offline_study.navigation_replication native \
  --vendor /workspace/jepa_steering/vendor/jepa-wms \
  --freeze "$runtime/navigation-development-freeze-20260907-v1/wall" \
  --checkpoint "$runtime/navigation-assets-20260907-v3/downloads/model/jepa_wm_wall.pth.tar" \
  --logical-ranks 0 1 2 3 4 5 6 7 \
  --output "$runtime/navigation-development-20260907-v1/wall/native/shard-all"

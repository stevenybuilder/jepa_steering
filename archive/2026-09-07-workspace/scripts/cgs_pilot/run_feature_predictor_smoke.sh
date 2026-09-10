#!/bin/bash
# Smoke chain for train_feature_predictor.py on the remote 4090 (< 4 GB VRAM, ~5 min):
#   1. train depth-6/width-512 predictor on SYNTHETIC features (N=256, T=4, D=1024, A=2),
#      frozen DINOv3 encoder instantiated on CPU via vendor init_video_model, never run;
#   2. reload through model_action_sensitivity.load_model(..., "jepa_wm_driving") and run
#      encode / unroll / predictor_hooks capture (smoke_feature_predictor.py);
#   3. on-disk store round trip (16 clips, ~34 MB, deleted afterwards);
#   4. ms/iter benchmarks for the 10k-clip per-epoch estimate (6/512 and 12/1024).
# Env: BENCH=1 to run stage 4 (default 0), ROUNDTRIP=0 to skip stage 3.
# Launch: ssh -n -p 20566 root@HOST 'setsid nohup bash /root/cgs-pilot/run_feature_predictor_smoke.sh > /root/cgs-pilot/logs/feature_predictor_smoke.log 2>&1 < /dev/null &'
set -u
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs
export PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:/root/cgs-pilot/code/cgs_pilot
MPY=/opt/conda/bin/python
REPO=/root/cgs-pilot/vendor/jepa-wms
OUT=/root/cgs-pilot/artifacts/feature_predictor_smoke
mkdir -p $OUT
cd /root/cgs-pilot/code/cgs_pilot
echo "[$(date -u +%FT%TZ)] smoke start"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; df -h /root | tail -1
echo "=== 1. train d6/w512 on synthetic (encoder present on CPU, never run)"
$MPY train_feature_predictor.py --repo $REPO --synthetic 256,4,2 --with-encoder --pred-depth 6 --pred-embed-dim 512 \
  --batch-size 8 --epochs 40 --max-minutes 2 --arm smoke_synthetic --seed 0 --out $OUT/d6w512 2>&1 | grep -v "^INFO:\|^🔧\|^🧠\|^🔮\|^📉\|Encoder: \|Predictor: "
echo "=== 2. reload through the pilot loader + hooks"
$MPY smoke_feature_predictor.py --repo $REPO --out-dir $OUT/d6w512 --report $OUT/d6w512/smoke_report.json --site L03.mlp_out 2>&1 | grep -v "^INFO:"
if [ "${ROUNDTRIP:-1}" = 1 ]; then
echo "=== 3. on-disk store round trip"
$MPY train_feature_predictor.py --repo $REPO --synthetic 16,4,2 --write-synthetic $OUT/tiny_store.npz --epochs 1 --batch-size 8 --arm tiny_write --out $OUT/tiny_write 2>&1 | grep "^{\"epoch\|stopping\|Error\|error"
$MPY train_feature_predictor.py --repo $REPO --features $OUT/tiny_store.npz --epochs 2 --batch-size 8 --arm tiny_from_file --out $OUT/tiny_read 2>&1 | grep "^data:\|^{\"epoch\|Error\|error"
ls -la $OUT/tiny_store.npz $OUT/tiny_store.clips.json; rm -f $OUT/tiny_store.npz; rm -rf $OUT/tiny_write $OUT/tiny_read
fi
if [ "${BENCH:-0}" = 1 ]; then
echo "=== 4. benchmarks (bench_n=10000)"
for B in 8 16; do
  echo "--- d6/w512 batch $B"; $MPY train_feature_predictor.py --repo $REPO --synthetic 32,4,2 --pred-depth 6 --pred-embed-dim 512 --batch-size $B --bench-iters 20 --seed 0 --out $OUT/bench 2>&1 | grep "^{\"batch"
done
for B in 2 3; do
  echo "--- d12/w1024 batch $B (no optimizer state, VRAM budget)"; $MPY train_feature_predictor.py --repo $REPO --synthetic 32,4,2 --pred-depth 12 --pred-embed-dim 1024 --batch-size $B --bench-iters 10 --bench-skip-optimizer --seed 0 --out $OUT/bench 2>&1 | grep "^{\"batch"
done
echo "--- d6/w1024 batch 2 (no optimizer state)"; $MPY train_feature_predictor.py --repo $REPO --synthetic 32,4,2 --pred-depth 6 --pred-embed-dim 1024 --batch-size 2 --bench-iters 10 --bench-skip-optimizer --seed 0 --out $OUT/bench 2>&1 | grep "^{\"batch"
fi
du -sh $OUT; ls -la $OUT/d6w512
echo "[$(date -u +%FT%TZ)] SMOKE_DONE"

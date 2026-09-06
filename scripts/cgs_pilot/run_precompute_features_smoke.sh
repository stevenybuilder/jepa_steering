#!/bin/bash
# Smoke chain for precompute_dinov3_features.py on the remote 4090 (< 6 GB VRAM, ~3 min):
#   1. fabricate 2 tiny generator shards (n=8, T=4, 256x256 random uint8 RGB + random actions + meta.json);
#   2. precompute with the pilot encoder (DROID JEPA-WM yaml + checkpoint) for pool none/fp16 (the trainer's
#      contract), pool patch2/fp16 and pool none/bf16 (size + rounding comparison), --resume no-op check;
#   3. load the none/fp16 and patch2/fp16 stores through train_feature_predictor.load_store + build the
#      vendor predictor and run ONE fwd/bwd step (jepa_wm_loss);
#   4. independent check: encode_frames(stored frame) == stored feature within fp16 rounding, plus bf16 decode;
#   5. report sizes, VRAM, throughput; delete everything under $OUT.
# Launch: ssh -n -p 20566 root@HOST 'setsid nohup bash /root/cgs-pilot/code/cgs_pilot/run_precompute_features_smoke.sh > /root/cgs-pilot/logs/precompute_features_smoke.log 2>&1 < /dev/null &'
set -u
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs
export PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:/root/cgs-pilot/code/cgs_pilot
export OMP_NUM_THREADS=8
MPY=/opt/conda/bin/python
REPO=/root/cgs-pilot/vendor/jepa-wms
CFG=$REPO/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
CKPT=/root/cgs-pilot/checkpoints/jepa_wm_droid.pth.tar
OUT=/root/cgs-pilot/artifacts/_smoke_features
FILTER='^INFO:|^🔧|^🧠|^🔮|^📉|Encoder: |Predictor: |^Loading|^Loaded|Warning|warn'
rm -rf $OUT; mkdir -p $OUT/shards
cd /root/cgs-pilot/code/cgs_pilot
echo "[$(date -u +%FT%TZ)] smoke start"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader; df -h /root | tail -1

echo "=== 1. fabricate shards"
$MPY - "$OUT/shards" <<'EOF'
import json, sys, numpy as np
from pathlib import Path
root = Path(sys.argv[1]); rng = np.random.default_rng(0)
for s in range(2):
    n, T = 8, 4
    frames = rng.integers(0, 256, size=(n, T, 256, 256, 3), dtype=np.uint8)
    actions = rng.uniform(-1, 1, size=(n, T - 1, 2)).astype(np.float32)
    np.savez(root / f"shard_{s:03d}.npz", frames=frames, actions=actions)
    meta = []
    for i in range(n):
        seed = 1000 * s + i
        hz = (i % 2 == 0)
        meta.append({"clip_id": f"clip_s{s}_{i:03d}", "seed": seed, "arm": "solid" if hz else "ghost",
                     "hazard": {"kind": "pedestrian" if hz else "none", "pose": [1.0 * i, 2.0, 0.0]},
                     "contact": bool(hz and i % 4 == 0), "contact_step": (2 if (hz and i % 4 == 0) else None), "min_distance": float(3.0 - 0.3 * i)})
    (root / f"shard_{s:03d}.meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", root / f"shard_{s:03d}.npz", frames.shape, actions.shape)
EOF
ls -la $OUT/shards

for CFGNAME in "none float16" "patch2 float16" "none bfloat16"; do
  set -- $CFGNAME; POOL=$1; DT=$2; TAG=${POOL}_${DT}
  echo "=== 2. precompute pool=$POOL dtype=$DT"
  $MPY precompute_dinov3_features.py --shards "$OUT/shards/shard_*.npz" --repo $REPO --config $CFG --checkpoint $CKPT \
    --out $OUT/$TAG --pool $POOL --dtype $DT --batch-frames 32 --verify-clips 2 --self-check 2 --split-seed 0 2>&1 | grep -Ev "$FILTER"
  echo "--- files"; ls -la $OUT/$TAG; du -sh $OUT/$TAG
done
echo "=== 2b. --resume no-op (all shards skipped)"
$MPY precompute_dinov3_features.py --shards "$OUT/shards/shard_*.npz" --repo $REPO --config $CFG --checkpoint $CKPT \
  --out $OUT/none_float16 --pool none --dtype float16 --resume 2>&1 | grep -E "resume|skipped|DONE" | cut -c1-300
echo "--- clips.json head"; head -c 700 $OUT/none_float16/clips.json; echo
echo "--- features.meta.json (selected)"; $MPY -c "
import json,sys; m=json.load(open('$OUT/none_float16/features.meta.json'))
print(json.dumps({k:m[k] for k in ('N','T','grid','D','dtype','pool','bytes_features','hazard_fraction','split_counts','peak_vram_gb','timing','verify_encoding','rounding_check','self_checks')}, indent=1, default=str))
print('size_table'); [print(f'  {k:18s} {v[\"gb_per_1k_clips\"]:.3f} GB / 1k clips   {v[\"gb_at_12000_clips\"]:.2f} GB @ 12k') for k,v in m['size_table'].items()]
print('shards'); [print('  ', s['path'].split('/')[-1], s['sha256'][:12], s['n'], s['start'], s['end']) for s in m['shards']]"

echo "=== 3. trainer loader + one fwd/bwd step (train_feature_predictor.py imports)"
$MPY - $REPO $OUT <<'EOF' 2>&1 | grep -Ev '^INFO:|^🔧|^🧠|^🔮|^📉|Encoder: |Predictor: '
import sys, json, numpy as np, torch
repo, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
import train_feature_predictor as tfp
dev = torch.device("cuda:0")
for tag, grid in (("none_float16", 16), ("patch2_float16", 8)):
    store = tfp.load_store(tfp.Path(f"{out}/{tag}"), None)  # picks up clips.json automatically
    print(tag, "store", store.n, store.t, store.grid, store.dim, store.action_dim, type(store.features).__name__, store.features.dtype,
          "hazard_frac", store.hazard_fraction(), "splits", sorted({c["split"] for c in store.clips}), "sha", {k: v[:10] for k, v in store.sha256.items()})
    tr, va, rule = tfp.split_indices(store, 0.25, 0)
    print("  split rule ->", rule, len(tr), len(va))
    cfg = tfp.ModelCfg(pred_depth=2, pred_embed_dim=256, pred_num_heads=4, embed_dim=store.dim, grid_size=store.grid, img_size=256, num_frames_pred=store.t, action_dim=store.action_dim)
    torch.manual_seed(0)
    pred, _ = tfp.build_vendor_predictor(cfg, dev)
    feats, acts = tfp.fetch_batch(store, np.arange(min(8, store.n)), dev)
    gen = torch.Generator().manual_seed(0)
    torch.cuda.reset_peak_memory_stats(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, stats = tfp.jepa_wm_loss(pred, feats, acts, 2, "random", 3, True, gen)
    loss.backward()
    gn = float(torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0))
    print(f"  fwd/bwd ok: feats {tuple(feats.shape)} acts {tuple(acts.shape)} loss {float(loss):.4f} stats {json.dumps({k: round(v, 4) for k, v in stats.items()})} grad_norm {gn:.3f} peak_vram_gb {torch.cuda.max_memory_allocated(dev)/1e9:.2f}")
    val = tfp.evaluate(pred, store, va, 8, dev, 2, lambda: torch.autocast("cuda", dtype=torch.bfloat16))
    print("  evaluate:", json.dumps({k: round(v, 4) for k, v in val.items()}))
EOF

echo "=== 4. independent check: pipeline encode_frames(stored frame) vs stored feature"
$MPY - $REPO $CFG $CKPT $OUT <<'EOF' 2>&1 | grep -Ev '^INFO:|^🔧|^🧠|^🔮|^📉|Encoder: |Predictor: '
import sys, json, numpy as np, torch
repo, cfg, ckpt, out = sys.argv[1:5]
from pathlib import Path
from model_action_sensitivity import encode_frames, load_model
from precompute_dinov3_features import load_features, from_storage, pool_tokens
dev = torch.device("cuda:0")
model, _ = load_model(Path(repo), Path(cfg), Path(ckpt), "jepa_wm_droid", "cuda:0")
clips = json.load(open(f"{out}/none_float16/clips.json"))
rng = np.random.default_rng(1)
worst = {}
for tag in ("none_float16", "patch2_float16", "none_bfloat16"):
    feats, meta = load_features(Path(out) / tag)
    half_ulp = {"float16": 2.0**-11, "bfloat16": 2.0**-8}[meta["dtype"]]
    for _ in range(4):
        i = int(rng.integers(len(clips))); t = int(rng.integers(meta["T"]))
        c = clips[i]
        with np.load(f"{out}/shards/{c['shard']}") as z:
            frame = np.ascontiguousarray(z["frames"][c["index_in_shard"], t:t+1])
        z = encode_frames(model, frame, dev)[0, 0, 0].float()          # pipeline path, [16,16,1024]
        ref = pool_tokens(z.unsqueeze(0), meta["pool"])[0].cpu().numpy()
        got = from_storage(feats[i, t], meta["dtype"])
        err = np.abs(got - ref); bound = half_ulp * np.abs(ref) + 1e-4   # half-ulp + fp32 batch-shape nondeterminism (<= 1.9e-5 measured)
        ok = bool(np.all(err <= bound)); worst[tag] = max(worst.get(tag, 0.0), float(err.max()))
        assert ok, (tag, i, t, float(err.max()), float((err - bound).max()))
    # a full clip through model.encode directly (the unroll entry point) vs stored
    i = 0
    with np.load(f"{out}/shards/{clips[i]['shard']}") as zz:
        clip = np.ascontiguousarray(zz["frames"][clips[i]["index_in_shard"]])
    with torch.inference_mode():
        whole = model.encode(torch.from_numpy(clip).permute(0, 3, 1, 2).unsqueeze(0).to(dev))[0, :, 0].float()
    ref = pool_tokens(whole, meta["pool"]).cpu().numpy(); got = from_storage(feats[i], meta["dtype"])
    e = float(np.abs(got - ref).max()); assert np.all(np.abs(got - ref) <= half_ulp * np.abs(ref) + 1e-4), (tag, e)
    print(f"{tag}: encode_frames/model.encode == stored within {meta['dtype']} half-ulp; max_abs_err {worst[tag]:.3e} (clip0 whole-clip {e:.3e}); feature |max| {float(np.abs(ref).max()):.2f} rms {float(np.sqrt((ref**2).mean())):.3f}")
# fp16 vs bf16 rounding on the same content
a = from_storage(np.load(f"{out}/none_float16/features.npy", mmap_mode="r")[:4], "float16")
b = from_storage(np.load(f"{out}/none_bfloat16/features.bf16.npy", mmap_mode="r")[:4], "bfloat16")
print(f"fp16 vs bf16 stored: max_abs_diff {float(np.abs(a-b).max()):.3e}, rms_diff {float(np.sqrt(((a-b)**2).mean())):.3e}, rms {float(np.sqrt((a**2).mean())):.3f}")
print("VERIFY_OK")
EOF

echo "=== 5. sizes"; du -sh $OUT/*; ls -la $OUT/*/features*.npy
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
rm -rf $OUT; echo "cleaned: $(ls -d $OUT 2>&1)"; df -h /root | tail -1
echo "[$(date -u +%FT%TZ)] SMOKE_DONE"

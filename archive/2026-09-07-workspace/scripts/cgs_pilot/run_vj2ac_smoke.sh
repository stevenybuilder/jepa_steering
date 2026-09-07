#!/bin/bash
set -u
cd /root/cgs-pilot
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs
export PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:/root/cgs-pilot/code/cgs_pilot
MPY=/opt/conda/bin/python
REPO=/root/cgs-pilot/vendor/jepa-wms
CFG=$REPO/configs/evals/simu_env_planning/droid/vj2ac/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml
CKPT=/root/cgs-pilot/checkpoints/droid_vj2ac_noprop.pth.tar
SMOKE=/root/cgs-pilot/artifacts/_smoke_vj2ac
echo "[$(date -u +%FT%TZ)] smoke_vjepa2_ac start"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
$MPY code/cgs_pilot/smoke_vjepa2_ac.py --repo $REPO --config $CFG --checkpoint $CKPT --model-name vjepa2_ac_droid \
  --cell /root/cgs-pilot/artifacts/heldout_v1/seed_202/cells/contact_seed000202__h1a1.npz --out $SMOKE/smoke_report.json
echo "[$(date -u +%FT%TZ)] smoke rc=$?"
echo "[$(date -u +%FT%TZ)] latent_cache start"
$MPY code/cgs_pilot/latent_cache.py --model-name vjepa2_ac_droid --config $CFG --checkpoint $CKPT --repo $REPO \
  --artifacts $SMOKE --cache $SMOKE/latent_cache_vj2ac.npz
echo "[$(date -u +%FT%TZ)] latent_cache rc=$?"
$MPY - <<PY
import numpy as np, json
z = np.load("$SMOKE/latent_cache_vj2ac.npz")
for k in z.files:
    v = z[k]
    if k == "meta": print(k, json.loads(str(v))); continue
    fin = bool(np.isfinite(v.astype(np.float32)).all()) if v.dtype.kind == "f" else None
    print(k, v.shape, v.dtype, "finite=", fin)
PY
echo SMOKE_DONE

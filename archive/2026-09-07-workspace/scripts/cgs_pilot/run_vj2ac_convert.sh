#!/bin/bash
# Download HF vjepa2 ViT-g safetensors, convert to native, verify, delete source.
set -u
cd /root/cgs-pilot
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints
export PYTHONPATH=/root/cgs-pilot/vendor/jepa-wms:/root/cgs-pilot/code/cgs_pilot
export HF_HUB_ENABLE_HF_TRANSFER=0
HFDIR=/root/cgs-pilot/checkpoints/vjepa2_hf
mkdir -p "$HFDIR"
echo "[$(date -u +%FT%TZ)] download start"; df -h /root | tail -1
/opt/conda/bin/python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("facebook/vjepa2-vitg-fpc64-256", allow_patterns=["config.json", "model.safetensors"], local_dir="$HFDIR")
print("downloaded to", p)
PY
rc=$?
echo "[$(date -u +%FT%TZ)] download rc=$rc"; ls -la "$HFDIR"; df -h /root | tail -1
[ $rc -ne 0 ] && { echo DOWNLOAD_FAILED; exit 1; }
sz=$(stat -c %s "$HFDIR/model.safetensors")
[ "$sz" -ne 4138311608 ] && { echo "SIZE_MISMATCH $sz"; exit 1; }
echo "[$(date -u +%FT%TZ)] convert start"
/opt/conda/bin/python /root/cgs-pilot/code/cgs_pilot/convert_vjepa2_hf_to_native.py \
  --repo /root/cgs-pilot/vendor/jepa-wms --hf-dir "$HFDIR" \
  --output /root/cgs-pilot/checkpoints/vjepa2_opensource/vjepa2_vit_giant.pth \
  --provenance-dir /root/cgs-pilot/artifacts/cgs_pilot/vjepa2_backbone \
  --expected-source-sha256 f205e77aa2ade168db6b09d4bc420d156141f64ab964278a9c181a2bdf2a232b \
  --delete-source
rc=$?
echo "[$(date -u +%FT%TZ)] convert rc=$rc"; df -h /root | tail -1
[ $rc -eq 0 ] && echo CONVERT_DONE || echo CONVERT_FAILED

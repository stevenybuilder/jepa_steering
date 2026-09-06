#!/bin/bash
# Panel B smoke driver (box). Everything under /root/cgs-pilot/artifacts/_panelB_dev; delete when done.
set -u
DEV=/root/cgs-pilot/artifacts/_panelB_dev
REPO=/root/cgs-pilot/vendor/jepa-wms
MPY=/opt/conda/bin/python
export JEPAWM_HOME=/root/cgs-pilot/vendor JEPAWM_OSSCKPT=/root/cgs-pilot/checkpoints JEPAWM_LOGS=/root/cgs-pilot/logs JEPAWM_DRIVING_ACTION_DIM=2
export PYTHONPATH=$DEV/code:$REPO:/root/cgs-pilot/code/cgs_pilot
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
FILTER='^INFO:|^🔧|^🧠|^🔮|^📉|Encoder: |Predictor: |^Loading|^Loaded'
ts() { date -u +%FT%TZ; }
cd $DEV/code
COMMON="--repo $REPO --synthetic 64,4,2 --epochs 2 --batch-size 16 --pred-depth 6 --pred-embed-dim 512 --seed 0 --no-save-opt --arm smoke"
run() { echo "[$(ts)] === $*"; local t0=$SECONDS; "$@" 2>&1 | grep -Ev "$FILTER"; echo "[$(ts)] rc=${PIPESTATUS[0]} wall=$((SECONDS - t0)) s"; }
echo "[$(ts)] START $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader); $(df -h /root | tail -1)"
run $MPY train_feature_predictor.py $COMMON --out $DEV/adaln_new
run $MPY /root/cgs-pilot/code/cgs_pilot/train_feature_predictor.py $COMMON --out $DEV/adaln_old
run $MPY train_feature_predictor.py $COMMON --pred-type token --write-synthetic $DEV/synth_store --out $DEV/token
run $MPY train_feature_predictor.py $COMMON --pred-type feature --action-emb-dim 16 --out $DEV/feature
run $MPY train_feature_predictor.py --repo $REPO --features $DEV/synth_store --epochs 2 --batch-size 16 --pred-depth 6 --pred-embed-dim 512 --seed 0 --no-save-opt --arm smoke_store --pred-type token --out $DEV/token_store
echo "[$(ts)] === old-vs-new AdaLN equivalence"
$MPY - $DEV <<'EOF'
import json, sys, torch
from pathlib import Path
dev = Path(sys.argv[1])
a = torch.load(dev / "adaln_new/jepa-latest.pth.tar", map_location="cpu", weights_only=True)
b = torch.load(dev / "adaln_old/jepa-latest.pth.tar", map_location="cpu", weights_only=True)
same = set(a["predictor"]) == set(b["predictor"]) and all(torch.equal(a["predictor"][k], b["predictor"][k]) for k in a["predictor"])
ma = json.loads((dev / "adaln_new/jepa-latest.meta.json").read_text()); mb = json.loads((dev / "adaln_old/jepa-latest.meta.json").read_text())
ha = [{k: v for k, v in r.items() if not k.endswith("_s") and k != "ms_per_iter" and k != "peak_vram_gb"} for r in ma["history"]]
hb = [{k: v for k, v in r.items() if not k.endswith("_s") and k != "ms_per_iter" and k != "peak_vram_gb"} for r in mb["history"]]
import yaml
def strip(p):
    d = yaml.safe_load(p.read_text()); d.pop("folder", None); return d
ev = strip(dev / "adaln_new/eval_config.yaml") == strip(dev / "adaln_old/eval_config.yaml")
tr_new = strip(dev / "adaln_new/train_config.yaml"); tr_old = strip(dev / "adaln_old/train_config.yaml")
extra_model_keys = sorted(set(ma["model"]) - set(mb["model"]))
extra_hyper_keys = sorted(set(ma["hyperparameters"]) - set(mb["hyperparameters"]))
print("ADALN_EQUIV", json.dumps({"predictor_tensors_equal": same, "history_equal": ha == hb, "eval_config_equal_modulo_folder": ev,
      "train_config_equal_modulo_folder": tr_new == tr_old, "meta_model_extra_keys": extra_model_keys, "meta_hyper_extra_keys": extra_hyper_keys,
      "new_hist": ha[-1], "old_hist": hb[-1]}))
# raw byte comparison of the eval yaml after replacing the folder line
ea = (dev / "adaln_new/eval_config.yaml").read_text().replace("adaln_new", "X"); eb = (dev / "adaln_old/eval_config.yaml").read_text().replace("adaln_old", "X")
print("ADALN_EVAL_YAML_BYTES_EQUAL", ea == eb)
ta = (dev / "adaln_new/train_config.yaml").read_text().replace("adaln_new", "X"); tb = (dev / "adaln_old/train_config.yaml").read_text().replace("adaln_old", "X")
print("ADALN_TRAIN_YAML_BYTES_EQUAL", ta == tb)
EOF
for d in adaln_new token feature token_store; do run $MPY check_panelb_loader.py --repo $REPO --model-dir $DEV/$d; done
echo "[$(ts)] === with-encoder build check + bench (batch 16)"
run $MPY train_feature_predictor.py $COMMON --pred-type token --with-encoder --bench-iters 10 --out $DEV/bench_token
run $MPY train_feature_predictor.py $COMMON --pred-type feature --action-emb-dim 16 --with-encoder --bench-iters 10 --out $DEV/bench_feature
run $MPY train_feature_predictor.py $COMMON --bench-iters 10 --out $DEV/bench_adaln
echo "[$(ts)] === pytest (vendor-free CPU tests of the trainer against the NEW file)"
cp /root/cgs-pilot/code/cgs_pilot/test_train_feature_predictor.py $DEV/code/
(cd $DEV/code && $MPY -m pytest -q test_train_feature_predictor.py 2>&1 | tail -5)
echo "[$(ts)] === token/feature meta summary"
$MPY - $DEV <<'EOF'
import json, sys
from pathlib import Path
dev = Path(sys.argv[1])
for d in ("adaln_new", "token", "feature", "token_store"):
    m = json.loads((dev / d / "jepa-latest.meta.json").read_text())
    print(d, json.dumps({"class": m["model"]["predictor_class"], "params": m["model"]["params"], "pred_type": m["model"].get("pred_type"), "block_width": m["model"].get("block_width"),
        "split": m["data"]["split_rule"], "source": m["data"]["source"][:60], "ms_per_iter": round(m["timing"]["ms_per_iter_mean"], 1), "elapsed_s": round(m["timing"]["elapsed_s"], 1),
        "peak_vram_gb": round(m["history"][-1].get("peak_vram_gb", 0), 2), "last": {k: round(v, 5) for k, v in m["history"][-1].items() if k in ("train_loss", "val_tf_l2", "val_unroll_l2", "val_copy_baseline_l2")}}))
EOF
echo "[$(ts)] DONE $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader); $(du -sh $DEV | cut -f1); $(df -h /root | tail -1)"

"""CPU tests for ``scripts/cgs_pilot/train_feature_predictor.py`` with tiny dims.

A fake AdaLN-style predictor (same forward signature as the vendor
``VisionTransformerAdaLN``: ``forward(x [B,T,1,G,G,D], actions [B,T,A]) -> ([B,T,G*G,D], None, None)``)
exercises the store, split, vendor-style loss weighting, the training loop, the vendor
checkpoint format, the metadata sidecar, the eval yaml and the probe reload.  Skips when
torch is not importable (e.g. /usr/local/bin/python3.9 without torch).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")
nn = torch.nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from train_feature_predictor import (  # noqa: E402
    MODEL_NAME,
    ModelCfg,
    build_parser,
    fetch_batch,
    jepa_wm_loss,
    load_store,
    make_synthetic,
    mse,
    split_indices,
    train,
    unroll_eval,
    write_store,
)

G, D, A, T, W, DEPTH = 2, 8, 2, 4, 16, 2
N_TOK = G * G


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(W)
        self.attn = nn.Linear(W, W)
        self.norm2 = nn.LayerNorm(W)
        self.mlp = nn.Sequential(nn.Linear(W, 2 * W), nn.GELU(), nn.Linear(2 * W, W))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(W, 6 * W))

    def forward(self, x, z):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(z).repeat_interleave(N_TOK, dim=1).chunk(6, dim=2)
        )
        x = x + gate_msa * self.attn(self.norm1(x) * (1 + scale_msa) + shift_msa)
        return x + gate_mlp * self.mlp(self.norm2(x) * (1 + scale_mlp) + shift_mlp)


class FakePredictor(nn.Module):
    """Frame-wise (no cross-frame attention) stand-in with the vendor module layout."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor_embed_dim = W
        self.predictor_embed = nn.Linear(D, W)
        self.action_encoder = nn.Linear(A, W)
        self.predictor_blocks = nn.ModuleList([FakeBlock() for _ in range(DEPTH)])
        self.predictor_norm = nn.LayerNorm(W)
        self.predictor_proj = nn.Linear(W, D)

    def forward(self, x, actions, proprio=None):
        inp = x.flatten(2, 4)  # residual around the input so the tiny model starts near "copy last frame"
        x = self.predictor_embed(x).flatten(2, 4)
        b, t, n, w = x.shape
        z = self.action_encoder(actions)
        x = x.flatten(1, 2)
        for blk in self.predictor_blocks:
            x = blk(x, z)
        x = self.predictor_norm(x).view(b, t, n, w)
        return inp + self.predictor_proj(x), None, None


def _args(out: Path, **over) -> "argparse.Namespace":  # noqa: F821
    base = ["--out", str(out), "--synthetic", f"32,{T},{A}", "--grid", str(G), "--embed-dim", str(D), "--epochs", "12",
            "--batch-size", "8", "--ref-lr", "1e-2", "--start-lr", "1e-3", "--final-lr", "1e-3", "--warmup-epochs", "0.5",
            "--dtype", "float32", "--device", "cpu", "--arm", "unit_test", "--seed", "1"]
    for k, v in over.items():
        base += [f"--{k.replace('_', '-')}", str(v)]
    return build_parser().parse_args(base)


def test_synthetic_store_split_and_roundtrip(tmp_path: Path) -> None:
    store = make_synthetic(20, T, G, D, A, seed=3, hazard_fraction=0.5)
    assert store.features.shape == (20, T, G, G, D) and store.features.dtype == np.float16
    assert store.actions.shape == (20, T - 1, A)
    assert 0.0 < store.hazard_fraction() < 1.0
    train_idx, val_idx, rule = split_indices(store, 0.25, seed=0)
    assert len(val_idx) == 5 and len(train_idx) == 15 and not set(train_idx) & set(val_idx)
    for target in (tmp_path / "store.npz", tmp_path / "store_dir"):
        write_store(store, target)
        again = load_store(target, None)
        np.testing.assert_array_equal(np.asarray(again.features), np.asarray(store.features))
        np.testing.assert_array_equal(again.actions, store.actions)
        assert again.clips == store.clips and again.sha256
    store.clips[0]["split"] = "val"
    for c in store.clips[1:]:
        c["split"] = "train"
    tr, va, rule = split_indices(store, 0.25, 0)
    assert rule == "meta.split" and list(va) == [0] and len(tr) == 19


def test_loss_weighting_matches_vendor_two_roll() -> None:
    torch.manual_seed(0)
    pred = FakePredictor().eval()
    store = make_synthetic(4, T, G, D, A, seed=0)
    feats, acts = fetch_batch(store, np.arange(4), torch.device("cpu"))
    assert feats.shape == (4, T, 1, G, G, D) and acts.shape == (4, T, A)
    gen = torch.Generator().manual_seed(0)
    total, stats = jepa_wm_loss(pred, feats, acts, rollout_steps=2, prefixes="first", ctxt_window=3, stop_gradient=True, generator=gen)
    # manual: teacher forced over all transitions / 3, plus one rollout step from prefix t=0 weighted 1/2
    out, _, _ = pred(feats, acts)
    out = out.view(4, T, 1, G, G, D)
    tf = mse(out[:, :-1], feats[:, 1:])
    ctx = torch.cat([feats[:, :1], out[:, 0:1]], dim=1)
    nxt, _, _ = pred(ctx, acts[:, :2])
    roll = mse(nxt.view(4, 2, 1, G, G, D)[:, -1:], feats[:, 2:3])
    assert torch.isclose(total, tf / 3 + roll / 2, atol=1e-6)
    assert stats["tf_l2"] == pytest.approx(float(tf), rel=1e-5) and stats["roll_l2_2"] == pytest.approx(float(roll), rel=1e-5)
    # rollout_steps=3 with T=4 is a full unroll from frame 0: prefixes must be {0}
    total3, stats3 = jepa_wm_loss(pred, feats, acts, 3, "random", 3, True, gen)
    assert set(stats3) == {"tf_l2", "roll_l2_2", "roll_l2_3"}
    with pytest.raises(ValueError):
        jepa_wm_loss(pred, feats, acts, 4, "random", 3, True, gen)


def test_training_writes_vendor_checkpoint_and_probe_reloads(tmp_path: Path) -> None:
    args = _args(tmp_path / "run")
    store = make_synthetic(32, T, G, D, A, seed=args.seed)
    torch.manual_seed(0)
    predictor = FakePredictor()
    cfg = ModelCfg(pred_depth=DEPTH, pred_embed_dim=W, pred_num_heads=1, embed_dim=D, grid_size=G, img_size=G * 16, num_frames_pred=T, action_dim=A)
    meta = train(args, predictor, store, torch.device("cpu"), cfg)
    hist = meta["history"]
    assert len(hist) == args.epochs
    assert hist[-1]["train_loss"] < 0.6 * hist[0]["train_loss"], hist
    assert hist[-1]["val_unroll_l2"] < hist[0]["val_unroll_l2"]
    assert hist[-1]["val_unroll_l2"] < hist[-1]["val_copy_baseline_l2"], hist[-1]
    assert meta["arm"] == "unit_test" and meta["model_name"] == MODEL_NAME
    assert 0.0 <= meta["data"]["hazard_fraction_all"] <= 1.0 and meta["data"]["sha256"]
    assert meta["hyperparameters"]["ref_lr"] == 1e-2 and meta["best"]["epoch"] >= 1

    out = tmp_path / "run"
    for name in ("jepa-latest.pth.tar", "jepa-best.pth.tar", "jepa-latest.meta.json", "jepa-best.meta.json", "eval_config.yaml", "train_config.yaml", "probe.npz"):
        assert (out / name).exists(), name
    latest = torch.load(out / "jepa-latest.pth.tar", map_location="cpu", weights_only=True)  # vendor fetch_checkpoint default on torch>=2.6
    best = torch.load(out / "jepa-best.pth.tar", map_location="cpu", weights_only=False)
    assert set(latest) == {"predictor", "opt", "scaler", "epoch", "cgs_meta"}
    assert latest["epoch"] == args.epochs and latest["scaler"] is None and latest["opt"]["param_groups"]
    assert best["opt"] is None and best["epoch"] == meta["best"]["epoch"]
    assert set(latest["predictor"]) == set(predictor.state_dict()) and "action_encoder.weight" in latest["predictor"]
    assert tuple(latest["predictor"]["action_encoder.weight"].shape) == (W, A)
    side = json.loads((out / "jepa-latest.meta.json").read_text())
    assert side["best"] == meta["best"] and side["data"]["n_train"] + side["data"]["n_val"] == 32

    ev = yaml.safe_load((out / "eval_config.yaml").read_text())
    pk = ev["model_kwargs"]["pretrain_kwargs"]
    assert ev["model_kwargs"]["checkpoint"] == "jepa-best.pth.tar" and ev["folder"] == str(out)
    assert pk["predictor"] == {"tubelet_size": 1, "pred_num_heads": 1, "pred_depth": DEPTH, "pred_embed_dim": W, "pred_use_extrinsics": False,
                               "pred_type": "AdaLN", "act_pred_projector": None, "use_SiLU": None, "use_rope": True}
    assert pk["visual_encoder"]["enc_version"] == "dinov3_vitl16" and pk["num_frames_pred"] == T
    assert pk["action_encoder"]["action_encoder_inpred"] is True and pk["proprio_encoder"]["proprio_tokens"] == 0
    assert pk["heads_cfg"] == {"architectures": {}, "pretrain_dec_path": {}}
    assert ev["model_kwargs"]["data"]["datasets"] == ["driving"] and ev["model_kwargs"]["wrapper_kwargs"]["ctxt_window"] == 2
    assert ev["cgs_pilot"]["action_dim"] == A
    tr = yaml.safe_load((out / "train_config.yaml").read_text())
    assert tr["model"]["rollout_cfg"]["rollout_steps"] == 2 and tr["optimization"]["transition_model"]["num_epochs"] == args.epochs

    # probe reload: a fresh predictor with the saved weights reproduces the eval-style unroll
    probe = np.load(out / "probe.npz")
    fresh = FakePredictor()
    fresh.load_state_dict(latest["predictor"])
    fresh.eval()
    z_ctx = torch.from_numpy(probe["context"])
    acts = torch.from_numpy(probe["actions"]).permute(1, 0, 2)  # [H, B, A] -> [B, H, A]
    with torch.no_grad():
        roll = unroll_eval(fresh, z_ctx, acts, int(probe["ctxt_window"]))
    assert roll.shape == (1, T - 1, 1, G, G, D)
    np.testing.assert_allclose(roll.permute(1, 0, 2, 3, 4, 5).numpy(), probe["prediction"], rtol=1e-5, atol=1e-5)


def test_max_minutes_and_val_fraction_zero(tmp_path: Path) -> None:
    args = _args(tmp_path / "run2", val_fraction=0.0, epochs=3, max_minutes=1e-9)
    store = make_synthetic(16, T, G, D, A, seed=1)
    meta = train(args, FakePredictor(), store, torch.device("cpu"), ModelCfg(pred_depth=DEPTH, pred_embed_dim=W, pred_num_heads=1, embed_dim=D, grid_size=G, num_frames_pred=T, action_dim=A))
    assert len(meta["history"]) == 1 and meta["data"]["split_rule"] == "val_is_train_subset"

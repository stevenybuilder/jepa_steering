#!/usr/bin/env python3
"""CPU-only spatial finite-response summaries of already captured action probes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8<<20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)


def norm(value):
    return np.linalg.norm(value, axis=-1)


def denominator_floor(denominator, baseline):
    """One declared scene/stage/horizon floor, not token-specific tuning."""
    return np.maximum(1e-10, np.maximum(1e-6*np.median(baseline, axis=-1), .01*np.median(denominator, axis=-1)))


def spatial_response(values, conditions, scale):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 4 or values.shape[:3] != (6, 17, 256):
        raise ValueError("Expected six horizons,17 conditions,256 native tokens")
    lookup = {(r["scale"], r["x_sign"], r["z_sign"]): i for i, r in enumerate(conditions)}
    zero = values[:, 0]
    odd2, even2 = np.zeros((6, 256)), np.zeros((6, 256))
    for plus, minus in (((1, 0), (-1, 0)), ((0, 1), (0, -1))):
        p, m = values[:, lookup[(scale, *plus)]], values[:, lookup[(scale, *minus)]]
        odd2 += norm((p-m)/2)**2
        even2 += norm(p+m-2*zero)**2
    mixed2, mixed_den2 = np.zeros_like(odd2), np.zeros_like(odd2)
    for sx, sz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        x, z, both = (values[:, lookup[(scale, *s)]] for s in ((sx, 0), (0, sz), (sx, sz)))
        mixed2 += norm(both-x-z+zero)**2/4
        mixed_den2 += (norm(x-zero)+norm(z-zero))**2/4
    odd, even = np.sqrt(odd2), np.sqrt(even2)
    mixed, mixed_den = np.sqrt(mixed2), np.sqrt(mixed_den2)
    odd_den = 2*odd
    curv_floor = denominator_floor(odd_den, norm(zero))
    mixed_floor = denominator_floor(mixed_den, norm(zero))
    return {"odd_l2": odd, "curvature_numerator_l2": even, "curvature_denominator_l2": odd_den,
            "mixed_numerator_l2": mixed, "mixed_denominator_l2": mixed_den,
            "curvature_floor": curv_floor, "mixed_floor": mixed_floor,
            "curvature_ratio": even/np.maximum(odd_den, curv_floor[:, None]),
            "mixed_ratio": mixed/np.maximum(mixed_den, mixed_floor[:, None])}


def region_metrics(response, horizon, indices):
    result = {"odd_l2": float(np.linalg.norm(response["odd_l2"][horizon, indices]))}
    for kind in ("curvature", "mixed"):
        numerator = float(np.linalg.norm(response[kind+"_numerator_l2"][horizon, indices]))
        denominator = float(np.linalg.norm(response[kind+"_denominator_l2"][horizon, indices]))
        floor = float(response[kind+"_floor"][horizon]*np.sqrt(len(indices)))
        result[kind] = {"raw_numerator_l2": numerator, "raw_denominator_l2": denominator,
                        "denominator_floor": floor, "ratio_of_aggregate_norms": numerator/max(denominator, floor),
                        "tokens_below_floor": int(np.sum(response[kind+"_denominator_l2"][horizon, indices] < response[kind+"_floor"][horizon]))}
    return result


def stability(values):
    ranks = {h+1: np.argsort(-values[h], kind="stable")[:64] for h in (0, 2, 5)}
    early = set(ranks[1].tolist())
    return {"top_quartile_token_ids": {str(k): v.tolist() for k, v in ranks.items()},
            "comparisons": [{"early": 1, "later": h, "intersection_fraction": len(early & set(ranks[h]))/64,
                             "jaccard": len(early & set(ranks[h]))/len(early | set(ranks[h]))} for h in (3, 6)]}


def svg_map(path, stage, responses):
    metrics = (("odd_l2", "Odd response L2"), ("curvature_ratio", "Even / signed difference"), ("mixed_ratio", "Mixed XZ / single effects"))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="750" height="790" viewBox="0 0 750 790">',
             '<rect width="750" height="790" fill="white"/>',
             f'<text x="20" y="26" font-family="sans-serif" font-size="17">{stage}: spatial action response, three fixed development scenes</text>',
             '<text x="20" y="48" font-family="sans-serif" font-size="11">Raw action amplitude0.05; first5 actions perturbed. Response maps, not individually edited-token importance.</text>']
    for col, (key, label) in enumerate(metrics):
        averaged = np.mean([r[key] for r in responses], axis=0)
        maximum = float(averaged[[0, 2, 5]].max()) or 1.
        x0 = 55+col*230
        parts.append(f'<text x="{x0}" y="77" font-family="sans-serif" font-size="12">{label}; max={maximum:.4g}</text>')
        for row, h in enumerate((0, 2, 5)):
            y0 = 105+row*215
            if col == 0:
                parts.append(f'<text x="10" y="{y0+90}" font-family="sans-serif" font-size="12">H{h+1}</text>')
            for token, value in enumerate(averaged[h]):
                fraction = max(0., min(1., float(value)/maximum))
                r, g, b = (int(245-215*fraction), int(248-130*fraction), int(255-65*fraction))
                parts.append(f'<rect x="{x0+(token%16)*12}" y="{y0+(token//16)*12}" width="12" height="12" fill="rgb({r},{g},{b})"><title>token{token}: {value:.6g}</title></rect>')
    parts.append('<text x="20" y="772" font-family="sans-serif" font-size="11">Fixed scene mean; image-grid token order. Color maxima shared over H1/H3/H6 within each metric and stage.</text></svg>')
    path.write_text("".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic(); torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.source/"DONE.json").read_text())
    if not receipt["complete"] or sorted(receipt["episodes"]) != [0, 4, 10]:
        raise RuntimeError("Expected exactly three completed predetermined scenes")
    grid = np.arange(256).reshape(16, 16)
    regions = {"global": grid.ravel(), "top_left": grid[:8, :8].ravel(), "top_right": grid[:8, 8:].ravel(),
               "bottom_left": grid[8:, :8].ravel(), "bottom_right": grid[8:, 8:].ravel()}
    arrays, rows, rank_rows, plot_values, sources = {}, [], [], {}, []
    for episode in (0, 4, 10):
        file = args.source/f"episode-{episode:03d}.pt"
        entry = next(row for row in receipt["outputs"] if row["path"] == file.name)
        if entry["episode"] != episode or sha(file) != entry["sha256"]:
            raise RuntimeError("Action probe checksum/episode mismatch before tensor access")
        value = torch.load(file, map_location="cpu", weights_only=False)
        if not value["actual_capture_nohooks_identity_exact"] or value["anchor_label"] != "baseline_zero_valid":
            raise RuntimeError("Wrong actual-network experiment")
        sources.append({"episode": episode, "path": str(file), "sha256": entry["sha256"]})
        for stage, captured in value["stages"].items():
            for scale in (.5, 1.):
                response = spatial_response(captured.numpy(), value["conditions"], scale)
                prefix = f"episode{episode}/{stage}/scale{scale}"
                arrays.update({prefix+"/"+key: data for key, data in response.items()})
                for h in range(6):
                    for region, indices in regions.items():
                        rows.append({"episode": episode, "stage": stage, "scale": scale, "raw_amplitude": .05*scale,
                                     "imagined_step": h+1, "region": region, **region_metrics(response, h, indices)})
                for key in ("odd_l2", "curvature_numerator_l2", "mixed_numerator_l2"):
                    rank_rows.append({"episode": episode, "stage": stage, "scale": scale, "ranking_metric": key, **stability(response[key])})
                if scale == 1. and stage in ("P3", "returned_visual"):
                    plot_values.setdefault(stage, []).append(response)
        del value
    np.savez_compressed(args.output_dir/"spatial_arrays.npz", **arrays)
    for stage, response in plot_values.items():
        svg_map(args.output_dir/(stage+"_H1_H3_H6.svg"), stage, response)
    result = {"complete": True, "scope": "CPU-only existing17condition H6 action-response tensors,3development scenes; no new model/simulator call",
              "sources": sources, "source_done_sha256": sha(args.source/"DONE.json"), "script_sha256": sha(Path(__file__)),
              "token_grid": [16, 16], "token_semantics": "row-major newest-frame patch positions; no head grid or individual-token causal edit",
              "odd_definition": "sqrt(sum over X,Z ||(f(+a)-f(-a))/2||channel^2)",
              "curvature_definition": "sqrt(sum over X,Z ||f(+a)+f(-a)-2f(0)||channel^2) / (2*odd)",
              "mixed_definition": "RMS over four signedXZ factorial residual norms divided by RMS(||singleX||+||singleZ||)",
              "floor_policy": "per scene/stage/horizon/scale max(1e-10,1e-6*median baseline tokennorm,0.01*median denominator tokennorm); region ratios use aggregate rawL2 and floor*sqrt(token_count), not mean tokenratios",
              "rows": rows, "spatial_rank_stability": rank_rows,
              "interpretation": "Spatial distribution of finite-dose input response; rank stability descriptive over3scenes, not individual-token importance, dense-manifold evidence or steering efficacy",
              "seconds": time.monotonic()-started}
    write(args.output_dir/"action_token_time.json", result)
    outputs = [{"path": p.name, "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(args.output_dir.iterdir()) if p.is_file()]
    write(args.output_dir/"DONE.json", {"complete": True, "outputs": outputs})
    print(json.dumps({"event": "action_token_time_complete", "seconds": result["seconds"], "region_rows": len(rows), "rank_rows": len(rank_rows), "outputs": outputs}), flush=True)


if __name__ == "__main__":
    main()

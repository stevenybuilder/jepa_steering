#!/usr/bin/env python3
"""Discovery-only full-spatial action Jacobian/PCA bases for all6 predictor blocks."""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from protocol import file_sha256, write_json_atomic

FRAMES = (0, 25, 50, 75)
EPISODES = tuple(range(8))
SITES = ("residual", "attention_preproj", "mlp_output")


def action_probes(actual):
    if actual.shape != (5, 4):
        raise ValueError("One five-raw-action chunk required")
    values = torch.zeros(32, 5, 4, dtype=actual.dtype)
    for index in range(15):
        raw_step, axis = divmod(index, 3)
        values[1+2*index, raw_step, axis] = .05
        values[2+2*index, raw_step, axis] = -.05
    values[-1] = actual
    return values


def orthonormal_rows(values, max_rank):
    """Thin SVD in row-space; preserve raw Euclidean activation coordinates."""
    values = np.asarray(values, dtype=np.float64)
    gram = values@values.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    rank = min(max_rank, int(np.sum(eigenvalues > max(float(eigenvalues[0])*1e-12, 1e-16))))
    if rank == 0:
        return np.empty((0, values.shape[1]), np.float32), eigenvalues
    basis = (eigenvectors[:, :rank].T@values)/np.sqrt(eigenvalues[:rank, None])
    return basis.astype(np.float32), eigenvalues


class CaptureAll:
    def __init__(self, blocks):
        self.blocks, self.handles, self.values = blocks, [], {}

    def save(self, name, value):
        if name in self.values or value.shape != (32, 256, 400):
            raise RuntimeError(f"Unexpected repeated/native capture shape:{name} {value.shape}")
        self.values[name] = value.detach().float().cpu().clone()

    def __enter__(self):
        for index, block in enumerate(self.blocks):
            self.handles.append(block.register_forward_hook(lambda m,a,v,i=index: self.save(f"{i}/residual",v)))
            self.handles.append(block.attn.proj.register_forward_pre_hook(lambda m,a,i=index: self.save(f"{i}/attention_preproj",a[0])))
            self.handles.append(block.mlp.register_forward_hook(lambda m,a,v,i=index: self.save(f"{i}/mlp_output",v)))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "parquet", "manifest", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    sys.path.insert(0, str(args.repo))
    from canary_capture import decode_selected
    from model_loader import load_headless_metaworld
    from capture_specificity_controls import parameters_sha
    from evals.simu_env_planning.planning.utils import make_td
    import pyarrow.parquet as pq
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    selected = [r for r in manifest if r["task"] == "mw-reach-wall" and r["seed"] == 2 and r["episode"] in EPISODES and r["split"] == "discovery"]
    if sorted(r["episode"] for r in selected) != list(EPISODES) or {r["parquet_sha256"] for r in selected} != {file_sha256(args.parquet)}:
        raise RuntimeError("Frozen discovery sources do not match official file")
    protocol = {"phase": "discovery-only action/PCA basis fit", "seed": 2, "episodes": EPISODES, "raw_frames": FRAMES,
                "probe": "zero-action anchor plus15 independent rawXYZ positions with symmetric+/-0.05; one unchanged native-action sample for natural PCA",
                "perturbed_raw_actions_are_bounded": True, "native_actions_preserved_not_clipped": True,
                "basis_coordinates": "raw Euclidean full256x400 spatial vectors; action Jacobian average across32 discovery contexts; natural PCA centered",
                "sources": selected, "source_manifest_sha256": file_sha256(args.manifest), "script_sha256": file_sha256(Path(__file__)),
                "future_physical_outcomes_used": False, "development_or_validation_fit_access": False}
    write_json_atomic(args.output_dir/"protocol.json", protocol)
    table = pq.read_table(args.parquet, columns=["task", "seed", "episode", "video", "states", "actions"],
                          filters=[("task", "=", "mw-reach-wall"), ("seed", "=", 2), ("episode", "in", list(EPISODES))])
    wm, preprocessor, model = load_headless_metaworld(args.repo)
    wm.eval().requires_grad_(False); before = parameters_sha(wm)
    blocks = wm.model.predictor.predictor_blocks
    if len(blocks) != 6:
        raise RuntimeError("Expected exactly6 released predictor blocks")
    architecture = []
    for index, block in enumerate(blocks):
        dim, heads = block.attn.proj.in_features, block.attn.num_heads
        if dim != 400 or heads != 16 or dim%heads:
            raise RuntimeError("Unexpected native head layout")
        architecture.append({"block": index, "heads": heads, "head_dim": dim//heads, "dim": dim,
                             "attention_class": type(block.attn).__name__, "mlp_class": type(block.mlp).__name__,
                             "block_source": inspect.getfile(type(block)), "block_source_sha256": file_sha256(Path(inspect.getfile(type(block)))),
                             "attention_source": inspect.getfile(type(block.attn)), "attention_source_sha256": file_sha256(Path(inspect.getfile(type(block.attn))))})
    write_json_atomic(args.output_dir/"architecture.json", {"actual_runtime": architecture, "model": model})
    print(json.dumps({"event": "search_architecture_verified", "actual_runtime": architecture}), flush=True)
    jacobians, natural, source_hashes = {}, {}, []
    started = time.monotonic()
    for episode in EPISODES:
        index = next(i for i in range(len(table)) if table["episode"][i].as_py() == episode)
        row = {k: table[k][index].as_py() for k in table.column_names}
        frames = decode_selected(row["video"], FRAMES)
        for frame_index, raw_frame in enumerate(FRAMES):
            raw = action_probes(torch.tensor(row["actions"][raw_frame:raw_frame+5], dtype=torch.float32))
            normalized = preprocessor.normalize_actions(raw).reshape(1, 32, 20).cuda()
            pixels = torch.from_numpy(frames[frame_index]).permute(2, 0, 1).unsqueeze(0)
            proprio = torch.tensor(row["states"][raw_frame][:4], dtype=torch.float32).unsqueeze(0)
            z = wm.encode(make_td(pixels, {"proprio": proprio}).cuda().unsqueeze(0), act=True)
            with CaptureAll(blocks) as capture:
                prediction = wm.unroll(z.clone(), act_suffix=normalized)
            if episode == 0 and raw_frame == 0:
                repeated = wm.unroll(z.clone(), act_suffix=normalized)
                if not all(torch.equal(repeated[k], prediction[k]) for k in ("visual", "proprio")):
                    raise RuntimeError("Actual all-site capture nohook identity failed")
            for name, values in capture.values.items():
                derivative = ((values[1:31:2]-values[2:31:2])/.1).numpy()
                jacobians[name] = jacobians.get(name, np.zeros_like(derivative, dtype=np.float64))+derivative
                natural.setdefault(name, []).append(values[-1].numpy())
            source_hashes.append({"episode": episode, "raw_frame": raw_frame})
            print(json.dumps({"event": "discovery_action_context_captured", "episode": episode, "raw_frame": raw_frame, "count": len(source_hashes)}), flush=True)
    arrays, ranks = {}, []
    for name in jacobians:
        jacobian = (jacobians[name]/len(source_hashes)).astype(np.float32)
        states = np.stack(natural[name]); mean = states.mean(0)
        action_basis, action_eigenvalues = orthonormal_rows(jacobian.reshape(15, -1), 16)
        pca_basis, pca_eigenvalues = orthonormal_rows((states-mean).reshape(len(states), -1), 16)
        arrays.update({name+"/jacobian": jacobian, name+"/action_basis": action_basis, name+"/pca_basis": pca_basis,
                       name+"/mean": mean, name+"/action_eigenvalues": action_eigenvalues, name+"/pca_eigenvalues": pca_eigenvalues})
        ranks.append({"site": name, "action_rank": len(action_basis), "pca_rank": len(pca_basis)})
        if name.endswith("/residual"):
            arrays[name.split('/')[0]+"/cap_radius"] = np.asarray(.005*np.sqrt(np.square(states.astype(np.float64)).sum((1, 2)).mean()))
    path = args.output_dir/"discovery_bases.npz"
    np.savez_compressed(path, **arrays)
    if parameters_sha(wm) != before:
        raise RuntimeError("Frozen weights changed")
    report = args.output_dir/"basis_report.json"
    write_json_atomic(report, {"complete": True, "protocol": protocol, "architecture": architecture, "ranks": ranks,
                              "parameters_unchanged": True, "parameter_sha256": before, "model": model, "contexts": source_hashes,
                              "seconds": time.monotonic()-started, "basis_sha256": file_sha256(path)})
    outputs = [{"path": p.name, "sha256": file_sha256(p), "bytes": p.stat().st_size} for p in (path, report, args.output_dir/"architecture.json", args.output_dir/"protocol.json")]
    write_json_atomic(args.output_dir/"DONE.json", {"complete": True, "outputs": outputs})
    print(json.dumps({"event": "discovery_search_bases_complete", "outputs": outputs, "seconds": time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    main()

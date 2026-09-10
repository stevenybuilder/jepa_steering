#!/usr/bin/env python3
"""Panel P provenance (Label-first principle, amendment 3): archive checkpoint URL + sha256, repo
revision and working-tree status, evaluator config sha256s, environment package versions and
dataset file digests BEFORE any inference. Written to ``artifacts/public_panel/provenance.json``."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

CKPT_URL = "https://dl.fbaipublicfiles.com/jepa-wms/{env}_{model}.pth.tar"
ENVS = ("mz", "wall", "pt", "mw")
MODELS = ("jepa-wm", "dino-wm")
PACKAGES = (
    "torch", "torchvision", "numpy", "gym", "gymnasium", "metaworld", "mujoco", "mujoco_py", "d4rl",
    "pymunk", "pygame", "shapely", "timm", "nevergrad", "torchrl", "tensordict", "omegaconf", "hydra",
    "cv2", "skimage", "lpips", "imageio", "decord", "datasets", "huggingface_hub", "yaml", "einops",
)
# Table 2 of arXiv:2512.24497 (JEPA-WM / DINO-WM), success rate in %, 96 episodes x 3 seeds
TABLE2 = {
    ("mz", "jepa-wm"): 83.9, ("mz", "dino-wm"): 81.6,
    ("wall", "jepa-wm"): 78.8, ("wall", "dino-wm"): 64.1,
    ("pt", "jepa-wm"): 70.2, ("pt", "dino-wm"): 66.0,
    ("mw-reach", "jepa-wm"): 58.2, ("mw-reach", "dino-wm"): 44.8,
    ("mw-reach-wall", "jepa-wm"): 41.6, ("mw-reach-wall", "dino-wm"): 35.1,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_set_manifest(paths: list[Path], root: Path) -> dict:
    """Hash both every file and the canonical ordered file set."""
    rows = []
    for path in sorted(paths):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    encoded = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return {
        "n_files": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "file_set_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "files": rows,
    }


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def pkg_version(name: str):
    try:
        m = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        return {"importable": False, "error": str(e)[:200]}
    v = getattr(m, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version

            v = version(name.replace("_", "-"))
        except Exception:  # noqa: BLE001
            v = None
    return {"importable": True, "version": v, "file": getattr(m, "__file__", None)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--datasets", default=os.environ.get("JEPAWM_DSET"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    ck = Path(args.checkpoints).resolve()
    sys.path.insert(0, str(repo))

    checkpoints = {}
    for env in ENVS:
        for model in MODELS:
            p = ck / f"{env}_{model}.pth.tar"
            checkpoints[f"{env}_{model}"] = {
                "url": CKPT_URL.format(env=env, model=model),
                "path": str(p),
                "exists": p.exists(),
                "bytes": p.stat().st_size if p.exists() else None,
                "sha256": sha256_file(p) if p.exists() else None,
            }

    configs = {}
    for env in ENVS:
        for model in MODELS:
            d = repo / "configs" / "evals" / "simu_env_planning" / env / model
            for c in sorted(d.glob("*.yaml")):
                configs[f"{env}_{model}"] = {"path": str(c.relative_to(repo)), "sha256": sha256_file(c)}

    evaluator_files = {}
    for rel in (
        "evals/main.py", "evals/scaffold.py", "evals/simu_env_planning/eval.py",
        "evals/simu_env_planning/planning/plan_evaluator.py", "evals/simu_env_planning/planning/gc_agent.py",
        "evals/simu_env_planning/planning/planning/planner.py", "evals/simu_env_planning/planning/planning/objectives.py",
        "evals/simu_env_planning/planning/utils.py", "evals/simu_env_planning/envs/init.py",
        "evals/simu_env_planning/envs/pointmaze_gym_wrap.py", "evals/simu_env_planning/envs/wall_gym_wrap.py",
        "evals/simu_env_planning/envs/pusht_gym_wrap.py", "evals/simu_env_planning/envs/metaworld.py",
        "evals/simu_env_planning/envs/pusht_env/pusht_env.py", "evals/simu_env_planning/envs/pointmaze_env/maze_model.py",
        "app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py",
    ):
        p = repo / rel
        evaluator_files[rel] = sha256_file(p) if p.exists() else None

    datasets = {}
    if args.datasets:
        droot = Path(args.datasets)
        for rel in (
            "point_maze/states.pth", "point_maze/actions.pth", "point_maze/seq_lengths.pth",
            "wall_single/train/states.pth", "wall_single/val/states.pth",
            "pusht_noise/train/states.pth", "pusht_noise/val/states.pth", "pusht_noise/val/seq_lengths.pkl",
        ):
            p = droot / rel
            datasets[rel] = {"exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None, "sha256": sha256_file(p) if p.exists() and p.stat().st_size < 3_000_000_000 else None}
        mw = droot / "Metaworld" / "data"
        mw_files = list(mw.glob("*.parquet")) if mw.exists() else []
        mw_manifest = file_set_manifest(mw_files, mw) if mw_files else {
            "n_files": 0,
            "total_bytes": 0,
            "file_set_sha256": None,
            "files": [],
        }
        datasets["Metaworld/data"] = {
            "exists": mw.exists(),
            **mw_manifest,
        }

    nvidia = ""
    try:
        nvidia = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    except Exception as e:  # noqa: BLE001
        nvidia = f"ERROR: {e}"
    cuda = None
    try:
        import torch

        cuda = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "available": torch.cuda.is_available()}
    except Exception as e:  # noqa: BLE001
        cuda = {"error": str(e)}

    prov = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "principle": "Performance-first frozen-world-model steering amendment (mechinterp-vla/cross model plan.md); native-label invariant",
        "substrate": "facebook/jepa-wms released checkpoints (Terver, Yang, Ponce, Bardes, LeCun, arXiv:2512.24497); evaluator configs/evals/simu_env_planning/<env>/<model>/*.yaml, succ_def: simu, executed unmodified",
        "table2_reference_success_pct": {f"{k[0]}|{k[1]}": v for k, v in TABLE2.items()},
        "protocol_reference": "Table 2: 96 episodes x 3 seeds per env (MetaWorld configs ship eval_episodes: 48); released configs carry meta.seed: 1 and tasks_per_node: 8",
        "host": {"hostname": platform.node(), "platform": platform.platform(), "python": sys.version, "nvidia_smi": nvidia, "torch": cuda},
        "repo": {
            "path": str(repo),
            "head": git(repo, "rev-parse", "HEAD"),
            "describe": git(repo, "log", "-1", "--format=%H %ad %s"),
            "status_porcelain": git(repo, "status", "--porcelain"),
            "diff_stat": git(repo, "diff", "--stat"),
            "remote": git(repo, "remote", "-v"),
        },
        "checkpoints": checkpoints,
        "evaluator_configs": configs,
        "evaluator_file_sha256": evaluator_files,
        "packages": {p: pkg_version(p) for p in PACKAGES},
        "mujoco210": {"path": str(Path.home() / ".mujoco" / "mujoco210"), "exists": (Path.home() / ".mujoco" / "mujoco210").exists()},
        "datasets": datasets,
        "env_vars": {k: os.environ.get(k) for k in ("JEPAWM_DSET", "JEPAWM_LOGS", "JEPAWM_CKPT", "JEPAWM_OSSCKPT", "JEPAWM_HOME", "MUJOCO_GL", "PYOPENGL_PLATFORM", "LD_LIBRARY_PATH", "OMP_NUM_THREADS", "PYTHONPATH")},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prov, indent=2))
    missing = [k for k, v in checkpoints.items() if not v["exists"]]
    print(json.dumps({"written": str(out), "missing_checkpoints": missing, "repo_head": prov["repo"]["head"], "dirty": bool(prov["repo"]["status_porcelain"])}))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

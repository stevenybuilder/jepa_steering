"""Populate a receiving host's released checkpoint and audited DINO cache only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import urllib.request

from huggingface_hub import hf_hub_download
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from offline_study.model_loader import DINO_SOURCE_SHA256, DINO_WEIGHT_SHA256
from offline_study.protocol import sha256


def main(root):
    metadata = json.loads((root / "STAGING.json").read_text())
    hub = Path(torch.hub.get_dir())
    hub.mkdir(parents=True, exist_ok=True)
    directory = hub / "facebookresearch_dinov2_main"
    if not directory.exists():
        with tempfile.TemporaryDirectory(prefix="dino-source-", dir=hub) as temporary:
            archive_path = Path(temporary) / "source.tgz"
            urllib.request.urlretrieve("https://codeload.github.com/facebookresearch/dinov2/tar.gz/refs/heads/main", archive_path)
            with tarfile.open(archive_path) as archive:
                archive.extractall(temporary, filter="data")
            os.rename(Path(temporary) / "dinov2-main", directory)
    digest = hashlib.sha256()
    files = sorted(directory.rglob("*.py"))
    for path in files:
        digest.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes())
    if len(files) != 157 or digest.hexdigest() != DINO_SOURCE_SHA256:
        raise ValueError("DINO source differs from audited cache; no model calls permitted")
    weights = hub / "checkpoints/dinov2_vits14_pretrain.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        temporary = weights.with_suffix(".part")
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth", temporary)
        if sha256(temporary) != DINO_WEIGHT_SHA256:
            raise ValueError("DINO weight download mismatch")
        os.replace(temporary, weights)
    if sha256(weights) != DINO_WEIGHT_SHA256:
        raise ValueError("DINO weight cache mismatch")
    # MetaWorld first: its receiving engineering need not await navigation weights.
    for task, digest in metadata["checkpoints"].items():
        name = f"jepa_wm_{task}.pth.tar"
        path = root / "checkpoints" / name
        if not path.exists():
            hf_hub_download(metadata["checkpoint_repository"], name,
                            revision=metadata["checkpoint_revision"], local_dir=root / "checkpoints")
        if sha256(path) != digest:
            raise ValueError("Released checkpoint mismatch: " + name)
        with (root / f"ASSET_READY_{task}.json").open("x") as output:
            json.dump({"checkpoint_sha256": digest, "dino_weights_sha256": DINO_WEIGHT_SHA256,
                       "dino_source_sha256": DINO_SOURCE_SHA256, "model_calls": 0,
                       "scientific_launch_ready": False}, output, indent=2)
    result = {"checkpoints": metadata["checkpoints"], "dino_source_sha256": DINO_SOURCE_SHA256,
              "dino_weights_sha256": DINO_WEIGHT_SHA256, "model_calls": 0,
              "scientific_launch_ready": False}
    with (root / "ASSETS_READY.json").open("x") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())

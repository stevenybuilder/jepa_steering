"""Remove only accidental Torch/CUDA overlays from our private receiving venv."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
prefix = root / "python"
if Path(sys.prefix).resolve() != prefix:
    raise ValueError("Must run inside the campaign's private venv")
names = []
for dist in importlib.metadata.distributions():
    name = dist.metadata["Name"]
    if (name in {"torch", "torchvision", "triton"} or name.startswith("nvidia-")) and Path(dist.locate_file("")).resolve().is_relative_to(prefix):
        names.append(name)
subprocess.run(["uv", "pip", "uninstall", "--python", sys.executable, *sorted(set(names))], check=True)
result = subprocess.check_output([sys.executable, "-c", "import torch,torchvision;assert torch.__version__=='2.7.1+cu128';assert torchvision.__version__=='0.22.1+cu128';print(torch.__version__,torchvision.__version__,torch.__file__)"], text=True)
with (root / "TORCH_RESTORED.json").open("x") as output:
    json.dump({"removed_private_overlays": names, "verified": result.strip(), "scientific_calls": 0}, output, indent=2)
print(result)

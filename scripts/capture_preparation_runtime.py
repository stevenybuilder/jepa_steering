"""Record reusable preparation runtime metadata without credentials or models."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'artifacts/offline_study/fresh-simulator-banks-20260912-v1/parallel-preparation/runtime.json'
if TARGET.exists():
    raise SystemExit('Existing runtime receipt; refusing overwrite')
data = {
    'python': sys.version, 'platform': platform.platform(),
    'packages': sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()),
    'base_image': 'pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime',
    'vendor_commit': subprocess.check_output(['git', '-C', str(ROOT / 'vendor/jepa-wms'), 'rev-parse', 'HEAD'], text=True).strip(),
    'd4rl_commit': subprocess.check_output(['git', '-C', '/workspace/preparation-runtime/D4RL', 'rev-parse', 'HEAD'], text=True).strip(),
    'ready_markers': [p.name for p in Path('/workspace/preparation-runtime').glob('*_READY')],
    'scientific_receiving_parity': False, 'learned_model_calls': 0,
}
TARGET.write_text(json.dumps(data, indent=2) + '\n')
print(TARGET)

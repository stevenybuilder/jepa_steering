"""Launch only today's audited four-task bank, never canonical/base-1 inputs.

Example after freeze and infrastructure smoke (one process per physical GPU):
  CUDA_VISIBLE_DEVICES=0 python scripts/run_fresh_confirmation.py worker \
    --project . --freeze fresh-freeze --task reach --worker-id 0 --workers 6 \
    --output fresh-results

For 24 GPUs use six workers per task, IDs 0..5 independently within each task.
Each worker performs receiving engineering first, then all eight paired arms for
each assigned scenario. This command rents nothing and never refits an edit.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from offline_study.fresh_confirmation import main

if __name__ == '__main__':
    main()

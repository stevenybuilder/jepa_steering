"""Suspend only the owned queue parent; let its current finite shard finish.

No model child is interrupted. The old panel remains a separately labelled partial
development run. Moving to a new GPU requires the new same-device native freeze.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/workspace/jepa-runtime')
PANEL = ROOT / 'droid-coupling-behavior-20260908-v2'
OUTPUT = ROOT / 'droid-device-handoff-20260908-v1'
QUEUE = 12066


def main():
    sys.path.insert(0, str(ROOT / 'droid-coupling-code-20260908-v2/src'))
    from offline_study.droid_native import verified_report
    from offline_study.protocol import sha256, write_json
    command = Path(f'/proc/{QUEUE}/cmdline').read_bytes().split(b'\0')
    if str(ROOT / 'droid-coupling-code-20260908-v2/run_droid_coupling_handoff.py').encode() not in command:
        raise ValueError('Owned queue identity changed')
    if subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() != 'GPU-5bbfd503-dbb3-c9ad-7b41-2a4f985b18fc':
        raise ValueError('Wrong owned GPU')
    children = Path(f'/proc/{QUEUE}/task/{QUEUE}/children').read_text().split()
    if len(children) != 1:
        raise ValueError('Need exactly one known scientific child')
    child = int(children[0])
    args = Path(f'/proc/{child}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    if 'offline_study.droid_coupling_behavior' not in args or 'run' not in args:
        raise ValueError('Do not pause during fitting/engineering or unknown work')
    target = Path(args[args.index('--output')+1])
    if not target.is_relative_to(PANEL / 'conditions'):
        raise ValueError('Unexpected scientific child output')
    OUTPUT.mkdir(exist_ok=False)
    write_json(OUTPUT / 'PLAN.json', {'queue_pid': QUEUE, 'child_pid': child, 'child_command': args,
        'reason': 'prospective four-device speed/cost reallocation, no outcome selection',
        'new_behavior_freeze_sha256': '369d632792b6b482f5c63bbcd599dd7576cf949001ac005baa1cb5f600681862',
        'old_panel_not_pooled_into_new_device_comparison': True, 'original_outputs_preserved': True,
        'pointmaze_resume_checkpoint_sha256': '35f1f4ba625d0dccf867d43303ee2c23e43daa5b72bb724e857c7e687d60c150',
        'pointmaze_resume_required_on_replacement_after_droid': True})
    os.kill(QUEUE, signal.SIGSTOP)
    deadline = time.monotonic() + 1800
    try:
        while not (target / 'DONE.json').exists():
            if (target / 'FAILED.json').exists() or time.monotonic() > deadline:
                raise ValueError('Current shard failed or did not finish; retain old worker')
            time.sleep(5)
        report, digest = verified_report(target)
        if report['status'] != 'droid_coupling_scientific_shard_complete' or report['episodes'] != 8:
            raise ValueError('Incomplete current scientific stream')
        for name, wanted in report['files_sha256'].items():
            if sha256(target / name) != wanted:
                raise ValueError('Completed stream evidence differs')
        while subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader']).strip():
            if time.monotonic() > deadline:
                raise ValueError('GPU remains occupied')
            time.sleep(2)
        write_json(OUTPUT / 'READY.json', {'status': 'owned_queue_suspended_after_complete_scientific_shard',
            'completed_output': str(target), 'report_sha256': digest, 'gpu_empty': True,
            'queue_parent_suspended_not_failed': True, 'new_worker_not_started': True,
            'requires_durable_archive_before_provider_stop': True})
        print(json.dumps({'status': 'droid_safe_shard_boundary', 'output': str(target)}), flush=True)
    except Exception:
        os.kill(QUEUE, signal.SIGCONT)
        raise


if __name__ == '__main__':
    main()

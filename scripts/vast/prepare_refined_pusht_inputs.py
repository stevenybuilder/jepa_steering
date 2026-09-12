"""CPU-only recovery of pinned fitting inputs; never logs authentication URLs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from offline_study.refined_fit_inputs import PUSHT_ARCHIVE_BYTES, PUSHT_ARCHIVE_SHA, extract, verify
from offline_study.protocol import sha256, write_json


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / 'pusht_noise.zip'
    if not archive.exists():
        if shutil.disk_usage(args.output).free < PUSHT_ARCHIVE_BYTES + 9 * 1024**3:
            raise ValueError('Require input bytes plus extraction and eight-GiB reserve')
        token = None
        env = PROJECT / '.env'
        if env.exists():
            for line in env.read_text().splitlines():
                key, _, value = line.partition('=')
                if key.strip() in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN'):
                    token = value.strip().strip('\"').strip("'")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = Path(hf_hub_download('facebook/jepa-wms', 'pusht/pusht_noise.zip',
                repo_type='dataset', revision='6116f042ae7ae4c8e3f1fd2f194f432615664182',
                token=token, local_dir=args.output / 'official_hf'))
            if downloaded.stat().st_size != PUSHT_ARCHIVE_BYTES or sha256(downloaded) != PUSHT_ARCHIVE_SHA:
                raise ValueError('Pinned archive checksum differs')
            if archive.exists():
                raise ValueError('Another process created the destination; no overwrite')
            downloaded.rename(archive)
        except Exception as error:
            print(json.dumps({'error_type': type(error).__name__, 'partial_retained': True}), flush=True)
            raise RuntimeError('Input download incomplete; authentication URL suppressed') from None
    if archive.stat().st_size != PUSHT_ARCHIVE_BYTES or sha256(archive) != PUSHT_ARCHIVE_SHA:
        raise ValueError('Existing archive differs')
    destination = args.output / 'fit-data'
    if not destination.exists():
        extract(archive, args.cohort, destination)
    proof = verify(args.cohort, destination)
    write_json(args.output / 'READY.json', {'fit_inputs': proof, 'archive_sha256': PUSHT_ARCHIVE_SHA,
        'gpu_calls': 0, 'behavioral_launch_ready': False})
    print(json.dumps({'status': 'exact_fit_inputs_ready', 'gpu_calls': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())

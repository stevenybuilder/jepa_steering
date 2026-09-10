"""CPU-only native eligibility then unchanged raw-loader audit; no GPU fitting."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/workspace/jepa-runtime')
CODE=ROOT/'droid-fit-audit-code-20260908-v2'
INPUTS=ROOT/'droid-fit-native-eligible-20260908-v1'
AUDIT=ROOT/'droid-fit-audit-20260908-v2'


def main():
    subprocess.run([sys.executable,'-u','-m','offline_study.droid_fit_eligible','run',
        '--freeze',str(CODE/'eligibility-freeze'),'--metadata',str(ROOT/'droid-fit-input-evidence-20260908-v1/metadata'),
        '--inputs',str(ROOT/'droid-fit-download-20260908-v1'),'--output',str(INPUTS)],check=True)
    subprocess.run([sys.executable,'-u','-m','offline_study.droid_fit_audit',
        '--inputs',str(INPUTS),'--assets',str(ROOT/'droid-assets-20260907-v1'),
        '--manifest',str(CODE/'configs/droid_assets.json'),'--vendor','/workspace/jepa_steering/vendor/jepa-wms',
        '--output',str(AUDIT)],check=True)
    print(json.dumps({'status':'native_droid_input_preparation_finished','gpu_jobs_launched':False}),flush=True)


if __name__=='__main__':main()

"""Only the registered missing native-rank reconstruction and fixed-response fit."""
import argparse
from pathlib import Path
from types import SimpleNamespace

from offline_study.tasks.navigation.navigation_rank_reconstruction import run as reconstruct
from offline_study.tasks.metaworld.refined_task_fit import run as fit


def main(a):
    root = a.root
    common = dict(cohort=root / 'cohort/cohort.json', data_root=root / 'data',
        vendor=Path('/workspace/metaworld-components-20260911-v1/code/vendor/jepa-wms'),
        checkpoint=root / 'checkpoint.pth.tar')
    reconstruct(SimpleNamespace(**common, original=root / 'original', output=root / 'rank-source-v1'))
    fit(SimpleNamespace(**common, fit=root / 'rank-source-v1', output=root / 'fit-v1', device='cuda:0'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args())

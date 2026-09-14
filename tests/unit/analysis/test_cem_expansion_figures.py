"""Plot contracts use synthetic tables; these are not scientific results."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"scripts"))
spec = importlib.util.spec_from_file_location("cem_expansion_figures", ROOT/"scripts/build_cem_expansion_figures.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    iterations, prefixes = [], []
    for task in m.TASKS:
        for episode in range(32):
            for ai, arm in enumerate(m.ARMS):
                key = dict(task=task, episode=episode, arm=arm, cohort="initial8" if episode < 4 else "extension56")
                prefixes.append(key | {"selected_prefix_delta_rms": (episode+1)*(ai+1)/100})
                for iteration in range(15):
                    iterations.append(key | dict(iteration=iteration, proposal_mean_delta_rms=iteration*(episode+1)/1000,
                                                 proposal_entropy_delta_nats=iteration*(episode-10)/100))
    iterations, prefixes = pd.DataFrame(iterations), pd.DataFrame(prefixes)
    summaries = []
    block = iterations[iterations.cohort == "extension56"]
    for (task, arm, iteration), group in block.groupby(["task", "arm", "iteration"]):
        for metric in ("proposal_mean_delta_rms", "proposal_entropy_delta_nats"):
            values = group[metric].to_numpy()
            summaries.append(dict(task=task, arm=arm, iteration=iteration, population="extension56", metric=metric,
                                  n=28, mean=values.mean(), scenario_se=values.std(ddof=1)/np.sqrt(28)))
    return iterations, prefixes, pd.DataFrame(summaries)


class ExpansionFigureTests(unittest.TestCase):
    def test_full_registry_and_signed_entropy(self):
        m.validate_frames(*fixture())

    def test_missing_case_not_silently_averaged(self):
        iterations, prefixes, summary = fixture()
        with self.assertRaisesRegex(ValueError, "complete64"):
            m.validate_frames(iterations.iloc[:-1], prefixes, summary)

    def test_duplicated_candidate_context(self):
        iterations, prefixes, summary = fixture()
        with self.assertRaisesRegex(ValueError, "complete64"):
            m.validate_frames(iterations, pd.concat([prefixes, prefixes.iloc[:1]]), summary)

    def test_original_pilot_not_relabelled_as_extension(self):
        iterations, prefixes, summary = fixture()
        iterations.loc[iterations.episode == 0, "cohort"] = "extension56"
        with self.assertRaisesRegex(ValueError, "cohorts"):
            m.validate_frames(iterations, prefixes, summary)

    def test_sd_cannot_be_labelled_as_se(self):
        iterations, prefixes, summary = fixture()
        summary.scenario_se *= np.sqrt(28)
        with self.assertRaisesRegex(ValueError, "mean/SE"):
            m.validate_frames(iterations, prefixes, summary)

    def test_missing_iteration_cell_rejected(self):
        iterations, prefixes, summary = fixture()
        with self.assertRaisesRegex(ValueError, "curve cell"):
            m.validate_frames(iterations, prefixes, summary.iloc[:-1])

    def test_no_nonfinite_or_negative_prefix_rms(self):
        for value in (float("nan"), -1):
            iterations, prefixes, summary = fixture()
            prefixes.loc[0, "selected_prefix_delta_rms"] = value
            with self.assertRaisesRegex(ValueError, "prefix RMS"):
                m.validate_frames(iterations, prefixes, summary)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("story_figures",ROOT/"scripts/build_story_figures.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(root):
    data = root/"paper/data";data.mkdir(parents=True)
    rows = [dict(task=t,episode=e,arm=a,iteration=i,proposal_mean_delta_rms=(e+1)*i/100,
                 same_candidate_actions_verified=True if i==0 else None,
                 shared_population_elite_overlap_count=10 if i==0 else None)
            for t in m.TASKS for e in range(4) for a in m.ARMS for i in range(15)]
    cases = pd.DataFrame(rows)
    summary = cases.groupby(["task","arm","iteration"]).proposal_mean_delta_rms.mean().reset_index(name="mean")
    summary["metric"] = m.METRIC;summary["n"] = 4
    elites = pd.DataFrame([dict(task=t,arm=a,metric="shared_population_elite_overlap_count",n=4,mean=10)
                           for t in m.TASKS for a in m.ARMS])
    tables = {"iteration_cases":cases,"iteration_summary":summary,"shared_iteration0_summary":elites,
              "selected_prefix_cases":pd.DataFrame({"unused":[0]}),"selected_prefix_summary":pd.DataFrame({"unused":[0]})}
    outputs = {}
    for name, frame in tables.items():
        path = data/f"cem_steering_{name}.csv";frame.to_csv(path,index=False)
        outputs[str(path.relative_to(root))] = dict(sha256=m.sha(path),rows=len(frame))
    receipt = dict(status="complete_fixed8_instrumentation_only",scenarios=8,n_per_task=4,outputs=outputs,
                   sources=[dict(task=t,episode=e) for t in m.TASKS for e in range(4)])
    (data/"cem_steering_summary.json").write_text(json.dumps(receipt))
    return cases


class StoryFigureTests(unittest.TestCase):
    def test_all_individual_lines_and_shared_unclipped_scale(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            cases, provenance=m.load_source(root)
            fig=m.make_figure(cases)
            lines=[line for ax in fig.axes for line in ax.lines if (line.get_gid() or "").startswith("individual-")]
            means=[line for ax in fig.axes for line in ax.lines if (line.get_gid() or "").startswith("mean-")]
            self.assertEqual(len(lines),16);self.assertEqual(len(means),4)
            self.assertEqual(fig.axes[0].get_ylim(),fig.axes[1].get_ylim())
            self.assertGreater(fig.axes[0].get_ylim()[1],provenance["maximum_individual_value"])
            for line in lines:
                self.assertEqual(len(line.get_ydata()),15)
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            footer=next(text for text in fig.texts if text.get_text().startswith("Thin lines:"))
            for ax in fig.axes:
                self.assertFalse(ax.xaxis.label.get_window_extent(renderer).overlaps(footer.get_window_extent(renderer)))
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_all_hashes_not_only_plotted_table_verified(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            (root/"paper/data/cem_steering_selected_prefix_summary.csv").write_text("tampered")
            with self.assertRaisesRegex(ValueError,"hash changed"):m.load_source(root)

    def test_partial_or_duplicate_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            path=root/"paper/data/cem_steering_summary.json"
            receipt=json.loads(path.read_text());receipt["sources"].pop();path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,"missing or duplicate"):m.load_source(root)

    def test_false_initial_elite_claim_fails_even_with_updated_table_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            table=root/"paper/data/cem_steering_iteration_cases.csv"
            frame=pd.read_csv(table);frame.loc[0,"shared_population_elite_overlap_count"]=9;frame.to_csv(table,index=False)
            path=root/"paper/data/cem_steering_summary.json";receipt=json.loads(path.read_text())
            receipt["outputs"][str(table.relative_to(root))]["sha256"]=m.sha(table);path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,"ten shared"):m.load_source(root)

    def test_export_all_formats_and_embedded_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            provenance=m.build(root)
            for extension in ("png","svg","pdf"):
                self.assertGreater((root/f"docs/figures/steering_search_story.{extension}").stat().st_size,1000)
            self.assertIn(provenance["source_receipt_sha256"],(root/"docs/figures/steering_search_story.svg").read_text())


if __name__ == "__main__":
    unittest.main()

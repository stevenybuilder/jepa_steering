"""Presentation guards: complete source, original intervals, no cropped uncertainty."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("precision_story", ROOT / "scripts/build_precision_story.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PrecisionStoryTests(unittest.TestCase):
    def copy_inputs(self, folder):
        root = Path(folder)
        (root / "paper/data").mkdir(parents=True)
        receipt = ROOT / "paper/data/pathway_geometry.json"
        shutil.copy(receipt, root / "paper/data/pathway_geometry.json")
        for name in json.loads(receipt.read_text())["outputs"]:
            shutil.copy(ROOT / name, root / name)
        return root

    def test_real_source_complete_and_unclipped(self):
        values, _ = m.load_source()
        self.assertEqual(len(values), 10)
        fig = m.make_figure(values)
        ax = fig.axes[0]
        self.assertEqual(sum((line.get_gid() or "").startswith("condition-") for line in ax.lines), 10)
        self.assertLess(ax.get_xlim()[0], min(v["low"] for v in values))
        self.assertGreater(ax.get_xlim()[1], max(v["high"] for v in values))
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            path = root / m.TABLE
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                m.load_source(root)

    def test_partial_source_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            path = root / m.TABLE
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
            receipt_path = root / "paper/data/pathway_geometry.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["outputs"][m.TABLE] = dict(sha256=m.sha(path), rows=len(lines)-2)
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "ten task/precision"):
                m.load_source(root)

    def test_export_embeds_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            provenance = m.build(root)
            for extension in ("png", "svg", "pdf"):
                self.assertGreater((root / f"docs/figures/precision_reconstruction_story.{extension}").stat().st_size, 1000)
            svg = (root / "docs/figures/precision_reconstruction_story.svg").read_text()
            self.assertIn(provenance["source_receipt_sha256"], svg)


if __name__ == "__main__":
    unittest.main()

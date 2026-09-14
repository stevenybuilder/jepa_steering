"""Check display/provenance consistency, not simulator reproducibility."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('public_results', ROOT/'scripts/check_public_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicResultsTests(unittest.TestCase):
    def test_published_aggregates(self):
        module.check()

    def copy_inputs(self, directory):
        path = Path(directory)
        shutil.copy(ROOT/'README.md', path/'README.md')
        shutil.copytree(ROOT/'reports/fresh-confirmation', path/'reports/fresh-confirmation')
        (path/'paper/data').mkdir(parents=True)
        shutil.copy(ROOT/'paper/data/benchmark_comparison_sources.json', path/'paper/data/benchmark_comparison_sources.json')
        return path

    def test_wrong_readme_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            readme = root/'README.md'
            readme.write_text(readme.read_text().replace('| Protected | Unsteered | 54.17', '| Protected | Unsteered | 44.79'))
            with self.assertRaises(AssertionError):
                module.check(root)

    def test_wrong_headline_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            path = root/'README.md'
            path.write_text(path.read_text().replace('| Published | DINO-WM | 44.80',
                                                      '| Published | DINO-WM | 99.99'))
            with self.assertRaisesRegex(AssertionError, 'Table value'):
                module.check(root)

    def test_development_value_cannot_fill_protected_blank(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            path = root/'README.md'
            path.write_text(path.read_text().replace('| Protected | Unsteered | 54.17 | 29.17 | —',
                                                     '| Protected | Unsteered | 54.17 | 29.17 | 59.38'))
            with self.assertRaisesRegex(AssertionError, 'Table value'):
                module.check(root)

    def test_stage_cannot_be_silently_relabelled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            path = root/'README.md'
            path.write_text(path.read_text().replace('| Development | Unsteered |', '| Protected | Unsteered |'))
            with self.assertRaisesRegex(AssertionError, 'stage/label'):
                module.check(root)

    def test_wrong_paired_delta_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            path = root/'README.md'
            path.write_text(path.read_text().replace('0.00 / −2.08 / +2.08 / +2.08', '0.00 / −2.08 / +3.00 / +2.08'))
            with self.assertRaisesRegex(AssertionError, 'paired delta'):
                module.check(root)

    def test_best_edit_selection_disclosure_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            path = root/'README.md'
            path.write_text(path.read_text().replace('post hoc', 'fixed-method'))
            with self.assertRaisesRegex(AssertionError, 'selection disclosure'):
                module.check(root)

    def test_changed_report_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.copy_inputs(folder)
            report = root/'reports/fresh-confirmation/report.json'
            report.write_bytes(report.read_bytes()+b'\n')
            with self.assertRaisesRegex(AssertionError, 'report changed'):
                module.check(root)


if __name__ == '__main__':
    unittest.main()

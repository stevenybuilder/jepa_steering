"""Portable source checks; synthetic failures need neither TeX nor model assets."""
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from scripts.check_manuscript import (
    ARMS, HISTORICAL_TASKS, ROOT, TASKS, CheckError, check, check_citations,
    check_figures, check_tables, rounded, without_comments,
)


def table(tasks, marker=False):
    labels = dict(zip(ARMS, ("Native", "Refined", "Random subspace", "Coupling",
                           "Random directions", "Joint", "Visual", "Action")))
    headers = {"reach": "Reach", "reach-wall": "R.-Wall", "pusht": "Push-T",
               "pointmaze": "P.Maze", "wall": "Wall", "droid": "DROID"}
    rows = ["Arm & " + " & ".join(headers[t] for t in tasks)]
    for arm in ARMS:
        cells = ["12.50" + (r"\(^{*}\)" if marker and arm == "joint" and t == "reach" else "")
                 for t in tasks]
        rows.append(labels[arm] + " & " + " & ".join(cells))
    return "\\begin{tabular}{lrrrrrr}\n\\toprule\n" + "\\\\\n".join(rows) + "\\\\\n\\bottomrule\n\\end{tabular}\n"


class ManuscriptTests(unittest.TestCase):
    def setUp(self):
        self.fresh = {(a, t): Decimal("12.50") for a in ARMS for t in TASKS}
        self.old = {(a, t): Decimal("12.50") for a in ARMS for t in HISTORICAL_TASKS}
        self.tex = table(TASKS) + table(HISTORICAL_TASKS, marker=True)

    def test_complete_public_manuscript(self):
        result = check(ROOT)
        self.assertEqual((result["fresh_cells"], result["historical_cells"]), (32, 48))

    def test_all_cells_and_component_marker(self):
        self.assertEqual(check_tables(self.tex, self.fresh, self.old), 80)

    def test_fresh_corruption_reports_exact_cell(self):
        with self.assertRaisesRegex(CheckError, "fresh table / Native / reach: found 13"):
            check_tables(self.tex.replace("12.50", "13.00", 1), self.fresh, self.old)

    def test_historical_corruption(self):
        tex = table(TASKS) + table(HISTORICAL_TASKS).replace("12.50", "13.00", 1)
        with self.assertRaisesRegex(CheckError, "historical table / Native / reach"):
            check_tables(tex, self.fresh, self.old)

    def test_missing_table(self):
        with self.assertRaisesRegex(CheckError, "Missing complete historical"):
            check_tables(table(TASKS), self.fresh, self.old)

    def test_duplicate_table(self):
        with self.assertRaisesRegex(CheckError, "Duplicate complete fresh"):
            check_tables(self.tex + table(TASKS), self.fresh, self.old)

    def test_duplicate_or_unknown_arm(self):
        for replacement in ("Native", "Invented"):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(CheckError, "arm label"):
                check_tables(self.tex.replace("Refined &", replacement+" &", 1), self.fresh, self.old)

    def test_bad_task_header(self):
        with self.assertRaisesRegex(CheckError, "task headers"):
            check_tables(self.tex.replace("P.Maze", "Mars", 1), self.fresh, self.old)

    def test_rounding_is_half_up(self):
        self.assertEqual(rounded("84.375"), Decimal("84.38"))
        with self.assertRaises(CheckError):
            rounded("nan")

    def test_macro_graphics_and_missing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "figure.pdf").touch()
            tex = r"\newcommand{\figroot}{.}\includegraphics[width=\textwidth]{\figroot/figure.pdf}"
            self.assertEqual(check_figures(tex, base), 1)
            with self.assertRaisesRegex(CheckError, "Missing figure"):
                check_figures(tex.replace("figure.pdf", "missing.pdf"), base)

    def test_unresolved_graphics_macro(self):
        with self.assertRaisesRegex(CheckError, "unresolved path macro"):
            check_figures(r"\includegraphics{\unknown/f.pdf}", ROOT)

    def test_optional_citation_notes_and_undefined_key(self):
        tex = r"\citep[see][p. 2]{a,b} \citet*{a}"
        bib = "@article{a, title={A}}\n@misc{b, title={B}}"
        self.assertEqual(check_citations(tex, bib), 2)
        with self.assertRaisesRegex(CheckError, "Undefined.*b"):
            check_citations(tex, "@article{a, title={A}}")

    def test_comments_do_not_create_citations(self):
        tex = without_comments("\\citep{a} % \\citep{missing}\nvalue 50\\%")
        self.assertEqual(check_citations(tex, "@misc{a,title={A}}"), 1)
        self.assertIn(r"50\%", tex)


if __name__ == "__main__":
    unittest.main()

"""The focused manuscript must not drift from its immutable result source."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_lcfm_manuscript import ROOT, CheckError, check


class FocusedManuscriptTest(unittest.TestCase):
    def test_complete_paper(self):
        self.assertEqual(check()['replication_macros'], 12)

    def test_rejects_changed_result(self):
        p = ROOT/'paper/lcfm/replication_numbers.tex'
        bad = p.read_text().replace('{42}', '{43}')
        with self.assertRaisesRegex(CheckError, 'numbers differ'):
            check(numbers=bad)

    def test_rejects_stale_section_reference(self):
        with self.assertRaisesRegex(CheckError, 'Undefined section'):
            check(checklist=r'\ref{old_broad_paper_section}')

    def test_rejects_missing_figure(self):
        tex = (ROOT/'paper/lcfm/main.tex').read_text().replace('lcfm_context_lifetime.pdf', 'missing_figure.pdf')
        with self.assertRaisesRegex(CheckError, 'Missing figure'):
            check(tex=tex)


if __name__ == '__main__':
    unittest.main()

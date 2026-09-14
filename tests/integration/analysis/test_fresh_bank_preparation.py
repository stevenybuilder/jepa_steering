import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[3] / 'scripts/run_fresh_bank_preparation.py'
spec = importlib.util.spec_from_file_location('bank_runner', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BankVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        original = runner.BANK
        self.addCleanup(setattr, runner, 'BANK', original)
        runner.BANK = Path(self.tmp.name)
        self.target = runner.BANK / 'reach/rank-00'
        self.target.mkdir(parents=True)
        (runner.BANK / 'protocol.json').write_text('{}')
        rows = []
        for i in range(13):
            tensor = self.target / f'{i}.pt'
            tensor.write_bytes(str(i).encode())
            rows.append({'tensor_file': tensor.name, 'tensor_sha256': runner.digest(tensor)})
        (self.target / 'records.json').write_text(json.dumps(rows))
        report = {'task': 'reach', 'rank': 0, 'scientific_candidates': 12,
            'excluded_engineering': 1, 'learned_policy_calls': 0,
            'protocol_sha256': runner.digest(runner.BANK / 'protocol.json'),
            'records_sha256': runner.digest(self.target / 'records.json')}
        (self.target / 'report.json').write_text(json.dumps(report))

    def test_valid_shard_is_reused_without_execution(self):
        self.assertEqual(runner.run(('reach', 0))['scientific_candidates'], 12)

    def test_corrupt_tensor_is_not_reused(self):
        (self.target / '0.pt').write_bytes(b'changed')
        with self.assertRaises(AssertionError):
            runner.run(('reach', 0))

    def test_partial_shard_is_not_retried(self):
        (self.target / 'report.json').unlink()
        with self.assertRaises(FileNotFoundError):
            runner.run(('reach', 0))


if __name__ == '__main__':
    unittest.main()

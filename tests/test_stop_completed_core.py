"""Stop boundary tests: mocked provider only, no live API or SSH."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/vast'))
import stop_completed_core as stop


class CompletedCoreStopTests(unittest.TestCase):
    def test_missing_drive_proof_blocks_before_provider(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(stop, 'ROOT', Path(directory)), \
                patch.object(sys, 'argv', ['stop', '--execute']), \
                patch.object(stop.subprocess, 'check_output') as provider:
            with self.assertRaises(FileNotFoundError): stop.main()
            provider.assert_not_called()

    def test_read_only_mode_cannot_stop(self):
        with patch.object(stop, 'verify_preservation', return_value={}), \
                patch.object(sys, 'argv', ['stop']), patch.object(stop.subprocess, 'check_output') as provider:
            stop.main(); provider.assert_not_called()

    def test_live_identity_mismatch_blocks(self):
        with patch.object(stop, 'verify_preservation', return_value={}), \
                patch.object(sys, 'argv', ['stop', '--execute']), \
                patch.object(stop.subprocess, 'check_output', return_value=json.dumps([
                    {'id':50233992,'label':'another-owner'}])) as provider:
            with self.assertRaises(ValueError): stop.main()
            self.assertEqual(provider.call_count, 1)

    def test_exact_stop_after_complete_and_empty_checks_no_destroy(self):
        current={'id':50233992,'label':'jepa-fixed-behavior-us-v1','cur_state':'running',
                 'actual_status':'running','num_gpus':8,'geolocation':'United States, US',
                 'public_ipaddr':'98.142.241.142'}
        complete={'complete_shards':40,'published_files':960,'analysis_done':True,'live':[],'failed':[]}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); calls=[]
            def provider(command, **kwargs):
                calls.append(command)
                if command == [stop.CLI,'show','instances','--raw']: return json.dumps([current])
                self.assertEqual(command,[stop.CLI,'stop','instance','50233992'])
                self.assertTrue((root/'STOP_INTENT.json').is_file())
                return 'stopping instance 50233992.'
            with patch.object(stop,'ROOT',root), patch.object(stop,'verify_preservation',return_value={}), \
                    patch.object(sys,'argv',['stop','--execute']), \
                    patch.object(stop.subprocess,'check_output',side_effect=provider), \
                    patch.object(stop,'remote',side_effect=[complete,{'all_gpus_empty':True}]):
                stop.main()
            self.assertEqual(len(calls),2)
            self.assertTrue(json.loads((root/'STOP_INTENT.json').read_text())['source_volume_retained'])
            self.assertTrue((root/'STOP_RESPONSE.json').is_file())


if __name__ == '__main__': unittest.main()

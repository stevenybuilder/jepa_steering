import json
import hashlib
import signal
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/vast'))
import component_fit_behavior_chain as chain


class ChainTests(unittest.TestCase):
    def test_timer_adoption_never_signals_or_restarts_active_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'ops'; root.mkdir(); component = Path(tmp) / 'component'; component.mkdir()
            panel = Path(tmp) / 'panel'; panel.mkdir(); tail = Path(tmp) / 'tail'; tail.mkdir()
            queue_args = ['python', '-u', '/ops/component_extension_queue.py', '--root', str(component), '--deadline', '1000000']
            spec = {'component_root': str(component), 'panel_root': str(panel), 'tail_root': str(tail),
                    'queue_pid': 10, 'queue_args': queue_args, 'boundary_pid': 20, 'boundary_args': ['boundary'],
                    'panel_pid': 30, 'panel_args': ['active-panel'], 'tail_pid': 40, 'tail_args': ['tail'], 'deadline': 1000000}
            (root / 'PLAN.json').write_text(json.dumps(spec)); (panel / 'PANEL_STARTED.json').write_text('{}')
            (panel / 'PANEL_DONE.json').write_text('{}')
            (panel / 'JOB.json').write_text(json.dumps({'timeout_seconds': 21600}))
            lookups = {30: 0}
            def lookup(pid):
                if pid == 30:
                    lookups[30] += 1
                    return {'state': 'R' if lookups[30] <= 2 else 'Z', 'args': ['active-panel']}
                return {10: {'state': 'T', 'args': queue_args}, 20: {'state': 'S', 'args': ['boundary']},
                        40: {'state': 'S', 'args': ['tail']}}[pid]
            with patch.object(chain, 'process', side_effect=lookup), \
                 patch.object(chain, 'direct_children', side_effect=lambda pid: [(30, {})] if pid == 20 else []), \
                 patch.object(chain, 'wait_gpu_release'), patch.object(chain.time, 'time', return_value=1000), \
                 patch.object(chain.os, 'kill') as signals, patch.object(chain, 'validate_job'), \
                 patch.object(chain.subprocess, 'Popen') as launch:
                chain.adopt_panel_timer(root)
            self.assertEqual([c.args for c in signals.call_args_list], [
                (20, signal.SIGSTOP), (20, signal.SIGKILL), (10, signal.SIGCONT)])
            launch.assert_not_called()
            self.assertEqual(json.loads((panel / 'COMPONENTS_RESUMED.json').read_text())['queue_pid'], 10)

    def test_fit_and_panel_share_one_hold_without_an_intervening_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fit = root / 'fit'; panel = root / 'panel'; previous = root / 'previous'
            for p in (fit, panel, previous): p.mkdir()
            (previous / 'report.json').write_text('{}')
            (previous / 'DONE.json').write_text(json.dumps({'report_sha256': hashlib.sha256(b'{}').hexdigest()}))
            args = ['queue']; spec = {'fit_root': str(fit), 'panel_root': str(panel),
                'queue_pid': 10, 'scientific_pid': 20, 'scientific_output': str(previous),
                'scientific_args': ['scientific'], 'deadline': 1000000}
            (root / 'PLAN.json').write_text(json.dumps(spec))
            job = {'task': 'wall', 'gpu_uuid': 'GPU-test', 'command': ['fit'], 'timeout_seconds': 1000}
            (fit / 'JOB.json').write_text(json.dumps(job))
            events = []
            def command(cmd, *unused):
                events.append(cmd[0])
                if cmd == ['fit']:
                    (fit / 'fit-v1').mkdir(); (fit / 'fit-v1/DONE.json').write_text('{}')
                else: (panel / 'PANEL_DONE.json').write_text('{}')
            with patch.object(chain, 'validate_adoption', return_value={'args': args}), \
                 patch.object(chain, 'validate_job'), patch.object(chain, 'wait_gpu_release'), \
                 patch.object(chain.subprocess, 'check_output', return_value='GPU-test'), \
                 patch.object(chain.signal, 'signal'), \
                 patch.object(chain, 'process', side_effect=lambda pid: {'state': 'Z'} if pid == 20 else {'args': args}), \
                 patch.object(chain, 'run_command', side_effect=command), \
                 patch.object(chain, 'finalize_behavior', side_effect=lambda *unused: (events.append('freeze') or {'command': ['panel'], 'timeout_seconds': 1000})), \
                 patch.object(chain.os, 'kill', side_effect=lambda pid, sig: events.append(('signal', pid, sig))):
                chain.main(root)
            self.assertEqual(events, ['fit', 'freeze', 'panel', ('signal', 10, signal.SIGCONT)])
            self.assertEqual(json.loads((root / 'TERMINAL.json').read_text())['status'], 'fit_and_behavior_complete')

    def test_failed_command_terminates_owned_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Mock(pid=51); child.wait.side_effect = [RuntimeError('interrupted'), 0]
            with patch.object(chain.time, 'time', return_value=0), \
                 patch.object(chain.subprocess, 'Popen', return_value=child), patch.object(chain.os, 'killpg') as kill:
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    chain.run_command(['fit'], {}, Path(tmp), Path(tmp) / 'log', 1000)
                kill.assert_called_once_with(51, signal.SIGTERM)

    def test_adoption_requires_exact_held_coordinator_and_unstarted_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fit = root / 'fit'; fit.mkdir()
            output = root / 'results/reach/native/shard-0'
            args = ['python', '-u', '/ops/component_extension_queue.py', '--root', str(root)]
            child = ['python', '-m', 'offline_study.metaworld_component_behavior', '--output', str(output)]
            spec = {'component_root': str(root), 'queue_pid': 10, 'queue_args': args,
                    'scientific_pid': 20, 'scientific_args': child, 'scientific_output': str(output), 'fit_root': str(fit)}
            (fit / 'WAITING_BOUNDARY.json').write_text(json.dumps({'queue_pid': 10,
                'continuing_scientific_pid': 20, 'continuing_output': str(output)}))
            with patch.object(chain, 'process', side_effect=lambda pid: {'state': 'T' if pid == 10 else 'R', 'args': args if pid == 10 else child}):
                chain.validate_adoption(spec)
                (fit / 'FIT_STARTED.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'already started'): chain.validate_adoption(spec)
            with patch.object(chain, 'process', return_value={'state': 'S', 'args': args}):
                with self.assertRaisesRegex(ValueError, 'suspended'): chain.validate_adoption(spec)

    def test_finalize_refuses_unready_or_unverified_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'preparation'):
                chain.finalize_behavior(root, root / 'fit', {}, 100)

    def test_no_new_command_beyond_deadline(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(chain.time, 'time', return_value=100):
            with self.assertRaises(TimeoutError):
                chain.run_command(['false'], {}, Path(tmp), Path(tmp) / 'log', 101)
            self.assertFalse((Path(tmp) / 'log').exists())


if __name__ == '__main__': unittest.main()

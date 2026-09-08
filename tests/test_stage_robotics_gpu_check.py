"""Pure CPU staging helpers; no SSH, provider or GPU work."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/vast"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("stage_robotics_gpu_check", SCRIPTS / "stage_robotics_gpu_check.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


class RoboticsStageTests(unittest.TestCase):
    def test_manifest_binds_bytes_not_only_names(self):
        one = s.make_manifest({"src/offline_study/a.py": b"pass\n"})
        two = s.make_manifest({"src/offline_study/a.py": b"pass \n"})
        self.assertEqual(one["src/offline_study/a.py"]["bytes"], 5)
        self.assertNotEqual(one, two)

    def test_flat_science_hash_matches_receiving_contract(self):
        import hashlib
        expected = hashlib.sha256(b"a.py\0firstb.py\0second").hexdigest()
        members = {"src/offline_study/b.py": b"second", "tests/test.py": b"test",
                   "src/offline_study/a.py": b"first"}
        self.assertEqual(s.science_hash(members), expected)
        members["src/offline_study/a.py"] = b"changed"
        self.assertNotEqual(s.science_hash(members), expected)
        with self.assertRaises(ValueError):
            s.science_hash({"tests/test.py": b"test"})

    def test_fixed_instance_and_existing_batch_no_raw_pull(self):
        self.assertEqual(s.ROOT, "/workspace/jepa-runtime/robotics-training-gpu-check-20260908-v1")
        self.assertEqual(s.UUID, "GPU-1541ee72-c8fc-ca75-7fea-470ae793c81e")
        self.assertEqual(len(s.FIXED_INPUTS), 5)
        self.assertTrue(all(path.startswith(s.BATCH + "/") for path in s.FIXED_INPUTS))
        self.assertEqual(s.FIXED_INPUTS[s.BATCH + "/batch.pt"],
                         "2ada0bc241424e26899300b77da849f38c9c056ecab14295c6c99be27526bcc4")

    def test_remote_programs_compile_without_execution(self):
        tree = ast.parse((SCRIPTS / "stage_robotics_gpu_check.py").read_text())
        scripts = [node.value.value for node in ast.walk(tree)
                   if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                   and isinstance(node.value.value, str) and node.value.value.startswith("import ")]
        self.assertGreaterEqual(len(scripts), 5)
        for script in scripts:
            compile(script, "<remote-script-no-execution>", "exec")

    def test_activation_does_not_duplicate_long_scientific_verifier(self):
        calls = [node for node in ast.walk(ast.parse(s.ACTIVATE)) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(call.func, ast.Attribute) and
                             call.func.attr == "verify_predecessor" for call in calls))
        # Full scientific verification remains the frozen waiter's prerequisite.
        self.assertIn("q.predecessor_pending()", s.ACTIVATE)
        self.assertIn("'may_have_live_waiter':child is not None", s.ACTIVATE)

    def test_status_preserves_truncated_spawn_and_uses_known_failure_handle(self):
        tree = ast.parse((SCRIPTS / "stage_robotics_gpu_check.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "status")
        script = next(node.value.value for node in function.body if isinstance(node, ast.Assign)
                      and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "SPAWNED.json").write_text('{"pid":')
            (root / "ACTIVATION_FAILED.json").write_text(json.dumps({"pid": 999999999,
                "argv": ["fixture"], "may_have_live_waiter": True}))
            result = json.loads(subprocess.check_output([sys.executable, "-c", script, str(root)], text=True))
            self.assertIn("SPAWNED.json", result["observed_receipt_errors"])
            self.assertEqual(result["waiter_pid"], 999999999)
            self.assertFalse(result["identity_bound"])
            self.assertFalse(result["waiter_live"])
            self.assertEqual((root / "SPAWNED.json").read_text(), '{"pid":')


if __name__ == "__main__":
    unittest.main()

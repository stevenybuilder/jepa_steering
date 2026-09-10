import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location("monitor",Path(__file__).parents[1]/"scripts/geometry_map/monitor_steering_preparation.py")
monitor=importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def fixture(self,root):
        p=root/"bank"
        p.mkdir()
        (p/"episode-001.pt").write_bytes(b"not a torchpickle; must neverdeserialize")
        (p/"DONE.json").write_bytes(b"native bytes maycontainheldoutcomes andneednotbeparsed")
        (p/"SEALED_DONE.json").write_text(json.dumps({"complete":True,"outputs":[{"episode":1,"path":"episode-001.pt","sha256":monitor.sha(p/"episode-001.pt")}],"native_done_sha256":monitor.sha(p/"DONE.json")}))
        return {"local_directory":"bank","episode_ids":[1],"instance_id":1,"task":"reach_wall"}

    def test_verifies_bytes_without_parsing_native_outcomes_or_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); job=self.fixture(root)
            self.assertTrue(monitor.verify_local(job,root)["complete"])

    def test_corrupt_tensor_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); job=self.fixture(root)
            (root/"bank/episode-001.pt").write_bytes(b"corrupt")
            with self.assertRaises(RuntimeError):monitor.verify_local(job,root)

    def test_incomplete_copy_not_claimed_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); job=self.fixture(root)
            (root/"bank/episode-001.pt").unlink()
            self.assertIsNone(monitor.verify_local(job,root))

    def test_wrong_episode_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); job=self.fixture(root); job["episode_ids"]=[2]
            with self.assertRaises(RuntimeError):monitor.verify_local(job,root)


if __name__=="__main__":unittest.main()

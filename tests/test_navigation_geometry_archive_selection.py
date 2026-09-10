import hashlib
import inspect
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/vast"
sys.path.insert(0, str(SCRIPTS))
from navigation_geometry_archive_selection import select_files
from backup_results_to_google import verify_archive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return digest(path)


class GeometryArchiveSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bindings = {"sources": {}, "cohorts": {}, "scopes": {}, "parity": {}}
        arms = ["native", "zero_dose", "equal_anchor_linear", "cubic",
                "projected_cubic", "reflected_curvature", "matched_random"]
        for label in ("navigation-evaluation-code-v1", "navigation-analysis-code-v1"):
            base = self.root / label
            put(base / "src/offline_study/example.py", {"source": label})
            put(base / "tests/test_example.py", {"source": label})
            h = hashlib.sha256()
            for path in sorted(base.rglob("*.py")):
                h.update(str(path.relative_to(base)).encode() + b"\0" + path.read_bytes())
            src = base / "src/offline_study/example.py"
            harness = hashlib.sha256(src.name.encode() + b"\0" + src.read_bytes()).hexdigest()
            self.bindings["sources"][label] = (harness, h.hexdigest(), 2)
        for task in ("wall", "pointmaze"):
            base = self.root / "navigation-offline-cohorts-20260907-v1" / task
            inputs = put(base / "input_files.json", {"not_copied_dataset": "0" * 64})
            manifest = put(base / "source_manifest.json", {"task": task})
            cohort = put(base / "cohort.json", {"task": task, "fresh_confirmation": False,
                         "source_input_files_sha256": inputs, "source_manifest_sha256": manifest})
            self.bindings["cohorts"][task] = cohort
            put(base / "FROZEN.json", {"cohort_sha256": cohort, "model_runs": 0,
                                      "confirmation_authorized": False, "input_files_sha256": inputs})
        for precision in ("bfloat16", "float32"):
            for task in ("wall", "pointmaze"):
                scope = precision + "/" + task
                fit, raw, aggregate = self.paths(scope)
                common = {"task": task, "category": "action_response_geometry", "precision": precision,
                          "cohort_sha256": self.bindings["cohorts"][task]}
                self.bindings["parity"][precision] = put(fit.parent / "PARITY.json", {"precision": precision})
                receipt = put(fit / "fit_receipt.json", {**common,
                              "status": "author_train_fit_only_complete", "checkpoint_sha256": "checkpoint"})
                protocol = {"status": "frozen", "category": "action_response_geometry",
                            "evaluation_split": "development", "fit_receipt_sha256": receipt,
                            "checkpoint_sha256": "checkpoint", "arms": [{"name": a} for a in arms],
                            "author_validation": {"cohort_sha256": common["cohort_sha256"], "precision": precision}}
                protocol_hash = put(fit / "protocol.json", protocol)
                bank = put(fit / "operator_bank.pt", {"fixture_only": True})
                put(fit / "DONE.json", {"status": "author_protocol_frozen", "protocol_sha256": protocol_hash,
                     "fit_receipt_sha256": receipt, "operator_bank_sha256": bank,
                     "cohort_sha256": common["cohort_sha256"]})
                shared = {**common, "protocol_sha256": protocol_hash, "fit_receipt_sha256": receipt,
                          "protected_outcomes_accessed": False, "fresh_confirmation": False}
                inputs = {}
                for index in range(8):
                    shard = raw / f"shard-{index}"
                    report = {**shared, "status": "author_offline_shard_complete", "checkpoint_sha256": "checkpoint",
                              "shard_index": index, "shard_count": 8, "zero_dose_identity": True,
                              "native_instrumentation_fidelity": True,
                              "source_sha256": self.bindings["sources"]["navigation-evaluation-code-v1"][0]}
                    done = {"status": "author_offline_shard_complete"}
                    for name, value in (("report", report), ("protocol", protocol), ("selection", {}),
                                        ("window_metrics", []), ("mechanism_diagnostics", [])):
                        path = shard / (name + ".json")
                        done[name + "_sha256"] = put(path, value)
                        inputs[str(Path("/workspace/jepa-runtime") / path.relative_to(self.root))] = digest(path)
                    put(shard / "DONE.json", done)
                aggregate_hash = put(aggregate / "report.json", {**shared,
                    "status": "verified_author_corrected_development_complete", "input_sha256": inputs,
                    "rollout_trajectories": 192 if task == "wall" else 200,
                    "prefix_rollouts_per_arm": 4224 if task == "wall" else 24400})
                put(aggregate / "DONE.json", {"status": "verified_author_corrected_development_complete",
                                             "report_sha256": aggregate_hash})
                self.bindings["scopes"][scope] = (protocol_hash, receipt, aggregate_hash)

    def paths(self, scope="float32/wall"):
        return tuple(self.root / (label + "-20260907-v1") / scope / "action_response_geometry"
                     for label in ("navigation-fits", "navigation-comparisons", "navigation-analysis"))

    def select(self):
        return select_files(self.root, _test_bindings=self.bindings)

    def test_exact_complete_scope_and_allowlisted_members(self):
        for name in ("cache/big.pt", "worker.log", "credentials.pem", "progress.json"):
            put(self.paths()[1] / "shard-0" / name, {"excluded": True})
        put(self.root / "navigation-assets-20260907-v1/checkpoint.pt", {})
        put(self.root / "navigation-evaluation-code-v1/src/offline_study/__pycache__/ignored.pyc", {})
        put(self.root / "navigation-evaluation-code-v1/worker.log", {})
        files = self.select()
        self.assertEqual(len(files), 232)  # 228 evidence files + 4 fixture source files
        self.assertEqual(files, sorted(set(files)))
        self.assertEqual(sum("comparisons" in p and p.endswith("DONE.json") for p in files), 32)
        self.assertEqual(sum("analysis-20260907" in p and p.endswith("DONE.json") for p in files), 4)
        self.assertFalse(any(term in p for p in files for term in ("cache", ".log", "credential", "checkpoint", "progress")))

    def test_production_pins_reject_fixture_and_function_is_self_contained(self):
        with self.assertRaisesRegex(ValueError, "source snapshot changed"):
            select_files(self.root)
        namespace = {}
        exec(inspect.getsource(select_files), namespace)
        self.assertEqual(namespace["select_files"](self.root, _test_bindings=self.bindings), self.select())

    def test_missing_or_extra_shard_fails(self):
        raw = self.paths()[1]
        (raw / "shard-8").mkdir()
        with self.assertRaisesRegex(ValueError, "extra geometry shard"):
            self.select()
        (raw / "shard-8").rmdir()
        (raw / "shard-7").rename(raw / "not-a-shard")
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            self.select()

    def test_missing_done_and_failure_marker_fail(self):
        raw = self.paths()[1]
        done = raw / "shard-0/DONE.json"
        previous = done.read_bytes()
        done.unlink()
        with self.assertRaisesRegex(ValueError, "Missing member"):
            self.select()
        done.write_bytes(previous)
        put(raw / "shard-0/FAILED.json", {})
        with self.assertRaisesRegex(ValueError, "Failed evidence"):
            self.select()

    def test_every_done_bound_raw_file_is_verified(self):
        raw = self.paths()[1]
        for name in ("report", "protocol", "selection", "window_metrics", "mechanism_diagnostics"):
            with self.subTest(name=name):
                path = raw / "shard-0" / (name + ".json")
                before = path.read_bytes()
                path.write_bytes(before + b" ")
                with self.assertRaisesRegex(ValueError, "Shard checksum"):
                    self.select()
                path.write_bytes(before)

    def test_fit_bank_and_cohort_inputs_are_bound(self):
        for path in (self.paths()[0] / "operator_bank.pt",
                     self.root / "navigation-offline-cohorts-20260907-v1/wall/input_files.json"):
            with self.subTest(path=path):
                before = path.read_bytes()
                path.write_bytes(before + b" ")
                with self.assertRaisesRegex(ValueError, "Binding mismatch"):
                    self.select()
                path.write_bytes(before)

    def test_rehashed_shard_with_wrong_source_or_identity_still_fails(self):
        shard = self.paths()[1] / "shard-0"
        report = json.loads((shard / "report.json").read_text())
        for key, value in (("source_sha256", "wrong"), ("shard_index", 7), ("precision", "bfloat16"),
                           ("zero_dose_identity", False), ("protected_outcomes_accessed", True)):
            with self.subTest(key=key):
                changed = dict(report, **{key: value})
                done = json.loads((shard / "DONE.json").read_text())
                done["report_sha256"] = put(shard / "report.json", changed)
                put(shard / "DONE.json", done)
                with self.assertRaisesRegex(ValueError, "Binding mismatch: shard"):
                    self.select()

    def test_aggregate_input_map_must_be_exact_even_with_new_pin(self):
        aggregate = self.paths()[2]
        report = json.loads((aggregate / "report.json").read_text())
        report["input_sha256"].pop(next(iter(report["input_sha256"])))
        new_hash = put(aggregate / "report.json", report)
        self.bindings["scopes"]["float32/wall"] = (*self.bindings["scopes"]["float32/wall"][:2], new_hash)
        put(aggregate / "DONE.json", {"status": report["status"], "report_sha256": new_hash})
        with self.assertRaisesRegex(ValueError, "Binding mismatch: aggregate"):
            self.select()

    def test_symlink_file_ancestor_and_source_directory_fail(self):
        shard = self.paths()[1] / "shard-0"
        path = shard / "report.json"
        path.rename(shard / "original.json")
        path.symlink_to(shard / "original.json")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.select()
        path.unlink()
        (shard / "original.json").rename(path)
        moved = shard.with_name("moved")
        shard.rename(moved)
        shard.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.select()
        shard.unlink()
        moved.rename(shard)
        (self.root / "navigation-evaluation-code-v1/linked").symlink_to(shard, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.select()

    def test_unexpected_credential_in_source_fails_closed(self):
        put(self.root / "navigation-analysis-code-v1/credentials.py", {"never_archive": True})
        with self.assertRaisesRegex(ValueError, "credential-like"):
            self.select()

    def test_manifest_and_existing_full_archive_readback_compatible(self):
        files = self.select()
        manifest = {name: {"sha256": digest(self.root / name), "bytes": (self.root / name).stat().st_size}
                    for name in files}
        archive = self.root / "fixture.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name in files:
                stream.add(self.root / name, arcname=name, recursive=False)
        command = [sys.executable, "-c", "import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())", str(archive)]
        verify_archive(command, digest(archive), archive.stat().st_size, manifest)


if __name__ == "__main__":
    unittest.main()

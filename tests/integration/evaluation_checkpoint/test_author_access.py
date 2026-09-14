import copy
import json
import tempfile
import unittest
from pathlib import Path

from offline_study.evaluation.checkpoint.author_access import CATEGORIES, PRECISIONS, digest, verify_runtime_access
from offline_study.evaluation.checkpoint.author_runtime import validate_cohort
from offline_study.core.protocol import sha256, write_json


class AuthorAccessTests(unittest.TestCase):
    def fixture(self, directory):
        fit = [{"trajectory_id": f"train:{i}", "lineage_group": f"train:{i}", "index": i,
                "source_pool": "train", "split": "fit", "task": "pusht"} for i in range(128)]
        evaluation = [{"trajectory_id": f"val:{i}", "lineage_group": f"val:{i}", "index": i,
                       "source_pool": "val", "split": "external_reserve", "task": "pusht"} for i in range(21)]
        manifest = directory / "source_manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in fit + evaluation))
        registry = {"manifest_sha256": sha256(manifest), "trajectories": {
            row["trajectory_id"]: {"use": "protected"} for row in evaluation}}
        write_json(directory / "source_registry.json", registry)
        cohort = {"dataset": "pusht", "task": "pusht", "fit": fit, "evaluation": [],
                  "protected_official_validation": evaluation,
                  "all_protected_groups": [row["lineage_group"] for row in evaluation],
                  "all_author_validation_groups": [row["lineage_group"] for row in evaluation],
                  "source_manifest_sha256": sha256(manifest),
                  "source_registry_sha256": sha256(directory / "source_registry.json"),
                  "coverage": {"evaluated_rows": 0, "official_rows": 21}}
        write_json(directory / "source_cohort.json", cohort)
        access = {"purpose": "one_shot_released_author_validation_replication", "authority": "user_20260907",
                  "retune_from_outcomes": False, "fresh_confirmation": False,
                  "manifest_sha256": sha256(manifest), "registry_sha256": cohort["source_registry_sha256"],
                  "original_cohort_sha256": sha256(directory / "source_cohort.json"),
                  "evaluation_rows_sha256": digest(evaluation), "fit_rows_sha256": digest(fit),
                  "frozen_protocols": {f"{p}/{c}": {} for p in PRECISIONS for c in CATEGORIES}}
        cohort.update(evaluation=evaluation, protected_official_validation=[], measurement_role="author_replication",
                      replication_access=access, access_authorization_sha256=digest(access),
                      coverage={"evaluated_rows": 21, "official_rows": 21})
        write_json(directory / "access_authorization.json", access)
        return cohort

    def test_authorized_replication_retains_original_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort = self.fixture(root)
            validate_cohort(cohort)
            self.assertTrue(verify_runtime_access(root / "cohort.json", cohort))
            self.assertEqual(len(cohort["all_protected_groups"]), 21)
            registry = json.loads((root / "source_registry.json").read_text())
            self.assertTrue(all(row["use"] == "protected" for row in registry["trajectories"].values()))

    def test_a_permission_flag_alone_never_opens_protected_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = self.fixture(Path(temporary))
            del cohort["replication_access"]
            cohort["allow_holdout"] = True
            with self.assertRaisesRegex(ValueError, "Protected family"):
                validate_cohort(cohort)

    def test_changed_membership_purpose_or_unfrozen_sweep_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = self.fixture(Path(temporary))
            for mutate in (
                lambda c: c["evaluation"].pop(),
                lambda c: c["fit"][0].update(lineage_group="val:0"),
                lambda c: c["replication_access"].update(fresh_confirmation=True),
                lambda c: c["replication_access"]["frozen_protocols"].pop("float32/operator_rank"),
                lambda c: c.update(task="mw-reach"),
            ):
                cohort = copy.deepcopy(original)
                mutate(cohort)
                # Even recomputing a digest cannot broaden the fixed authorization.
                cohort["access_authorization_sha256"] = digest(cohort["replication_access"])
                with self.assertRaises(ValueError):
                    validate_cohort(cohort)

    def test_runtime_requires_unchanged_evidence_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort = self.fixture(root)
            registry = json.loads((root / "source_registry.json").read_text())
            registry["trajectories"]["val:0"]["use"] = "development"
            write_json(root / "source_registry.json", registry)
            with self.assertRaisesRegex(ValueError, "exposure evidence changed"):
                verify_runtime_access(root / "cohort.json", cohort)


if __name__ == "__main__":
    unittest.main()

import copy
import json
import tempfile
import unittest
from pathlib import Path

from offline_study.evaluation.checkpoint.author_access import CATEGORIES, PRECISIONS, digest, verify_runtime_access
from offline_study.evaluation.checkpoint.author_extension import INDICES, FULL_COUNTS, SCOPE
from offline_study.evaluation.checkpoint.author_runtime import validate_cohort
from offline_study.core.protocol import sha256, write_json


class AuthorExtensionTests(unittest.TestCase):
    def fixture(self, root, task="mw-reach"):
        def row(i, split):
            return {"trajectory_id": f"metaworld:all:{i}", "lineage_group": f"group:{i}",
                    "index": i, "source_pool": "all", "split": split, "task": task}
        fit = [row(i, "fit") for i in range(128)]
        extra = [row(i, "holdout") for i in sorted(INDICES[task])]
        reused = [row(i + 300, "development") for i in range(FULL_COUNTS[task] - len(extra))]
        manifest = root / "source_manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(r) for r in fit + extra + reused))
        registry = {"manifest_sha256": sha256(manifest), "trajectories": {
            r["trajectory_id"]: {"use": "protected" if r["split"] == "holdout" else "development"}
            for r in fit + extra + reused}}
        write_json(root / "source_registry.json", registry)
        original = {"task": task, "dataset": "metaworld", "fit": fit, "evaluation": reused,
            "protected_official_validation": extra, "all_protected_groups": [r["lineage_group"] for r in extra],
            "all_author_validation_groups": [r["lineage_group"] for r in extra + reused],
            "source_manifest_sha256": sha256(manifest), "source_registry_sha256": sha256(root / "source_registry.json"),
            "coverage": {"evaluated_rows": len(reused), "official_rows": FULL_COUNTS[task]}}
        validate_cohort(original)
        write_json(root / "source_cohort.json", original)
        access = {"scope": SCOPE, "purpose": "one_shot_released_author_validation_replication",
            "authority": "user_20260907_metaworld_conditional_approval", "retune_from_outcomes": False,
            "fresh_confirmation": False, "manifest_sha256": sha256(manifest),
            "registry_sha256": original["source_registry_sha256"], "original_cohort_sha256": sha256(root / "source_cohort.json"),
            "fit_rows_sha256": digest(fit), "evaluation_rows_sha256": digest(extra),
            "reused_evaluation_rows_sha256": digest(reused),
            "frozen_protocols": {f"{p}/{c}": {} for p in PRECISIONS for c in CATEGORIES}}
        cohort = {**original, "evaluation": extra, "reused_evaluation": reused, "protected_official_validation": [],
            "measurement_role": "author_replication", "replication_access": access,
            "access_authorization_sha256": digest(access),
            "coverage": {"evaluated_rows": len(extra), "official_rows": FULL_COUNTS[task]}}
        write_json(root / "access_authorization.json", access)
        return cohort

    def test_only_missing_rows_execute_with_unchanged_protection(self):
        for task in INDICES:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                cohort = self.fixture(root, task)
                validate_cohort(cohort)
                self.assertTrue(verify_runtime_access(root / "cohort.json", cohort))
                self.assertEqual({r["split"] for r in cohort["evaluation"]}, {"holdout"})
                self.assertEqual(len(cohort["evaluation"]) + len(cohort["reused_evaluation"]), FULL_COUNTS[task])

    def test_unapproved_row_cannot_enter_even_with_rehashed_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = self.fixture(Path(temporary))
            cohort["evaluation"][0].update(index=9999, trajectory_id="metaworld:all:9999")
            cohort["replication_access"]["evaluation_rows_sha256"] = digest(cohort["evaluation"])
            cohort["access_authorization_sha256"] = digest(cohort["replication_access"])
            with self.assertRaisesRegex(ValueError, "exact seven approved"):
                validate_cohort(cohort)

    def test_scope_flag_without_authorization_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            cohort = self.fixture(Path(temporary))
            del cohort["replication_access"]
            with self.assertRaisesRegex(ValueError, "Protected family"):
                validate_cohort(cohort)

    def test_reused_population_cannot_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort = self.fixture(root)
            original = json.loads((root / "source_cohort.json").read_text())
            original["evaluation"][0]["index"] = 9999
            write_json(root / "source_cohort.json", original)
            with self.assertRaisesRegex(ValueError, "Original extension evidence changed"):
                verify_runtime_access(root / "cohort.json", cohort)


if __name__ == "__main__":
    unittest.main()

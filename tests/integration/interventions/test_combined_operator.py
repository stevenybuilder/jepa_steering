import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from offline_study.evaluation.combined_contract import validate_contract
from offline_study.interventions.combined_operator import PAIRS, RECIPE_ARMS, remap_edits
from offline_study.interventions.interventions import CompiledEdit
from offline_study.evaluation.combined_run import run_batch


class CombinedOperatorTests(unittest.TestCase):
    def test_source_fields_and_pairing_are_preserved_exactly(self):
        for column in (1, 2):
            names = list(dict.fromkeys(row[column] for row in RECIPE_ARMS))
            source = torch.arange(2 * len(names) * 6.).reshape(2 * len(names), 2, 3)
            edit = CompiledEdit("block_output", 3, 3, -256, None, source)
            actual = remap_edits([edit], names, column, 2)[0]
            expected = torch.stack([source[b * len(names) + names.index(row[column])]
                                    for b in range(2) for row in RECIPE_ARMS])
            self.assertTrue(torch.equal(actual.delta, expected))
            self.assertEqual(actual.delivered_l2, expected.double().flatten(1).norm(dim=1).tolist())
            self.assertEqual(actual.applications, 0)
            self.assertIsNone(actual.realized_l2)

    def fixture(self):
        return {"schema_version": 1, "status": "frozen", "category": "combined_development",
            "task": "mw-reach", "evaluation_role": "previously_exposed_full_primary_development", "fresh_confirmation": False,
            "recipe_arms": [list(row) for row in RECIPE_ARMS],
            "primary_contrasts": [{"name": a + "_vs_" + b, "candidate": a, "control": b} for a, b in PAIRS],
            "selected_choices": {"coupling": "joint_equal_standardized_energy", "rank": "rank4", "layer_change": None,
                                 "spatial_change": None, "reference_block": 3, "reference_patches": 256},
            "shard_count": 4, "precisions": ["bfloat16", "float32"], "primary_precision": "bfloat16",
            "selection_between_precisions": False, "trajectory_count": 33, "newly_opened_untouched_rows": 0}

    def test_closed_registry_full_population_and_no_protection_override(self):
        contract = self.fixture()
        validate_contract(contract)
        changes = [("task", "mw-reach-wall"), ("fresh_confirmation", True), ("trajectory_count", 29),
                   ("primary_precision", "float32"), ("newly_opened_untouched_rows", 1)]
        for key, value in changes:
            with self.assertRaises(ValueError):
                validate_contract({**contract, key: value})
        changed = copy.deepcopy(contract)
        changed["recipe_arms"][4][1] = "joint"
        with self.assertRaises(ValueError):
            validate_contract(changed)

    def test_every_candidate_has_a_scope_matched_combined_control(self):
        pairs = set(PAIRS)
        self.assertIn(("combined", "matched_random_combined"), pairs)
        self.assertIn(("combined_rank1", "matched_random_combined_rank1"), pairs)
        self.assertEqual(len(RECIPE_ARMS), 10)

    def test_dispatch_keeps_true_noops_and_source_singletons_separate(self):
        # Distinct values identify each window/component. The second window has
        # a degenerate rank correction: it remains an observation and a no-op.
        source = {}
        for column, site in ((1, "predictor_visual"), (2, "block_output")):
            names = list(dict.fromkeys(row[column] for row in RECIPE_ARMS))
            values = [0. if name in ("native", "zero_dose") or (column == 2 and b == 1)
                      else float(10 * column + j + b) for b in range(2) for j, name in enumerate(names)]
            delta = torch.tensor(values).reshape(-1, 1, 1)
            source[column] = ([CompiledEdit(site, 3, 3 if column == 2 else None,
                                            -256, None, delta)], names)
        edits = sum([remap_edits(*source[c], c, 2) for c in (1, 2)], [])

        class Backend:
            device = torch.device("cpu")
            active = []
            calls = []

            def predict(self, context, actions):
                self.calls.append((len(actions[0]), [e.site for e in self.active]))
                value = context["visual"][:, 0, 0].clone()
                for edit in self.active:
                    # No zero-only windows should pass through an active scope.
                    value += edit.delta.flatten(1).sum(1)
                    edit.realized_l2 = edit.delta.flatten(1).norm(dim=1)
                return {k: value[None, :, None].repeat(7, 1, 1) for k in ("visual", "proprio")}

            def metrics(self, pred, target):
                return [{"value": float(v)} for v in pred["visual"][6, :, 0]]

        backend = Backend()
        backend.predictor = backend

        class Hooks:
            def __init__(self, predictor, fields):
                self.predictor, self.fields = predictor, fields

            def __enter__(self):
                self.predictor.active = self.fields

            def __exit__(self, *args):
                self.predictor.active = []

        components = {key: (None, None, {}) for key in ("vision_action_coupling", "operator_rank")}
        context = {"visual": torch.tensor([100., 200.]).reshape(2, 1, 1)}
        with patch("offline_study.evaluation.combined_run.prepare_combined", return_value=(edits, [{}, {}], source)), \
                patch("offline_study.evaluation.combined_run.PredictorIntervention", Hooks):
            measured, diagnostic = run_batch(backend, components, {k: {} for k in components},
                [{"trajectory_id": "a"}, {"trajectory_id": "b"}], context, torch.zeros(6, 2, 1), {}, True)
        self.assertEqual(len(measured), 20)
        self.assertEqual(len(diagnostic), 2)
        for b in range(2):
            for j in range(10):
                row = measured[b * 10 + j]
                expected = 100. * (b + 1) + sum(float(e.delta[b * 10 + j].sum()) for e in edits)
                self.assertEqual(row["metrics"]["value"], expected)
                self.assertTrue(row["realized_energy_within_fp32_tolerance"])
        self.assertEqual(backend.calls[:2], [(2, []), (2, [])])
        self.assertIn((4, ["predictor_visual"]), backend.calls)
        self.assertIn((2, ["block_output"]), backend.calls)
        self.assertIn((8, ["predictor_visual", "block_output"]), backend.calls)

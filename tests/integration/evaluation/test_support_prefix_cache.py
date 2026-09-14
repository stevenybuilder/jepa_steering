import unittest

import torch

from offline_study.evaluation.support_prefix_cache import PrefixMemo, identical, snapshot


class Counter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, factor=2):
        self.calls += 1
        return x * factor, None


class PrefixCacheTests(unittest.TestCase):
    def test_snapshots_preserve_noncontiguous_layout_and_do_not_alias(self):
        x = torch.arange(60.).reshape(3, 20)[:, ::2]
        y = snapshot(x)
        self.assertTrue(identical(x, y))
        x[0, 0] += 1
        self.assertFalse(identical(x, y))

    @torch.no_grad()
    def test_only_prefix_is_cached_and_call_hooks_still_execute(self):
        model = Counter().eval()
        hooks = []
        handle = model.register_forward_pre_hook(lambda *_: hooks.append(1))
        with PrefixMemo(model) as memo:
            for probe in range(4):
                for horizon in range(6):
                    x = torch.ones(3) * (horizon if horizon < 2 else horizon + probe)
                    value, _ = model(x)
                    self.assertTrue(torch.equal(value, x * 2))
                    value.zero_()  # Cannot mutate the stored snapshot.
        handle.remove()
        self.assertEqual(len(hooks), 24)
        self.assertEqual(memo.hits, 6)
        self.assertEqual(model.calls, 18)
        model(torch.ones(3))
        self.assertEqual(model.calls, 19)

    @torch.no_grad()
    def test_changed_input_misses_and_incomplete_sequence_restores(self):
        model = Counter().eval()
        with PrefixMemo(model) as memo:
            for probe in range(2):
                for horizon in range(6):
                    model(torch.ones(3) * probe)
        self.assertEqual(memo.hits, 0)
        with self.assertRaises(ValueError):
            with PrefixMemo(model):
                model(torch.ones(3))
        self.assertEqual(model.forward.__func__, Counter.forward)

    def test_training_and_grad_paths_are_rejected(self):
        with self.assertRaises(ValueError):
            with PrefixMemo(Counter().eval()):
                pass


if __name__ == "__main__":
    unittest.main()

import argparse
import unittest
from pathlib import Path

from offline_study.intervention_multigpu import intervention_command


class InterventionMultigpuTests(unittest.TestCase):
    def test_command_carries_global_shard_identity_and_frozen_inputs(self):
        args = argparse.Namespace(
            vendor=Path("vendor"), checkpoint=Path("checkpoint"), checkpoint_sha256="a" * 64,
            manifest=Path("manifest"), exposure_registry=Path("registry"), data_root=Path("data"),
            protocol=Path("protocol"), operator_bank=Path("bank"), fit_receipt=Path("receipt"),
            output=Path("run"), tasks=["pusht"], max_trajectories=100000, batch_size=1,
            warmup=2, prefetch_batches=2, max_runtime_seconds=100,
        )
        command = intervention_command(args, "cuda:3", 11, 16)
        self.assertEqual(command[command.index("--device") + 1], "cuda:3")
        self.assertEqual(command[command.index("--shard-index") + 1], "11")
        self.assertEqual(command[command.index("--num-shards") + 1], "16")
        self.assertEqual(command[command.index("--protocol") + 1], "protocol")
        self.assertEqual(command[command.index("--operator-bank") + 1], "bank")


if __name__ == "__main__":
    unittest.main()

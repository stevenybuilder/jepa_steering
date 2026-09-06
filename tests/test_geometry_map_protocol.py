import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str):
    path = ROOT / "scripts" / "geometry_map" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_shard_metadata_and_hash_are_deterministic(tmp_path):
    module = load_module("discover_metaworld_shards")
    path = tmp_path / "sample.parquet"
    table = pa.table(
        {
            "task": ["mw-reach-wall", "mw-reach-wall"],
            "seed": [1, 1],
            "episode": [1, 2],
        }
    )
    pq.write_table(table, path)
    metadata = module.inspect_shard(path)
    assert metadata == {
        "rows": 2,
        "row_groups": 1,
        "tasks_first_row_group": ["mw-reach-wall"],
        "seeds_first_row_group": [1],
        "episode_min_first_row_group": 1,
        "episode_max_first_row_group": 2,
    }
    assert module.sha256(path) == module.sha256(path)


def test_reach_wall_splits_and_labels():
    module = load_module("protocol")
    assert [module.split_for_episode(i) for i in (0, 49, 50, 74, 75, 99)] == [
        "discovery",
        "discovery",
        "validation",
        "validation",
        "confirmation",
        "confirmation",
    ]
    assert module.shard_for(1, 0) == 0
    assert module.shard_for(3, 99) == 11
    states = np.zeros((100, 39), dtype=np.float32)
    states[:, :3] = np.array([0.0, 0.6, 0.2])
    states[:, -3:] = np.array([0.0, 0.9, 0.2])
    actions = np.zeros((99, 4), dtype=np.float32)
    rewards = np.arange(99, dtype=np.float32)
    labels = module.reach_wall_labels(states, actions, rewards)
    assert labels["hand_xyz"].shape == (20, 3)
    assert labels["goal_distance"].shape == (20,)
    assert labels["realized_hand_delta"].shape == (19, 3)
    assert labels["action_mean"].shape == (19, 4)
    assert labels["reward_sum"].shape == (19,)

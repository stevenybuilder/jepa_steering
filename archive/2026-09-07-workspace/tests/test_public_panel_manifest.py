from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cgs_pilot"))

from public_panel_manifest import (  # noqa: E402
    SCHEMA_VERSION,
    SeedAllocator,
    generate_rows,
    legacy_episode_seeds,
    parse_int_set,
    rows_sha256,
    validate_paired_realizations,
    validate_rows,
)
from public_panel_eval import (  # noqa: E402
    explicit_environment_seed,
    install_manifest_seeding,
    install_realization_hashing,
    simulator_state_trace,
    stable_value_sha256,
)


COMMON = {
    "task": "mw-reach-wall",
    "namespace": "performance-first-v1",
    "checkpoint_sha256": "c" * 64,
    "config_sha256": "f" * 64,
    "repo_commit": "1" * 40,
}


def _bundle():
    alloc = SeedAllocator(COMMON["namespace"], excluded=range(1, 63))
    fit = generate_rows(split="fit", count=15, allocator=alloc, **COMMON)
    evaluation = generate_rows(split="evaluation", count=30, allocator=alloc, **COMMON)
    return fit, evaluation


def test_legacy_seed_two_is_not_held_out_from_seed_one() -> None:
    overlap = set(legacy_episode_seeds(1, 30)) & set(legacy_episode_seeds(2, 30))
    assert overlap == set(range(4, 31, 2))


def test_manifest_generation_is_deterministic_and_disjoint() -> None:
    fit, evaluation = _bundle()
    fit2, evaluation2 = _bundle()
    assert fit == fit2 and evaluation == evaluation2
    assert all(row["schema_version"] == SCHEMA_VERSION for row in fit + evaluation)
    assert not {r["env_seed"] for r in fit} & {r["env_seed"] for r in evaluation}
    assert not {r["planner_seed"] for r in fit} & {r["planner_seed"] for r in evaluation}
    assert not ({r["env_seed"] for r in fit + evaluation} | {r["planner_seed"] for r in fit + evaluation}) & set(range(1, 63))
    verdict = validate_rows((("fit", fit), ("evaluation", evaluation)))
    assert verdict["passed"] and verdict["n_rows"] == 45


def test_three_way_development_fit_evaluation_split_is_disjoint() -> None:
    alloc = SeedAllocator(COMMON["namespace"], excluded=range(1, 63))
    development = generate_rows(split="development", count=30, allocator=alloc, **COMMON)
    fit = generate_rows(split="fit", count=30, allocator=alloc, **COMMON)
    evaluation = generate_rows(split="evaluation", count=60, allocator=alloc, **COMMON)
    groups = (("development", development), ("fit", fit), ("evaluation", evaluation))
    verdict = validate_rows(groups)
    assert verdict["passed"] and verdict["n_rows"] == 120
    all_rows = development + fit + evaluation
    assert len({row["pair_id"] for row in all_rows}) == len(all_rows)
    assert len({row["env_seed"] for row in all_rows}) == len(all_rows)
    assert len({row["planner_seed"] for row in all_rows}) == len(all_rows)


def test_validator_rejects_cross_split_seed_reuse() -> None:
    fit, evaluation = _bundle()
    evaluation[0]["env_seed"] = fit[0]["env_seed"]
    verdict = validate_rows((("fit", fit), ("evaluation", evaluation)))
    assert not verdict["passed"]
    assert any("env_seed" in error and "reused" in error for error in verdict["errors"])


def test_realization_validator_requires_exact_paired_hashes() -> None:
    hashes = {
        "initial_observation_sha256": "a",
        "goal_observation_sha256": "b",
        "initial_visual_sha256": "c",
        "goal_visual_sha256": "d",
        "initial_proprio_sha256": "e",
        "goal_proprio_sha256": "f",
        "initial_simulator_state_sha256": "h",
        "goal_state_sha256": "g",
    }
    arm_a = [{"pair_id": "p0", **hashes}]
    arm_b = [{"pair_id": "p0", **hashes}]
    assert validate_paired_realizations((("a", arm_a), ("b", arm_b)))["passed"]
    arm_b[0]["goal_state_sha256"] = "changed"
    verdict = validate_paired_realizations((("a", arm_a), ("b", arm_b)))
    assert not verdict["passed"]
    assert any("goal_state_sha256" in error for error in verdict["errors"])


def test_hash_covers_order_and_every_field() -> None:
    fit, _ = _bundle()
    original = rows_sha256(fit)
    assert rows_sha256(list(reversed(fit))) != original
    fit[0]["planner_seed"] += 1
    assert rows_sha256(fit) != original


def test_realization_hash_is_layout_invariant_and_value_sensitive() -> None:
    import numpy as np

    value = {
        "visual": np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
        "proprio": np.array([0.1, 0.2], dtype=np.float32),
    }
    same = {
        "proprio": value["proprio"].copy(),
        "visual": np.asfortranarray(value["visual"]),
    }
    changed = {key: array.copy() for key, array in value.items()}
    changed["visual"][0, 0, 0] += 1
    assert stable_value_sha256(value) == stable_value_sha256(same)
    assert stable_value_sha256(value) != stable_value_sha256(changed)


def test_realization_hashing_captures_policy_warmup_state() -> None:
    import numpy as np

    stale_state = np.array([99.0, 99.0], dtype=np.float32)
    realized_state = np.array([1.0, 2.0], dtype=np.float32)

    class Env:
        def reset_warmup(self, seed=None):
            return "obs", {"state": realized_state}

    class Evaluator:
        def set_episode(self, cfg, agent, env, ep_seed, task_idx=-1):
            env.reset_warmup(seed=ep_seed)
            self.state_g = np.array([3.0, 4.0], dtype=np.float32)
            obs = {"visual": np.zeros((1, 2), dtype=np.uint8), "proprio": np.zeros(2, dtype=np.float32)}
            return obs, obs, [], 1

    class PE:
        PlanEvaluator = Evaluator

    install_realization_hashing(PE)
    evaluator = Evaluator()
    evaluator.set_episode(None, None, Env(), 123)
    assert np.array_equal(evaluator._panelp_initial_simulator_state, realized_state)
    assert evaluator._panelp_realization_hashes["initial_simulator_state_sha256"] == stable_value_sha256(
        realized_state
    )
    trace = simulator_state_trace(
        evaluator,
        {"state": stale_state},
        [{"state": np.array([5.0, 6.0], dtype=np.float32)}],
    )
    assert np.array_equal(trace, np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32))


def test_integer_set_parser() -> None:
    assert parse_int_set("1,4-6,9") == {1, 4, 5, 6, 9}
    assert parse_int_set(None) == set()


class _FakeEnv:
    def __init__(self):
        self.calls = []

    def reset(self, seed=None, task_idx=-1):
        self.calls.append(("reset", seed, task_idx))

    def reset_warmup(self, seed=None):
        self.calls.append(("reset_warmup", seed))

    def seed(self, seed=None):
        self.calls.append(("seed", seed))

    def prepare(self, seed, state):
        self.calls.append(("prepare", seed, state))

    def sample_random_init_goal_states(self, seed):
        self.calls.append(("sample", seed))


def test_explicit_environment_seed_substitutes_and_restores() -> None:
    env = _FakeEnv()
    with explicit_environment_seed(env, 1234):
        env.reset(seed=9, task_idx=2)
        env.reset_warmup(seed=9)
        env.seed(9)
        env.prepare(9, "state")
        env.sample_random_init_goal_states(9)
    assert env.calls == [
        ("reset", 1234, 2),
        ("reset_warmup", 1234),
        ("seed", 1234),
        ("prepare", 1234, "state"),
        ("sample", 1234),
    ]
    env.reset(seed=77, task_idx=0)
    assert env.calls[-1] == ("reset", 77, 0)


def test_manifest_wrapper_resets_planner_and_environment_rngs(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def manual_seed(seed):
            return seed

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    class Generator:
        value = None

        def manual_seed(self, seed):
            self.value = seed

    class Evaluator:
        def eval(self, cfg, agent, env, task_idx=-1, ep=0):
            env.reset(seed=999, task_idx=task_idx)
            env.seed(999)
            return agent.local_generator.value

    class PE:
        PlanEvaluator = Evaluator

    class Cfg:
        tasks = ["mw-reach-wall"]

    class Agent:
        local_generator = Generator()

    rows = [{"task": "mw-reach-wall", "env_seed": 444, "planner_seed": 555, "pair_id": "fit-000"}]
    install_manifest_seeding(PE, rows)
    env = _FakeEnv()
    first = Evaluator().eval(Cfg(), Agent(), env, task_idx=0, ep=0)
    second = Evaluator().eval(Cfg(), Agent(), _FakeEnv(), task_idx=0, ep=0)
    assert first == second
    assert env.calls == [("reset", 444, 0), ("seed", 444)]

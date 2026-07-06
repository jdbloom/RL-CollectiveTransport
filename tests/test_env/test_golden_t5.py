"""T5 golden-equivalence gate — ZMQ parser vectorization.

Protocol: this test is committed on the UNMODIFIED baseline.  The frozen
references are captured by running ``_freeze_all()`` exactly once (on baseline)
and the resulting .pkl files are committed alongside this test.  After the T5
patch is applied the vectorized parsers must reproduce those frozen arrays /
bytes exactly — byte-identical for integer/float32 data, bytearray-identical
for serialize_actions.

The test exercises:
  - parse_obs       → list[np.float32 (num_obs,)]
  - parse_failures  → list[np.intc (1,)]
  - parse_rewards   → list[np.float32 (1,)]
  - parse_stats     → list[np.float32 (num_stats,)]
  - parse_robot_stats → list[np.float32 (6,)]
  - serialize_actions → bytearray

Seeds 0, 1, 7, 42, 99 and robot counts R ∈ {3, 4, 6} are exercised.
"""
from __future__ import annotations

import pickle
import struct
from pathlib import Path

import numpy as np
import pytest

# The import works whether tests are run from the repo root or from rl_code/
try:
    from rl_code.src.env import ZMQ_Utility
except ImportError:
    from src.env import ZMQ_Utility  # type: ignore[import]

GOLDEN_DIR = Path(__file__).parent / "golden_refs_t5"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_util(R: int, num_stats: int = 4) -> ZMQ_Utility:
    """Return a ZMQ_Utility pre-seeded with params for R robots."""
    util = ZMQ_Utility()
    params_bytes = struct.pack(
        "9f",
        float(R),   # num_robots
        2.0,        # num_obstacles
        31.0,       # num_obs
        3.0,        # num_actions
        float(num_stats),  # num_stats
        9.0,        # alphabet_size
        0.0,        # use_gate
        10.0,       # distance_to_goal_normalization_factor
        0.0,        # num_prisms
    )
    util.get_params(params_bytes)
    util.set_obstacles_fields()
    return util


def _make_obs_msg(R: int, num_obs: int, rng: np.random.Generator) -> bytes:
    vals = rng.uniform(-5.0, 5.0, R * num_obs).astype("<f4")
    return vals.tobytes()


def _make_failure_msg(R: int, rng: np.random.Generator) -> bytes:
    vals = rng.integers(0, 2, size=R).astype("<u4")
    return vals.tobytes()


def _make_reward_msg(R: int, rng: np.random.Generator) -> bytes:
    vals = rng.uniform(-3.0, 0.0, R).astype("<f4")
    return vals.tobytes()


def _make_stats_msg(R: int, num_stats: int, rng: np.random.Generator) -> bytes:
    vals = rng.uniform(-2.0, 2.0, R * num_stats).astype("<f4")
    return vals.tobytes()


def _make_robot_stats_msg(R: int, rng: np.random.Generator) -> bytes:
    vals = rng.uniform(-1.0, 1.0, R * 6).astype("<f4")
    return vals.tobytes()


def _make_actions(R: int, rng: np.random.Generator) -> list:
    return [
        [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), float(rng.integers(0, 2))]
        for _ in range(R)
    ]


def _capture_all() -> dict:
    """Run every parser on all (seed, R) combinations and collect outputs."""
    results = {}
    for seed in (0, 1, 7, 42, 99):
        for R in (3, 4, 6):
            rng = np.random.default_rng(seed)
            util = _make_util(R)
            num_obs = util.params["num_obs"]
            num_stats = util.params["num_stats"]

            obs_msg = _make_obs_msg(R, num_obs, rng)
            fail_msg = _make_failure_msg(R, rng)
            rew_msg = _make_reward_msg(R, rng)
            stat_msg = _make_stats_msg(R, num_stats, rng)
            rs_msg = _make_robot_stats_msg(R, rng)
            actions = _make_actions(R, rng)

            key = f"s{seed}_r{R}"
            results[key] = {
                "obs": [arr.copy() for arr in util.parse_obs(obs_msg)],
                "failures": [arr.copy() for arr in util.parse_failures(fail_msg)],
                "rewards": [arr.copy() for arr in util.parse_rewards(rew_msg)],
                "stats": [arr.copy() for arr in util.parse_stats(stat_msg)],
                "robot_stats": [arr.copy() for arr in util.parse_robot_stats(rs_msg)],
                "serialize_actions": bytes(util.serialize_actions(actions)),
                # Store messages + actions so the gate can re-run the parser under test
                "_obs_msg": obs_msg,
                "_fail_msg": fail_msg,
                "_rew_msg": rew_msg,
                "_stat_msg": stat_msg,
                "_rs_msg": rs_msg,
                "_actions": actions,
            }
    return results


def _freeze_all() -> None:
    """Capture on BASELINE and write frozen references.  Run exactly once, pre-patch."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    data = _capture_all()
    with open(GOLDEN_DIR / "t5_parsers.pkl", "wb") as f:
        pickle.dump(data, f)
    print(f"Frozen {len(data)} cases to {GOLDEN_DIR / 't5_parsers.pkl'}")


def _load_frozen() -> dict:
    path = GOLDEN_DIR / "t5_parsers.pkl"
    if not path.exists():
        pytest.skip(
            f"Golden reference {path} not found. "
            "Run _freeze_all() on the baseline to generate it."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# The golden gate test
# ---------------------------------------------------------------------------

class TestGoldenT5:
    """Assert that every parser returns results byte-identical to the frozen baseline."""

    def test_parse_obs_matches_frozen(self):
        frozen = _load_frozen()
        for key, ref in frozen.items():
            seed, r = key.split("_")
            R = int(r[1:])
            util = _make_util(R)
            result = util.parse_obs(ref["_obs_msg"])
            assert len(result) == len(ref["obs"]), f"{key}: obs list length mismatch"
            for i, (got, want) in enumerate(zip(result, ref["obs"])):
                assert np.array_equal(got, want), (
                    f"{key} robot {i}: obs mismatch\n  got:  {got}\n  want: {want}"
                )

    def test_parse_failures_matches_frozen(self):
        frozen = _load_frozen()
        for key, ref in frozen.items():
            R = int(key.split("_r")[1])
            util = _make_util(R)
            result = util.parse_failures(ref["_fail_msg"])
            assert len(result) == len(ref["failures"]), f"{key}: failures list length mismatch"
            for i, (got, want) in enumerate(zip(result, ref["failures"])):
                assert np.array_equal(got, want), (
                    f"{key} robot {i}: failure mismatch got={got} want={want}"
                )

    def test_parse_rewards_matches_frozen(self):
        frozen = _load_frozen()
        for key, ref in frozen.items():
            R = int(key.split("_r")[1])
            util = _make_util(R)
            result = util.parse_rewards(ref["_rew_msg"])
            assert len(result) == len(ref["rewards"]), f"{key}: rewards list length mismatch"
            for i, (got, want) in enumerate(zip(result, ref["rewards"])):
                assert np.array_equal(got, want), (
                    f"{key} robot {i}: reward mismatch got={got} want={want}"
                )

    def test_parse_stats_matches_frozen(self):
        frozen = _load_frozen()
        for key, ref in frozen.items():
            R = int(key.split("_r")[1])
            util = _make_util(R)
            result = util.parse_stats(ref["_stat_msg"])
            assert len(result) == len(ref["stats"]), f"{key}: stats list length mismatch"
            for i, (got, want) in enumerate(zip(result, ref["stats"])):
                assert np.array_equal(got, want), (
                    f"{key} robot {i}: stats mismatch got={got} want={want}"
                )

    def test_parse_robot_stats_matches_frozen(self):
        frozen = _load_frozen()
        for key, ref in frozen.items():
            R = int(key.split("_r")[1])
            util = _make_util(R)
            result = util.parse_robot_stats(ref["_rs_msg"])
            assert len(result) == len(ref["robot_stats"]), (
                f"{key}: robot_stats list length mismatch"
            )
            for i, (got, want) in enumerate(zip(result, ref["robot_stats"])):
                assert np.array_equal(got, want), (
                    f"{key} robot {i}: robot_stats mismatch\n  got:  {got}\n  want: {want}"
                )

    def test_serialize_actions_matches_frozen(self):
        frozen = _load_frozen()
        for key, ref in frozen.items():
            R = int(key.split("_r")[1])
            util = _make_util(R)
            result = bytes(util.serialize_actions(ref["_actions"]))
            assert result == ref["serialize_actions"], (
                f"{key}: serialize_actions bytes mismatch"
            )

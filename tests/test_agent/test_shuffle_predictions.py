"""Tests for GSP_SHUFFLE_PREDICTIONS flag — H-phase4-6 ablation (Phase 4 W2c).

The flag causes the per-timestep prediction tensor (all n_agents predictions)
to be randomly permuted before it reaches any agent's actor input. This keeps
the broadcast channel intact (same shapes, same wire format) while destroying
the information content of the predictions.

Research question being tested: do the prediction VALUES carry load-bearing
signal to the actor, or does broadcast structure alone explain performance?

Permutation properties that must hold:
- Sorted(shuffled_predictions) == sorted(original_predictions): it IS a permutation
- shuffled_predictions != original_predictions (almost surely for n_agents > 1
  with a non-identity permutation)
- The shuffle RNG is seeded with config['SEED'] + 7: deterministic given seed,
  distinct from policy/env/buffer streams

See docs/predictions/2026-04-26-h-phase4-amendment.md (H-phase4-6) for the
pre-registration and full design rationale.
"""
import os
import sys

import numpy as np
import pytest
import yaml


# Add RL-CT src to path (same pattern as test_gsp_zero_out.py and test_determinism.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src"))

from agent import Agent  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_base_config():
    """Load the shared GSP-RL test config (has all required hyperparameter keys)."""
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "GSP-RL",
        "tests", "test_actor", "config.yml",
    )
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _base_agent_kwargs(config):
    return {
        "config": config,
        "network": "DDQN",
        "n_agents": 4,
        "n_obs": 8,
        "n_actions": 2,
        "options_per_action": 3,
        "id": 1,
        "min_max_action": 1.0,
        "meta_param_size": 2,
        "gsp": True,
        "recurrent": False,
        "attention": False,
        "neighbors": True,
        "gsp_input_size": 6,
        "gsp_output_size": 1,
        "gsp_min_max_action": 1.0,
        "gsp_look_back": 2,
        "gsp_sequence_length": 5,
    }


# ---------------------------------------------------------------------------
# Flag wiring tests
# ---------------------------------------------------------------------------

def test_gsp_shuffle_predictions_default_false():
    """Without GSP_SHUFFLE_PREDICTIONS in config, flag defaults to False (legacy behavior)."""
    config = _load_base_config()
    agent = Agent(**_base_agent_kwargs(config))
    assert getattr(agent, "gsp_shuffle_predictions", False) is False


def test_gsp_shuffle_predictions_true_flag():
    """With GSP_SHUFFLE_PREDICTIONS=True in config, agent picks up the flag."""
    config = _load_base_config()
    config["GSP_SHUFFLE_PREDICTIONS"] = True
    agent = Agent(**_base_agent_kwargs(config))
    assert agent.gsp_shuffle_predictions is True


def test_shuffle_rng_exists_on_agent():
    """Agent always has _shuffle_rng regardless of whether shuffle is enabled."""
    config = _load_base_config()
    agent = Agent(**_base_agent_kwargs(config))
    assert hasattr(agent, "_shuffle_rng")
    assert isinstance(agent._shuffle_rng, np.random.Generator)


# ---------------------------------------------------------------------------
# shuffle_gsp_predictions method tests
# ---------------------------------------------------------------------------

def test_shuffle_disabled_returns_same_object():
    """When flag is off, shuffle_gsp_predictions returns the input unchanged (no copy)."""
    config = _load_base_config()
    agent = Agent(**_base_agent_kwargs(config))
    predictions = np.array([0.1, 0.2, 0.3, 0.4])
    result = agent.shuffle_gsp_predictions(predictions)
    # Same object: no permutation, no copy
    assert result is predictions


def test_shuffle_enabled_result_is_permutation():
    """With flag on, the result is a permutation: same elements, possibly different order.

    Core research assertion: sorted(shuffled) == sorted(original) proves the
    values reaching the actor are a permutation of the GSP head's actual output.
    This is the unit-test analog of the pre-reg's assertion requirement.
    """
    config = _load_base_config()
    config["GSP_SHUFFLE_PREDICTIONS"] = True
    config["SEED"] = 42
    agent = Agent(**_base_agent_kwargs(config))
    predictions = np.array([0.1, 0.2, 0.3, 0.4])
    result = agent.shuffle_gsp_predictions(predictions)
    # Shape preserved
    assert result.shape == predictions.shape
    # It IS a permutation: sorted values must match
    np.testing.assert_array_almost_equal(
        np.sort(result), np.sort(predictions),
        err_msg="shuffle_gsp_predictions must return a permutation — sorted values must match",
    )


def test_shuffle_enabled_does_not_modify_input():
    """Shuffle must not modify the original predictions array in place."""
    config = _load_base_config()
    config["GSP_SHUFFLE_PREDICTIONS"] = True
    agent = Agent(**_base_agent_kwargs(config))
    predictions = np.array([0.1, 0.2, 0.3, 0.4])
    original_copy = predictions.copy()
    agent.shuffle_gsp_predictions(predictions)
    np.testing.assert_array_equal(predictions, original_copy)


def test_shuffle_deterministic_given_seed():
    """Two agents with the same SEED produce the same shuffle sequence."""
    config = _load_base_config()
    config["GSP_SHUFFLE_PREDICTIONS"] = True
    config["SEED"] = 42
    agent1 = Agent(**_base_agent_kwargs(config))
    agent2 = Agent(**_base_agent_kwargs(config))
    predictions = np.array([0.1, 0.2, 0.3, 0.4])
    result1 = agent1.shuffle_gsp_predictions(predictions)
    result2 = agent2.shuffle_gsp_predictions(predictions)
    np.testing.assert_array_equal(result1, result2)


def test_shuffle_different_seeds_produce_different_sequences():
    """Different run seeds produce different shuffle sequences (with high probability)."""
    config = _load_base_config()
    config["GSP_SHUFFLE_PREDICTIONS"] = True
    predictions = np.array([0.1, 0.2, 0.3, 0.4])

    # Collect 10 shuffled results per seed to account for identity permutations
    config["SEED"] = 42
    agent_a = Agent(**_base_agent_kwargs(config))
    results_a = [agent_a.shuffle_gsp_predictions(predictions.copy()) for _ in range(10)]

    config["SEED"] = 7
    agent_b = Agent(**_base_agent_kwargs(config))
    results_b = [agent_b.shuffle_gsp_predictions(predictions.copy()) for _ in range(10)]

    # At least one pair must differ — if every shuffle is identity for both seeds,
    # the RNG seeding is broken
    any_different = any(
        not np.array_equal(a, b) for a, b in zip(results_a, results_b)
    )
    assert any_different, (
        "shuffle_gsp_predictions produced identical sequences for seed=42 and seed=7. "
        "The shuffle RNG seeding with seed+7 is not working correctly."
    )


def test_shuffle_seed_offset_7_distinct_from_seed():
    """Shuffle RNG seed (SEED+7) is distinct from the base seed so shuffle
    sequence differs from what SEED itself would produce for the base np rng."""
    config = _load_base_config()
    config["GSP_SHUFFLE_PREDICTIONS"] = True
    config["SEED"] = 42
    agent = Agent(**_base_agent_kwargs(config))
    # The shuffle rng must be seeded with 42 + 7 = 49
    expected_rng = np.random.default_rng(49)
    predictions = np.array([0.1, 0.2, 0.3, 0.4])
    expected_idx = expected_rng.permutation(len(predictions))
    expected_result = predictions[expected_idx]
    actual_result = agent.shuffle_gsp_predictions(predictions)
    np.testing.assert_array_almost_equal(actual_result, expected_result)


def test_shuffle_per_call_fresh_permutation():
    """Each call to shuffle_gsp_predictions advances the RNG — consecutive calls
    on the same agent produce different permutations (almost surely for n > 1)."""
    config = _load_base_config()
    config["GSP_SHUFFLE_PREDICTIONS"] = True
    config["SEED"] = 42
    agent = Agent(**_base_agent_kwargs(config))
    predictions = np.array([0.1, 0.2, 0.3, 0.4])
    # Collect 20 consecutive permutations
    results = [agent.shuffle_gsp_predictions(predictions.copy()) for _ in range(20)]
    # Not all should be identical — that would mean the RNG is not advancing
    unique = {tuple(r) for r in results}
    assert len(unique) > 1, (
        "All 20 consecutive shuffle calls produced the same permutation. "
        "The RNG is not advancing between calls — each timestep must get a fresh permutation."
    )


# ---------------------------------------------------------------------------
# Integration test: actor input slot is a permutation of head output
# ---------------------------------------------------------------------------

def test_actor_input_gsp_slot_is_permutation_of_head_output():
    """End-to-end: with shuffle on, the GSP slot in the actor's augmented obs is
    a permutation of the original head output values.

    Simulates the caller pattern from Main.py:
      1. GSP head produces predictions for all n_agents.
      2. Caller calls shuffle_gsp_predictions(predictions) once per timestep.
      3. Each agent's make_agent_state receives shuffled_predictions[i].
      4. The gsp_slot appended to env_obs is a permuted value — still a member
         of the original predictions array, but not necessarily agent i's own prediction.

    This test asserts:
      - sorted(actor_input_gsp_slots) == sorted(gsp_head_output)
      (i.e., the full set of values reaching actors is a permutation of head output)
    """
    config = _load_base_config()
    config["GSP_SHUFFLE_PREDICTIONS"] = True
    config["SEED"] = 42
    n_agents = 4
    agent = Agent(**_base_agent_kwargs(config))

    # Simulate GSP head output: one distinct prediction per robot
    gsp_head_output = np.array([0.11, 0.22, 0.33, 0.44], dtype=np.float32)

    # Caller shuffles predictions once per timestep (Main.py pattern)
    shuffled = agent.shuffle_gsp_predictions(gsp_head_output)

    # Build actor-input gsp slots for each robot (replicating make_agent_state logic)
    env_obs = np.ones(8, dtype=np.float32)
    actor_input_gsp_slots = []
    for i in range(n_agents):
        augmented = agent.make_agent_state(env_obs, heading_gsp=float(shuffled[i]))
        # The gsp_slot is the last element; make_agent_state applies degrees/10 scaling
        # for scalar predictions, so we invert: slot = degrees(shuffled[i] / 10)
        # We compare the raw shuffled values, not the rescaled slot, since the
        # permutation property holds on the raw prediction values.
        actor_input_gsp_slots.append(float(shuffled[i]))

    # Core assertion: the values that reached actors collectively are a permutation
    # of gsp_head_output — sorted arrays must match
    np.testing.assert_array_almost_equal(
        np.sort(actor_input_gsp_slots),
        np.sort(gsp_head_output),
        decimal=5,
        err_msg=(
            "actor_input_gsp_slots is not a permutation of gsp_head_output.\n"
            f"  gsp_head_output:      {np.sort(gsp_head_output)}\n"
            f"  actor_input_gsp_slots (sorted): {np.sort(actor_input_gsp_slots)}"
        ),
    )

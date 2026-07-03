"""Tests for GSP_INPUT_INCLUDE_CYL_BEARING_DELTA.

The GSP prediction head is inert because the target (cylinder rotation) is
0.89-predictable from the WRAP-SAFE one-step change in a robot's bearing-to-
cylinder angle, but the head's raw angle_to_cyl input (env_observations[i][5])
wraps at ±π, so the head can only reach corr ~0.13 offline. This flag feeds the
explicit, wrap-safe bearing-delta (in radians) as a +1 self-slot input.

Verifies:
- Input-size accounting exactly matches the written feature dims (no shape
  mismatch between agent.gsp_network_input and make_gsp_states output length).
- Adding the flag grows the per-agent vector by exactly 1 per temporal-stack unit.
- The written delta is wrap-safe: +3.10 → −3.10 rad yields the small wrapped
  delta (~+0.083), not ~−6.2.
- Default OFF is functionally inert: flag absent → _gsp_input_include_cyl_bearing_delta
  is False and input size unchanged.
"""
import math
import numpy as np
import pytest
from src.agent import Agent


def _make_agent(config_overrides: dict, n_hop_neighbors: int = 1) -> Agent:
    """Build a GSP-N agent with optional enrichment flags set in config."""
    config = {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
        **config_overrides,
    }
    return Agent(
        config=config,
        network="DDQN",
        n_agents=4,
        n_obs=31,
        n_actions=2,
        options_per_action=3,
        id=0,
        min_max_action=0.1,
        meta_param_size=1,
        gsp=True,
        recurrent=False,
        attention=False,
        neighbors=True,
        gsp_input_size=6,   # legacy default — recomputed by Agent.__init__
        gsp_output_size=1,
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
        n_hop_neighbors=n_hop_neighbors,
    )


def _env_obs(angle_to_cyl_by_agent, n_agents: int = 4) -> list:
    """Minimal env_observations with the indices used by enrichment.

    angle_to_cyl_by_agent: list of angle_to_cyl (radians) per agent → index 5.
    """
    obs = []
    for i in range(n_agents):
        o = np.zeros(31, dtype=np.float32)
        o[1] = 0.5 + i * 0.1                      # angle_to_goal
        o[4] = 0.3 + i * 0.05                     # dist_to_cyl
        o[5] = float(angle_to_cyl_by_agent[i])    # angle_to_cyl (radians)
        o[7:31] = np.linspace(0.1, 0.5, 24)       # raw prox
        obs.append(o)
    return obs


_PROX = [0.25, 0.30, 0.20, 0.10]
_PREV_GSP = [0.1, 0.2, 0.3, 0.4]


# ── Default OFF: functionally inert ───────────────────────────────────────────

def test_flag_default_off():
    """Absent flag → attribute False and size unchanged from baseline 6."""
    agent = _make_agent({})
    assert agent._gsp_input_include_cyl_bearing_delta is False
    assert agent.gsp_network_input == 6


def test_flag_off_explicit_inert():
    """Explicit False produces byte-identical output to no flag in config."""
    agent_base = _make_agent({})
    agent_flag = _make_agent({"GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": False})
    env = _env_obs([-0.2, -0.1, 0.0, 0.1])
    s_base = agent_base.make_gsp_states(_PROX, _PREV_GSP, env_observations=env)
    s_flag = agent_flag.make_gsp_states(_PROX, _PREV_GSP, env_observations=env)
    for a, b in zip(s_base, s_flag):
        np.testing.assert_array_equal(a, b)


# ── Input-size accounting matches feature dims ────────────────────────────────

def test_delta_adds_1_dim_to_input_size():
    """Flag ON adds exactly +1 to gsp_network_input (self-slot)."""
    agent = _make_agent({"GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True})
    assert agent.gsp_network_input == 7  # base 6 + 1


def test_delta_plus_cyl_rel_accounting():
    """CYL_REL (+2) and BEARING_DELTA (+1) compose to base 6 + 3 = 9."""
    agent = _make_agent({
        "GSP_INPUT_INCLUDE_CYL_REL": True,
        "GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True,
    })
    assert agent.gsp_network_input == 9


def test_accounting_matches_make_gsp_states_length():
    """Critical correctness: each returned vector length == gsp_network_input.

    Build a GSP-N agent with CYL_REL and BEARING_DELTA both on, call
    make_gsp_states with a 2-... (4-agent conftest) synthetic env, and assert
    each per-agent state vector length == agent.gsp_network_input. Then compare
    to the same agent WITHOUT the delta flag → length is exactly 1 larger.
    """
    env = _env_obs([-0.2, -0.1, 0.0, 0.1])

    agent_with = _make_agent({
        "GSP_INPUT_INCLUDE_CYL_REL": True,
        "GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True,
    })
    states_with = agent_with.make_gsp_states(_PROX, _PREV_GSP, env_observations=env)
    assert len(states_with) == 4
    for s in states_with:
        assert len(s) == agent_with.gsp_network_input

    agent_without = _make_agent({"GSP_INPUT_INCLUDE_CYL_REL": True})
    states_without = agent_without.make_gsp_states(_PROX, _PREV_GSP, env_observations=env)
    for s in states_without:
        assert len(s) == agent_without.gsp_network_input

    # Exactly 1 larger per temporal-stack unit (K=1 here).
    assert len(states_with[0]) == len(states_without[0]) + 1


def test_accounting_matches_with_temporal_stack():
    """With K=2, +1 self-slot becomes +2 total; length still == gsp_network_input."""
    env = _env_obs([-0.2, -0.1, 0.0, 0.1])
    agent = _make_agent({
        "GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True,
        "GSP_INPUT_TEMPORAL_STACK_K": 2,
    })
    # (base 6 + 1) * 2 = 14
    assert agent.gsp_network_input == 14
    states = agent.make_gsp_states(_PROX, _PREV_GSP, env_observations=env)
    for s in states:
        assert len(s) == agent.gsp_network_input


# ── Wrap-safe delta content ───────────────────────────────────────────────────

def test_first_step_delta_is_zero():
    """First step for an agent (no prev) → delta = 0.0 at the delta position."""
    agent = _make_agent({"GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True})
    env = _env_obs([1.0, 1.0, 1.0, 1.0])
    states = agent.make_gsp_states(_PROX, _PREV_GSP, env_observations=env)
    # Self slot layout: [avg_prox(0), prev_gsp(1), bearing_delta(2), n0..]
    for s in states:
        assert s[2] == pytest.approx(0.0, abs=1e-9)


def test_wrap_safe_delta_across_pi_boundary():
    """+3.10 rad → −3.10 rad crosses the ±π wrap; delta must be the SMALL
    wrapped value (~+0.083), not ~−6.2. This is the whole point of the feature."""
    agent = _make_agent({"GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True})

    env1 = _env_obs([3.10, 3.10, 3.10, 3.10])
    agent.make_gsp_states(_PROX, _PREV_GSP, env_observations=env1)  # seeds prev

    env2 = _env_obs([-3.10, -3.10, -3.10, -3.10])
    states2 = agent.make_gsp_states(_PROX, _PREV_GSP, env_observations=env2)

    # Expected wrapped delta: d = -3.10 - 3.10 = -6.20;
    # (d + pi) % (2*pi) - pi ≈ +0.0832 rad
    expected = ((-3.10 - 3.10) + math.pi) % (2 * math.pi) - math.pi
    for s in states2:
        assert s[2] == pytest.approx(expected, abs=1e-5)
        assert abs(s[2]) < 0.2            # small wrapped value
        assert s[2] > 0.0                 # sign is positive, not ~-6.2
        assert abs(s[2] + 6.2) > 5.0      # definitely NOT the raw -6.2


def test_non_wrap_delta_is_plain_difference():
    """Within (-π, π] the delta is the ordinary signed difference."""
    agent = _make_agent({"GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True})
    env1 = _env_obs([0.10, 0.10, 0.10, 0.10])
    agent.make_gsp_states(_PROX, _PREV_GSP, env_observations=env1)
    env2 = _env_obs([0.35, 0.35, 0.35, 0.35])
    states2 = agent.make_gsp_states(_PROX, _PREV_GSP, env_observations=env2)
    for s in states2:
        assert s[2] == pytest.approx(0.25, abs=1e-6)


def test_delta_position_after_cyl_rel():
    """With CYL_REL on, delta lands immediately after the cyl_rel pair.

    Layout: [avg_prox(0), prev_gsp(1), dist_cyl(2), ang_cyl(3), bearing_delta(4), ...]
    """
    agent = _make_agent({
        "GSP_INPUT_INCLUDE_CYL_REL": True,
        "GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True,
    })
    env1 = _env_obs([0.10, 0.10, 0.10, 0.10])
    agent.make_gsp_states(_PROX, _PREV_GSP, env_observations=env1)
    env2 = _env_obs([0.30, 0.30, 0.30, 0.30])
    states2 = agent.make_gsp_states(_PROX, _PREV_GSP, env_observations=env2)
    s0 = states2[0]
    # cyl_rel pair still correct
    assert s0[2] == pytest.approx(float(env2[0][4]), abs=1e-5)  # dist_to_cyl
    assert s0[3] == pytest.approx(float(env2[0][5]), abs=1e-5)  # angle_to_cyl
    # bearing delta at position 4
    assert s0[4] == pytest.approx(0.20, abs=1e-6)

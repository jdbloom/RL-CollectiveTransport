"""Tests for GSP_INPUT_INCLUDE_CYL_BEARING_DELTA.

The GSP prediction head is inert because the target (cylinder rotation) is
~0.77-0.89 predictable from the WRAP-SAFE one-step change in the robot's
WORLD-FRAME bearing around the cylinder (atan2(robot_y - cyl_y, robot_x - cyl_x)),
verified on run h5. An earlier (buggy) version used the delta of the BODY-FRAME
angle_to_cyl (env_observations[i][5]) which correlates only ~0.003 with the target.

The world-frame bearing is computed in Main.py from world positions and passed to
make_gsp_states via the cyl_bearing_delta arg. This +1 self-slot dim feeds that
pre-computed wrap-safe delta (radians).

Verifies:
- Input-size accounting exactly matches the written feature dims (no shape
  mismatch between agent.gsp_network_input and make_gsp_states output length).
- Adding the flag grows the per-agent vector by exactly 1 per temporal-stack unit.
- The written delta equals the value passed via the cyl_bearing_delta arg, at the
  correct self-slot position (right after cyl_rel, or after the self-slot base when
  cyl_rel is off).
- Default OFF is functionally inert: flag absent → attribute False, size unchanged.
- cyl_bearing_delta=None (arg absent) → the delta dim is written as 0.0.
- The wrap-safe world-frame computation (now living in Main.py) is unit-tested
  directly against a small reference implementation.
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
    """Minimal env_observations with the indices used by the goal/cyl_rel enrichment.

    angle_to_cyl_by_agent: list of angle_to_cyl (radians) per agent → index 5.
    (No longer used for the bearing delta — that comes via the arg — but cyl_rel
    still reads index 4/5, and full_prox reads 7:31.)
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
    make_gsp_states with a 4-agent synthetic env, and assert each per-agent state
    vector length == agent.gsp_network_input. Then compare to the same agent
    WITHOUT the delta flag → length is exactly 1 larger.
    """
    env = _env_obs([-0.2, -0.1, 0.0, 0.1])
    delta = {'delta': [0.01, 0.02, 0.03, 0.04]}

    agent_with = _make_agent({
        "GSP_INPUT_INCLUDE_CYL_REL": True,
        "GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True,
    })
    states_with = agent_with.make_gsp_states(
        _PROX, _PREV_GSP, env_observations=env, cyl_bearing_delta=delta)
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
    delta = {'delta': [0.01, 0.02, 0.03, 0.04]}
    agent = _make_agent({
        "GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True,
        "GSP_INPUT_TEMPORAL_STACK_K": 2,
    })
    # (base 6 + 1) * 2 = 14
    assert agent.gsp_network_input == 14
    states = agent.make_gsp_states(_PROX, _PREV_GSP, cyl_bearing_delta=delta)
    for s in states:
        assert len(s) == agent.gsp_network_input


# ── Arg-write contract: the value flows through unchanged ─────────────────────

def test_delta_written_from_arg_per_agent():
    """The delta dim equals the value passed via cyl_bearing_delta, per agent.

    Uses a 2-agent-style contract on the 4-agent fixture: assert agent 0 gets
    0.05 and agent 1 gets -0.03 at the self-slot delta position (index 2, right
    after avg_prox and prev_gsp; cyl_rel off).
    """
    agent = _make_agent({"GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True})
    delta = {'delta': [0.05, -0.03, 0.11, -0.07]}
    states = agent.make_gsp_states(_PROX, _PREV_GSP, cyl_bearing_delta=delta)
    # Self slot layout: [avg_prox(0), prev_gsp(1), bearing_delta(2), n0..]
    assert states[0][2] == pytest.approx(0.05, abs=1e-9)
    assert states[1][2] == pytest.approx(-0.03, abs=1e-9)
    assert states[2][2] == pytest.approx(0.11, abs=1e-9)
    assert states[3][2] == pytest.approx(-0.07, abs=1e-9)


def test_delta_none_writes_zero():
    """cyl_bearing_delta=None → the delta dim is 0.0; size unchanged."""
    agent = _make_agent({"GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True})
    states = agent.make_gsp_states(_PROX, _PREV_GSP, cyl_bearing_delta=None)
    for s in states:
        assert len(s) == agent.gsp_network_input
        assert s[2] == pytest.approx(0.0, abs=1e-12)


def test_delta_position_after_cyl_rel():
    """With CYL_REL on, delta lands immediately after the cyl_rel pair.

    Layout: [avg_prox(0), prev_gsp(1), dist_cyl(2), ang_cyl(3), bearing_delta(4), ...]
    """
    agent = _make_agent({
        "GSP_INPUT_INCLUDE_CYL_REL": True,
        "GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True,
    })
    env = _env_obs([0.30, 0.30, 0.30, 0.30])
    delta = {'delta': [0.20, 0.21, 0.22, 0.23]}
    states = agent.make_gsp_states(
        _PROX, _PREV_GSP, env_observations=env, cyl_bearing_delta=delta)
    s0 = states[0]
    # cyl_rel pair still correct
    assert s0[2] == pytest.approx(float(env[0][4]), abs=1e-5)  # dist_to_cyl
    assert s0[3] == pytest.approx(float(env[0][5]), abs=1e-5)  # angle_to_cyl
    # bearing delta at position 4 (right after cyl_rel), from the arg
    assert s0[4] == pytest.approx(0.20, abs=1e-9)
    assert states[1][4] == pytest.approx(0.21, abs=1e-9)


# ── World-frame wrap-safe computation (mirrors the Main.py block) ──────────────

def _worldframe_bearing_delta(robot_xy, cyl_xy, prev_bearing):
    """Reference implementation of the Main.py per-step computation.

    Returns (deltas, bearings_now). prev_bearing is None on the first step.
    """
    bearings_now = []
    deltas = []
    for (rx, ry) in robot_xy:
        b = math.atan2(ry - cyl_xy[1], rx - cyl_xy[0])
        if prev_bearing is None:
            d = 0.0
        else:
            d = b - prev_bearing[len(bearings_now)]
            d = (d + math.pi) % (2 * math.pi) - math.pi
        bearings_now.append(b)
        deltas.append(d)
    return deltas, bearings_now


def test_worldframe_bearing_uses_world_positions_not_body_frame():
    """The feature is the world-frame bearing atan2(ry-cy, rx-cx), NOT env_obs[5]."""
    cyl = (0.0, 0.0)
    # Robot due east of the cylinder → bearing 0; due north → bearing +pi/2.
    deltas0, prev = _worldframe_bearing_delta([(1.0, 0.0)], cyl, None)
    assert deltas0[0] == pytest.approx(0.0)          # first step
    # Move to due north: bearing goes 0 → +pi/2, delta = +pi/2.
    deltas1, _ = _worldframe_bearing_delta([(0.0, 1.0)], cyl, prev)
    assert deltas1[0] == pytest.approx(math.pi / 2, abs=1e-9)


def test_worldframe_bearing_wrap_safe_across_pi():
    """Crossing the ±π branch cut yields the SMALL wrapped delta, not ~±2π."""
    cyl = (0.0, 0.0)
    eps = 0.05
    # Just below +pi (2nd quadrant, near the cut) then just above -pi (3rd quadrant).
    p_below = (math.cos(math.pi - eps), math.sin(math.pi - eps))
    p_above = (math.cos(-math.pi + eps), math.sin(-math.pi + eps))
    _, prev = _worldframe_bearing_delta([p_below], cyl, None)
    deltas, _ = _worldframe_bearing_delta([p_above], cyl, prev)
    # True angular step is +2*eps across the branch cut, not ~-2*pi.
    assert deltas[0] == pytest.approx(2 * eps, abs=1e-6)
    assert abs(deltas[0]) < 0.2
    assert deltas[0] > 0.0


def test_worldframe_arg_matches_agent_write():
    """End-to-end: feed the reference-computed delta arg into make_gsp_states and
    confirm the agent writes exactly that value."""
    agent = _make_agent({"GSP_INPUT_INCLUDE_CYL_BEARING_DELTA": True})
    cyl = (0.5, -0.3)
    robots_t0 = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    robots_t1 = [(1.1, 0.2), (0.1, 1.2), (-1.2, 0.1), (0.05, -1.1)]
    _, prev = _worldframe_bearing_delta(robots_t0, cyl, None)
    deltas, _ = _worldframe_bearing_delta(robots_t1, cyl, prev)
    states = agent.make_gsp_states(
        _PROX, _PREV_GSP, cyl_bearing_delta={'delta': deltas})
    for i in range(4):
        assert states[i][2] == pytest.approx(deltas[i], abs=1e-9)

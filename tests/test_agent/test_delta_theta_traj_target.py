"""Tests for the horizon-K payload-rotation TRAJECTORY GSP target
(delta_theta_traj).

Motivation (2026-07-05): the default target `delta_theta_1d` predicts the
payload's rotation ONE step ahead (K=1). A one-step rotation is nearly
self-computable by the actor (redundant with the observation), so ablations
show it is causally inert — the actor ignores it. Predicting the whole
anticipated rotation PATH over the next K steps requires integrating the
payload dynamics and the OTHER robots' actions, so it is non-redundant. The
head predicts and SHARES the size-K path: neighbors ingest each other's size-K
prediction and the actor Q-net head ingests it too.

Label semantics (at maturity step t+K, SAME K-vector for every robot):
    label = [ Δθ(t→t+1), Δθ(t+1→t+2), …, Δθ(t+K-1→t+K) ]
where each per-step Δθ is wrap-safe to [-180,180) degrees and
K = GSP_PREDICTION_HORIZON. This is a DELAYED / future label reusing the same
K-step FIFO as future_prox / neighbor_force — push the payload angle (and the
per-robot state snapshot) at every step; at maturity the pop returns the
ordered K+1-angle window [angle(t), …, angle(t+K)] so consecutive entries can
be differenced.

Output size is COUPLED to the horizon: gsp_output_size_effective == K ==
GSP_PREDICTION_HORIZON. The registry stores None for this kind and resolves the
size from GSP_PREDICTION_HORIZON (in BOTH agent.py and GSP-RL, from the same
config key, so the head output width and the actor/neighbor input width agree).

Tests:
  (a) delta_theta_traj is an accepted GSP_OUTPUT_KIND with output size K, and
      'delta_theta_traj' is a delayed-label target;
  (b) the size-K output is wired through to the actor input width and the
      neighbor-sharing width (input size scales with K);
  (c) with a synthetic angle sequence and K=3, the matured label is the correct
      3-vector of consecutive wrapped per-step rotations for the right
      (state_{t-K}, window_t) pairing, shared across robots;
  (d) each per-step rotation is wraparound-handled (179→-179 in one step is a
      small delta ~+2, not ~358);
  (e) K=1 reduces to a 1-vector single-step rotation;
  (f) legacy delta_theta_1d behavior is unchanged (FIFO is a strict no-op).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src"))

from agent import Agent  # noqa: E402
from env import angle_normalize_signed_deg  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_config(**overrides):
    cfg = {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }
    cfg.update(overrides)
    return cfg


def _make_agent(gsp_output_kind="delta_theta_1d", prediction_target=None, K=5,
                n_agents=4, n_hop_neighbors=1):
    cfg = _base_config(
        GSP_OUTPUT_KIND=gsp_output_kind,
        GSP_PREDICTION_HORIZON=K,
    )
    if prediction_target is not None:
        cfg["GSP_PREDICTION_TARGET"] = prediction_target
    return Agent(
        config=cfg,
        network="DDQN",
        n_agents=n_agents,
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
        gsp_input_size=6,   # overridden in __init__ when neighbors=True
        gsp_output_size=1,  # overridden by GSP_OUTPUT_KIND
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
        n_hop_neighbors=n_hop_neighbors,
    )


def _consecutive_wrapped(angles):
    """Reference size-(len-1) per-step wrapped-rotation trajectory."""
    return np.array([
        angle_normalize_signed_deg(float(angles[k + 1]) - float(angles[k]))
        for k in range(len(angles) - 1)
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# (a) accepted kind + delayed-label registration + size == K
# ---------------------------------------------------------------------------

def test_delta_theta_traj_output_size_equals_horizon():
    for K in (1, 2, 3, 5):
        agent = _make_agent(gsp_output_kind="delta_theta_traj", K=K)
        assert agent.gsp_output_kind == "delta_theta_traj"
        assert agent.gsp_output_size_effective == K, f"K={K}"
        assert agent.gsp_network_output == K, f"K={K}"


def test_delta_theta_traj_is_delayed_label_target():
    assert "delta_theta_traj" in Agent._DELAYED_LABEL_TARGETS
    agent = _make_agent(
        gsp_output_kind="delta_theta_traj",
        prediction_target="delta_theta_traj",
        K=3,
    )
    assert agent._is_delayed_label_target() is True


def test_unknown_kind_still_raises():
    with pytest.raises(ValueError, match="Unknown GSP_OUTPUT_KIND"):
        _make_agent(gsp_output_kind="not_a_real_kind_xyz")


def test_horizon_zero_rejected():
    with pytest.raises(ValueError, match="GSP_PREDICTION_HORIZON"):
        _make_agent(gsp_output_kind="delta_theta_traj", K=0)


# ---------------------------------------------------------------------------
# (b) size-K wired through actor input width + neighbor-sharing width
# ---------------------------------------------------------------------------

def test_input_size_scales_with_K():
    """GSP-N per-agent layout: input = (1+K) self + (1+K) per neighbor slot.
    With n_hop_neighbors=1 there are 2 neighbor slots → input = (1+K)*3.
    K=1 gives the legacy 6; K=3 gives 12; the neighbor-sharing width scales."""
    for K, expected in ((1, 6), (2, 9), (3, 12), (5, 18)):
        agent = _make_agent(
            gsp_output_kind="delta_theta_traj", K=K, n_hop_neighbors=1
        )
        assert agent.gsp_network_input == expected, (
            f"K={K}: expected input {expected}, got {agent.gsp_network_input}"
        )


def test_actor_input_grows_by_K():
    """The actor's augmented observation grows by the GSP output width (K)."""
    a1 = _make_agent(gsp_output_kind="delta_theta_1d", K=1)   # scalar head, width 1
    a3 = _make_agent(gsp_output_kind="delta_theta_traj", K=3)  # width 3
    # network_input_size = base obs + gsp_network_output.
    assert a3.network_input_size - a1.network_input_size == (3 - 1)


# ---------------------------------------------------------------------------
# (c) matured label == size-K trajectory of consecutive wrapped rotations
# ---------------------------------------------------------------------------

def test_fifo_returns_none_until_K_plus_one_pushes():
    agent = _make_agent(
        gsp_output_kind="delta_theta_traj",
        prediction_target="delta_theta_traj",
        K=3,
    )
    n = agent.n_agents
    for t in range(3):  # K=3 -> need K+1=4 pushes before first maturity
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states, payload_angle_deg=float(t))
        matured = agent.pop_matured_gsp_label(None)
        assert matured is None, f"should not mature before K+1 pushes; t={t}"


def test_matured_label_is_size_K_trajectory():
    """Drive a synthetic payload-angle sequence. The matured label (paired with
    the state pushed at t) must equal the size-K vector of consecutive wrapped
    per-step rotations over [t, t+1, …, t+K], and be the SAME K-vector for every
    robot."""
    K = 3
    n = 4
    agent = _make_agent(
        gsp_output_kind="delta_theta_traj",
        prediction_target="delta_theta_traj",
        K=K,
        n_agents=n,
    )
    angle_seq = [10.0, 25.0, 47.0, 80.0, 120.0, 175.0]

    matured_by_state_val = {}
    for t in range(len(angle_seq)):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states, payload_angle_deg=angle_seq[t])
        matured = agent.pop_matured_gsp_label(None)
        if matured is not None:
            win = matured["payload_angle_window"]
            traj = _consecutive_wrapped(win)
            label = [traj.copy() for _ in range(n)]
            state_val = float(matured["state_per_robot"][0][0])
            matured_by_state_val[state_val] = label

    for src_t in (0, 1, 2):
        mat_t = src_t + K
        assert src_t in matured_by_state_val, (
            f"state from t={src_t} never matured (expected at t={mat_t})"
        )
        expected = _consecutive_wrapped(angle_seq[src_t:mat_t + 1])
        assert expected.shape == (K,)
        # Same K-vector for every robot.
        for i in range(n):
            np.testing.assert_allclose(
                matured_by_state_val[src_t][i], expected, rtol=0, atol=1e-5,
                err_msg=(
                    f"robot {i} label for state@t={src_t} should be the size-{K} "
                    f"consecutive-rotation trajectory over [{src_t}..{mat_t}]"
                ),
            )
        # Spot-check the first element == first per-step rotation.
        assert matured_by_state_val[src_t][0][0] == pytest.approx(
            angle_normalize_signed_deg(angle_seq[src_t + 1] - angle_seq[src_t]),
            abs=1e-5,
        )


def test_K1_reduces_to_single_step_trajectory():
    """K=1 -> label is a 1-vector holding the single-step rotation."""
    K = 1
    n = 4
    agent = _make_agent(
        gsp_output_kind="delta_theta_traj",
        prediction_target="delta_theta_traj",
        K=K,
        n_agents=n,
    )
    angle_seq = [30.0, 34.0, 39.0, 45.0]
    got = {}
    for t in range(len(angle_seq)):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states, payload_angle_deg=angle_seq[t])
        matured = agent.pop_matured_gsp_label(None)
        if matured is not None:
            win = matured["payload_angle_window"]
            got[float(matured["state_per_robot"][0][0])] = _consecutive_wrapped(win)
    for src_t in (0, 1, 2):
        expected = _consecutive_wrapped(angle_seq[src_t:src_t + 2])
        assert expected.shape == (1,)
        np.testing.assert_allclose(got[src_t], expected, rtol=0, atol=1e-5)


# ---------------------------------------------------------------------------
# (d) per-step wraparound handling
# ---------------------------------------------------------------------------

def test_per_step_wraparound_is_small_delta():
    """A per-step 179 -> -179 crossing must yield a small per-step rotation
    (~+2), never ~358, within the trajectory vector."""
    K = 2
    n = 3
    agent = _make_agent(
        gsp_output_kind="delta_theta_traj",
        prediction_target="delta_theta_traj",
        K=K,
        n_agents=n,
    )
    # t=0:170, t=1:179, t=2:-179 → per-step rotations [+9, +2]; the second step
    # crosses the +/-180 boundary and must wrap to +2, not -358.
    angle_seq = [170.0, 179.0, -179.0, -170.0]
    labels = {}
    for t in range(len(angle_seq)):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states, payload_angle_deg=angle_seq[t])
        matured = agent.pop_matured_gsp_label(None)
        if matured is not None:
            win = matured["payload_angle_window"]
            labels[float(matured["state_per_robot"][0][0])] = _consecutive_wrapped(win)
    assert 0.0 in labels
    traj = labels[0.0]
    assert traj.shape == (K,)
    assert traj[0] == pytest.approx(9.0, abs=1e-5)
    assert traj[1] == pytest.approx(2.0, abs=1e-5)  # boundary crossing wrapped
    assert np.all(np.abs(traj) < 20.0), "no per-step delta should be ~358"


def test_per_step_wraparound_negative_direction():
    """-179 -> 179 in one step is a small NEGATIVE per-step rotation (~-2)."""
    K = 2
    n = 3
    agent = _make_agent(
        gsp_output_kind="delta_theta_traj",
        prediction_target="delta_theta_traj",
        K=K,
        n_agents=n,
    )
    angle_seq = [-170.0, -179.0, 179.0, 170.0]
    labels = {}
    for t in range(len(angle_seq)):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states, payload_angle_deg=angle_seq[t])
        matured = agent.pop_matured_gsp_label(None)
        if matured is not None:
            win = matured["payload_angle_window"]
            labels[float(matured["state_per_robot"][0][0])] = _consecutive_wrapped(win)
    assert 0.0 in labels
    traj = labels[0.0]
    assert traj[0] == pytest.approx(-9.0, abs=1e-5)
    assert traj[1] == pytest.approx(-2.0, abs=1e-5)


# ---------------------------------------------------------------------------
# (f) legacy delta_theta unchanged
# ---------------------------------------------------------------------------

def test_fifo_noop_for_default_target():
    """Default target (delta_theta) -> push/pop are strict no-ops even with the
    delta_theta_traj output kind selected."""
    agent = _make_agent(gsp_output_kind="delta_theta_traj", K=3)
    assert agent.gsp_prediction_target == "delta_theta"
    assert agent._is_delayed_label_target() is False
    n = agent.n_agents
    for _ in range(10):
        states = [np.zeros(8, dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states, payload_angle_deg=1.0)
        assert agent.pop_matured_gsp_label(None) is None


def test_delta_theta_1d_output_size_unchanged():
    """Legacy delta_theta_1d still has output size 1 regardless of horizon."""
    agent = _make_agent(gsp_output_kind="delta_theta_1d", K=7)
    assert agent.gsp_network_output == 1
    assert agent.gsp_network_input == 6  # legacy GSP-N width


def test_push_without_angle_still_works_for_neighbor_force():
    """The payload_angle_deg kwarg is optional; neighbor_force (which does not
    pass it) keeps working, and the returned window is all-None."""
    K = 2
    n = 3
    agent = _make_agent(
        gsp_output_kind="neighbor_force_1d",
        prediction_target="neighbor_force",
        K=K,
        n_agents=n,
    )
    for t in range(K + 1):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states)  # no payload_angle_deg
    label_now = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    matured = agent.pop_matured_gsp_label(label_now)
    assert matured is not None
    assert matured["payload_angle_deg"] is None
    assert all(a is None for a in matured["payload_angle_window"])
    np.testing.assert_array_equal(matured["label_per_robot"], label_now)

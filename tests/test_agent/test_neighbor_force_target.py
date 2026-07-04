"""Tests for the neighbor-force GSP prediction target (neighbor_force_1d).

Motivation (2026-07-04 target-relevance analysis): the mean applied
force-magnitude of the OTHER robots K steps in the future correlates with
per-robot reward-to-go at |0.33| (n=3), versus the current default delta_theta
at 0.06 — i.e. neighbor-future-force is the coordination-relevant quantity a
coordination method should learn to predict.

Label semantics (for robot i, at maturity step t+K):
    label_i = mean_{j != i} force_magnitude[t+K, j]
where K = GSP_PREDICTION_HORIZON. This is a DELAYED / future label: it reuses
the exact FIFO mechanism that `future_prox` uses — push the per-robot state
snapshot at t, and K steps later pop it paired with the label observed at t+K.
The ONLY difference from future_prox is the VALUE that fills the label slot
(neighbor mean force magnitude instead of the robot's own proximity).

Tests:
  (a) neighbor_force_1d is an accepted GSP_OUTPUT_KIND with output size 1;
  (b) driving a synthetic sequence with known per-robot force magnitudes, the
      matured label for robot i equals the mean of the OTHER robots' force
      magnitudes K steps later;
  (c) an unknown GSP_OUTPUT_KIND still raises ValueError.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src"))

from agent import Agent  # noqa: E402


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
                n_agents=4):
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
        n_hop_neighbors=1,
    )


def _neighbor_mean_force(force_mags):
    """Reference label: for each robot i, mean of the OTHER robots' force mags."""
    force_mags = np.asarray(force_mags, dtype=np.float32)
    n = len(force_mags)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        others = np.delete(force_mags, i)
        out[i] = float(np.mean(others))
    return out


# ---------------------------------------------------------------------------
# (a) accepted kind with size 1
# ---------------------------------------------------------------------------

def test_neighbor_force_1d_is_accepted_kind_size_1():
    agent = _make_agent(gsp_output_kind="neighbor_force_1d")
    assert agent.gsp_output_kind == "neighbor_force_1d"
    assert agent.gsp_output_size_effective == 1
    assert agent.gsp_network_output == 1


def test_neighbor_force_1d_input_size_unchanged():
    """K=1 output → gsp_input_size is the legacy 6 (same as delta_theta_1d)."""
    agent = _make_agent(gsp_output_kind="neighbor_force_1d")
    assert agent.gsp_network_input == 6


# ---------------------------------------------------------------------------
# (c) unknown kind still raises
# ---------------------------------------------------------------------------

def test_unknown_kind_still_raises():
    with pytest.raises(ValueError, match="Unknown GSP_OUTPUT_KIND"):
        _make_agent(gsp_output_kind="not_a_real_kind_xyz")


# ---------------------------------------------------------------------------
# (b) matured label == neighbor mean force K steps later
# ---------------------------------------------------------------------------

def test_fifo_active_for_neighbor_force_target():
    """The delayed-label FIFO must activate for GSP_PREDICTION_TARGET='neighbor_force'
    (parallel to 'future_prox'), and be a no-op for the default target."""
    agent = _make_agent(
        gsp_output_kind="neighbor_force_1d",
        prediction_target="neighbor_force",
        K=3,
    )
    n = agent.n_agents
    # Below K+1 pushes → no maturation yet.
    for t in range(3):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states)
        matured = agent.pop_matured_gsp_label(np.zeros(n, dtype=np.float32))
        assert matured is None, f"should not mature before K+1 pushes; t={t}"


def test_fifo_noop_for_default_target():
    """Default target (delta_theta) → push/pop are strict no-ops even with the
    neighbor_force_1d output kind selected."""
    agent = _make_agent(gsp_output_kind="neighbor_force_1d", K=3)
    n = agent.n_agents
    for _ in range(10):
        states = [np.zeros(8, dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states)
        assert agent.pop_matured_gsp_label(np.zeros(n, dtype=np.float32)) is None


def test_matured_label_is_neighbor_mean_force_K_steps_later():
    """Drive a synthetic sequence with known per-robot force magnitudes. The
    matured label for robot i (paired with the state pushed at t) must equal the
    mean of the OTHER robots' force magnitudes at t+K.

    The Main.py driver, at maturity step t+K, computes the per-robot
    neighbor-mean-force vector and passes it to pop_matured_gsp_label as the
    label. This test replicates that contract at the Agent level.
    """
    K = 3
    n = 4
    agent = _make_agent(
        gsp_output_kind="neighbor_force_1d",
        prediction_target="neighbor_force",
        K=K,
        n_agents=n,
    )

    # A per-step per-robot force-magnitude sequence with distinct values so the
    # neighbor-mean is uniquely identifiable.  force_seq[t][j] = force mag of
    # robot j at step t.
    force_seq = [
        [1.0, 2.0, 3.0, 4.0],    # t=0
        [10.0, 20.0, 30.0, 40.0],  # t=1
        [0.5, 1.5, 2.5, 3.5],    # t=2
        [100.0, 200.0, 300.0, 400.0],  # t=3  (matures state from t=0)
        [11.0, 13.0, 17.0, 19.0],  # t=4  (matures state from t=1)
        [2.0, 4.0, 6.0, 8.0],    # t=5  (matures state from t=2)
    ]

    matured_by_state_val = {}
    for t in range(len(force_seq)):
        # State snapshot tagged with t so we can recover which step it came from.
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states)
        # At maturity, the driver passes the per-robot neighbor-mean force of the
        # CURRENT step (t) as the label.
        label_now = _neighbor_mean_force(force_seq[t])
        matured = agent.pop_matured_gsp_label(label_now)
        if matured is not None:
            state_val = float(matured["state_per_robot"][0][0])
            matured_by_state_val[state_val] = np.asarray(
                matured["label_per_robot"], dtype=np.float32
            )

    # State pushed at t matures at t+K.  Check the three matured pairs.
    for src_t in (0, 1, 2):
        mat_t = src_t + K
        assert src_t in matured_by_state_val, (
            f"state from t={src_t} never matured (expected at t={mat_t})"
        )
        expected = _neighbor_mean_force(force_seq[mat_t])
        np.testing.assert_allclose(
            matured_by_state_val[src_t], expected, rtol=0, atol=1e-6,
            err_msg=(
                f"robot label for state@t={src_t} should be neighbor-mean force "
                f"at t={mat_t}"
            ),
        )
        # Spot-check the semantics for robot 0 explicitly: mean of robots 1,2,3.
        others_mean = float(np.mean(force_seq[mat_t][1:]))
        assert matured_by_state_val[src_t][0] == pytest.approx(others_mean, abs=1e-6)

"""Tests for the future-prox delayed-label buffer.

Candidate A retargets the GSP head to predict each robot's own proximity
K steps in the future. Because the label at step t isn't known until step
t+K, the agent maintains a FIFO buffer of (state, gsp_obs) snapshots and
emits matured (state_t, gsp_obs_t, prox_{t+K}) tuples once K more steps
have elapsed.

The buffer:
  - is per-agent indexed (each robot's own gsp_obs and state get paired
    with that same robot's own future prox).
  - is empty/no-op for legacy GSP_PREDICTION_TARGET='delta_theta' runs.
  - is reset at episode boundaries so labels never bleed across resets.

All tests assume GSP_PREDICTION_TARGET='future_prox' and
GSP_PREDICTION_HORIZON=K=3 (small K for clean assertions).
"""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src"))

from agent import Agent  # noqa: E402


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


def _load_base_config():
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "GSP-RL",
        "tests", "test_actor", "config.yml",
    )
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def _make_future_prox_agent(K=3):
    config = _load_base_config()
    config["GSP_PREDICTION_TARGET"] = "future_prox"
    config["GSP_PREDICTION_HORIZON"] = K
    return Agent(**_base_agent_kwargs(config))


# ---- attribute / config plumbing -----------------------------------------

def test_default_target_is_delta_theta():
    config = _load_base_config()
    agent = Agent(**_base_agent_kwargs(config))
    assert agent.gsp_prediction_target == "delta_theta"
    assert agent.gsp_prediction_horizon == 5


def test_future_prox_flags_picked_up():
    agent = _make_future_prox_agent(K=3)
    assert agent.gsp_prediction_target == "future_prox"
    assert agent.gsp_prediction_horizon == 3


# ---- buffer mechanics ----------------------------------------------------

def test_buffer_returns_none_until_K_plus_one_pushes():
    """K-step lookahead semantics: label at t+K paired with state at t.
    Buffer must hold K+1 entries before the oldest matures.
    For K=3: pushes at t=0,1,2,3 → pop after the 4th push returns t=0 paired with prox at t=3.
    """
    agent = _make_future_prox_agent(K=3)
    n = agent.n_agents
    # 3 pushes (K=3): not enough yet — need K+1=4 to start maturing.
    for t in range(3):
        states = [np.full(8, t, dtype=np.float32) for _ in range(n)]
        gobs = [np.full(6, t, dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, gobs)
        matured = agent.pop_matured_gsp_label(np.zeros(n, dtype=np.float32))
        assert matured is None, f"Should not mature before K+1 pushes; got {matured} at t={t}"


def test_buffer_returns_oldest_after_K_plus_one_pushes():
    """After K+1 pushes (s_0..s_K), pop returns s_0 paired with prox at t=K (i.e. current prox)."""
    K = 3
    agent = _make_future_prox_agent(K=K)
    n = agent.n_agents
    # Push K+1=4 entries
    for t in range(K + 1):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        gobs = [np.full(6, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, gobs)
    current_prox = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    matured = agent.pop_matured_gsp_label(current_prox)
    assert matured is not None
    assert "state_per_robot" in matured
    assert "gsp_obs_per_robot" in matured
    assert "label_per_robot" in matured
    # state_per_robot[i] should be the t=0 push — full of 0.0
    np.testing.assert_array_equal(matured["state_per_robot"][0], np.full(8, 0.0, dtype=np.float32))
    np.testing.assert_array_equal(matured["gsp_obs_per_robot"][0], np.full(6, 0.0, dtype=np.float32))
    np.testing.assert_array_equal(matured["label_per_robot"], current_prox)


def test_buffer_FIFO_ordering_under_pop_after_push():
    """At each step we push then attempt to pop. With K=2:
    - t=0 push, pop None (buf=1, need 3)
    - t=1 push, pop None (buf=2, need 3)
    - t=2 push, pop returns s_0 (buf=3 → pop oldest, leaving 2)
    - t=3 push, pop returns s_1 (buf=3 again → pop)
    """
    K = 2
    agent = _make_future_prox_agent(K=K)
    n = agent.n_agents
    for t in range(4):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        gobs = [np.full(6, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, gobs)
        prox_arr = np.full(n, float(t) * 10, dtype=np.float32)
        matured = agent.pop_matured_gsp_label(prox_arr)
        if t < K:  # pushes 0..K-1: buffer below K+1
            assert matured is None, f"t={t}: expected None, got {matured}"
        else:
            expected_state_val = float(t - K)
            np.testing.assert_array_equal(
                matured["state_per_robot"][0],
                np.full(8, expected_state_val, dtype=np.float32),
            )
            np.testing.assert_array_equal(matured["label_per_robot"], prox_arr)


def test_reset_buffer_clears_pending():
    agent = _make_future_prox_agent(K=3)
    n = agent.n_agents
    for _ in range(5):
        states = [np.zeros(8, dtype=np.float32) for _ in range(n)]
        gobs = [np.zeros(6, dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, gobs)
    agent.reset_gsp_label_buffer()
    # After reset, popping immediately should return None
    matured = agent.pop_matured_gsp_label(np.zeros(n, dtype=np.float32))
    assert matured is None


def test_buffer_no_op_for_delta_theta_mode():
    """In legacy delta_theta mode, push/pop are no-ops (return None)."""
    config = _load_base_config()
    agent = Agent(**_base_agent_kwargs(config))  # default delta_theta
    n = agent.n_agents
    for _ in range(10):
        states = [np.zeros(8, dtype=np.float32) for _ in range(n)]
        gobs = [np.zeros(6, dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, gobs)
        matured = agent.pop_matured_gsp_label(np.zeros(n, dtype=np.float32))
        assert matured is None

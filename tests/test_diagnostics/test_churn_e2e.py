"""Integration test: churn diagnostic fires end-to-end.

Verifies that the start-of-episode / end-of-episode state_dict snapshot
pattern produces ``diag_actor_churn_*`` and ``diag_gsp_churn_*`` keys in
the compute_diagnostics() output when DIAGNOSE_CHURN is enabled.

The test:
  - Constructs a minimal DDPG + GSP-N agent (1 robot, 5-dim obs, 2-dim action,
    4-dim GSP input).
  - Stuffs the replay buffers with synthetic random transitions so learn() can
    fire on every step.
  - Takes a "before" snapshot, runs several learn steps (which updates the
    weights), takes an "after" snapshot.
  - Freezes a diagnostic eval batch and calls compute_diagnostics() with both
    snapshots.
  - Asserts diag_actor_churn_output and diag_gsp_churn_output are in the result
    and are finite non-NaN floats. Also asserts they are non-zero (the network
    had at least one update, so weights changed).
  - Runs again with DIAGNOSE_CHURN=False and asserts no churn keys appear.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.agent import Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(diagnose_churn: bool = True) -> Agent:
    """Build the smallest viable DDPG + GSP-N agent for testing."""
    config = {
        # Core RL hyperparameters
        "GAMMA": 0.99,
        "TAU": 0.005,
        "ALPHA": 0.001,
        "BETA": 0.001,
        "LR": 0.001,
        "EPSILON": 0.0,
        "EPS_MIN": 0.0,
        "EPS_DEC": 0.0,
        "BATCH_SIZE": 8,
        "MEM_SIZE": 200,
        "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0,
        "UPDATE_ACTOR_ITER": 1,
        "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 1,
        "GSP_BATCH_SIZE": 8,
        # Diagnostics
        "DIAGNOSTICS_ENABLED": True,
        "DIAGNOSTICS_FREEZE_EPISODE": 0,
        "DIAGNOSTICS_CADENCE": 1,
        "DIAGNOSTICS_BATCH_SIZE": 16,
        "DIAGNOSE_GRAD_ZERO": False,  # keep test fast — only churn matters
        "DIAGNOSE_KFAC": False,
        "DIAGNOSE_CHURN": diagnose_churn,
    }
    agent = Agent(
        config=config,
        network="DDPG",
        n_agents=1,
        n_obs=5,
        n_actions=2,
        options_per_action=3,
        id=0,
        min_max_action=1.0,
        meta_param_size=1,
        gsp=True,
        recurrent=False,
        attention=False,
        neighbors=True,   # GSP-N: gsp_input_size will be 2+2*(1*2)=6
        gsp_input_size=6,
        gsp_output_size=1,
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
        broadcast=False,
        prox_filter_angle_deg=45.0,
    )
    return agent


def _fill_replay(agent: Agent, n: int = 50) -> None:
    """Push ``n`` random SARSD transitions into both replay buffers."""
    obs_dim = agent.network_input_size
    gsp_dim = agent.gsp_network_input
    act_dim = agent.output_size
    rng = np.random.default_rng(42)

    for _ in range(n):
        s = rng.standard_normal(obs_dim).astype(np.float32)
        s_ = rng.standard_normal(obs_dim).astype(np.float32)
        a_num = rng.standard_normal(act_dim).astype(np.float32)
        a_take = a_num
        r = float(rng.standard_normal())
        agent.store_agent_transition(s, (a_num, a_take), r, s_, False)

        g_s = rng.standard_normal(gsp_dim).astype(np.float32)
        g_s_ = rng.standard_normal(gsp_dim).astype(np.float32)
        label = float(rng.uniform(-1, 1))
        agent.store_gsp_transition(g_s, label, 0.0, g_s_, 0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_churn_keys_present_and_nonzero():
    """After at least one learn step, churn keys must appear in diagnostics output
    and be non-zero finite floats (weight update != zero => activations changed)."""
    agent = _make_agent(diagnose_churn=True)
    _fill_replay(agent, n=80)

    # Freeze the diagnostic eval batch directly from the replay buffer
    # (mirrors what Main.py does via freeze_diagnostic_batch).
    # We need at least diagnostics_batch_size (16) samples in the GSP pool too.
    rng = np.random.default_rng(1)
    gsp_pool = rng.standard_normal(
        (agent.diagnostics_batch_size, agent.gsp_network_input)
    ).astype(np.float32)
    agent.freeze_diagnostic_batch(gsp_obs_pool=gsp_pool)
    assert agent.diag_actor_eval_batch is not None, "Actor eval batch not frozen"
    assert agent.diag_gsp_eval_batch is not None, "GSP eval batch not frozen"

    # Take the "before" snapshot (mirrors episode start in Main.py).
    actor_net = agent._main_network(agent.networks)
    gsp_net = agent._main_network(agent.gsp_networks)
    before_actor = copy.deepcopy(actor_net.state_dict())
    before_gsp = copy.deepcopy(gsp_net.state_dict())

    # Run several learn steps so both actor and GSP weights change.
    for _ in range(10):
        agent.learn()

    # Take the "after" snapshot (mirrors just before compute_diagnostics in Main.py).
    after_actor = copy.deepcopy(actor_net.state_dict())
    after_gsp = copy.deepcopy(gsp_net.state_dict())

    # Run diagnostics with both snapshots.
    result = agent.compute_diagnostics(
        actor_before_state_dict=before_actor,
        actor_after_state_dict=after_actor,
        gsp_before_state_dict=before_gsp,
        gsp_after_state_dict=after_gsp,
    )

    assert result, "compute_diagnostics returned empty dict — diagnostics not enabled or batch not frozen"

    # Actor churn key must be present.
    assert "diag_actor_churn_output" in result, (
        f"diag_actor_churn_output missing from result keys: {sorted(result.keys())}"
    )
    # GSP churn key must be present.
    assert "diag_gsp_churn_output" in result, (
        f"diag_gsp_churn_output missing from result keys: {sorted(result.keys())}"
    )

    actor_churn = result["diag_actor_churn_output"]
    gsp_churn = result["diag_gsp_churn_output"]

    # Values must be finite.
    assert np.isfinite(actor_churn), f"diag_actor_churn_output is not finite: {actor_churn}"
    assert np.isfinite(gsp_churn), f"diag_gsp_churn_output is not finite: {gsp_churn}"

    # Values must be non-zero (network updated, so churn > 0).
    assert actor_churn > 0.0, (
        f"diag_actor_churn_output is zero — weights did not change during learn steps"
    )
    assert gsp_churn > 0.0, (
        f"diag_gsp_churn_output is zero — GSP weights did not change during learn steps"
    )


def test_churn_absent_when_flag_disabled():
    """With DIAGNOSE_CHURN=False, no churn keys should appear even when snapshots
    are provided. Backward compat guarantee for legacy runs."""
    agent = _make_agent(diagnose_churn=False)
    _fill_replay(agent, n=80)

    rng = np.random.default_rng(2)
    gsp_pool = rng.standard_normal(
        (agent.diagnostics_batch_size, agent.gsp_network_input)
    ).astype(np.float32)
    agent.freeze_diagnostic_batch(gsp_obs_pool=gsp_pool)

    actor_net = agent._main_network(agent.networks)
    gsp_net = agent._main_network(agent.gsp_networks)
    before_actor = copy.deepcopy(actor_net.state_dict())
    before_gsp = copy.deepcopy(gsp_net.state_dict())

    for _ in range(5):
        agent.learn()

    after_actor = copy.deepcopy(actor_net.state_dict())
    after_gsp = copy.deepcopy(gsp_net.state_dict())

    result = agent.compute_diagnostics(
        actor_before_state_dict=before_actor,
        actor_after_state_dict=after_actor,
        gsp_before_state_dict=before_gsp,
        gsp_after_state_dict=after_gsp,
    )

    churn_keys = [k for k in result if "churn" in k]
    assert churn_keys == [], (
        f"Expected no churn keys with DIAGNOSE_CHURN=False, got: {churn_keys}"
    )


def test_churn_skipped_when_no_snapshots():
    """With DIAGNOSE_CHURN=True but no snapshots passed, churn keys must be absent
    (silently skipped, not NaN or error)."""
    agent = _make_agent(diagnose_churn=True)
    _fill_replay(agent, n=80)

    rng = np.random.default_rng(3)
    gsp_pool = rng.standard_normal(
        (agent.diagnostics_batch_size, agent.gsp_network_input)
    ).astype(np.float32)
    agent.freeze_diagnostic_batch(gsp_obs_pool=gsp_pool)

    for _ in range(5):
        agent.learn()

    # Pass no snapshots — should silently skip churn computation.
    result = agent.compute_diagnostics()

    churn_keys = [k for k in result if "churn" in k]
    assert churn_keys == [], (
        f"Expected no churn keys when snapshots not provided, got: {churn_keys}"
    )

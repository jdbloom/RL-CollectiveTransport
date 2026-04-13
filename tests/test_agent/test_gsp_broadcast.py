"""Tests for GSP-B (full-broadcast variant) state construction.

GSP-B: each agent's input is [self_prox, self_prev_gsp, other_0_prox,
other_0_prev_gsp, other_1_prox, other_1_prev_gsp, ..., other_{n-1}_prox,
other_{n-1}_prev_gsp], length 2*n_agents. Self-first ordering.

Known limitation (inherited from plain GSP): the network input size is
coupled to n_agents, so a trained GSP-B policy only transfers to the same
team size. This is the tradeoff vs GSP-N's fixed (self + n_hop_neighbors)
input which transfers across team sizes.
"""

import numpy as np
import pytest

from src.agent import Agent


BASE_CONFIG = {
    "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.002, "LR": 0.0001,
    "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
    "BATCH_SIZE": 16, "MEM_SIZE": 1000, "REPLACE_TARGET_COUNTER": 10,
    "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 1, "GSP_BATCH_SIZE": 16,
}


def make_agent(n_agents=4, network="DDQN", broadcast=True):
    return Agent(
        config=BASE_CONFIG,
        network=network,
        n_agents=n_agents,
        n_obs=8,
        n_actions=4,
        options_per_action=3,
        id=0,
        min_max_action=1.0,
        meta_param_size=1,
        gsp=True,
        recurrent=False,
        attention=False,
        neighbors=False,
        broadcast=broadcast,
        gsp_input_size=4,  # overridden when broadcast=True
        gsp_output_size=1,
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
    )


def test_broadcast_agent_has_gsp_broadcast_property_true():
    agent = make_agent()
    assert agent.gsp_broadcast is True


def test_broadcast_agent_gsp_input_size_is_two_times_n_agents():
    """For 4 agents, the broadcast input is [self_prox, self_prev_gsp, +3×(prox, prev_gsp)] = 8."""
    agent = make_agent(n_agents=4)
    assert agent.gsp_network_input == 8


def test_broadcast_agent_gsp_input_size_scales_with_n_agents():
    """For 8 agents, input is 16. Known limitation: coupled to team size."""
    agent = make_agent(n_agents=8)
    assert agent.gsp_network_input == 16


def test_make_gsp_states_broadcast_returns_one_state_per_agent():
    agent = make_agent(n_agents=4)
    prox = [0.1, 0.2, 0.3, 0.4]
    prev_gsp = [-0.5, 0.0, 0.25, 0.75]
    states = agent.make_gsp_states_broadcast(prox, prev_gsp)
    assert len(states) == 4
    for s in states:
        assert len(s) == 8


def test_make_gsp_states_broadcast_self_first_ordering():
    """For each agent i, the first two entries must be (prox[i], prev_gsp[i])."""
    agent = make_agent(n_agents=4)
    prox = [0.11, 0.22, 0.33, 0.44]
    prev_gsp = [-0.1, -0.2, -0.3, -0.4]
    states = agent.make_gsp_states_broadcast(prox, prev_gsp)
    for i in range(4):
        assert states[i][0] == pytest.approx(prox[i]), f"agent {i} self_prox"
        assert states[i][1] == pytest.approx(prev_gsp[i]), f"agent {i} self_prev_gsp"


def test_make_gsp_states_broadcast_others_in_order():
    """After the self-pair, the remaining entries are other agents in ascending id order (skipping self)."""
    agent = make_agent(n_agents=4)
    prox = [0.10, 0.20, 0.30, 0.40]
    prev_gsp = [0.01, 0.02, 0.03, 0.04]
    states = agent.make_gsp_states_broadcast(prox, prev_gsp)
    # Agent 0: self=0, others=[1, 2, 3]
    assert list(states[0]) == pytest.approx([0.10, 0.01, 0.20, 0.02, 0.30, 0.03, 0.40, 0.04])
    # Agent 2: self=2, others=[0, 1, 3]
    assert list(states[2]) == pytest.approx([0.30, 0.03, 0.10, 0.01, 0.20, 0.02, 0.40, 0.04])
    # Agent 3: self=3, others=[0, 1, 2]
    assert list(states[3]) == pytest.approx([0.40, 0.04, 0.10, 0.01, 0.20, 0.02, 0.30, 0.03])


def test_broadcast_is_mutually_exclusive_with_neighbors():
    """Can't have both neighbors=True and broadcast=True; they overload gsp_input_size."""
    with pytest.raises((ValueError, AssertionError)):
        Agent(
            config=BASE_CONFIG,
            network="DDQN", n_agents=4, n_obs=8, n_actions=4,
            options_per_action=3, id=0, min_max_action=1.0, meta_param_size=1,
            gsp=True, recurrent=False, attention=False,
            neighbors=True, broadcast=True,
            gsp_input_size=4, gsp_output_size=1,
            gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
        )


def test_plain_gsp_without_broadcast_unchanged():
    """Plain GSP (neighbors=False, broadcast=False) keeps the legacy input size."""
    agent = make_agent(broadcast=False)
    # Should fall through to the config-provided gsp_input_size=4
    assert agent.gsp_network_input == 4
    assert agent.gsp_broadcast is False

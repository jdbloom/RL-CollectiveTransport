"""Tests for ring neighbor topology construction."""

import numpy as np
import pytest
from src.agent import Agent


def _make_agent(agent_config, n_agents):
    return Agent(
        config=agent_config, network='DQN', n_agents=n_agents, n_obs=31,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=True, recurrent=False, attention=False,
        neighbors=True, gsp_input_size=6, gsp_output_size=1,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
    )


class TestBuildNeighbors:
    def test_4_agents_each_has_2_neighbors(self, agent_config):
        agent = _make_agent(agent_config, 4)
        for i in range(4):
            assert len(agent.neighbors_dict[i]) == 2

    def test_4_agents_circular_topology(self, agent_config):
        agent = _make_agent(agent_config, 4)
        assert 3 in agent.neighbors_dict[0]
        assert 1 in agent.neighbors_dict[0]
        assert 2 in agent.neighbors_dict[3]
        assert 0 in agent.neighbors_dict[3]

    def test_2_agents(self, agent_config):
        agent = _make_agent(agent_config, 2)
        assert 1 in agent.neighbors_dict[0]
        assert 0 in agent.neighbors_dict[1]

    def test_8_agents_wraparound(self, agent_config):
        agent = _make_agent(agent_config, 8)
        assert 7 in agent.neighbors_dict[0]
        assert 1 in agent.neighbors_dict[0]
        assert 0 in agent.neighbors_dict[7]
        assert 6 in agent.neighbors_dict[7]

    def test_all_agents_have_entries(self, agent_config):
        agent = _make_agent(agent_config, 4)
        assert set(agent.neighbors_dict.keys()) == {0, 1, 2, 3}

    def test_no_self_in_neighbors(self, agent_config):
        agent = _make_agent(agent_config, 4)
        for i in range(4):
            assert i not in agent.neighbors_dict[i]


class TestMakeGSPStates:
    def test_state_shape(self, agent_config):
        agent = _make_agent(agent_config, 4)
        prox = [0.1, 0.2, 0.3, 0.4]
        prev_gsp = np.array([0.0, 0.0, 0.0, 0.0])
        states = agent.make_gsp_states(prox, prev_gsp)
        assert len(states) == 4
        assert len(states[0]) == 6  # 2 + 2*(1*2)

    def test_own_values_first(self, agent_config):
        agent = _make_agent(agent_config, 4)
        prox = [0.1, 0.2, 0.3, 0.4]
        prev_gsp = np.array([0.5, 0.6, 0.7, 0.8])
        states = agent.make_gsp_states(prox, prev_gsp)
        assert states[0][0] == pytest.approx(0.1)
        assert states[0][1] == pytest.approx(0.5)

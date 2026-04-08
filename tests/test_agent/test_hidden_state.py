"""Tests for per-agent LSTM hidden state management."""

import pytest
from src.agent import Agent


def _make_recurrent_neighbors_agent(agent_config, n_agents):
    return Agent(
        config=agent_config, network='DQN', n_agents=n_agents, n_obs=31,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=True, recurrent=True, attention=False,
        neighbors=True, gsp_input_size=6, gsp_output_size=1,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
    )


def _make_nonrecurrent_neighbors_agent(agent_config, n_agents):
    return Agent(
        config=agent_config, network='DQN', n_agents=n_agents, n_obs=31,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=True, recurrent=False, attention=False,
        neighbors=True, gsp_input_size=6, gsp_output_size=1,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
    )


class TestAgentHiddenStateInit:
    def test_recurrent_neighbors_initializes_hidden_dict(self, agent_config):
        agent = _make_recurrent_neighbors_agent(agent_config, 4)
        assert hasattr(agent, '_agent_hidden_states')
        assert len(agent._agent_hidden_states) == 4

    def test_recurrent_neighbors_all_none_initially(self, agent_config):
        agent = _make_recurrent_neighbors_agent(agent_config, 4)
        for i in range(4):
            assert agent._agent_hidden_states[i] is None

    def test_recurrent_neighbors_keys_match_n_agents(self, agent_config):
        agent = _make_recurrent_neighbors_agent(agent_config, 6)
        assert set(agent._agent_hidden_states.keys()) == {0, 1, 2, 3, 4, 5}

    def test_nonrecurrent_neighbors_empty_hidden_dict(self, agent_config):
        agent = _make_nonrecurrent_neighbors_agent(agent_config, 4)
        assert hasattr(agent, '_agent_hidden_states')
        assert len(agent._agent_hidden_states) == 0


class TestResetHiddenStates:
    def test_reset_sets_all_to_none(self, agent_config):
        agent = _make_recurrent_neighbors_agent(agent_config, 4)
        # Manually set a non-None value to simulate post-inference state
        agent._agent_hidden_states[0] = ('fake_h', 'fake_c')
        agent._agent_hidden_states[2] = ('fake_h', 'fake_c')
        agent.reset_hidden_states()
        for i in range(4):
            assert agent._agent_hidden_states[i] is None

    def test_reset_on_empty_dict_does_not_raise(self, agent_config):
        agent = _make_nonrecurrent_neighbors_agent(agent_config, 4)
        # Should not raise even with empty dict
        agent.reset_hidden_states()

    def test_reset_is_idempotent(self, agent_config):
        agent = _make_recurrent_neighbors_agent(agent_config, 4)
        agent.reset_hidden_states()
        agent.reset_hidden_states()
        for i in range(4):
            assert agent._agent_hidden_states[i] is None

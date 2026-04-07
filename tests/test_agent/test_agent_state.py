"""Tests for agent observation state construction."""

import numpy as np
import pytest
from src.agent import Agent


@pytest.fixture
def agent(agent_config):
    return Agent(
        config=agent_config, network='DQN', n_agents=4, n_obs=31,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=False, recurrent=False, attention=False,
        neighbors=False, gsp_input_size=6, gsp_output_size=1,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
    )


class TestMakeAgentState:
    def test_no_augmentation(self, agent):
        obs = np.ones(31)
        state = agent.make_agent_state(obs)
        assert len(state) == 31

    def test_with_gsp_heading(self, agent):
        obs = np.ones(31)
        state = agent.make_agent_state(obs, heading_gsp=0.5)
        assert len(state) == 32
        expected_heading = np.degrees(0.5 / 10)
        assert state[31] == pytest.approx(expected_heading)

    def test_with_global_knowledge(self, agent):
        obs = np.ones(31)
        gk = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        state = agent.make_agent_state(obs, global_knowledge=gk)
        assert len(state) == 43

    def test_with_both(self, agent):
        obs = np.ones(31)
        gk = np.array([1.0, 2.0, 3.0, 4.0])
        state = agent.make_agent_state(obs, heading_gsp=0.5, global_knowledge=gk)
        assert len(state) == 36

    def test_original_obs_unchanged(self, agent):
        obs = np.arange(31, dtype=float)
        state = agent.make_agent_state(obs.copy(), heading_gsp=0.5)
        np.testing.assert_array_equal(state[:31], obs)

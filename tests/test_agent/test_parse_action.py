"""Tests for discrete action parsing (DQN/DDQN)."""

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


class TestParseAction:
    def test_action_0(self, agent):
        result = agent.parse_action(0)
        np.testing.assert_array_almost_equal(result, [-0.1, -0.1, 0.0])

    def test_action_1(self, agent):
        result = agent.parse_action(1)
        np.testing.assert_array_almost_equal(result, [-0.1, 0.0, 0.0])

    def test_action_4(self, agent):
        result = agent.parse_action(4)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_action_8(self, agent):
        result = agent.parse_action(8)
        np.testing.assert_array_almost_equal(result, [0.1, 0.1, 0.0])

    def test_all_9_actions_valid(self, agent):
        for i in range(9):
            result = agent.parse_action(i)
            assert len(result) == 3
            assert result[2] == 0.0
            assert -0.1 <= result[0] <= 0.1
            assert -0.1 <= result[1] <= 0.1

    def test_out_of_range_raises(self, agent):
        with pytest.raises(Exception, match="Action Number Out of Range"):
            agent.parse_action(9)
        with pytest.raises(Exception, match="Action Number Out of Range"):
            agent.parse_action(-1)

    def test_gripper_always_zero(self, agent):
        for i in range(9):
            assert agent.parse_action(i)[2] == 0.0

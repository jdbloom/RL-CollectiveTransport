"""Tests for proximity sensor filtering around the cylinder."""

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
        prox_filter_angle_deg=45.0,
    )


class TestFilterProxValues:
    def test_cylinder_at_zero_filters_front_sensors(self, agent):
        prox = np.ones(24)
        filtered, indices = agent.filter_prox_values(prox, 0.0)
        assert len(indices) > 0
        assert len(filtered) + len(indices) == 24

    def test_cylinder_at_90_filters_side_sensors(self, agent):
        prox = np.ones(24)
        filtered, indices = agent.filter_prox_values(prox, 90.0)
        assert len(indices) > 0
        assert len(filtered) + len(indices) == 24

    def test_filtered_plus_indices_equals_24(self, agent):
        prox = np.ones(24)
        for angle in [0, 45, 90, 135, 170, -30, -90, -170]:
            filtered, indices = agent.filter_prox_values(prox, float(angle))
            assert len(filtered) + len(indices) == 24, f"Failed at angle {angle}"

    def test_wraparound_positive_near_180(self, agent):
        prox = np.ones(24)
        filtered, indices = agent.filter_prox_values(prox, 170.0)
        assert len(indices) > 0
        assert len(filtered) + len(indices) == 24

    def test_wraparound_negative_near_minus_180(self, agent):
        prox = np.ones(24)
        filtered, indices = agent.filter_prox_values(prox, -170.0)
        assert len(indices) > 0
        assert len(filtered) + len(indices) == 24

    def test_filtered_values_are_from_unfiltered_sensors(self, agent):
        prox = np.arange(24, dtype=float)
        filtered, indices = agent.filter_prox_values(prox, 0.0)
        remaining_indices = [i for i in range(24) if i not in indices]
        expected = [prox[i] for i in remaining_indices]
        assert filtered == pytest.approx(expected)

    def test_all_zeros_input(self, agent):
        prox = np.zeros(24)
        filtered, indices = agent.filter_prox_values(prox, 45.0)
        assert all(v == 0.0 for v in filtered)
        assert len(filtered) + len(indices) == 24

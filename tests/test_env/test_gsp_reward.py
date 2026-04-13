"""Tests for GSP reward computation."""

import math
import numpy as np
import pytest
from src.env import calculate_gsp_reward


class TestGSPRewardDisabled:
    def test_gsp_false_returns_zeros(self):
        rewards, label, _ = calculate_gsp_reward(False, 10.0, 15.0, [0.0, 0.0], 2)
        assert rewards == [0, 0]
        assert label == 0

    def test_gsp_false_any_num_robots(self):
        rewards, label, _ = calculate_gsp_reward(False, 0, 0, [0]*8, 8)
        assert len(rewards) == 8
        assert all(r == 0 for r in rewards)


class TestGSPRewardEnabled:
    def test_perfect_prediction_zero_change(self):
        rewards, label, _ = calculate_gsp_reward(True, 45.0, 45.0, [0.0], 1)
        assert label == pytest.approx(0.0)
        assert rewards[0] == pytest.approx(0.0)

    def test_perfect_prediction_nonzero_change(self):
        # 5 deg change: radians(5)=0.0873, x100=8.73, clipped to 1.0
        rewards, label, _ = calculate_gsp_reward(True, 0.0, 5.0, [1.0], 1)
        assert label == pytest.approx(1.0)
        assert rewards[0] == pytest.approx(0.0)

    def test_bad_prediction_negative_reward(self):
        rewards, label, _ = calculate_gsp_reward(True, 0.0, 0.0, [1.0], 1)
        assert label == pytest.approx(0.0)
        assert rewards[0] == pytest.approx(-1.0)

    def test_reward_clipped_at_minus_2(self):
        rewards, label, _ = calculate_gsp_reward(True, 0.0, 0.0, [2.0], 1)
        assert rewards[0] == pytest.approx(-2.0)

    def test_reward_per_robot(self):
        rewards, label, _ = calculate_gsp_reward(True, 0.0, 0.0, [0.0, 1.0, 0.5], 3)
        assert len(rewards) == 3
        assert rewards[0] == pytest.approx(0.0)
        assert rewards[1] == pytest.approx(-1.0)
        assert rewards[2] == pytest.approx(-0.25)

    def test_wraparound_angles(self):
        rewards, label, _ = calculate_gsp_reward(True, 350.0, 10.0, [0.0], 1)
        assert label == pytest.approx(1.0)  # 20 deg -> radians -> x100 -> clipped to 1.0

    def test_label_type(self):
        _, label, _ = calculate_gsp_reward(True, 0.0, 1.0, [0.0], 1)
        assert isinstance(label, (float, np.floating))


class TestGSPSquaredErrorReturn:
    """The function returns per-robot squared prediction error alongside the clipped reward.

    Needed for information-collapse diagnosis: the raw error carries more signal than the
    clipped reward, since the reward saturates at -2 and loses the magnitude of large errors.
    """

    def test_returns_three_tuple(self):
        result = calculate_gsp_reward(True, 0.0, 0.0, [0.0], 1)
        assert len(result) == 3

    def test_squared_error_is_zero_for_perfect_prediction(self):
        _, _, squared = calculate_gsp_reward(True, 45.0, 45.0, [0.0, 0.0], 2)
        assert squared == pytest.approx([0.0, 0.0])

    def test_squared_error_is_unclipped(self):
        """Reward is clipped to [-2, 0] but squared_error is the raw (diff - pred)^2."""
        _, _, squared = calculate_gsp_reward(True, 0.0, 0.0, [3.0], 1)
        # diff=0, pred=3.0 -> raw squared error = 9.0 (much bigger than the clipped reward -2)
        assert squared[0] == pytest.approx(9.0)

    def test_squared_error_per_robot(self):
        _, _, squared = calculate_gsp_reward(True, 0.0, 0.0, [0.0, 1.0, 0.5], 3)
        assert len(squared) == 3
        assert squared[0] == pytest.approx(0.0)
        assert squared[1] == pytest.approx(1.0)
        assert squared[2] == pytest.approx(0.25)

    def test_squared_error_is_zero_list_when_gsp_disabled(self):
        _, _, squared = calculate_gsp_reward(False, 0, 0, [0.0] * 4, 4)
        assert squared == [0, 0, 0, 0]

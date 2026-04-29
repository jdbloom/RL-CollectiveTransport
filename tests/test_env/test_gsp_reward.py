"""Tests for GSP reward computation."""

import math
import numpy as np
import pytest
from src.env import calculate_gsp_reward


class TestGSPRewardDisabled:
    def test_gsp_false_returns_zeros(self):
        rewards, label, _, _raw = calculate_gsp_reward(False, 10.0, 15.0, [0.0, 0.0], 2)
        assert rewards == [0, 0]
        assert label == 0

    def test_gsp_false_any_num_robots(self):
        rewards, label, _, _raw = calculate_gsp_reward(False, 0, 0, [0]*8, 8)
        assert len(rewards) == 8
        assert all(r == 0 for r in rewards)


class TestGSPRewardEnabled:
    def test_perfect_prediction_zero_change(self):
        rewards, label, _, _raw = calculate_gsp_reward(True, 45.0, 45.0, [0.0], 1)
        assert label == pytest.approx(0.0)
        assert rewards[0] == pytest.approx(0.0)

    def test_perfect_prediction_nonzero_change(self):
        # 5 deg change: radians(5)=0.0873, x100=8.73, clipped to 1.0
        rewards, label, _, _raw = calculate_gsp_reward(True, 0.0, 5.0, [1.0], 1)
        assert label == pytest.approx(1.0)
        assert rewards[0] == pytest.approx(0.0)

    def test_bad_prediction_negative_reward(self):
        rewards, label, _, _raw = calculate_gsp_reward(True, 0.0, 0.0, [1.0], 1)
        assert label == pytest.approx(0.0)
        assert rewards[0] == pytest.approx(-1.0)

    def test_reward_clipped_at_minus_2(self):
        rewards, label, _, _raw = calculate_gsp_reward(True, 0.0, 0.0, [2.0], 1)
        assert rewards[0] == pytest.approx(-2.0)

    def test_reward_per_robot(self):
        rewards, label, _, _raw = calculate_gsp_reward(True, 0.0, 0.0, [0.0, 1.0, 0.5], 3)
        assert len(rewards) == 3
        assert rewards[0] == pytest.approx(0.0)
        assert rewards[1] == pytest.approx(-1.0)
        assert rewards[2] == pytest.approx(-0.25)

    def test_wraparound_angles(self):
        rewards, label, _, _raw = calculate_gsp_reward(True, 350.0, 10.0, [0.0], 1)
        assert label == pytest.approx(1.0)  # 20 deg -> radians -> x100 -> clipped to 1.0

    def test_label_type(self):
        _, label, _, _raw = calculate_gsp_reward(True, 0.0, 1.0, [0.0], 1)
        assert isinstance(label, (float, np.floating))


class TestGSPSquaredErrorReturn:
    """The function returns per-robot squared prediction error alongside the clipped reward.

    Needed for information-collapse diagnosis: the raw error carries more signal than the
    clipped reward, since the reward saturates at -2 and loses the magnitude of large errors.
    """

    def test_returns_four_tuple(self):
        """Function returns (rewards, label, squared_errors, raw_diff_rad)."""
        result = calculate_gsp_reward(True, 0.0, 0.0, [0.0], 1)
        assert len(result) == 4

    def test_squared_error_is_zero_for_perfect_prediction(self):
        _, _, squared, _raw = calculate_gsp_reward(True, 45.0, 45.0, [0.0, 0.0], 2)
        assert squared == pytest.approx([0.0, 0.0])

    def test_squared_error_is_unclipped(self):
        """Reward is clipped to [-2, 0] but squared_error is the raw (diff - pred)^2."""
        _, _, squared, _raw = calculate_gsp_reward(True, 0.0, 0.0, [3.0], 1)
        # diff=0, pred=3.0 -> raw squared error = 9.0 (much bigger than the clipped reward -2)
        assert squared[0] == pytest.approx(9.0)

    def test_squared_error_per_robot(self):
        _, _, squared, _raw = calculate_gsp_reward(True, 0.0, 0.0, [0.0, 1.0, 0.5], 3)
        assert len(squared) == 3
        assert squared[0] == pytest.approx(0.0)
        assert squared[1] == pytest.approx(1.0)
        assert squared[2] == pytest.approx(0.25)

    def test_squared_error_is_zero_list_when_gsp_disabled(self):
        _, _, squared, _raw = calculate_gsp_reward(False, 0, 0, [0.0] * 4, 4)
        assert squared == [0, 0, 0, 0]

    def test_raw_diff_rad_returned(self):
        """raw_diff_rad is the pre-scale, pre-clip radian rotation."""
        _, _, _, raw = calculate_gsp_reward(True, 0.0, 5.0, [0.0], 1)
        # 5 deg → radians ≈ 0.0873
        assert raw == pytest.approx(math.radians(5.0), abs=1e-6)

    def test_raw_diff_rad_zero_when_disabled(self):
        _, _, _, raw = calculate_gsp_reward(False, 0.0, 5.0, [0.0], 1)
        assert raw == 0.0


class TestGSPRewardMultiDim:
    """calculate_gsp_reward accepts 2D next_heading_gsp (num_robots, K).

    For multi-dim output (cyl_kinematics_3d/goal_4d), each robot's prediction is a
    K-dim vector. The reward uses the LAST dim (the cylinder Δθ component) per plan §3.
    For K=1, behavior must be bit-identical to the legacy scalar path.
    """

    def test_1d_array_per_robot_identical_to_scalar(self):
        """K=1: passing np.array([v]) per robot gives same reward as scalar v."""
        scalar_next = [0.5, 0.3]
        array_next = np.array([[0.5], [0.3]], dtype=np.float32)
        r_scalar, l_scalar, sq_scalar, raw_scalar = calculate_gsp_reward(
            True, 0.0, 2.0, scalar_next, 2
        )
        r_array, l_array, sq_array, raw_array = calculate_gsp_reward(
            True, 0.0, 2.0, array_next, 2
        )
        assert r_scalar == pytest.approx(r_array, abs=1e-6)
        assert l_scalar == pytest.approx(l_array)
        assert sq_scalar == pytest.approx(sq_array, abs=1e-6)

    def test_3d_uses_last_dim(self):
        """K=3: reward is computed from next_heading_gsp[i][-1], not [0] or [1]."""
        # Set up: diff will be small (near-zero), so perfect prediction (reward≈0)
        # only when we pick the correct dim (last).
        # Use old=0 new=0 → diff=0 after normalize.
        pred_3d = np.array([[9.9, 9.9, 0.0], [9.9, 9.9, 0.0]], dtype=np.float32)
        rewards, label, squared, _ = calculate_gsp_reward(True, 0.0, 0.0, pred_3d, 2)
        # diff=0, last dim=0.0 → perfect prediction → reward=0
        assert rewards[0] == pytest.approx(0.0, abs=1e-6)
        assert rewards[1] == pytest.approx(0.0, abs=1e-6)

    def test_4d_uses_last_dim(self):
        """K=4: reward uses dim index 3 (last)."""
        pred_4d = np.array([[9.9, 9.9, 9.9, 0.0]], dtype=np.float32)
        rewards, label, squared, _ = calculate_gsp_reward(True, 0.0, 0.0, pred_4d, 1)
        assert rewards[0] == pytest.approx(0.0, abs=1e-6)

    def test_2d_array_shape_r_k(self):
        """next_heading_gsp of shape (R, K) works without crash for K=3."""
        next_gsp = np.zeros((4, 3), dtype=np.float32)
        rewards, label, squared, raw = calculate_gsp_reward(True, 0.0, 0.0, next_gsp, 4)
        assert len(rewards) == 4
        assert len(squared) == 4

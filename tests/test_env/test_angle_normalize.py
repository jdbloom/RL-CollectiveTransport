"""Tests for angle normalization utility functions."""

import pytest
from src.env import angle_normalize_unsigned_deg, angle_normalize_signed_deg


class TestAngleNormalizeUnsigned:
    def test_zero(self):
        assert angle_normalize_unsigned_deg(0) == 0

    def test_positive_within_range(self):
        assert angle_normalize_unsigned_deg(90) == 90

    def test_360_wraps_to_zero(self):
        assert angle_normalize_unsigned_deg(360) == 0

    def test_negative_wraps_positive(self):
        assert angle_normalize_unsigned_deg(-90) == 270

    def test_large_positive(self):
        assert angle_normalize_unsigned_deg(720) == 0

    def test_large_negative(self):
        assert angle_normalize_unsigned_deg(-450) == 270

    def test_just_below_360(self):
        assert angle_normalize_unsigned_deg(359.9) == pytest.approx(359.9)


class TestAngleNormalizeSigned:
    def test_zero(self):
        assert angle_normalize_signed_deg(0) == 0

    def test_positive_within_range(self):
        assert angle_normalize_signed_deg(90) == 90

    def test_180_wraps_to_negative(self):
        assert angle_normalize_signed_deg(180) == -180

    def test_negative_within_range(self):
        assert angle_normalize_signed_deg(-90) == -90

    def test_270_wraps(self):
        assert angle_normalize_signed_deg(270) == -90

    def test_negative_270_wraps(self):
        assert angle_normalize_signed_deg(-270) == 90

    def test_small_positive_diff(self):
        assert angle_normalize_signed_deg(5) == 5

    def test_crosses_180_boundary(self):
        assert angle_normalize_signed_deg(340) == -20

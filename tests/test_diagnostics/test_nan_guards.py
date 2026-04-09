import numpy as np
import torch as T
import pytest
from gsp_rl.src.actors.learning_aids import _check_nan


class TestNaNDetection:
    def test_check_nan_detects_nan_float(self):
        with pytest.raises(RuntimeError, match="NaN detected"):
            _check_nan(float("nan"), "test_loss")

    def test_check_nan_passes_normal_float(self):
        _check_nan(0.5, "normal_loss")

    def test_check_nan_detects_inf(self):
        with pytest.raises(RuntimeError, match="NaN detected"):
            _check_nan(float("inf"), "inf_loss")

    def test_check_nan_detects_nan_tensor(self):
        nan_tensor = T.tensor(float("nan"))
        with pytest.raises(RuntimeError, match="NaN detected"):
            _check_nan(nan_tensor, "nan_tensor")

    def test_check_nan_passes_normal_tensor(self):
        _check_nan(T.tensor(0.5), "normal_tensor")

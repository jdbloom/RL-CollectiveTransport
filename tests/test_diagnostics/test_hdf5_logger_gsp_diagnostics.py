"""Tests for GSP information-collapse diagnostic fields added to HDF5Logger.

See Stelaris docs/specs/2026-04-12-dispatcher-diagnostic-batch.md for the hypothesis
these fields exist to test: GSP output variance, prediction/target correlation,
and per-step GSP-specific training loss.
"""

import os
import numpy as np
import pytest

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

pytestmark = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


if HAS_H5PY:
    from src.hdf5_logger import HDF5Logger


def _base_writerow_kwargs():
    return dict(
        rewards=[0.1] * 4,
        epsilons=0.5,
        terminations=False,
        losses=0.0,
        force_magnitudes=[0.0] * 4,
        force_angles=[0.0] * 4,
        average_force_vectors=[0.0, 0.0],
        cyl_x_poses=0.0,
        cyl_y_poses=0.0,
        cyl_angles=0.0,
        gate_stats=0,
        obstacle_stats=0,
        gsp_rewards=[0.0] * 4,
        gsp_headings=[0.0] * 4,
        run_times=0.0,
        robots_x_poses=[0.0] * 4,
        robots_y_poses=[0.0] * 4,
        robot_angles=[0.0] * 4,
        robot_failure=[False] * 4,
    )


def test_gsp_target_and_squared_error_stored_per_step(tmp_path):
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)
    for t in range(5):
        kwargs = _base_writerow_kwargs()
        kwargs["gsp_headings"] = [0.1 * t, 0.2 * t, 0.3 * t, 0.4 * t]
        kwargs["gsp_target"] = [0.15 * t, 0.25 * t, 0.35 * t, 0.45 * t]
        kwargs["gsp_squared_error"] = [
            (0.05 * t) ** 2,
            (0.05 * t) ** 2,
            (0.05 * t) ** 2,
            (0.05 * t) ** 2,
        ]
        logger.writerow(**kwargs)
    logger.write_episode(0)

    with h5py.File(path) as f:
        grp = f["episode_0000"]
        assert grp["gsp_target"].shape == (5, 4)
        assert grp["gsp_squared_error"].shape == (5, 4)
        np.testing.assert_allclose(
            grp["gsp_target"][4], [0.6, 1.0, 1.4, 1.8], rtol=1e-5
        )


def test_gsp_loss_recorded_per_learning_step(tmp_path):
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)
    for t in range(3):
        logger.writerow(**_base_writerow_kwargs())
    logger.record_gsp_loss(0.5)
    logger.record_gsp_loss(0.4)
    logger.record_gsp_loss(0.3)
    logger.write_episode(0)

    with h5py.File(path) as f:
        grp = f["episode_0000"]
        assert "gsp_loss" in grp
        assert grp["gsp_loss"].shape == (3,)
        np.testing.assert_allclose(grp["gsp_loss"][...], [0.5, 0.4, 0.3], rtol=1e-6)


def test_episode_attrs_include_output_std_and_correlation(tmp_path):
    """write_episode computes gsp_output_std and gsp_pred_target_corr."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)

    predictions = np.array([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6], [0.4, 0.8]], dtype=np.float32)
    targets = np.array([[0.11, 0.21], [0.19, 0.39], [0.31, 0.61], [0.40, 0.82]], dtype=np.float32)

    for t in range(4):
        kwargs = _base_writerow_kwargs()
        kwargs["gsp_headings"] = predictions[t].tolist() + [0.0, 0.0]
        kwargs["gsp_target"] = targets[t].tolist() + [0.0, 0.0]
        kwargs["gsp_squared_error"] = [
            float((predictions[t, 0] - targets[t, 0]) ** 2),
            float((predictions[t, 1] - targets[t, 1]) ** 2),
            0.0,
            0.0,
        ]
        logger.writerow(**kwargs)
    logger.write_episode(0)

    with h5py.File(path) as f:
        grp = f["episode_0000"]
        assert "gsp_output_std" in grp.attrs
        assert "gsp_pred_target_corr" in grp.attrs
        assert float(grp.attrs["gsp_output_std"]) > 0.0
        assert float(grp.attrs["gsp_pred_target_corr"]) > 0.95


def test_collapsed_gsp_signature_detectable(tmp_path):
    """A collapsed predictor (constant output) shows near-zero std and NaN correlation."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)

    constant_prediction = 0.05
    targets = [0.1, -0.2, 0.3, -0.4, 0.15]

    for t in range(5):
        kwargs = _base_writerow_kwargs()
        kwargs["gsp_headings"] = [constant_prediction] * 4
        kwargs["gsp_target"] = [targets[t]] * 4
        kwargs["gsp_squared_error"] = [(constant_prediction - targets[t]) ** 2] * 4
        logger.writerow(**kwargs)
    logger.write_episode(0)

    with h5py.File(path) as f:
        grp = f["episode_0000"]
        assert float(grp.attrs["gsp_output_std"]) < 1e-6
        corr = float(grp.attrs["gsp_pred_target_corr"])
        assert np.isnan(corr), f"expected NaN correlation for collapsed predictor, got {corr}"


def test_degenerate_task_signature_distinct_from_collapsed(tmp_path):
    """Both std=0 case (degenerate task) also produces NaN."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)

    for t in range(5):
        kwargs = _base_writerow_kwargs()
        kwargs["gsp_headings"] = [0.1] * 4
        kwargs["gsp_target"] = [0.1] * 4
        kwargs["gsp_squared_error"] = [0.0] * 4
        logger.writerow(**kwargs)
    logger.write_episode(0)

    with h5py.File(path) as f:
        grp = f["episode_0000"]
        assert np.isnan(float(grp.attrs["gsp_pred_target_corr"]))


def test_nan_prediction_does_not_poison_episode_summary(tmp_path):
    """A single NaN in predictions should not propagate through the std/corr attrs."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)

    predictions = [[0.1, 0.2], [0.2, 0.4], [float("nan"), 0.6], [0.4, 0.8]]
    targets = [[0.11, 0.21], [0.19, 0.39], [0.31, 0.61], [0.40, 0.82]]

    for t in range(4):
        kwargs = _base_writerow_kwargs()
        kwargs["gsp_headings"] = predictions[t] + [0.0, 0.0]
        kwargs["gsp_target"] = targets[t] + [0.0, 0.0]
        kwargs["gsp_squared_error"] = [0.0] * 4
        logger.writerow(**kwargs)
    logger.write_episode(0)

    with h5py.File(path) as f:
        grp = f["episode_0000"]
        std = float(grp.attrs["gsp_output_std"])
        corr = float(grp.attrs["gsp_pred_target_corr"])
        assert not np.isnan(std), "nanstd should handle NaN predictions"
        assert not np.isnan(corr), "corrcoef should handle NaN predictions after masking"


def test_desynced_target_buffer_raises(tmp_path):
    """Passing gsp_target on only some writerow calls in an episode is a contract violation."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)

    kwargs = _base_writerow_kwargs()
    kwargs["gsp_headings"] = [0.1] * 4
    kwargs["gsp_target"] = [0.2] * 4
    logger.writerow(**kwargs)

    kwargs = _base_writerow_kwargs()
    kwargs["gsp_headings"] = [0.15] * 4
    logger.writerow(**kwargs)

    with pytest.raises(ValueError, match="buffer length"):
        logger.write_episode(0)


def test_writerow_backwards_compatible_without_gsp_diagnostics(tmp_path):
    """Existing callers that don't pass the new kwargs still work."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)
    logger.writerow(**_base_writerow_kwargs())
    logger.write_episode(0)

    with h5py.File(path) as f:
        grp = f["episode_0000"]
        assert "gsp_target" not in grp
        assert "gsp_squared_error" not in grp
        assert "gsp_loss" not in grp
        assert "gsp_output_std" not in grp.attrs
        assert "gsp_pred_target_corr" not in grp.attrs


def test_record_gsp_loss_is_optional(tmp_path):
    """If record_gsp_loss is never called, no gsp_loss dataset is written."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)
    for t in range(3):
        logger.writerow(**_base_writerow_kwargs())
    logger.write_episode(0)

    with h5py.File(path) as f:
        grp = f["episode_0000"]
        assert "gsp_loss" not in grp

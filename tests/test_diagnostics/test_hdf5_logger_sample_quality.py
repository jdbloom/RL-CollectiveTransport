"""Tests for Phase 1 sample-quality attrs added to HDF5Logger in schema v4.

See docs/specs/2026-04-21-phase1-verification.md §4. These attrs answer:
- gsp_loss distribution per episode (was previously stored as dataset only)
- gsp_label distribution from stored transitions (the input side the external
  gsp_rl replay buffer would see)
- gsp_train_input std (whether stored input features are diverse)
"""

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


def _writerow_kwargs():
    return dict(
        rewards=[0.0] * 4,
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


def test_schema_bumped_to_v4(tmp_path):
    p = str(tmp_path / "e.h5")
    logger = HDF5Logger(p)
    logger.writerow(**_writerow_kwargs())
    logger.write_episode(0)
    with h5py.File(p, "r") as f:
        assert int(f["episode_0000"].attrs["log_schema_version"]) == 4


def test_gsp_label_attrs_written_when_transitions_stored(tmp_path):
    p = str(tmp_path / "e.h5")
    logger = HDF5Logger(p)
    logger.writerow(**_writerow_kwargs())
    labels = [0.1, 0.2, 0.3, 0.4, 0.5]
    for lbl in labels:
        logger.record_stored_transition(label=lbl, input_vec=np.zeros(6))
    logger.write_episode(0)
    with h5py.File(p, "r") as f:
        g = f["episode_0000"]
        assert int(g.attrs["gsp_label_count"]) == 5
        assert float(g.attrs["gsp_label_mean"]) == pytest.approx(np.mean(labels), rel=1e-5)
        assert float(g.attrs["gsp_label_std"]) == pytest.approx(np.std(labels), rel=1e-5)
        assert float(g.attrs["gsp_label_min"]) == pytest.approx(min(labels))
        assert float(g.attrs["gsp_label_max"]) == pytest.approx(max(labels))


def test_gsp_label_attrs_absent_when_no_transitions_stored(tmp_path):
    """Schema stays lean when the cell is IC (no GSP training) — no stored transitions."""
    p = str(tmp_path / "e.h5")
    logger = HDF5Logger(p)
    logger.writerow(**_writerow_kwargs())
    logger.write_episode(0)
    with h5py.File(p, "r") as f:
        g = f["episode_0000"]
        assert "gsp_label_count" not in g.attrs
        assert "gsp_label_mean" not in g.attrs


def test_gsp_loss_attrs_written_when_losses_recorded(tmp_path):
    p = str(tmp_path / "e.h5")
    logger = HDF5Logger(p)
    logger.writerow(**_writerow_kwargs())
    losses = [0.5, 0.3, 0.2, 0.15, 0.1]
    for ls in losses:
        logger.record_gsp_loss(ls)
    logger.write_episode(0)
    with h5py.File(p, "r") as f:
        g = f["episode_0000"]
        assert int(g.attrs["gsp_loss_count"]) == 5
        assert float(g.attrs["gsp_loss_mean"]) == pytest.approx(np.mean(losses), rel=1e-5)
        assert float(g.attrs["gsp_loss_std"]) == pytest.approx(np.std(losses), rel=1e-5)


def test_gsp_loss_attrs_ignore_nan_and_inf(tmp_path):
    p = str(tmp_path / "e.h5")
    logger = HDF5Logger(p)
    logger.writerow(**_writerow_kwargs())
    logger.record_gsp_loss(0.1)
    logger.record_gsp_loss(float("nan"))
    logger.record_gsp_loss(0.3)
    logger.record_gsp_loss(float("inf"))
    logger.record_gsp_loss(0.2)
    logger.write_episode(0)
    with h5py.File(p, "r") as f:
        g = f["episode_0000"]
        assert int(g.attrs["gsp_loss_count"]) == 3
        assert float(g.attrs["gsp_loss_mean"]) == pytest.approx(0.2, rel=1e-5)


def test_gsp_train_input_std_aggregates_per_dim(tmp_path):
    p = str(tmp_path / "e.h5")
    logger = HDF5Logger(p)
    logger.writerow(**_writerow_kwargs())
    # 3 inputs of dim 4; set std to be predictable per dim.
    inputs = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 2.0, 0.0],
        [2.0, 0.0, 4.0, 0.0],
    ])
    for vec in inputs:
        logger.record_stored_transition(label=0.0, input_vec=vec)
    logger.write_episode(0)
    with h5py.File(p, "r") as f:
        g = f["episode_0000"]
        assert int(g.attrs["gsp_train_input_count"]) == 3
        # mean-over-dims of std per dim
        expected = float(np.mean(np.std(inputs, axis=0)))
        assert float(g.attrs["gsp_train_input_std"]) == pytest.approx(expected, rel=1e-5)


def test_vector_label_reduced_to_mean(tmp_path):
    """cyl_kinematics_3d/4d targets are vectors; record_stored_transition should
    store their mean as a scalar proxy without crashing."""
    p = str(tmp_path / "e.h5")
    logger = HDF5Logger(p)
    logger.writerow(**_writerow_kwargs())
    logger.record_stored_transition(label=np.array([0.1, 0.3, 0.5]), input_vec=np.zeros(6))
    logger.record_stored_transition(label=np.array([0.2, 0.4, 0.6]), input_vec=np.zeros(6))
    logger.write_episode(0)
    with h5py.File(p, "r") as f:
        g = f["episode_0000"]
        assert int(g.attrs["gsp_label_count"]) == 2
        # Mean of means: mean([0.3, 0.4]) = 0.35
        assert float(g.attrs["gsp_label_mean"]) == pytest.approx(0.35, rel=1e-5)


def test_buffers_reset_between_episodes(tmp_path):
    p = str(tmp_path / "e.h5")
    logger = HDF5Logger(p)
    # Episode 0: 3 stored transitions
    logger.writerow(**_writerow_kwargs())
    for _ in range(3):
        logger.record_stored_transition(label=0.5, input_vec=np.zeros(6))
    logger.write_episode(0)
    # Episode 1: 1 stored transition — must not include ep0's 3
    logger.writerow(**_writerow_kwargs())
    logger.record_stored_transition(label=0.9, input_vec=np.zeros(6))
    logger.write_episode(1)
    with h5py.File(p, "r") as f:
        assert int(f["episode_0000"].attrs["gsp_label_count"]) == 3
        assert int(f["episode_0001"].attrs["gsp_label_count"]) == 1
        assert float(f["episode_0001"].attrs["gsp_label_mean"]) == pytest.approx(0.9)

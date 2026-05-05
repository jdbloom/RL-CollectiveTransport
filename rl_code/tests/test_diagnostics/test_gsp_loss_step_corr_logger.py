"""Unit tests for the Phase 4 gsp_loss_step_corr attrs in HDF5Logger.

Verifies:
1. record_gsp_loss_step_corr() accumulates finite floats and drops NaN.
2. write_episode() writes gsp_loss_step_corr_{mean,std,min,max,n_batches}
   attrs when samples are present.
3. Non-GSP episodes (no samples) produce no gsp_loss_step_corr_* attrs —
   backward compat is preserved.
4. _reset() clears gsp_loss_step_corr_samples between episodes.
5. Attrs coexist with gsp_pred_target_corr (the existing actor-input-path
   metric) without interfering.
"""

import math
import tempfile
import os

import h5py
import numpy as np
import pytest

from src.hdf5_logger import HDF5Logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logger(tmp_path):
    """Return an HDF5Logger writing to a temp file."""
    path = os.path.join(tmp_path, "test_run.h5")
    return HDF5Logger(path), path


def _write_minimal_episode(logger, ep_num=1):
    """Write the minimum required fields to complete an episode."""
    rewards = [0.0]
    logger.writerow(
        rewards=rewards,
        epsilons=[0.0],
        terminations=[False],
        losses=[0.0],
        force_magnitudes=[0.0],
        force_angles=[0.0],
        average_force_vectors=[0.0],
        cyl_x_poses=[0.0],
        cyl_y_poses=[0.0],
        cyl_angles=[0.0],
        gate_stats=0,
        obstacle_stats=0,
        gsp_rewards=[0.0],
        gsp_headings=[0.0],
        run_times=[0.0],
        robots_x_poses=[0.0],
        robots_y_poses=[0.0],
        robot_angles=[0.0],
        robot_failure=[False],
    )
    return logger.write_episode(ep_num)


# ---------------------------------------------------------------------------
# Test 1: record_gsp_loss_step_corr accumulates finite floats, drops NaN
# ---------------------------------------------------------------------------

def test_record_accumulates_finite_drops_nan():
    with tempfile.TemporaryDirectory() as tmp:
        logger, _ = _make_logger(tmp)
        logger.record_gsp_loss_step_corr(0.3)
        logger.record_gsp_loss_step_corr(float("nan"))  # must be dropped
        logger.record_gsp_loss_step_corr(0.5)

        assert len(logger.gsp_loss_step_corr_samples) == 2
        assert logger.gsp_loss_step_corr_samples == pytest.approx([0.3, 0.5])


# ---------------------------------------------------------------------------
# Test 2: write_episode writes all five attrs when samples present
# ---------------------------------------------------------------------------

def test_write_episode_writes_attrs_when_samples_present():
    with tempfile.TemporaryDirectory() as tmp:
        logger, path = _make_logger(tmp)
        corr_values = [0.1, 0.3, 0.5, 0.7]
        for c in corr_values:
            logger.record_gsp_loss_step_corr(c)

        _write_minimal_episode(logger, ep_num=1)

        with h5py.File(path, "r") as f:
            grp = f["episode_0001"]
            assert "gsp_loss_step_corr_mean" in grp.attrs, (
                "gsp_loss_step_corr_mean attr must be written when samples present"
            )
            assert "gsp_loss_step_corr_std" in grp.attrs
            assert "gsp_loss_step_corr_min" in grp.attrs
            assert "gsp_loss_step_corr_max" in grp.attrs
            assert "gsp_loss_step_corr_n_batches" in grp.attrs

            arr = np.array(corr_values)
            assert grp.attrs["gsp_loss_step_corr_mean"] == pytest.approx(float(np.mean(arr)))
            assert grp.attrs["gsp_loss_step_corr_std"] == pytest.approx(float(np.std(arr)))
            assert grp.attrs["gsp_loss_step_corr_min"] == pytest.approx(float(np.min(arr)))
            assert grp.attrs["gsp_loss_step_corr_max"] == pytest.approx(float(np.max(arr)))
            assert int(grp.attrs["gsp_loss_step_corr_n_batches"]) == 4


# ---------------------------------------------------------------------------
# Test 3: no attrs when no samples (non-GSP episode — backward compat)
# ---------------------------------------------------------------------------

def test_write_episode_no_attrs_when_no_samples():
    with tempfile.TemporaryDirectory() as tmp:
        logger, path = _make_logger(tmp)
        # Do NOT call record_gsp_loss_step_corr
        _write_minimal_episode(logger, ep_num=1)

        with h5py.File(path, "r") as f:
            grp = f["episode_0001"]
            assert "gsp_loss_step_corr_mean" not in grp.attrs, (
                "gsp_loss_step_corr_mean must NOT be written when no samples present"
            )
            assert "gsp_loss_step_corr_n_batches" not in grp.attrs


# ---------------------------------------------------------------------------
# Test 4: _reset() clears samples between episodes
# ---------------------------------------------------------------------------

def test_reset_clears_samples_between_episodes():
    with tempfile.TemporaryDirectory() as tmp:
        logger, path = _make_logger(tmp)

        # Episode 1: record some samples
        logger.record_gsp_loss_step_corr(0.2)
        logger.record_gsp_loss_step_corr(0.4)
        _write_minimal_episode(logger, ep_num=1)

        # After write_episode, _reset() must have cleared gsp_loss_step_corr_samples
        assert len(logger.gsp_loss_step_corr_samples) == 0, (
            "gsp_loss_step_corr_samples must be empty after write_episode/_reset()"
        )

        # Episode 2: no samples recorded → no attrs on this episode
        _write_minimal_episode(logger, ep_num=2)

        with h5py.File(path, "r") as f:
            assert "gsp_loss_step_corr_mean" in f["episode_0001"].attrs
            assert "gsp_loss_step_corr_mean" not in f["episode_0002"].attrs


# ---------------------------------------------------------------------------
# Test 5: gsp_loss_step_corr attrs coexist with gsp_pred_target_corr
# ---------------------------------------------------------------------------

def test_both_corr_attrs_coexist():
    """gsp_loss_step_corr_* and gsp_pred_target_corr can appear on the same episode."""
    with tempfile.TemporaryDirectory() as tmp:
        logger, path = _make_logger(tmp)

        # Populate actor-input-path corr (gsp_heading + gsp_target)
        logger.writerow(
            rewards=[0.0],
            epsilons=[0.0],
            terminations=[False],
            losses=[0.0],
            force_magnitudes=[0.0],
            force_angles=[0.0],
            average_force_vectors=[0.0],
            cyl_x_poses=[0.0],
            cyl_y_poses=[0.0],
            cyl_angles=[0.0],
            gate_stats=0,
            obstacle_stats=0,
            gsp_rewards=[0.0],
            gsp_headings=[0.3],
            run_times=[0.0],
            robots_x_poses=[0.0],
            robots_y_poses=[0.0],
            robot_angles=[0.0],
            robot_failure=[False],
            gsp_target=[0.4],
        )
        # Populate loss-step-corr
        logger.record_gsp_loss_step_corr(0.25)

        logger.write_episode(1)

        with h5py.File(path, "r") as f:
            grp = f["episode_0001"]
            assert "gsp_pred_target_corr" in grp.attrs, (
                "gsp_pred_target_corr must be present (actor-input path)"
            )
            assert "gsp_loss_step_corr_mean" in grp.attrs, (
                "gsp_loss_step_corr_mean must be present (loss-step path)"
            )

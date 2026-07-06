"""Tests for M4 candidate-target logging (GSP_LOG_CANDIDATE_TARGETS).

Scientific contract: when GSP_LOG_CANDIDATE_TARGETS=1, ALL FOUR candidate GSP
targets are computed every timestep regardless of the active GSP_OUTPUT_KIND and
written to the episode HDF5 group as per-step datasets:

    cand_target_delta_theta   (T,)     collective Δθ scalar
    cand_target_future_prox   (T,)     mean per-robot proximity (future-prox source)
    cand_target_cyl_kin       (T, 3)   cylinder (Δx, Δy, Δθ)
    cand_target_centroid_goal (T,)     centroid-to-goal delta

Each dataset has length == number of timesteps. With the flag OFF (default),
NONE of these datasets exist — the candidate block is skipped and behavior is
bit-identical to a run without the instrumentation.

The candidate quantities are buffered via HDF5Logger.record_candidate_targets()
(driven from Main.py every step behind the flag) and written in write_episode.
This test drives the logger directly — Main.py runs argparse/ZMQ/ARGoS at module
import and cannot be imported in a unit test. The four candidate names are hard
enumerated here (not read from the implementation) so a buggy implementation
cannot make a wrong test pass.
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


_CAND_1D = ("cand_target_delta_theta", "cand_target_future_prox",
            "cand_target_centroid_goal")
_CAND_2D = ("cand_target_cyl_kin",)  # (T, 3)


def _base_writerow_kwargs():
    return dict(
        rewards=[0.1] * 4, epsilons=0.5, terminations=False, losses=0.0,
        force_magnitudes=[0.0] * 4, force_angles=[0.0] * 4,
        average_force_vectors=[0.0, 0.0],
        cyl_x_poses=0.0, cyl_y_poses=0.0, cyl_angles=0.0,
        gate_stats=0, obstacle_stats=0,
        gsp_rewards=[0.0] * 4, gsp_headings=[0.0] * 4,
        run_times=0.0, robots_x_poses=[0.0] * 4, robots_y_poses=[0.0] * 4,
        robot_angles=[0.0] * 4, robot_failure=[False] * 4,
    )


def _drive_episode(logger, timesteps, log_candidates, active_kind='delta_theta_1d'):
    """Drive a short synthetic episode. When log_candidates is True, record the
    four candidate targets every step regardless of active_kind."""
    for t in range(timesteps):
        logger.writerow(**_base_writerow_kwargs())
        if log_candidates:
            logger.record_candidate_targets(
                delta_theta=0.1 * t,
                future_prox=0.05 * t + 0.2,
                cyl_kin=[0.01 * t, -0.02 * t, 0.03 * t],
                centroid_goal=0.5 - 0.01 * t,
            )
    logger.write_episode(0)


class TestFlagOnDatasetsPresent:
    @pytest.mark.parametrize("active_kind",
                             ['delta_theta_1d', 'cyl_kinematics_3d',
                              'cyl_kinematics_goal_4d', 'future_prox_1d',
                              'time_to_goal_1d'])
    def test_all_four_candidates_present_regardless_of_kind(self, tmp_path, active_kind):
        path = str(tmp_path / f"ep_{active_kind}.h5")
        logger = HDF5Logger(path)
        T = 6
        _drive_episode(logger, T, log_candidates=True, active_kind=active_kind)
        with h5py.File(path) as f:
            grp = f["episode_0000"]
            for name in _CAND_1D:
                assert name in grp, f"{name} missing (active_kind={active_kind})"
                assert grp[name].shape == (T,), (
                    f"{name} shape {grp[name].shape} != ({T},)"
                )
            for name in _CAND_2D:
                assert name in grp, f"{name} missing (active_kind={active_kind})"
                assert grp[name].shape == (T, 3), (
                    f"{name} shape {grp[name].shape} != ({T}, 3)"
                )

    def test_candidate_values_roundtrip(self, tmp_path):
        path = str(tmp_path / "ep_vals.h5")
        logger = HDF5Logger(path)
        T = 5
        _drive_episode(logger, T, log_candidates=True)
        with h5py.File(path) as f:
            grp = f["episode_0000"]
            np.testing.assert_allclose(
                grp["cand_target_delta_theta"][:],
                [0.1 * t for t in range(T)], rtol=1e-5)
            np.testing.assert_allclose(
                grp["cand_target_future_prox"][:],
                [0.05 * t + 0.2 for t in range(T)], rtol=1e-5)
            np.testing.assert_allclose(
                grp["cand_target_centroid_goal"][:],
                [0.5 - 0.01 * t for t in range(T)], rtol=1e-5)
            expected_kin = np.array(
                [[0.01 * t, -0.02 * t, 0.03 * t] for t in range(T)],
                dtype=np.float32)
            np.testing.assert_allclose(
                grp["cand_target_cyl_kin"][:], expected_kin, rtol=1e-5)


class TestFlagOffNoDatasets:
    def test_no_candidate_datasets_when_flag_off(self, tmp_path):
        """Default (flag off): none of the candidate datasets exist."""
        path = str(tmp_path / "ep_off.h5")
        logger = HDF5Logger(path)
        _drive_episode(logger, 6, log_candidates=False)
        with h5py.File(path) as f:
            grp = f["episode_0000"]
            for name in (*_CAND_1D, *_CAND_2D):
                assert name not in grp, (
                    f"{name} present when GSP_LOG_CANDIDATE_TARGETS off"
                )

    def test_reset_clears_candidate_buffer_between_episodes(self, tmp_path):
        """After write_episode the candidate buffers reset; a following episode
        with the flag off must not carry the previous episode's candidates."""
        path = str(tmp_path / "ep_reset.h5")
        logger = HDF5Logger(path)
        # Episode 0: candidates on
        for t in range(4):
            logger.writerow(**_base_writerow_kwargs())
            logger.record_candidate_targets(
                delta_theta=float(t), future_prox=float(t),
                cyl_kin=[float(t)] * 3, centroid_goal=float(t))
        logger.write_episode(0)
        # Episode 1: candidates off
        for t in range(4):
            logger.writerow(**_base_writerow_kwargs())
        logger.write_episode(1)
        with h5py.File(path) as f:
            grp1 = f["episode_0001"]
            for name in (*_CAND_1D, *_CAND_2D):
                assert name not in grp1, (
                    f"{name} bled from episode 0 into episode 1"
                )

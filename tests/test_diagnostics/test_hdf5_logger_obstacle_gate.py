"""Regression test for obstacle_stats + gate_stats persistence (task #110).

The pkl format stored these per-episode static layout fields. The HDF5 logger
refactor (rl_ct@44d709e) silently dropped them — collected them at writerow but
never wrote to the h5 file. Reintroduced 2026-05-05 as episode-level datasets
(stored once per episode since the layout doesn't change within an episode).
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


def _writerow_kwargs(obstacle_stats, gate_stats):
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
        gate_stats=gate_stats,
        obstacle_stats=obstacle_stats,
        gsp_rewards=[0.0] * 4,
        gsp_headings=[0.0] * 4,
        run_times=0.0,
        robots_x_poses=[0.0] * 4,
        robots_y_poses=[0.0] * 4,
        robot_angles=[0.0] * 4,
        robot_failure=[False] * 4,
    )


def test_obstacle_stats_persisted_when_present(tmp_path):
    """2 obstacles → flat array of length 4 ([x0,y0,x1,y1]) stored once per ep."""
    p = str(tmp_path / "obs.h5")
    logger = HDF5Logger(p)
    obs = np.array([1.0, 2.0, -3.0, 4.5], dtype=np.float32)
    for _ in range(5):
        logger.writerow(**_writerow_kwargs(obstacle_stats=obs, gate_stats=0))
    logger.write_episode(0)

    with h5py.File(p, "r") as h:
        ep = h["episode_0000"]
        assert "obstacle_stats" in ep, "obstacle_stats dataset missing"
        loaded = np.array(ep["obstacle_stats"])
        np.testing.assert_array_almost_equal(loaded, obs)
        # gate=0 sentinel should NOT create a dataset
        assert "gate_stats" not in ep, "gate_stats should be absent when sentinel 0"


def test_gate_stats_persisted_when_present(tmp_path):
    """gate stats: flat array of length 4."""
    p = str(tmp_path / "gate.h5")
    logger = HDF5Logger(p)
    gate = np.array([5.0, 1.0, 5.0, 2.0], dtype=np.float32)
    for _ in range(3):
        logger.writerow(**_writerow_kwargs(obstacle_stats=0, gate_stats=gate))
    logger.write_episode(0)

    with h5py.File(p, "r") as h:
        ep = h["episode_0000"]
        assert "gate_stats" in ep
        loaded = np.array(ep["gate_stats"])
        np.testing.assert_array_almost_equal(loaded, gate)
        assert "obstacle_stats" not in ep


def test_both_absent_when_sentinel(tmp_path):
    """No obstacles + no gate → neither dataset present (don't litter)."""
    p = str(tmp_path / "empty.h5")
    logger = HDF5Logger(p)
    logger.writerow(**_writerow_kwargs(obstacle_stats=0, gate_stats=0))
    logger.write_episode(0)
    with h5py.File(p, "r") as h:
        ep = h["episode_0000"]
        assert "obstacle_stats" not in ep
        assert "gate_stats" not in ep

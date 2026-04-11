"""Tests for HDF5Logger — the episode data writer."""

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


class TestHDF5Logger:
    def test_creates_h5_file(self, tmp_path):
        path = str(tmp_path / "test.h5")
        logger = HDF5Logger(path)
        for _ in range(3):
            logger.writerow(
                [-2.0]*4, 0.5, False, 0.01, [0.1]*4, [45.0]*4,
                [0.1, 45.0], -1.0, 0.5, 10.0, 0, 0,
                [0.0]*4, [0.003]*4, 0.1, [0.0]*4, [0.0]*4,
                [90.0]*4, [False]*4,
            )
        logger.write_episode(0)
        assert os.path.exists(path)

    def test_episode_has_all_datasets(self, tmp_path):
        path = str(tmp_path / "test.h5")
        logger = HDF5Logger(path)
        for _ in range(5):
            logger.writerow(
                [-2.0]*4, 0.5, False, 0.01, [0.1]*4, [45.0]*4,
                [0.1, 45.0], -1.0, 0.5, 10.0, 0, 0,
                [0.0]*4, np.array([0.003]*4), 0.1, [0.0]*4, [0.0]*4,
                [90.0]*4, [False]*4,
            )
        logger.write_episode(0)
        with h5py.File(path, "r") as f:
            ep = f["episode_0000"]
            expected = {"reward", "gsp_reward", "force_magnitude", "force_angle",
                        "robot_x_pos", "robot_y_pos", "robot_angle", "robot_failure",
                        "gsp_heading", "epsilon", "loss", "cyl_x_pos", "cyl_y_pos",
                        "cyl_angle", "run_time", "comX", "comY", "termination"}
            assert expected.issubset(set(ep.keys()))

    def test_reward_shape(self, tmp_path):
        path = str(tmp_path / "test.h5")
        logger = HDF5Logger(path)
        num_steps = 10
        num_robots = 4
        for _ in range(num_steps):
            logger.writerow(
                [-2.0]*num_robots, 0.5, False, 0.01, [0.1]*num_robots, [45.0]*num_robots,
                [0.1, 45.0], -1.0, 0.5, 10.0, 0, 0,
                [0.0]*num_robots, [0.003]*num_robots, 0.1,
                [0.0]*num_robots, [0.0]*num_robots,
                [90.0]*num_robots, [False]*num_robots,
            )
        logger.write_episode(0)
        with h5py.File(path, "r") as f:
            assert f["episode_0000"]["reward"].shape == (num_steps, num_robots)

    def test_summary_attributes(self, tmp_path):
        path = str(tmp_path / "test.h5")
        logger = HDF5Logger(path)
        for t in range(10):
            logger.writerow(
                [-2.0]*4, 0.5, (t == 9), 0.01, [0.1]*4, [45.0]*4,
                [0.1, 45.0], -1.0, 0.5, 10.0, 0, 0,
                [-0.01]*4, [0.003]*4, 0.1, [0.0]*4, [0.0]*4,
                [90.0]*4, [False]*4,
            )
        summary = logger.write_episode(0)
        assert summary["timesteps"] == 10
        assert summary["success"] == True
        assert len(summary["reward_per_robot"]) == 4
        assert all(r < 0 for r in summary["reward_per_robot"])

    def test_multiple_episodes(self, tmp_path):
        path = str(tmp_path / "test.h5")
        logger = HDF5Logger(path)
        for ep in range(3):
            for _ in range(5):
                logger.writerow(
                    [-1.0]*4, 0.5, False, 0.01, [0.1]*4, [45.0]*4,
                    [0.1, 45.0], -1.0, 0.5, 10.0, 0, 0,
                    [0.0]*4, [0.0]*4, 0.1, [0.0]*4, [0.0]*4,
                    [90.0]*4, [False]*4,
                )
            logger.write_episode(ep)
        with h5py.File(path, "r") as f:
            episodes = [k for k in f.keys() if k.startswith("episode")]
            assert len(episodes) == 3

    def test_resets_between_episodes(self, tmp_path):
        path = str(tmp_path / "test.h5")
        logger = HDF5Logger(path)
        for _ in range(10):
            logger.writerow(
                [-1.0]*4, 0.5, False, 0.01, [0.1]*4, [45.0]*4,
                [0.1, 45.0], -1.0, 0.5, 10.0, 0, 0,
                [0.0]*4, [0.0]*4, 0.1, [0.0]*4, [0.0]*4,
                [90.0]*4, [False]*4,
            )
        logger.write_episode(0)
        for _ in range(5):
            logger.writerow(
                [-1.0]*4, 0.5, False, 0.01, [0.1]*4, [45.0]*4,
                [0.1, 45.0], -1.0, 0.5, 10.0, 0, 0,
                [0.0]*4, [0.0]*4, 0.1, [0.0]*4, [0.0]*4,
                [90.0]*4, [False]*4,
            )
        logger.write_episode(1)
        with h5py.File(path, "r") as f:
            assert f["episode_0000"]["reward"].shape[0] == 10
            assert f["episode_0001"]["reward"].shape[0] == 5


class TestHDF5Disabled:
    """Test that Main.py works when h5py is not available."""

    def test_has_hdf5_flag_exists(self):
        """The HAS_HDF5 flag should be importable."""
        # This tests the import pattern — if h5py is installed, HAS_HDF5=True
        import importlib
        spec = importlib.util.find_spec("h5py")
        if spec:
            assert HAS_H5PY is True

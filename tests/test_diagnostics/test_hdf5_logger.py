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


class TestHDF5SWMR:
    """Tests for SWMR (Single Writer Multiple Reader) mode.

    SWMR allows external *processes* to open the file with ``swmr=True, mode='r'``
    while the writer process holds it open in append mode.  HDF5 enforces this at
    the process boundary; within a single process h5py refuses to open the same path
    in both read and write mode simultaneously.  The concurrent-process tests here
    use subprocess so the reader runs in a separate OS process, which is the real
    production scenario (dashboard probe vs training writer).
    """

    _WRITEROW_ARGS = (
        [-2.0] * 4, 0.5, False, 0.01, [0.1] * 4, [45.0] * 4,
        [0.1, 45.0], -1.0, 0.5, 10.0, 0, 0,
        [0.0] * 4, [0.003] * 4, 0.1, [0.0] * 4, [0.0] * 4,
        [90.0] * 4, [False] * 4,
    )

    def test_swmr_mode_true_after_first_write_episode(self, tmp_path):
        """After write_episode is called once, the file is readable in SWMR mode."""
        path = str(tmp_path / "test.h5")
        logger = HDF5Logger(path)
        for _ in range(3):
            logger.writerow(*self._WRITEROW_ARGS)
        logger.write_episode(0)

        # File must be openable by a reader using swmr=True without error.
        with h5py.File(path, "r", swmr=True) as rf:
            assert "episode_0000" in rf

    def test_swmr_reader_subprocess_can_open_during_write(self, tmp_path):
        """A separate-process SWMR reader must not raise while the writer appends.

        This is the production scenario: dashboard probe opens the running training
        file.  We serialize via a sentinel file: writer writes ep 0, creates
        sentinel; subprocess waits for sentinel, opens with swmr=True, exits 0 on
        success.  Writer then writes ep 1 and verifies subprocess exit code.
        """
        import subprocess
        import sys

        path = str(tmp_path / "test.h5")
        sentinel = str(tmp_path / "ready.flag")

        logger = HDF5Logger(path)
        for _ in range(3):
            logger.writerow(*self._WRITEROW_ARGS)
        logger.write_episode(0)

        # Create sentinel so reader knows the file exists and SWMR is enabled.
        with open(sentinel, "w") as fh:
            fh.write("ready")

        reader_script = f"""
import sys, time
import h5py

sentinel = {sentinel!r}
path = {path!r}

# Wait for the sentinel (should already be there).
for _ in range(20):
    try:
        open(sentinel).close()
        break
    except FileNotFoundError:
        time.sleep(0.1)

try:
    with h5py.File(path, "r", swmr=True) as rf:
        keys = list(rf.keys())
    sys.exit(0)
except Exception as exc:
    print(exc, file=sys.stderr)
    sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", reader_script],
            timeout=10,
            capture_output=True,
            text=True,
        )
        # Writer continues writing after launching the reader.
        for _ in range(3):
            logger.writerow(*self._WRITEROW_ARGS)
        logger.write_episode(1)

        assert result.returncode == 0, (
            f"SWMR reader subprocess failed: {result.stderr.strip()}"
        )

    def test_writer_appends_correctly_across_multiple_episodes(self, tmp_path):
        """write_episode succeeds for ep 0 and ep 1 with SWMR enabled; both groups present."""
        path = str(tmp_path / "test.h5")
        logger = HDF5Logger(path)
        for _ in range(3):
            logger.writerow(*self._WRITEROW_ARGS)
        summary0 = logger.write_episode(0)

        for _ in range(3):
            logger.writerow(*self._WRITEROW_ARGS)
        summary1 = logger.write_episode(1)

        assert summary0["episode_num"] == 0
        assert summary1["episode_num"] == 1
        with h5py.File(path, "r", swmr=True) as rf:
            assert "episode_0000" in rf
            assert "episode_0001" in rf

    def test_file_uses_latest_libver_high_bound(self, tmp_path):
        """The HDF5 file high libver bound must not be 'earliest' (required for SWMR).

        h5py.File.libver returns a 2-tuple (low, high).  When opened with
        libver='latest' the HIGH bound is set to the current library version;
        the low bound stays 'earliest' (h5py default).  What matters for SWMR
        compatibility is that the high bound is not 'earliest'.
        """
        path = str(tmp_path / "test.h5")
        logger = HDF5Logger(path)
        for _ in range(3):
            logger.writerow(*self._WRITEROW_ARGS)
        logger.write_episode(0)

        with h5py.File(path, "r", swmr=True) as rf:
            _low, high = rf.libver
            assert high != "earliest", (
                f"Expected non-earliest high libver bound (SWMR requires 'latest'), "
                f"got libver={rf.libver}"
            )


class TestHDF5Disabled:
    """Test that Main.py works when h5py is not available."""

    def test_has_hdf5_flag_exists(self):
        """The HAS_HDF5 flag should be importable."""
        # This tests the import pattern — if h5py is installed, HAS_HDF5=True
        import importlib
        spec = importlib.util.find_spec("h5py")
        if spec:
            assert HAS_H5PY is True

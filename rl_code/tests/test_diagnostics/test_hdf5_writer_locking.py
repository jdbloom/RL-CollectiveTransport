"""Regression: the training h5 writer must not be crashable by a concurrent reader.

Root cause (2026-07-08): a passive reader (the dispatcher stale-check every
tick, push_metrics every 3 min, an analysis script) opens a running cell's h5
read-only, which takes an HDF5 shared file lock. The writer's next episode open
then fails to acquire its exclusive lock with
``BlockingIOError: [Errno 11] unable to lock file`` and the whole training run
crashes mid-episode — killing the ep~270 headline stop-grad cells before ep500.

Each run writes its OWN unique h5 (never two writers to one file), so the lock
protects nothing and only causes harm. `_open_h5_writer` therefore opens with
file locking disabled. This test reproduces the exact cross-process scenario:
an external process holds a default-locking read handle while the writer opens.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import time

import h5py

from src.hdf5_logger import _open_h5_writer


def _reader_holding_lock(path: str, hold_s: float) -> subprocess.Popen:
    """Spawn a separate process that opens `path` read-only with DEFAULT locking
    (the pre-fix reader behavior) and holds it, taking a shared file lock."""
    src = textwrap.dedent(
        f"""
        import h5py, time
        f = h5py.File({path!r}, "r")   # default locking -> shared lock held
        time.sleep({hold_s})
        f.close()
        """
    )
    return subprocess.Popen([sys.executable, "-c", src])


def test_writer_opens_despite_external_read_lock(tmp_path):
    path = str(tmp_path / "run.h5")
    with h5py.File(path, "w", libver="latest") as f:
        f.create_group("episode_0000")

    proc = _reader_holding_lock(path, hold_s=4.0)
    try:
        time.sleep(1.0)  # let the external reader grab the lock

        # Sanity: a DEFAULT-locking writer open IS blocked in this window — this
        # is the crash we are fixing. (If the platform doesn't lock, skip the
        # assertion rather than fail spuriously.)
        blocked = False
        try:
            h5py.File(path, "a", libver="latest").close()
        except (BlockingIOError, OSError):
            blocked = True

        # The fixed writer (locking disabled) MUST open and write regardless.
        fw = _open_h5_writer(path, "a")
        fw.create_group("episode_0001")
        fw.close()
    finally:
        proc.wait()

    with h5py.File(path, "r", locking=False) as chk:
        assert "episode_0001" in chk.keys()
    # Document intent: on a locking platform the old writer would have crashed.
    assert blocked or os.name == "nt"


def test_writer_basic_open_and_write(tmp_path):
    """The lock-free open still produces a normal, readable file."""
    path = str(tmp_path / "basic.h5")
    fw = _open_h5_writer(path, "a")
    fw.create_group("episode_0000")
    fw.close()
    with h5py.File(path, "r", locking=False) as f:
        assert "episode_0000" in f.keys()

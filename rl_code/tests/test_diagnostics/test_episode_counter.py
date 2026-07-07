import os
import importlib.util
from pathlib import Path

SPEC = Path(__file__).resolve().parents[2] / "src" / "episode_counter.py"
spec = importlib.util.spec_from_file_location("episode_counter", SPEC)
ec = importlib.util.module_from_spec(spec); spec.loader.exec_module(ec)


def test_record_and_total_high_water_mark(tmp_path):
    db = str(tmp_path / "c.db")
    ec.record_episode("/data/run_a/foo.h5", 0, db)   # episode 0 -> 1
    ec.record_episode("/data/run_a/foo.h5", 4, db)   # episode 4 -> 5
    ec.record_episode("/data/run_a/foo.h5", 2, db)   # stale resume, must NOT lower
    ec.record_episode("/data/run_b/bar.h5", 9, db)   # different run -> +10
    assert ec.node_total(db) == 15                    # 5 + 10
    assert ec.run_count(db) == 2


def test_missing_db_is_zero(tmp_path):
    assert ec.node_total(str(tmp_path / "nope.db")) == 0


def test_record_never_raises(tmp_path):
    # unwritable path must be swallowed, not raised
    ec.record_episode("/data/x/foo.h5", 1, "/nonexistent-dir/c.db")  # no exception

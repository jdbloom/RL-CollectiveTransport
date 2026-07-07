"""Durable per-node ledger of executed episodes, keyed by h5 file basename.

Updated once per episode from hdf5_logger.write_episode(). Stores a high-water
mark per run (episode_num + 1) so resumes/re-runs cannot double-count and h5
pruning cannot subtract. All writes are fail-safe: any error is swallowed so
telemetry can never break the training loop. No network I/O.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

_DEFAULT = "~/stelaris/data/episode_counter.db"


def _db_path(db_path: str | None = None, h5_path: str | None = None) -> str:
    # 1. explicit arg wins. 2. env override. 3. derive <repo>/data from the h5
    # path so the writer lands where the website flusher reads (<repo>/data/…)
    # regardless of checkout location. 4. hardcoded default.
    if db_path:
        return os.path.expanduser(db_path)
    env = os.environ.get("STELARIS_EPISODE_COUNTER_DB")
    if env:
        return os.path.expanduser(env)
    marker = "/code/phd_code/"
    if h5_path and marker in h5_path:
        root = h5_path.split(marker, 1)[0]
        return os.path.join(root, "data", "episode_counter.db")
    return os.path.expanduser(_DEFAULT)


def _connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=10)
    con.execute("PRAGMA journal_mode=WAL")           # safe concurrent writers
    con.execute("PRAGMA busy_timeout=10000")
    con.execute(
        "CREATE TABLE IF NOT EXISTS episode_hwm ("
        "  h5_key TEXT PRIMARY KEY, episodes INTEGER NOT NULL, updated REAL)"
    )
    return con


def record_episode(h5_path: str, episode_num: int, db_path: str | None = None) -> None:
    """Record that `episode_num` completed for the run identified by `h5_path`.
    High-water-mark upsert (episodes = max(existing, episode_num+1)). Never raises."""
    try:
        key = os.path.basename(h5_path)
        hwm = int(episode_num) + 1
        con = _connect(_db_path(db_path, h5_path=h5_path))
        try:
            con.execute(
                "INSERT INTO episode_hwm (h5_key, episodes, updated) VALUES (?,?,?) "
                "ON CONFLICT(h5_key) DO UPDATE SET "
                "  episodes = MAX(episode_hwm.episodes, excluded.episodes), "
                "  updated = excluded.updated",
                (key, hwm, time.time()),
            )
            con.commit()
        finally:
            con.close()
    except Exception:                                # telemetry must never break training
        pass


def node_total(db_path: str | None = None) -> int:
    path = _db_path(db_path)
    if not os.path.exists(path):
        return 0
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return int(con.execute("SELECT COALESCE(SUM(episodes),0) FROM episode_hwm").fetchone()[0] or 0)
        finally:
            con.close()
    except Exception:
        return 0


def run_count(db_path: str | None = None) -> int:
    path = _db_path(db_path)
    if not os.path.exists(path):
        return 0
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return int(con.execute("SELECT COUNT(*) FROM episode_hwm").fetchone()[0] or 0)
        finally:
            con.close()
    except Exception:
        return 0

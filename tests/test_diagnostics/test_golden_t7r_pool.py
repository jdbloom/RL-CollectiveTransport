"""T7R golden-equivalence gate — diagnostics GSP-obs pool ring buffer.

Main.py maintains ``diag_gsp_obs_pool`` as a plain list trimmed with
``while len(pool) > _DIAG_POOL_MAX_SIZE: pool.pop(0)`` after each append batch
(Main.py lines 1489-1493) — O(n) per evicted element. The T7-residual
optimization replaces it with ``collections.deque(maxlen=_DIAG_POOL_MAX_SIZE)``
(O(1) eviction).

Main.py is not importable in tests (module-level argparse + ZMQ), so — like
tests/integration/test_golden_t7.py — the legacy implementation is reproduced
VERBATIM inline and compared against the deque semantics for random append
batches crossing the cap. The two consumers Main.py has are asserted directly:
``len(pool)`` (the diagnostics_batch_size threshold) and ``np.stack(pool)``
(the frozen diagnostic batch).
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _legacy_pool_step(pool: list, batch, max_size: int):
    """Verbatim Main.py maintenance: append batch, then while-pop trim."""
    for obs in batch:
        pool.append(np.asarray(obs, dtype=np.float32))
    while len(pool) > max_size:
        pool.pop(0)


def _deque_pool_step(pool: deque, batch):
    """deque(maxlen) maintenance: append only; eviction is implicit."""
    for obs in batch:
        pool.append(np.asarray(obs, dtype=np.float32))


def test_deque_pool_matches_legacy_list_pool():
    rng = np.random.default_rng(7)
    max_size = 64  # scaled-down _DIAG_POOL_MAX_SIZE; semantics are size-agnostic
    legacy: list = []
    ring: deque = deque(maxlen=max_size)

    for _ in range(200):
        batch = rng.standard_normal((int(rng.integers(0, 9)), 6))
        _legacy_pool_step(legacy, batch, max_size)
        _deque_pool_step(ring, batch)

        assert len(ring) == len(legacy)
        if legacy:
            np.testing.assert_array_equal(np.stack(ring), np.stack(legacy))


def test_deque_pool_single_giant_batch_keeps_newest():
    rng = np.random.default_rng(11)
    max_size = 32
    legacy: list = []
    ring: deque = deque(maxlen=max_size)
    batch = rng.standard_normal((100, 6))

    _legacy_pool_step(legacy, batch, max_size)
    _deque_pool_step(ring, batch)

    np.testing.assert_array_equal(np.stack(ring), np.stack(legacy))
    np.testing.assert_array_equal(np.stack(ring)[-1], batch[-1].astype(np.float32))

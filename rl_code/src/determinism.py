"""Determinism utilities for reproducible RL training.

This module provides ``apply_determinism_settings``, a one-call helper that
seeds all relevant RNGs and activates PyTorch's deterministic algorithm mode.

Usage
-----
Call ``apply_determinism_settings(seed)`` AFTER the torch import (so all
backend modules are loaded), but BEFORE constructing any Agent or building
any network. The caller is responsible for setting::

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

**BEFORE** importing torch, since that environment variable is only read once
when the CUDA workspace is initialised.  In Main.py the environment variable
is set at the very top of the file when ``determinism_enabled=true`` is
detected in the config.

Backend coverage
----------------
- CPU  : ``torch.manual_seed`` + ``torch.use_deterministic_algorithms(True)``
- CUDA : ``torch.cuda.manual_seed_all`` + ``cudnn.deterministic = True`` +
         ``cudnn.benchmark = False``
- MPS  : ``torch.mps.manual_seed``  (Apple Silicon)
- NumPy: ``np.random.seed`` (seeds the legacy global RandomState used by
         ``np.random.choice`` inside ``ReplayBuffer.sample_buffer``)
- Python random: ``random.seed``

Replay buffer RNG
-----------------
``ReplayBuffer.sample_buffer`` calls ``np.random.choice`` which draws from the
global NumPy RandomState.  Seeding NumPy with ``seed + 1`` (distinct from the
policy/env seed) ensures replay sampling is deterministic independently of
other NumPy usage in the training loop.

MPS determinism note
--------------------
``torch.use_deterministic_algorithms(True)`` is supported on MPS as of
PyTorch 2.x.  In practice, with the seeding applied here, forward passes and
Adam optimizer steps on MPS produce bit-exact results across independent runs
with the same seed (verified on PyTorch 2.3.1 / macOS 14 / M-series).  If a
future PyTorch version introduces non-determinism on MPS for any operation
used by this codebase, the call site will raise an error at runtime, which is
the correct failure mode (loud > silent).  See ``KNOWN_NONDETERMINISM.md`` at
the repo root for any documented exceptions.
"""
import logging
import random

import numpy as np
import torch as T

_log = logging.getLogger("stelaris.determinism")


def apply_determinism_settings(seed: int, enabled: bool = True) -> None:
    """Seed all RNGs and activate PyTorch deterministic mode.

    Args:
        seed: Base integer seed.  The replay-buffer RNG is seeded with
            ``seed + 1`` to keep replay sampling independent of other
            NumPy usage.
        enabled: When ``False`` this function is a no-op, preserving legacy
            (non-deterministic) behavior for all existing batches.

    Side effects (when ``enabled=True``):
        - ``random.seed(seed)``
        - ``np.random.seed(seed + 1)``   (replay buffer sampling)
        - ``T.manual_seed(seed)``
        - ``T.use_deterministic_algorithms(True)``
        - If CUDA available: ``T.cuda.manual_seed_all(seed)``
        - ``T.backends.cudnn.deterministic = True``
        - ``T.backends.cudnn.benchmark = False``
        - If MPS available: ``T.mps.manual_seed(seed)``
    """
    if not enabled:
        return

    # Python stdlib
    random.seed(seed)

    # NumPy global RandomState — used by ReplayBuffer.sample_buffer via
    # np.random.choice.  Seed with seed+1 so replay sampling is independent.
    np.random.seed(seed + 1)

    # PyTorch CPU (also seeds the default generator for most ops)
    T.manual_seed(seed)

    # Deterministic algorithm mode — raises an error if a non-deterministic
    # op is used, giving a loud failure rather than silent variance.
    T.use_deterministic_algorithms(True)

    # CUDA
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)
    # cuDNN — always set even when no CUDA device is present, so the flags
    # are correct if CUDA becomes available mid-session (e.g., via device
    # migration).  These are no-ops on CPU/MPS.
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

    # MPS (Apple Silicon)
    if T.backends.mps.is_available():
        try:
            T.mps.manual_seed(seed)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "determinism: torch.mps.manual_seed(%d) failed (%s); "
                "MPS operations may not be bit-exact.",
                seed, exc,
            )

    _log.info(
        "determinism: applied seed=%d (replay buffer seed=%d); "
        "use_deterministic_algorithms=True",
        seed, seed + 1,
    )

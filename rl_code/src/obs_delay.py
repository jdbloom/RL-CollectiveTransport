"""Axis-1 CONDITION lever — OBS_DELAY_K (actor-observation delay).

Pre-reg: docs/predictions/2026-07-14-delay-sweep-prereg.md (Option B).

The hypothesis under test: a k-step delay on the actor's egocentric observation
removes the actor's ability to read the CURRENT collective state directly, so a
live forward GSP prediction becomes the only way to recover it. The delay is a
shared handicap applied identically to IC and GSP-N; the prediction is the only
difference between arms.

Scope (what this module is and is NOT):

    * It delays ONLY the ``env_observations`` snapshot that the actor-observation
      block consumes (``make_agent_state`` / the IC ``agent_state =
      env_observations[i]`` lines). At delay k the actor sees the observation from
      step ``t-k``.
    * It does NOT touch the reward path, the prox-filter / GSP-head prox-flag
      computation, the GSP head input (which stays LIVE — the head predicts
      forward from the current input), or the label / store (obj_stats) paths.
      Those all keep reading the live ``env_observations`` in Main.py.

Correctness contract (the two validity-critical invariants):

    1. Default k=0 is a STRICT no-op: the buffer holds one slot, so the value
       pushed this step is the value returned this step — byte-identical to
       passing the live ``env_observations`` straight through. Main.py's default
       ``OBS_DELAY_K=0`` therefore reproduces current behavior exactly.
    2. RESET at every episode boundary — no cross-episode leakage. Main.py owns
       the per-episode lifecycle: it calls :meth:`reset` (or constructs a fresh
       buffer) at each episode boundary so episode N never sees the tail of
       episode N-1. This is the #1 correctness risk and is asserted by the unit
       tests.

The buffer stores a DEEP copy of the pushed observation list. The Main.py
prox-filter mutates ``env_observations[i]`` in place *before* the push, so the
stored snapshot is the post-filter observation exactly as the actor would have
consumed it at that step; copying prevents any later in-place mutation of a live
``env_observations`` from retroactively altering an already-buffered snapshot.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Sequence


class ObsDelayBuffer:
    """Per-episode ring buffer that delays the actor observation by k steps.

    One instance is held by Main.py across a single episode. Each acting step it
    is fed the current (post-prox-filter) ``env_observations`` and returns the
    observation the actor should consume this step: the one from ``t-k`` once the
    buffer has filled past k entries, else the current one (warm-up).

    Parameters
    ----------
    k:
        Delay in steps. Must be a non-negative int. ``k=0`` is a strict pass-through.

    Notes
    -----
    ``deque(maxlen=k+1)`` holds the last ``k+1`` observations. Element ``[0]`` is
    the oldest retained (i.e. ``t-k`` once full); element ``[-1]`` is the current.
    During warm-up (fewer than ``k+1`` entries pushed so far) the current
    observation is returned, so early steps degrade gracefully to no delay rather
    than raising or returning a partial snapshot.
    """

    __slots__ = ("_k", "_buf")

    def __init__(self, k: int) -> None:
        if not isinstance(k, (int,)) or isinstance(k, bool):
            raise TypeError(f"OBS_DELAY_K must be an int, got {type(k).__name__}")
        if k < 0:
            raise ValueError(f"OBS_DELAY_K must be >= 0, got {k}")
        self._k: int = k
        self._buf: deque = deque(maxlen=k + 1)

    @property
    def k(self) -> int:
        return self._k

    def reset(self) -> None:
        """Clear all buffered observations. Call at every episode boundary."""
        self._buf.clear()

    def push_and_get(self, env_observations: Sequence[Any]) -> Any:
        """Append a copy of the current observation list; return the delayed one.

        Parameters
        ----------
        env_observations:
            The current, post-prox-filter per-robot observation list (the exact
            object the actor block would otherwise consume).

        Returns
        -------
        The observation list the actor should consume this step: the ``t-k``
        snapshot once the buffer holds more than ``k`` entries, else the current
        one (warm-up). A deep copy is always stored; the returned value is the
        stored copy (never the live argument once buffered), so downstream
        in-place mutation of the live ``env_observations`` cannot corrupt it.
        """
        self._buf.append(copy.deepcopy(env_observations))
        # len grows 1..k+1 during warm-up; once it reaches k+1 the deque's maxlen
        # evicts the oldest, so buf[0] is exactly t-k. For k=0 the buffer holds a
        # single slot and buf[0] IS the just-pushed current -> strict pass-through.
        if len(self._buf) > self._k:
            return self._buf[0]
        return self._buf[-1]

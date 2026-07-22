"""Blindfold Fusion Probe — OBS_MASK_INDICES (actor egocentric-obs masking).

Pre-reg: docs/research/2026-07-22-blindfold-fusion-probe-spec.md (masking pilot,
step 1).

The hypothesis under test: masking (zeroing) one decision-critical channel from
the actor's NATIVE 31-dim egocentric observation creates a real, restorable
performance deficit. Train IC-masked; a channel whose masking drops IC by
>= 0.15 is a deficit that can later be re-injected through the GSP splice path
(the fusion probe). This module owns ONLY the masking; Main.py owns the single
apply() call per actor-observation site.

The 31-dim obs layout (source of truth: the ARGoS controller
argos/collectiveRlTransport.cpp OBS_DESCRIPTIONS + the m_vecObs assignment
block). Indices into each per-robot observation vector:

    0   robot2goal_dist      distance robot->goal
    1   robot2goal_angle     GOAL-RELATIVE BEARING (deg, robot-local frame)
    2   lwheel               left wheel speed
    3   rwheel               right wheel speed
    4   robot2object_dist    distance robot->payload/cylinder
    5   robot2object_angle   payload-relative bearing (deg, robot-local)
    6   object2goal_dist     distance payload->goal
    7..30 proximity[0..23]   24 foot-bot IR proximity readings (own contact/
                             obstacle sensing)

NOTE (from the same source): the robot's OWN CONTACT FORCE (magnitude, angle,
deltaX, deltaY) is NOT part of this 31-dim actor observation — it is carried on a
SEPARATE ZMQ stats channel (m_vecStats / ZMQSendForceStats) used for logging/MME,
never fed to the actor. The nearest own-contact signal available IN the actor's
native obs is the 24-dim proximity block (indices 7..30).

Scope (what this module is and is NOT):

    * It masks ONLY the ``env_observations`` snapshot the actor-observation block
      consumes (the ``_actor_env_obs`` returned by the obs-delay buffer, i.e. the
      exact object make_agent_state / the IC pass-through consumes).
    * It does NOT touch the reward path (env_observations[i][7:] prox penalty),
      the GSP head input, the prox-filter / GSP-head prox-flag computation, or the
      label / store (obj_stats) paths. Those keep reading the live, UNMASKED
      env_observations in Main.py. Masking here severs the channel ONLY from the
      actor's decision input, which is precisely the pilot's intent.

Correctness contract (the validity-critical invariants):

    1. Empty mask (default) is a STRICT no-op: apply() returns the input list
      unchanged (same objects), so OBS_MASK_INDICES=[] / None reproduces current
      behavior BYTE-FOR-BYTE.
    2. A non-empty mask zeros EXACTLY the listed indices in a COPY of each
      per-robot observation and nothing else; the live env_observations object is
      never mutated (the reward/GSP/label paths must keep the true values).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


class ObsMask:
    """Zeros a fixed set of indices in each per-robot actor observation.

    One instance is held by Main.py for the whole run (the mask is a static
    per-run condition, not per-episode). Each acting step it is fed the actor's
    per-robot observation list and returns the observation the actor should
    consume: unchanged when the mask is empty, else a copy with the listed
    indices zeroed.

    Parameters
    ----------
    indices:
        The obs-vector indices to zero, or None / empty for a strict no-op. Each
        must be a non-negative int strictly less than ``obs_dim``. Duplicates are
        rejected loudly (a duplicated index is almost always a config typo).
    obs_dim:
        Width of each per-robot observation vector (31 for the transport task).
        Used to bounds-check the requested indices at construction so an
        out-of-range mask fails LOUD at startup, not silently mid-run.
    """

    __slots__ = ("_indices", "_obs_dim")

    def __init__(self, indices: Optional[Sequence[int]], obs_dim: int) -> None:
        if not isinstance(obs_dim, int) or isinstance(obs_dim, bool) or obs_dim < 1:
            raise ValueError(f"obs_dim must be a positive int, got {obs_dim!r}")
        self._obs_dim = obs_dim
        # None -> empty (strict no-op). Never use `indices or []`: an empty list is
        # already the no-op and a falsy-remap would be a fail-loud violation.
        idx = [] if indices is None else list(indices)
        clean: list = []
        for v in idx:
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise TypeError(
                    f"OBS_MASK_INDICES entries must be ints, got {v!r} ({type(v).__name__})"
                )
            v = int(v)
            if v < 0 or v >= obs_dim:
                raise ValueError(
                    f"OBS_MASK_INDICES entry {v} out of range for obs_dim={obs_dim} "
                    f"(valid 0..{obs_dim - 1})"
                )
            clean.append(v)
        if len(set(clean)) != len(clean):
            raise ValueError(f"OBS_MASK_INDICES has duplicate indices: {clean}")
        self._indices = tuple(clean)

    @property
    def indices(self) -> tuple:
        return self._indices

    @property
    def enabled(self) -> bool:
        """True when at least one index is masked (the ENGAGED path)."""
        return len(self._indices) > 0

    def apply(self, actor_env_obs: Sequence[Any]) -> Any:
        """Return the actor observation list with the masked indices zeroed.

        Parameters
        ----------
        actor_env_obs:
            The per-robot observation list the actor is about to consume (the
            output of the obs-delay buffer's push_and_get).

        Returns
        -------
        When the mask is empty: the SAME list object, unchanged (strict no-op).
        When non-empty: a NEW list of per-robot copies with exactly ``indices``
        set to 0.0; the input and its arrays are never mutated, so the live
        env_observations the reward / GSP / label paths read stay intact.
        """
        if not self._indices:
            return actor_env_obs
        masked = []
        for obs in actor_env_obs:
            arr = np.array(obs, dtype=np.float32, copy=True)
            arr[list(self._indices)] = 0.0
            masked.append(arr)
        return masked

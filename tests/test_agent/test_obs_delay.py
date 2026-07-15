"""Tests for the OBS_DELAY_K actor-observation delay lever (Axis-1 CONDITION).

Pre-reg: docs/predictions/2026-07-14-delay-sweep-prereg.md (Option B).

Scientific contract (validity-critical): the actor's egocentric observation is
delayed by k steps — at delay k the actor consumes the observation from step
t-k, while every LIVE channel (GSP head input, reward, prox-flags, labels) still
reads step t. The delay is the shared handicap; the prediction is the only
between-arm difference.

The two invariants these tests defend:

    1. k=0 is a STRICT no-op — the value pushed this step is the value returned
       this step, so the default path reproduces current behavior exactly.
    2. RESET at every episode boundary — episode N never sees the tail of episode
       N-1 (the #1 correctness risk called out in the pre-reg).

The delay logic is factored into ``src.obs_delay.ObsDelayBuffer`` (a pure ring
buffer, no Main.py / ARGoS dependency) precisely so it is unit-testable here in
isolation; Main.py owns only the per-episode lifecycle (reset) and the single
push_and_get call in each acting block.
"""

import numpy as np
import pytest

from src.obs_delay import ObsDelayBuffer


# Synthetic per-robot observation list for "step t": a length-R list of length-D
# float arrays whose values encode the timestep, so a delayed read is unambiguous.
def _obs_at_step(t, num_robots=4, obs_dim=31):
    return [
        np.full(obs_dim, float(t) + 0.01 * i, dtype=np.float32)
        for i in range(num_robots)
    ]


def _step_index_of(actor_obs):
    """Recover the integer step encoded in a returned observation list."""
    # Robot 0's value is exactly float(t); undo the encoding.
    return int(round(float(actor_obs[0][0])))


# ── (a) k=0 is a strict no-op ────────────────────────────────────────────────

def test_k0_is_strict_noop_returns_current_step():
    buf = ObsDelayBuffer(0)
    for t in range(10):
        obs = _obs_at_step(t)
        out = buf.push_and_get(obs)
        # The actor sees the CURRENT step's observation — no delay whatsoever.
        assert _step_index_of(out) == t
        np.testing.assert_array_equal(out[0], obs[0])


def test_k0_values_identical_to_live_channel():
    """A 'live' reference channel (the raw pushed obs) equals the delayed read."""
    buf = ObsDelayBuffer(0)
    for t in range(6):
        live = _obs_at_step(t)
        actor = buf.push_and_get(live)
        for r in range(len(live)):
            np.testing.assert_array_equal(actor[r], live[r])


# ── (b) k=2 delays the actor obs while the live channel stays current ─────────

def test_k2_actor_sees_t_minus_2_after_warmup():
    k = 2
    buf = ObsDelayBuffer(k)
    seen = []
    for t in range(8):
        live = _obs_at_step(t)          # the LIVE channel would see step t
        actor = buf.push_and_get(live)  # the actor sees the delayed step
        seen.append(_step_index_of(actor))
    # Warm-up: for the first k steps (t=0,1) the buffer is not yet full, so the
    # actor degrades to the current step. From t=k onward it lags by exactly k.
    assert seen == [0, 1, 0, 1, 2, 3, 4, 5]
    # Explicitly: at every step past warm-up the actor is stale by k, while a
    # live channel would be current.
    for t in range(k, 8):
        assert seen[t] == t - k, f"step {t}: expected t-{k}={t-k}, got {seen[t]}"


def test_k2_live_and_delayed_differ_by_k():
    """The core hypothesis mechanic: live=t, actor=t-k, so they differ by k.

    Every step is fed to the buffer (as Main.py does, once per acting step); the
    live/actor gap of exactly k only holds once the buffer has warmed past k.
    """
    k = 2
    buf = ObsDelayBuffer(k)
    for t in range(6):
        live_step = t
        actor = buf.push_and_get(_obs_at_step(t))
        actor_step = _step_index_of(actor)
        if t >= k:
            assert live_step - actor_step == k
        else:  # warm-up: actor degrades to the current (live) step
            assert live_step - actor_step == 0


def test_k4_lag_is_four():
    k = 4
    buf = ObsDelayBuffer(k)
    seen = [_step_index_of(buf.push_and_get(_obs_at_step(t))) for t in range(10)]
    # warm-up 0..3 return current; from t=4 lag by 4.
    assert seen == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5]


# ── (c) episode reset — no cross-episode leakage ─────────────────────────────

def test_reset_clears_cross_episode_tail():
    k = 2
    buf = ObsDelayBuffer(k)
    # Episode 1: steps 100, 101, 102, 103 (offset so they are unmistakable).
    for t in (100, 101, 102, 103):
        buf.push_and_get(_obs_at_step(t))
    # Boundary: Main.py resets the buffer at every episode boundary.
    buf.reset()
    # Episode 2: fresh steps 0, 1, 2, 3. The early (warm-up) steps must NOT
    # return episode 1's tail (101, 102, ...) — they must return episode 2's own
    # current observation.
    ep2_seen = [_step_index_of(buf.push_and_get(_obs_at_step(t))) for t in range(4)]
    assert ep2_seen == [0, 1, 0, 1]
    assert 101 not in ep2_seen and 102 not in ep2_seen and 103 not in ep2_seen


def test_without_reset_tail_would_leak_control():
    """Control proving the reset is load-bearing: WITHOUT reset the buffer's tail
    from episode 1 leaks into episode 2's first read (this is exactly what
    reset() prevents)."""
    k = 2
    buf = ObsDelayBuffer(k)
    for t in (100, 101, 102):
        buf.push_and_get(_obs_at_step(t))
    # No reset. Episode 2 step 0: the buffer is full, so buf[0] is the (t-2)
    # entry from episode 1 — a leak. This asserts the failure mode reset guards.
    leaked = _step_index_of(buf.push_and_get(_obs_at_step(0)))
    assert leaked == 101, "expected episode-1 tail to leak without reset"


# ── buffer stores a copy — later in-place mutation cannot corrupt history ─────

def test_buffered_snapshot_is_deep_copied():
    k = 2
    buf = ObsDelayBuffer(k)
    live = _obs_at_step(5)
    buf.push_and_get(live)
    # Mutate the live obs in place AFTER it was buffered (Main.py's prox-filter
    # mutates env_observations in place between steps).
    live[0][:] = -999.0
    # Push two more so the mutated-step snapshot surfaces as the delayed read.
    buf.push_and_get(_obs_at_step(6))
    out = buf.push_and_get(_obs_at_step(7))
    # The delayed read must be the ORIGINAL step-5 values, not the -999 mutation.
    assert _step_index_of(out) == 5
    assert not np.any(out[0] == -999.0)


# ── validation guards ────────────────────────────────────────────────────────

def test_negative_k_rejected():
    with pytest.raises(ValueError):
        ObsDelayBuffer(-1)


def test_non_int_k_rejected():
    with pytest.raises(TypeError):
        ObsDelayBuffer(2.0)
    with pytest.raises(TypeError):
        ObsDelayBuffer(True)


def test_k_property_exposed():
    assert ObsDelayBuffer(0).k == 0
    assert ObsDelayBuffer(7).k == 7

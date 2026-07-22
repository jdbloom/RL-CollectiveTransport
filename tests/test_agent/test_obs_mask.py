"""Tests for the OBS_MASK_INDICES actor egocentric-obs masking lever.

Pre-reg: docs/research/2026-07-22-blindfold-fusion-probe-spec.md (masking pilot).

Scientific contract (validity-critical): masking zeros one decision-critical
channel from the actor's NATIVE 31-dim observation ONLY on the actor's decision
input, while every LIVE channel (reward, GSP head input, labels) still reads the
true env_observations. The invariants:

    1. Empty mask (default) is a STRICT no-op — apply() returns the input list
       unchanged, so OBS_MASK_INDICES=[] / None reproduces current behavior
       byte-for-byte.
    2. A non-empty mask zeros EXACTLY the listed indices and NOTHING else, on a
       COPY (the live obs is never mutated).

The mask logic is factored into ``src.obs_mask.ObsMask`` (a pure transform, no
Main.py / ARGoS dependency) precisely so it is unit-testable here in isolation;
Main.py owns only the single apply() call at each actor-observation site.
"""

import numpy as np
import pytest

from src.obs_mask import ObsMask


OBS_DIM = 31


def _obs(num_robots=4, obs_dim=OBS_DIM):
    """Per-robot observation list with distinct, all-nonzero values per index so a
    zeroed index is unambiguous (no value equals 0.0 by construction)."""
    return [
        np.arange(1, obs_dim + 1, dtype=np.float32) + 100.0 * i
        for i in range(num_robots)
    ]


# ── (a) empty mask -> identity (strict no-op) ────────────────────────────────

def test_empty_mask_returns_same_object_unchanged():
    m = ObsMask([], obs_dim=OBS_DIM)
    assert not m.enabled
    obs = _obs()
    out = m.apply(obs)
    # Strict no-op: the SAME list object flows through untouched.
    assert out is obs
    for r in range(len(obs)):
        assert out[r] is obs[r]


def test_none_mask_is_empty_and_noop():
    m = ObsMask(None, obs_dim=OBS_DIM)
    assert m.indices == ()
    assert not m.enabled
    obs = _obs()
    out = m.apply(obs)
    assert out is obs


def test_empty_mask_values_identical_to_input():
    m = ObsMask([], obs_dim=OBS_DIM)
    obs = _obs()
    ref = [a.copy() for a in obs]
    out = m.apply(obs)
    for r in range(len(obs)):
        np.testing.assert_array_equal(out[r], ref[r])


# ── (b) non-empty mask zeros EXACTLY the listed indices, nothing else ─────────

def test_single_index_goal_bearing_zeroed():
    # Goal-relative bearing = obs index 1 (robot2goal_angle).
    m = ObsMask([1], obs_dim=OBS_DIM)
    assert m.enabled
    assert m.indices == (1,)
    obs = _obs()
    ref = [a.copy() for a in obs]
    out = m.apply(obs)
    for r in range(len(obs)):
        assert out[r][1] == 0.0
        # Every OTHER index is preserved exactly.
        for j in range(OBS_DIM):
            if j != 1:
                assert out[r][j] == ref[r][j], f"robot {r} idx {j} changed"


def test_multi_index_proximity_block_zeroed():
    # Own-contact proxy = proximity block, obs indices 7..30.
    prox = list(range(7, 31))
    m = ObsMask(prox, obs_dim=OBS_DIM)
    obs = _obs()
    ref = [a.copy() for a in obs]
    out = m.apply(obs)
    for r in range(len(obs)):
        for j in range(OBS_DIM):
            if j in prox:
                assert out[r][j] == 0.0, f"robot {r} idx {j} not zeroed"
            else:
                assert out[r][j] == ref[r][j], f"robot {r} idx {j} changed"


def test_mask_does_not_mutate_input_live_obs():
    m = ObsMask([1, 5], obs_dim=OBS_DIM)
    obs = _obs()
    ref = [a.copy() for a in obs]
    out = m.apply(obs)
    # The LIVE input arrays are untouched — reward / GSP / label paths keep the
    # true values.
    for r in range(len(obs)):
        np.testing.assert_array_equal(obs[r], ref[r])
    # And the returned copies are genuinely masked.
    for r in range(len(obs)):
        assert out[r][1] == 0.0 and out[r][5] == 0.0
    assert out is not obs


def test_out_is_new_object_when_masked():
    m = ObsMask([1], obs_dim=OBS_DIM)
    obs = _obs()
    out = m.apply(obs)
    assert out is not obs
    for r in range(len(obs)):
        assert out[r] is not obs[r]


# ── (c) construction / validation guards ─────────────────────────────────────

def test_indices_property_and_dedup_order_preserved():
    m = ObsMask([7, 8, 9], obs_dim=OBS_DIM)
    assert m.indices == (7, 8, 9)


def test_duplicate_indices_rejected():
    with pytest.raises(ValueError):
        ObsMask([1, 1], obs_dim=OBS_DIM)


def test_out_of_range_index_rejected():
    with pytest.raises(ValueError):
        ObsMask([31], obs_dim=OBS_DIM)  # valid 0..30
    with pytest.raises(ValueError):
        ObsMask([-1], obs_dim=OBS_DIM)


def test_non_int_index_rejected():
    with pytest.raises(TypeError):
        ObsMask([1.0], obs_dim=OBS_DIM)
    with pytest.raises(TypeError):
        ObsMask([True], obs_dim=OBS_DIM)


def test_bad_obs_dim_rejected():
    with pytest.raises(ValueError):
        ObsMask([], obs_dim=0)
    with pytest.raises(ValueError):
        ObsMask([], obs_dim=-5)


def test_numpy_int_indices_accepted():
    # Config loaders / yaml may hand np.int64; those must be accepted.
    m = ObsMask([np.int64(1), np.int64(5)], obs_dim=OBS_DIM)
    assert m.indices == (1, 5)


# ── default-config coercion (None -> empty, bit-exact off) ────────────────────

def test_config_default_none_coerces_to_noop():
    # Mirrors Main.py: config.get('OBS_MASK_INDICES', None) with the key absent.
    cfg = {}
    raw = cfg.get("OBS_MASK_INDICES", None)
    m = ObsMask(raw, obs_dim=OBS_DIM)
    assert m.indices == () and not m.enabled
    obs = _obs()
    assert m.apply(obs) is obs

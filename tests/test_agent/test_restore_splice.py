"""Tests for the RESTORE-VIA-SPLICE lever (blindfold fusion probe, step 2).

Pre-reg: docs/research/2026-07-22-blindfold-fusion-probe-spec.md (step 2,
"Restore via the splice path").

Scientific contract (validity-critical):

    1. Default (RESTORE_SPLICE_SOURCE_INDEX=None) is a STRICT no-op —
       bit-exact-off, every method a pass-through.
    2. late_splice appends the TRUE (pre-mask) channel value through the SAME
       apparatus the GSP prediction uses — verified against a REAL Agent's
       make_agent_state (no mocks): slot position, width, and value identity
       through the untouched legacy scalar transform.
    3. early_fuse writes the true value back INTO the masked native slot (no
       appended slot, no width change) without mutating the input.
    4. Warm-noise carries range-matched uniform noise for episodes < M and the
       true value from episode M on (late_splice only; deterministic per seed).
    5. Invalid combinations are rejected loudly (early_fuse + warm-noise,
       dangling keys, GSP-enabled, share-prox, out-of-range index).

The transform logic lives in ``src.restore_splice.RestoreSplice`` (pure, no
Main.py / ARGoS dependency) so it is unit-testable here in isolation; Main.py
owns the construction, the ``n_obs + extra_actor_inputs`` width extension, the
per-episode ``set_episode`` and the call sites at both actor-obs assemblies.
"""

import numpy as np
import pytest

from src.agent import Agent
from src.obs_mask import ObsMask
from src.restore_splice import RestoreSplice, RANGE_LOG_EPISODES


OBS_DIM = 31
GOAL_BEARING_IDX = 1  # robot2goal_angle — the pilot's masked channel.


def _obs(num_robots=4, obs_dim=OBS_DIM):
    """Per-robot observation list with distinct, all-nonzero values per index so
    a zeroed/restored index is unambiguous."""
    return [
        np.arange(1, obs_dim + 1, dtype=np.float32) + 100.0 * i
        for i in range(num_robots)
    ]


def _splice(source_index=GOAL_BEARING_IDX, mode=None, warm=None, warm_range=None,
            mask=(GOAL_BEARING_IDX,), gsp=False, share_prox=False,
            train=True, seed=42, obs_dim=OBS_DIM):
    return RestoreSplice(
        source_index=source_index, mode=mode, warm_noise_episodes=warm,
        warm_noise_range=warm_range, mask_indices=mask, obs_dim=obs_dim,
        gsp_enabled=gsp, share_prox_values=share_prox, train_mode=train,
        seed=seed,
    )


@pytest.fixture
def agent(agent_config):
    """A REAL IC Agent (gsp=False) — the exact object whose make_agent_state
    the late_splice arm rides in Main.py. n_obs=32 mirrors Main.py's
    ``num_obs + extra_actor_inputs`` width extension for late_splice runs."""
    return Agent(
        config=agent_config, network='DQN', n_agents=4, n_obs=32,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=False, recurrent=False, attention=False,
        neighbors=False, gsp_input_size=6, gsp_output_size=1,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
    )


# ── (1) default-off identity — bit-exact no-op ───────────────────────────────

def test_default_off_is_strict_noop():
    rs = _splice(source_index=None, mask=())
    assert not rs.enabled
    assert not rs.late_splice_engaged and not rs.early_fuse_engaged
    assert rs.extra_actor_inputs == 0
    obs = _obs()
    assert rs.true_values(obs) is None
    # early-fuse pass-through returns the SAME object.
    assert rs.apply_early_fuse(obs, None) is obs
    assert rs.slot_values(None) is None
    # set_episode is inert when off.
    rs.set_episode(0)
    assert rs.true_values(obs) is None


def test_off_with_dangling_keys_rejected():
    # Configured-but-not-engaged is a misconfig, never a silent skip.
    with pytest.raises(ValueError):
        _splice(source_index=None, mode='late_splice', mask=())
    with pytest.raises(ValueError):
        _splice(source_index=None, warm=200, mask=())
    with pytest.raises(ValueError):
        _splice(source_index=None, warm_range=[-180.0, 180.0], mask=())


# ── (2) late_splice appends the TRUE value; native slot stays masked ─────────

def test_late_splice_true_value_through_real_make_agent_state(agent):
    """The restored value must ride the REAL GSP fusion: same call, same slot
    position (obs | gsp_slot), same width (+1), and the value the actor
    consumes equals the TRUE pre-mask channel value at native scale."""
    rs = _splice()  # late_splice default
    assert rs.late_splice_engaged and rs.extra_actor_inputs == 1
    rs.set_episode(0)

    obs = _obs()
    true_vals = rs.true_values(obs)                      # pre-mask truth
    masked = ObsMask([GOAL_BEARING_IDX], obs_dim=OBS_DIM).apply(obs)
    slot = rs.slot_values(true_vals)

    for i in range(len(obs)):
        state = agent.make_agent_state(
            masked[i], heading_gsp=rs.splice_arg(slot[i]))
        assert len(state) == OBS_DIM + 1                 # [31 obs | 1 restore]
        # Native slot STAYS masked — the restore rides the appended slot only.
        assert state[GOAL_BEARING_IDX] == 0.0
        # Appended slot carries the TRUE pre-mask value (float32 round-trip
        # through the untouched legacy scalar transform, degrees(x/10)).
        assert state[OBS_DIM] == pytest.approx(obs[i][GOAL_BEARING_IDX], rel=1e-5)
        # Everything else is untouched.
        np.testing.assert_array_equal(state[:OBS_DIM], masked[i])


def test_splice_arg_is_identity_through_legacy_scalar_transform():
    # make_agent_state's scalar branch computes degrees(heading_gsp / 10);
    # splice_arg pre-inverts it so the slot equals the raw value.
    for v in (-180.0, -37.25, 0.0, 1.0, 179.9):
        out = np.degrees(RestoreSplice.splice_arg(v) / 10.0)
        assert out == pytest.approx(v, abs=1e-9)


def test_late_splice_does_not_touch_native_obs(agent):
    rs = _splice()
    rs.set_episode(0)
    obs = _obs()
    ref = [a.copy() for a in obs]
    rs.true_values(obs)
    for r in range(len(obs)):
        np.testing.assert_array_equal(obs[r], ref[r])


# ── (3) early_fuse un-masks in place, no appended slot ───────────────────────

def test_early_fuse_restores_masked_slot_in_place(agent):
    rs = _splice(mode='early_fuse')
    assert rs.early_fuse_engaged
    assert rs.extra_actor_inputs == 0                    # no width change
    rs.set_episode(0)

    obs = _obs()
    true_vals = rs.true_values(obs)
    masked = ObsMask([GOAL_BEARING_IDX], obs_dim=OBS_DIM).apply(obs)
    fused = rs.apply_early_fuse(masked, true_vals)

    for i in range(len(obs)):
        # The masked native slot carries the TRUE value again...
        assert fused[i][GOAL_BEARING_IDX] == obs[i][GOAL_BEARING_IDX]
        # ...every other index is the masked copy's value...
        for j in range(OBS_DIM):
            if j != GOAL_BEARING_IDX:
                assert fused[i][j] == masked[i][j]
        # ...and the input masked copies were NOT mutated.
        assert masked[i][GOAL_BEARING_IDX] == 0.0
    # No appended slot: a real Agent consumes the 31-wide vector unchanged.
    assert len(fused[0]) == OBS_DIM
    # slot_values is None in early_fuse mode (nothing rides the appended path).
    assert rs.slot_values(true_vals) is None


def test_early_fuse_passthrough_for_late_splice_mode():
    rs = _splice()  # late_splice
    rs.set_episode(0)
    obs = _obs()
    out = rs.apply_early_fuse(obs, rs.true_values(obs))
    assert out is obs                                    # strict pass-through


# ── (4) warm-noise: noise for episodes < M, truth after ──────────────────────

def test_warm_noise_carries_noise_then_truth():
    M = 3
    lo, hi = -180.0, 180.0
    rs = _splice(warm=M, warm_range=[lo, hi], seed=42)
    obs = _obs()

    for ep in range(M):
        rs.set_episode(ep)
        true_vals = rs.true_values(obs)
        slot = rs.slot_values(true_vals)
        # Noise: range-matched, NOT the true values.
        assert np.all(slot >= lo) and np.all(slot <= hi)
        assert not np.allclose(slot, true_vals)

    rs.set_episode(M)
    true_vals = rs.true_values(obs)
    slot = rs.slot_values(true_vals)
    np.testing.assert_array_equal(slot, true_vals)       # truth from ep M on


def test_warm_noise_is_deterministic_per_seed():
    obs = _obs()
    draws = []
    for _ in range(2):
        rs = _splice(warm=1, warm_range=[-180.0, 180.0], seed=123)
        rs.set_episode(0)
        draws.append(rs.slot_values(rs.true_values(obs)))
    np.testing.assert_array_equal(draws[0], draws[1])
    rs_other = _splice(warm=1, warm_range=[-180.0, 180.0], seed=124)
    rs_other.set_episode(0)
    assert not np.allclose(rs_other.slot_values(rs_other.true_values(obs)), draws[0])


def test_warm_noise_forced_inert_in_test_mode():
    # Eval configs are cloned from training configs: a test run must consume
    # the TRUE value from episode 0, never the warm noise.
    rs = _splice(warm=200, warm_range=[-180.0, 180.0], train=False)
    assert rs.warm_noise_episodes == 0
    rs.set_episode(0)
    obs = _obs()
    true_vals = rs.true_values(obs)
    np.testing.assert_array_equal(rs.slot_values(true_vals), true_vals)


# ── (5) invalid combinations rejected loudly ─────────────────────────────────

def test_early_fuse_plus_warm_noise_rejected():
    with pytest.raises(ValueError):
        _splice(mode='early_fuse', warm=200, warm_range=[-180.0, 180.0])


def test_warm_noise_requires_range():
    with pytest.raises(ValueError):
        _splice(warm=200)                                # no range -> loud
    with pytest.raises(ValueError):
        _splice(warm=0, warm_range=[-180.0, 180.0])      # dangling range
    with pytest.raises(ValueError):
        _splice(warm=200, warm_range=[180.0, -180.0])    # lo >= hi
    with pytest.raises(ValueError):
        _splice(warm=200, warm_range=[0.0])              # not [lo, hi]


def test_gsp_enabled_rejected():
    # The GSP prediction owns the appended slot — a run cannot splice both.
    with pytest.raises(ValueError):
        _splice(gsp=True)


def test_share_prox_values_rejected():
    with pytest.raises(ValueError):
        _splice(share_prox=True)


def test_bad_source_index_rejected():
    with pytest.raises(ValueError):
        _splice(source_index=OBS_DIM)                    # out of range
    with pytest.raises(ValueError):
        _splice(source_index=-1)
    with pytest.raises(TypeError):
        _splice(source_index=True)                       # bool is not an index
    with pytest.raises(ValueError):
        _splice(mode='mid_fuse')                         # unknown mode
    with pytest.raises(TypeError):
        _splice(warm=1.5, warm_range=[-1.0, 1.0])        # non-int episodes


def test_restore_of_unmasked_channel_warns_hard(caplog):
    # Not fatal (a deliberate redundancy control is conceivable) but ERROR-loud:
    # restoring a channel the actor already sees makes the probe meaningless.
    import logging
    with caplog.at_level(logging.ERROR, logger='src.restore_splice'):
        rs = _splice(mask=(5,))                          # index 1 not masked
    assert rs.enabled
    assert any('NOT in OBS_MASK_INDICES' in r.message for r in caplog.records)


def test_ordering_guards_fail_loud():
    rs = _splice()
    obs = _obs()
    with pytest.raises(RuntimeError):
        rs.true_values(obs)                              # before set_episode
    rs.set_episode(0)
    with pytest.raises(RuntimeError):
        rs.slot_values(None)                             # truth not captured
    rs_ef = _splice(mode='early_fuse')
    rs_ef.set_episode(0)
    with pytest.raises(RuntimeError):
        rs_ef.apply_early_fuse(obs, None)                # truth not captured


# ── engaged-path logging (assert the ENGAGED path) ───────────────────────────

def test_first_episodes_range_logged(caplog):
    import logging
    rs = _splice()
    obs = _obs()
    with caplog.at_level(logging.INFO, logger='src.restore_splice'):
        for ep in range(RANGE_LOG_EPISODES + 1):
            rs.set_episode(ep)
            rs.true_values(obs)
    range_logs = [r for r in caplog.records
                  if 'engaged-path check' in r.getMessage()]
    assert len(range_logs) == RANGE_LOG_EPISODES
    # The logged range brackets the true channel values fed in.
    expected_min = min(float(o[GOAL_BEARING_IDX]) for o in obs)
    expected_max = max(float(o[GOAL_BEARING_IDX]) for o in obs)
    msg = range_logs[0].getMessage()
    assert f"[{expected_min:.4f}, {expected_max:.4f}]" in msg


def test_describe_names_mode_index_and_warm():
    rs = _splice(warm=200, warm_range=[-180.0, 180.0])
    d = rs.describe()
    assert 'ENGAGED' in d and 'late_splice' in d
    assert 'source_index=1' in d and 'warm_noise_episodes=200' in d
    assert _splice(source_index=None, mask=()).describe() == 'off (no-op)'

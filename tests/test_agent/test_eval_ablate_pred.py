"""Tests for the eval-time prediction-ablation flag GSP_EVAL_ABLATE_PRED (M2).

Scientific contract: the GSP prediction for every head variant funnels into
``next_heading_gsp[i]`` at the single injection site in Main.py. Immediately
after it is set, ``apply_pred_ablation`` transforms the per-robot prediction
vector according to the mode:

    none         -> identity (literal no-op; bit-exactness depends on this)
    zero         -> all-zeros, same shape/dtype
    shuffle      -> same multiset, order permuted deterministically under a seeded rng
    frozen_mean  -> the running mean of predictions accumulated across the episode

This mirrors the GSP_ZERO_OUT_SIGNAL / GSP_EVAL_ABLATE_NEIGHBORS ablation family:
the flag is host-side (parsed on the Agent, consumed in Main.py), and the default
`none` path is a strict bit-exact no-op vs the un-ablated run.

The helper is PURE (no Main.py import — Main.py runs argparse at module level and
needs ZMQ/ARGoS). It is tested here in isolation; the golden guard in
tests/integration/test_golden_t7.py proves the `none` path is a literal identity.
"""

import numpy as np
import pytest

from src.pred_ablation import apply_pred_ablation, RunningMeanState
from src.agent import Agent


# ── Flag parse (mirrors test_gsp_eval_ablate_neighbors / test_gsp_zero_out) ─────

def _base_config(**overrides):
    cfg = {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }
    cfg.update(overrides)
    return cfg


def _make_agent(**cfg_overrides):
    cfg = _base_config(**cfg_overrides)
    return Agent(
        config=cfg, network='DDQN', n_agents=4, n_obs=31, n_actions=2,
        options_per_action=3, id=0, min_max_action=0.1, meta_param_size=1,
        gsp=True, recurrent=False, attention=True, neighbors=True,
        gsp_input_size=6, gsp_output_size=1, gsp_min_max_action=1.0,
        gsp_look_back=2, gsp_sequence_length=5, n_hop_neighbors=1,
    )


class TestFlagParse:
    def test_flag_defaults_none(self):
        agent = _make_agent()
        assert agent.gsp_eval_ablate_pred == 'none'

    def test_flag_absent_is_none(self):
        cfg = _base_config()  # no GSP_EVAL_ABLATE_PRED key at all
        agent = Agent(
            config=cfg, network='DDQN', n_agents=4, n_obs=31, n_actions=2,
            options_per_action=3, id=0, min_max_action=0.1, meta_param_size=1,
            gsp=True, recurrent=False, attention=True, neighbors=True,
            gsp_input_size=6, gsp_output_size=1, gsp_min_max_action=1.0,
            gsp_look_back=2, gsp_sequence_length=5, n_hop_neighbors=1,
        )
        assert agent.gsp_eval_ablate_pred == 'none'

    @pytest.mark.parametrize("mode", ['none', 'zero', 'shuffle', 'frozen_mean'])
    def test_flag_picked_up(self, mode):
        agent = _make_agent(GSP_EVAL_ABLATE_PRED=mode)
        assert agent.gsp_eval_ablate_pred == mode


# ── Pure helper: apply_pred_ablation ────────────────────────────────────────────

class TestNoneIdentity:
    def test_none_returns_same_object(self):
        """`none` must be a LITERAL identity no-op — same object, not a copy.
        Bit-exactness of the default training path depends on this."""
        pred = np.array([0.3, -0.7, 1.1], dtype=np.float32)
        rng = np.random.default_rng(0)
        state = RunningMeanState()
        out = apply_pred_ablation(pred, 'none', rng, state)
        assert out is pred  # identical object, no allocation

    def test_none_leaves_running_mean_untouched(self):
        """`none` must not accumulate into the running-mean state."""
        pred = np.array([1.0, 2.0], dtype=np.float32)
        rng = np.random.default_rng(0)
        state = RunningMeanState()
        apply_pred_ablation(pred, 'none', rng, state)
        assert state.count == 0


class TestZero:
    def test_zero_all_zeros_same_shape_dtype(self):
        pred = np.array([0.3, -0.7, 1.1], dtype=np.float32)
        rng = np.random.default_rng(0)
        out = apply_pred_ablation(pred, 'zero', rng, RunningMeanState())
        assert out.shape == pred.shape
        assert out.dtype == pred.dtype
        np.testing.assert_array_equal(out, np.zeros_like(pred))

    def test_zero_does_not_mutate_input(self):
        pred = np.array([0.3, -0.7, 1.1], dtype=np.float32)
        orig = pred.copy()
        apply_pred_ablation(pred, 'zero', np.random.default_rng(0), RunningMeanState())
        np.testing.assert_array_equal(pred, orig)


class TestShuffle:
    def test_shuffle_same_multiset(self):
        pred = np.array([0.3, -0.7, 1.1, 4.2, -2.5], dtype=np.float32)
        rng = np.random.default_rng(123)
        out = apply_pred_ablation(pred, 'shuffle', rng, RunningMeanState())
        np.testing.assert_array_equal(np.sort(out), np.sort(pred))
        assert out.shape == pred.shape
        assert out.dtype == pred.dtype

    def test_shuffle_changes_order(self):
        """With a non-degenerate seed the order must actually change."""
        pred = np.array([0.3, -0.7, 1.1, 4.2, -2.5], dtype=np.float32)
        rng = np.random.default_rng(123)
        out = apply_pred_ablation(pred, 'shuffle', rng, RunningMeanState())
        assert not np.array_equal(out, pred)

    def test_shuffle_deterministic_under_seeded_rng(self):
        pred = np.array([0.3, -0.7, 1.1, 4.2, -2.5], dtype=np.float32)
        out_a = apply_pred_ablation(pred, 'shuffle', np.random.default_rng(7),
                                    RunningMeanState())
        out_b = apply_pred_ablation(pred, 'shuffle', np.random.default_rng(7),
                                    RunningMeanState())
        np.testing.assert_array_equal(out_a, out_b)

    def test_shuffle_does_not_mutate_input(self):
        pred = np.array([0.3, -0.7, 1.1, 4.2, -2.5], dtype=np.float32)
        orig = pred.copy()
        apply_pred_ablation(pred, 'shuffle', np.random.default_rng(7),
                            RunningMeanState())
        np.testing.assert_array_equal(pred, orig)


class TestFrozenMean:
    def test_frozen_mean_first_step_returns_itself(self):
        """First step: running mean == the single sample so far."""
        pred = np.array([2.0, 4.0], dtype=np.float32)
        state = RunningMeanState()
        out = apply_pred_ablation(pred, 'frozen_mean', np.random.default_rng(0), state)
        np.testing.assert_allclose(out, pred)

    def test_frozen_mean_accumulates_across_episode(self):
        """Every step returns the running mean of ALL predictions seen so far."""
        state = RunningMeanState()
        rng = np.random.default_rng(0)
        p1 = np.array([2.0, 4.0], dtype=np.float32)
        p2 = np.array([4.0, 8.0], dtype=np.float32)
        p3 = np.array([6.0, 12.0], dtype=np.float32)
        out1 = apply_pred_ablation(p1, 'frozen_mean', rng, state)
        out2 = apply_pred_ablation(p2, 'frozen_mean', rng, state)
        out3 = apply_pred_ablation(p3, 'frozen_mean', rng, state)
        np.testing.assert_allclose(out1, [2.0, 4.0])
        np.testing.assert_allclose(out2, [3.0, 6.0])   # mean of p1,p2
        np.testing.assert_allclose(out3, [4.0, 8.0])   # mean of p1,p2,p3

    def test_frozen_mean_shape_dtype_preserved(self):
        pred = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        out = apply_pred_ablation(pred, 'frozen_mean', np.random.default_rng(0),
                                  RunningMeanState())
        assert out.shape == pred.shape
        assert out.dtype == pred.dtype

    def test_frozen_mean_reset_state_starts_fresh(self):
        """A fresh RunningMeanState (per-episode reset) does not carry old samples."""
        state = RunningMeanState()
        apply_pred_ablation(np.array([100.0], dtype=np.float32), 'frozen_mean',
                            np.random.default_rng(0), state)
        state2 = RunningMeanState()  # simulate episode boundary reset
        out = apply_pred_ablation(np.array([5.0], dtype=np.float32), 'frozen_mean',
                                  np.random.default_rng(0), state2)
        np.testing.assert_allclose(out, [5.0])


class TestUnknownMode:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            apply_pred_ablation(np.array([1.0], dtype=np.float32), 'bogus',
                                np.random.default_rng(0), RunningMeanState())

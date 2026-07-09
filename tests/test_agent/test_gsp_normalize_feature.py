"""Tests for GSP_E2E_NORMALIZE_FEATURE at the ACTING splice (make_agent_state).

The lever standardizes the spliced GSP prediction to ~unit variance so it lands
on the scale of the O(1) egocentric obs. The RunningStandardizer that does it
lives on the Agent (which subclasses GSP-RL's Actor) as self.gsp_feature_stats,
and is the SAME instance the E2E learn splices update — so acting and learning
standardize identically. Acting READS frozen stats and never updates them.

Coverage here (acting side):
  (b)  flag OFF -> byte-identical: gsp_feature_stats is None, slot is the plain
       scaled prediction, exactly as today.
  (c)  the SAME shared stats object is used at acting (make_agent_state) and would
       be used at learning (both reach it via self.gsp_feature_stats).
  (n)  flag ON with warmed stats -> the slot is the standardized scaled pred.
  (d)  make_agent_state does NOT update the stats (eval reads frozen).
  (abl) frozen_mean ablation composes: feeding the per-episode mean prediction
        through standardization yields ~0.
  (zero) GSP_ZERO_OUT_SIGNAL still severs to exactly 0 even with the flag on.

Uses injected/synthetic values only -> zero-spend, no experiment data.
"""
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src"))

from agent import Agent  # noqa: E402
from gsp_rl.src.actors.feature_stats import RunningStandardizer  # noqa: E402


def _base_agent_kwargs(config):
    return {
        "config": config,
        "network": "DDQN",
        "n_agents": 4,
        "n_obs": 8,
        "n_actions": 2,
        "options_per_action": 3,
        "id": 1,
        "min_max_action": 1.0,
        "meta_param_size": 2,
        "gsp": True,
        "recurrent": False,
        "attention": False,
        "neighbors": True,
        "gsp_input_size": 6,
        "gsp_output_size": 1,
        "gsp_min_max_action": 1.0,
        "gsp_look_back": 2,
        "gsp_sequence_length": 5,
    }


def _load_base_config():
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "GSP-RL",
        "tests", "test_actor", "config.yml",
    )
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


ENV_OBS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


# --- (b) flag default off + byte-identical acting ---

def test_flag_default_off_stats_none():
    agent = Agent(**_base_agent_kwargs(_load_base_config()))
    assert getattr(agent, "gsp_e2e_normalize_feature", False) is False
    assert agent.gsp_feature_stats is None


def test_flag_off_slot_is_plain_scaled_pred():
    """Flag off -> the slot is exactly degrees(pred/10), byte-identical to today."""
    agent = Agent(**_base_agent_kwargs(_load_base_config()))
    aug = agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    # float32 slot storage (existing behavior) -> match the sibling zero-out test's
    # 1e-6 tolerance.
    assert abs(aug[-1] - np.degrees(0.5 / 10)) < 1e-6
    assert aug.shape == (9,)


def test_flag_off_bit_identical_regardless_of_explicit_false():
    """Setting GSP_E2E_NORMALIZE_FEATURE=False explicitly must give the identical
    slot as omitting it — proving the default path is untouched."""
    cfg = _load_base_config()
    cfg["GSP_E2E_NORMALIZE_FEATURE"] = False
    a_off = Agent(**_base_agent_kwargs(cfg))
    a_default = Agent(**_base_agent_kwargs(_load_base_config()))
    aug_off = a_off.make_agent_state(ENV_OBS.copy(), heading_gsp=0.37)
    aug_default = a_default.make_agent_state(ENV_OBS.copy(), heading_gsp=0.37)
    np.testing.assert_array_equal(aug_off, aug_default)


# --- (n) flag on -> standardized slot ---

def _agent_with_flag_on():
    cfg = _load_base_config()
    cfg["GSP_E2E_ENABLED"] = True
    cfg["GSP_E2E_NORMALIZE_FEATURE"] = True
    return Agent(**_base_agent_kwargs(cfg))


def test_flag_on_creates_stats_with_correct_dim():
    agent = _agent_with_flag_on()
    assert isinstance(agent.gsp_feature_stats, RunningStandardizer)
    assert agent.gsp_feature_stats.dim == agent.gsp_network_output == 1


def test_flag_on_slot_is_standardized_scaled_pred():
    agent = _agent_with_flag_on()
    # Warm the stats so it is not the count==0 identity. The acting slot value is
    # the SCALED pred (degrees(pred/10)), so warm the stats on that representation.
    scale = np.degrees(1.0) / 10.0
    rng = np.random.default_rng(0)
    warm = (rng.normal(0.31, 0.024, size=(1000, 1)) * scale)  # scaled pred distribution
    agent.gsp_feature_stats.update(warm)

    pred = 0.5
    scaled = np.degrees(pred / 10.0)
    expected = (np.float32(scaled) - agent.gsp_feature_stats.mean[0].astype(np.float32)) / \
        np.sqrt(agent.gsp_feature_stats.var[0] + agent.gsp_feature_stats.eps).astype(np.float32)

    aug = agent.make_agent_state(ENV_OBS.copy(), heading_gsp=pred)
    assert aug[-1] == pytest.approx(float(expected), rel=1e-4, abs=1e-5)
    # env_obs untouched.
    np.testing.assert_array_equal(aug[:-1], ENV_OBS)


def test_acting_does_not_update_stats():
    """(d) make_agent_state is READ-only w.r.t. the running stats — eval reads
    frozen stats. Only the learn splice updates them."""
    agent = _agent_with_flag_on()
    agent.gsp_feature_stats.update(np.array([[0.1], [0.3], [0.5]]))
    c0 = agent.gsp_feature_stats.count
    m0 = agent.gsp_feature_stats.mean.copy()
    for _ in range(5):
        agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.9)
    assert agent.gsp_feature_stats.count == c0
    np.testing.assert_array_equal(agent.gsp_feature_stats.mean, m0)


# --- (c) shared-stats identity ---

def test_same_stats_object_shared_across_calls():
    """The Agent holds ONE RunningStandardizer instance; every make_agent_state call
    (acting) and every learn_*_e2e call (learning) reads/writes that same object via
    self.gsp_feature_stats — the consistency contract between acting and learning."""
    agent = _agent_with_flag_on()
    s1 = agent.gsp_feature_stats
    agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    s2 = agent.gsp_feature_stats
    assert s1 is s2


# --- (abl) composition with frozen_mean ablation ---

def test_frozen_mean_ablation_composes_to_near_zero_after_standardization():
    """The GSP_EVAL_ABLATE_PRED=frozen_mean ablation replaces the live prediction
    with the per-episode running MEAN of predictions BEFORE make_agent_state runs.
    Standardizing that mean with stats whose mean ~matches it yields ~0 — the
    ablation still severs the signal, now on the normalized scale."""
    agent = _agent_with_flag_on()
    scale = np.degrees(1.0) / 10.0
    rng = np.random.default_rng(1)
    preds = rng.normal(0.31, 0.024, size=(2000, 1))
    agent.gsp_feature_stats.update(preds * scale)  # stats over the scaled slot values

    # frozen_mean would feed the mean prediction; make_agent_state scales it, then
    # standardizes. The standardized value must be near zero (signal severed).
    frozen_pred = float(preds.mean())
    aug = agent.make_agent_state(ENV_OBS.copy(), heading_gsp=frozen_pred)
    assert abs(aug[-1]) < 0.1


# --- (zero) zero-out still severs to exactly 0 with the flag on ---

def test_zero_out_signal_still_zero_with_flag_on():
    cfg = _load_base_config()
    cfg["GSP_E2E_ENABLED"] = True
    cfg["GSP_E2E_NORMALIZE_FEATURE"] = True
    cfg["GSP_ZERO_OUT_SIGNAL"] = True
    agent = Agent(**_base_agent_kwargs(cfg))
    agent.gsp_feature_stats.update(np.array([[1.0], [2.0], [3.0]]))  # non-zero mean
    aug = agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    # Zero-out severs to EXACTLY 0 — standardization must NOT be applied to a
    # deliberately-severed slot (else it would become -mean/std != 0).
    assert aug[-1] == 0.0

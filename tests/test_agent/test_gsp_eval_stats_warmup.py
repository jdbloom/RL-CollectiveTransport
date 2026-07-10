"""Tests for the eval-time feature-stats warm-up hook
(GSP_EVAL_FEATURE_STATS_WARMUP_EPISODES) at the acting splice.

Motivating incident (2026-07-10): checkpoints saved before GSP-RL#37 carry no
RunningStandardizer state and eval processes never learn, so standardize() was
the count==0 identity in every fresh-process ablation eval — the actor
received the raw tiny-scale feature it was not trained on, in both arms,
voiding the abl500r2 paired-gap verdict. The warm-up lets an eval cell rebuild
the stats from the live prediction stream during burn-in episodes: Main.py
sets `gsp_eval_stats_warmup_active` on each Agent for episodes < W, and
make_agent_state — the one place that sees the slot at the exact per-kind
scale the learn splice standardized — folds each prediction into the stats
before reading them.

Uses injected/synthetic values only -> zero-spend, no experiment data.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src"))

import yaml  # noqa: E402

from agent import Agent  # noqa: E402


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


def _agent_with_flag_on():
    cfg = _load_base_config()
    cfg["GSP_E2E_ENABLED"] = True
    cfg["GSP_E2E_NORMALIZE_FEATURE"] = True
    return Agent(**_base_agent_kwargs(cfg))


ENV_OBS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


def test_default_acting_never_updates_stats():
    """Without the warm-up flag, acting reads frozen stats — the pre-existing
    contract, byte-identical (count stays 0 across many acting steps)."""
    agent = _agent_with_flag_on()
    for v in (0.1, 0.2, 0.3):
        agent.make_agent_state(ENV_OBS.copy(), heading_gsp=v)
    assert agent.gsp_feature_stats.count == 0


def test_warmup_flag_folds_predictions_into_stats():
    """With the flag set (Main.py burn-in episodes), each acting step updates
    the stats with the post-scale slot; the stats converge to the stream's
    mean/std so a later standardized slot is on the ~unit scale."""
    agent = _agent_with_flag_on()
    agent.gsp_eval_stats_warmup_active = True
    scale = np.degrees(1.0) / 10.0
    rng = np.random.default_rng(3)
    preds = rng.normal(0.5, 0.05, size=400)
    for v in preds:
        agent.make_agent_state(ENV_OBS.copy(), heading_gsp=float(v))
    stats = agent.gsp_feature_stats
    assert stats.count == 400
    np.testing.assert_allclose(stats.mean, (preds * scale).mean(), rtol=1e-5)
    np.testing.assert_allclose(
        np.sqrt(stats.var), (preds * scale).std(), rtol=1e-2
    )

    # Freeze (warm-up over) and act: slot is standardized with the warm stats.
    agent.gsp_eval_stats_warmup_active = False
    aug = agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.55)
    expected = (0.55 * scale - stats.mean[0]) / np.sqrt(stats.var[0] + stats.eps)
    assert abs(aug[-1] - expected) < 1e-4
    assert stats.count == 400  # frozen again


def test_warmup_updates_use_post_scale_representation():
    """The stats must be fed the SAME representation the learn splice
    standardized (the scaled slot, degrees(pred/10) on the scalar path) —
    feeding raw preds would warm the stats on the wrong scale."""
    agent = _agent_with_flag_on()
    agent.gsp_eval_stats_warmup_active = True
    agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    scale = np.degrees(1.0) / 10.0
    np.testing.assert_allclose(agent.gsp_feature_stats.mean, [0.5 * scale], rtol=1e-6)


def test_warmup_with_zero_out_does_not_update():
    """GSP_ZERO_OUT_SIGNAL severs the slot; a severed constant must not be
    folded into the stats even during warm-up."""
    cfg = _load_base_config()
    cfg["GSP_E2E_ENABLED"] = True
    cfg["GSP_E2E_NORMALIZE_FEATURE"] = True
    cfg["GSP_ZERO_OUT_SIGNAL"] = True
    agent = Agent(**_base_agent_kwargs(cfg))
    agent.gsp_eval_stats_warmup_active = True
    agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    assert agent.gsp_feature_stats.count == 0


def test_warmup_standardizes_with_frozen_stats_then_updates():
    """Order contract (matches the learn splice): the returned slot is
    standardized with the PRE-update stats. The very first warm-up step must
    therefore pass through the count==0 identity (raw scaled slot), not
    (x-mean)/sqrt(eps) garbage."""
    agent = _agent_with_flag_on()
    agent.gsp_eval_stats_warmup_active = True
    scale = np.degrees(1.0) / 10.0
    aug = agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    assert abs(aug[-1] - 0.5 * scale) < 1e-6  # identity read, THEN update
    assert agent.gsp_feature_stats.count == 1


def test_warmup_skips_all_zero_placeholder_slot():
    """Main.py's episode-init make_agent_state call runs with the zero-
    initialized placeholder slot before any head inference — placeholders
    must not be folded into the stats."""
    agent = _agent_with_flag_on()
    agent.gsp_eval_stats_warmup_active = True
    agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.0)
    assert agent.gsp_feature_stats.count == 0
    agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    assert agent.gsp_feature_stats.count == 1


def test_warmup_skips_ema_mode_stats():
    """EMA-mode stats tick the half-life clock per update CALL; warm-up's
    per-robot-step single-sample calls are a different estimator than the
    per-learn-step batch updates the policy was trained with — EMA stats must
    come from the checkpoint npz, never from warm-up."""
    cfg = _load_base_config()
    cfg["GSP_E2E_ENABLED"] = True
    cfg["GSP_E2E_NORMALIZE_FEATURE"] = True
    cfg["GSP_E2E_NORMALIZE_EMA_HALFLIFE"] = 100.0
    agent = Agent(**_base_agent_kwargs(cfg))
    agent.gsp_eval_stats_warmup_active = True
    agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    assert agent.gsp_feature_stats.count == 0


def test_warmup_flag_without_normalize_flag_is_inert():
    """Warm-up flag set but normalize lever off -> gsp_feature_stats is None,
    make_agent_state stays byte-identical (no crash, plain scaled slot)."""
    cfg = _load_base_config()
    agent = Agent(**_base_agent_kwargs(cfg))
    agent.gsp_eval_stats_warmup_active = True
    aug = agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    assert abs(aug[-1] - np.degrees(0.5 / 10)) < 1e-6


# ---------------------------------------------------------------------------
# GSP_E2E_SPLICE_GAIN at the acting splice (GSP-RL#39 companion)
# ---------------------------------------------------------------------------

def test_splice_gain_default_is_noop():
    agent = _agent_with_flag_on()
    scale = np.degrees(1.0) / 10.0
    aug = agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    # gain defaults to 1.0 (parsed in GSP-RL NetworkAids) -> plain scaled slot
    # (stats cold -> standardize identity).
    assert abs(aug[-1] - 0.5 * scale) < 1e-6


def test_splice_gain_multiplies_slot_after_standardizer():
    cfg = _load_base_config()
    cfg["GSP_E2E_ENABLED"] = True
    cfg["GSP_E2E_NORMALIZE_FEATURE"] = True
    cfg["GSP_E2E_SPLICE_GAIN"] = 10.0
    agent = Agent(**_base_agent_kwargs(cfg))
    assert agent.gsp_e2e_splice_gain == 10.0
    scale = np.degrees(1.0) / 10.0
    # Stats cold -> standardize is identity -> slot = raw*scale*gain.
    aug = agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    assert abs(aug[-1] - 0.5 * scale * 10.0) < 1e-5


def test_splice_gain_zero_out_still_severs_to_zero():
    cfg = _load_base_config()
    cfg["GSP_E2E_ENABLED"] = True
    cfg["GSP_E2E_NORMALIZE_FEATURE"] = True
    cfg["GSP_E2E_SPLICE_GAIN"] = 10.0
    cfg["GSP_ZERO_OUT_SIGNAL"] = True
    agent = Agent(**_base_agent_kwargs(cfg))
    aug = agent.make_agent_state(ENV_OBS.copy(), heading_gsp=0.5)
    assert aug[-1] == 0.0

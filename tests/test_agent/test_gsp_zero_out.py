"""Tests for GSP_ZERO_OUT_SIGNAL flag — H-14 ablation.

The flag causes make_agent_state to ZERO OUT the GSP prediction slot in the
augmented observation, while leaving the GSP head's architecture and training
loop untouched. This is the QMIP-minus pattern: same architecture, same training,
signal removed — tests whether the GSP prediction contributes anything.

See docs/research/gsp-hypothesis-tracker.md H-14 entry.
"""
import os
import sys

import numpy as np
import pytest
import yaml


# Add RL-CT src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src"))

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
    # Reuse the GSP-RL test config (has all required keys)
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "GSP-RL",
        "tests", "test_actor", "config.yml",
    )
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def test_gsp_zero_out_signal_default_false():
    """Without GSP_ZERO_OUT_SIGNAL in config, agent.gsp_zero_out_signal must be False (legacy behavior)."""
    config = _load_base_config()
    agent = Agent(**_base_agent_kwargs(config))
    assert getattr(agent, "gsp_zero_out_signal", False) is False


def test_gsp_zero_out_signal_true_flag():
    """With GSP_ZERO_OUT_SIGNAL=True, the agent picks up the flag."""
    config = _load_base_config()
    config["GSP_ZERO_OUT_SIGNAL"] = True
    agent = Agent(**_base_agent_kwargs(config))
    assert agent.gsp_zero_out_signal is True


def test_make_agent_state_default_passes_gsp_through():
    """Default (flag off): make_agent_state concatenates heading_gsp scaled into env_obs."""
    config = _load_base_config()
    agent = Agent(**_base_agent_kwargs(config))
    env_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    heading_gsp = 0.5  # Non-zero GSP prediction
    augmented = agent.make_agent_state(env_obs, heading_gsp=heading_gsp)
    # The GSP slot in the augmented obs should be np.degrees(0.5/10) = ~2.865, NOT zero
    assert abs(augmented[-1] - np.degrees(0.5 / 10)) < 1e-6
    assert augmented.shape == (9,)  # 8 env_obs + 1 gsp


def test_make_agent_state_with_zero_out_forces_zero():
    """With GSP_ZERO_OUT_SIGNAL=True: the GSP slot in the augmented obs is ZERO regardless of
    the heading_gsp value passed in. Same output shape as normal (actor sees same-shape input)."""
    config = _load_base_config()
    config["GSP_ZERO_OUT_SIGNAL"] = True
    agent = Agent(**_base_agent_kwargs(config))
    env_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    heading_gsp = 0.5
    augmented = agent.make_agent_state(env_obs, heading_gsp=heading_gsp)
    # GSP slot must be EXACTLY zero
    assert augmented[-1] == 0.0
    # Shape must still be 8 + 1 = 9 (architecture unchanged — same input_size for the actor)
    assert augmented.shape == (9,)


def test_make_agent_state_with_zero_out_preserves_env_obs():
    """The 8 env_obs values must be unchanged; only the GSP slot is zeroed."""
    config = _load_base_config()
    config["GSP_ZERO_OUT_SIGNAL"] = True
    agent = Agent(**_base_agent_kwargs(config))
    env_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    augmented = agent.make_agent_state(env_obs, heading_gsp=0.5)
    np.testing.assert_array_equal(augmented[:-1], env_obs)


def test_make_agent_state_zero_out_with_global_knowledge():
    """When global_knowledge is also present, zero-out only zeros the GSP slot
    (not the global_knowledge slots)."""
    config = _load_base_config()
    config["GSP_ZERO_OUT_SIGNAL"] = True
    agent = Agent(**_base_agent_kwargs(config))
    env_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    gk = np.array([100.0, 200.0, 300.0])
    augmented = agent.make_agent_state(env_obs, heading_gsp=0.5, global_knowledge=gk)
    # Shape: 8 env_obs + 1 gsp + 3 gk = 12
    assert augmented.shape == (12,)
    # The GSP slot is zero; global_knowledge slots preserved
    assert augmented[8] == 0.0
    np.testing.assert_array_equal(augmented[9:], gk)

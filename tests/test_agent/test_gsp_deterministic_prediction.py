"""Behavioral tests for opt-in deterministic GSP prediction.

The GSP prediction head is a SUPERVISED regressor (trained by MSE against a
true environment label), yet during training rollouts its prediction is
produced via choose_action(..., test=False), which adds Gaussian exploration
noise (sigma=NOISE) to every prediction. That noise belongs on POLICY actions,
not on a supervised predictor.

GSP_PREDICTION_DETERMINISTIC (default False) makes the GSP prediction greedy
(no exploration noise) even during training, while leaving the policy action
path untouched. Default False keeps every existing experiment bit-identical.
"""

import numpy as np
import torch as T
import pytest

from src.agent import Agent


def _make_gsp_n_agent(config, n_agents=4):
    """Minimal GSP-N (gsp=True, neighbors=True) DDPG-headed Agent on CPU.

    gsp=True + attention=False + JEPA disabled => build_gsp_network('DDPG'),
    so the GSP head routes through choose_action's DDPG noise branch.
    """
    return Agent(
        config=config, network='DDPG', n_agents=n_agents, n_obs=31,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=True, recurrent=False, attention=False,
        neighbors=True, gsp_input_size=6, gsp_output_size=1,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
    )


@pytest.fixture
def noisy_config():
    """Config with NOISE > 0 so the exploration-noise path is observable."""
    return {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.1, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }


def _states(agent, n_agents=4):
    """A fixed per-agent GSP input list (one 6-d self-centric vector each)."""
    prox = [0.1, 0.2, 0.3, 0.4][:n_agents]
    prev_gsp = np.array([0.0] * n_agents, dtype=float)
    return agent.make_gsp_states(prox, prev_gsp)


def _preds(agent, states):
    """Flatten the per-agent GSP predictions into one float array."""
    out = agent.choose_agent_gsp(states, test=False)
    return np.concatenate([np.ravel(a) for a in out]).astype(float)


class TestDeterministicGSPPrediction:
    def test_default_is_off(self, noisy_config):
        agent = _make_gsp_n_agent(noisy_config)
        assert agent._gsp_prediction_deterministic is False

    def test_flag_off_train_mode_predictions_differ(self, noisy_config):
        """Flag OFF + test=False: exploration noise makes repeats differ."""
        agent = _make_gsp_n_agent(noisy_config)
        states = _states(agent)
        T.manual_seed(0)
        first = _preds(agent, states)
        second = _preds(agent, states)
        assert not np.allclose(first, second), (
            "With the flag OFF and test=False, successive GSP predictions must "
            "differ due to injected exploration noise."
        )

    def test_flag_on_train_mode_predictions_identical(self, noisy_config):
        """Flag ON + test=False: greedy predictor, repeats are identical."""
        config = dict(noisy_config)
        config["GSP_PREDICTION_DETERMINISTIC"] = True
        agent = _make_gsp_n_agent(config)
        assert agent._gsp_prediction_deterministic is True
        states = _states(agent)
        T.manual_seed(0)
        first = _preds(agent, states)
        second = _preds(agent, states)
        assert np.array_equal(first, second), (
            "With the flag ON, successive GSP predictions in train mode must be "
            "identical (no exploration noise on the supervised head)."
        )

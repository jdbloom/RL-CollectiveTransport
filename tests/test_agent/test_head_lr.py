"""Tests for GSP_HEAD_LR config flag — Phase 4 independent head learning rate.

Two acceptance criteria verified here:

1. HEAD RATIO: With GSP_HEAD_LR=3e-4 vs GSP_HEAD_LR=1e-5, after one optimizer
   step on identical data, the high-LR agent's GSP head parameters change by
   ~30x more than the low-LR agent's (within 10% tolerance of the 30:1 ratio).

2. TRUNK ISOLATION: Both agents use the same trunk/actor LR (config['LR']=1e-4).
   After one DDQN step on identical data, the trunk parameters change by the
   SAME amount in both agents (relative difference < 1%).

The GSP head already had its own optimizer (DDPGActorNetwork.optimizer, built in
Actor.build_DDPG_gsp) separate from the main policy optimizer. GSP_HEAD_LR
controls only that head optimizer's LR — the trunk is unaffected by construction.
"""
import os
import sys

import numpy as np
import pytest
import torch
import yaml

# Add rl_code/src to sys.path — consistent with other test_agent tests
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src")
sys.path.insert(0, _SRC)

from agent import Agent  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GSP_RL_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "GSP-RL",
    "tests", "test_actor", "config.yml",
)

# LR ratio used across tests. Must equal GSP_HEAD_LR_HIGH / GSP_HEAD_LR_LOW.
_LR_RATIO = 30.0
_GSP_HEAD_LR_HIGH = 3e-4
_GSP_HEAD_LR_LOW = 1e-5
_TRUNK_LR = 1e-4  # shared trunk LR for both agents

# Tolerances
_HEAD_RATIO_TOLERANCE = 0.10  # ratio must be within 10% of _LR_RATIO
_TRUNK_REL_DIFF_TOLERANCE = 0.01  # trunk delta relative diff must be < 1%


def _load_config() -> dict:
    with open(_GSP_RL_CONFIG) as f:
        base = yaml.safe_load(f)
    base.update({
        "BATCH_SIZE": 8,
        "MEM_SIZE": 200,
        "REPLACE_TARGET_COUNTER": 1,
        # Disable GSP background learning; we trigger it manually in tests.
        "GSP_LEARNING_FREQUENCY": 999999,
        "GSP_BATCH_SIZE": 8,
        "LR": _TRUNK_LR,
        "EPSILON": 0.0,
        "EPS_MIN": 0.0,
        "EPS_DEC": 0.0,
        "WARMUP": 0,
        "UPDATE_ACTOR_ITER": 1,
    })
    return base


def _agent_kwargs(config: dict) -> dict:
    return dict(
        config=config,
        network="DDQN",
        n_agents=4,
        n_obs=8,
        n_actions=2,
        options_per_action=3,
        id=1,
        min_max_action=1.0,
        meta_param_size=2,
        gsp=True,
        recurrent=False,
        attention=False,
        neighbors=True,
        gsp_input_size=6,
        gsp_output_size=1,
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
    )


def _make_agent(seed: int, gsp_head_lr: float, config: dict) -> Agent:
    """Construct an Agent with deterministic initial weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = dict(config)
    cfg["GSP_HEAD_LR"] = gsp_head_lr
    return Agent(**_agent_kwargs(cfg))


def _state_dict_cpu(net) -> dict:
    return {k: v.cpu().clone() for k, v in net.state_dict().items()}


def _total_abs_delta(before: dict, after: dict) -> float:
    """Sum of elementwise absolute parameter changes across all tensors."""
    return sum((after[k] - before[k]).abs().sum().item() for k in before)


def _fill_gsp_replay(agent: Agent, data_rng: np.random.Generator, n: int) -> None:
    """Store n GSP training transitions using data_rng-generated data."""
    gsp_input_size = 6  # base 2 (self_prox, self_prev_gsp) + 2 per 2 neighbors
    for _ in range(n):
        s = data_rng.standard_normal(gsp_input_size).astype(np.float32)
        label = float(data_rng.standard_normal())  # delta_theta label
        s_ = data_rng.standard_normal(gsp_input_size).astype(np.float32)
        agent.store_gsp_transition(s, label, 0.0, s_, False)


def _fill_main_replay(agent: Agent, data_rng: np.random.Generator, n: int) -> None:
    """Store n main-network transitions (augmented obs 9-dim)."""
    obs_size = 9  # 8 env_obs + 1 gsp output slot
    for _ in range(n):
        s = data_rng.standard_normal(obs_size).astype(np.float32)
        a_code = int(data_rng.integers(0, 9))
        r = float(data_rng.standard_normal())
        s_ = data_rng.standard_normal(obs_size).astype(np.float32)
        agent.store_agent_transition(s, (a_code,), r, s_, False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_gsp_head_lr_equals_trunk_lr():
    """Without GSP_HEAD_LR in config, gsp_head_lr must equal the trunk LR."""
    config = _load_config()
    # Do NOT set GSP_HEAD_LR — should default to config['LR']
    agent = _make_agent(42, _TRUNK_LR, config)
    # Remove the key we just set so we test the actual default path
    cfg_no_head_lr = dict(config)
    # Re-construct without GSP_HEAD_LR
    torch.manual_seed(42)
    np.random.seed(42)
    agent2 = Agent(**_agent_kwargs(cfg_no_head_lr))
    assert agent2.gsp_head_lr == pytest.approx(_TRUNK_LR)
    assert agent2.gsp_networks["actor"].optimizer.param_groups[0]["lr"] == pytest.approx(_TRUNK_LR)


def test_gsp_head_lr_set_on_agent():
    """GSP_HEAD_LR in config is stored as agent.gsp_head_lr."""
    config = _load_config()
    agent = _make_agent(42, _GSP_HEAD_LR_HIGH, config)
    assert agent.gsp_head_lr == pytest.approx(_GSP_HEAD_LR_HIGH)


def test_gsp_head_optimizer_uses_gsp_head_lr():
    """The GSP actor optimizer's LR must equal GSP_HEAD_LR, not the trunk LR."""
    config = _load_config()
    agent = _make_agent(42, _GSP_HEAD_LR_HIGH, config)
    gsp_opt_lr = agent.gsp_networks["actor"].optimizer.param_groups[0]["lr"]
    assert gsp_opt_lr == pytest.approx(_GSP_HEAD_LR_HIGH), (
        f"GSP actor optimizer LR is {gsp_opt_lr:.2e}, expected GSP_HEAD_LR={_GSP_HEAD_LR_HIGH:.2e}"
    )


def test_trunk_optimizer_uses_trunk_lr_regardless_of_head_lr():
    """The trunk (q_eval) optimizer LR must equal config['LR'], unaffected by GSP_HEAD_LR."""
    config = _load_config()
    agent = _make_agent(42, _GSP_HEAD_LR_HIGH, config)
    trunk_opt_lr = agent.networks["q_eval"].optimizer.param_groups[0]["lr"]
    assert trunk_opt_lr == pytest.approx(_TRUNK_LR), (
        f"Trunk optimizer LR is {trunk_opt_lr:.2e}, expected config['LR']={_TRUNK_LR:.2e}. "
        f"GSP_HEAD_LR must not affect trunk LR."
    )


def test_head_parameter_change_ratio():
    """After one GSP head step on identical data, high-LR agent's head parameters
    change by ~30x more than low-LR agent's (within 10% of the 30:1 ratio).

    At step 1, Adam reduces to a per-parameter sign step scaled by lr (momentum
    history is zero), so |Δθ_high| / |Δθ_low| ≈ lr_high / lr_low = 30.
    """
    config = _load_config()

    # Both agents start from identical weights (same seed)
    a_high = _make_agent(42, _GSP_HEAD_LR_HIGH, config)
    a_low = _make_agent(42, _GSP_HEAD_LR_LOW, config)

    # Confirm initial weights are identical
    sd_high_init = _state_dict_cpu(a_high.gsp_networks["actor"])
    sd_low_init = _state_dict_cpu(a_low.gsp_networks["actor"])
    for key in sd_high_init:
        assert torch.equal(sd_high_init[key], sd_low_init[key]), (
            f"Initial GSP head weights differ for '{key}' despite same seed. "
            f"Test setup error."
        )

    # Fill both replay buffers with identical GSP training data
    data_rng = np.random.default_rng(1337)
    _fill_gsp_replay(a_high, np.random.default_rng(1337), 50)
    _fill_gsp_replay(a_low, np.random.default_rng(1337), 50)

    # One GSP head MSE step — same batch (same np.random seed for sampling)
    np.random.seed(0)
    a_high.learn_gsp_mse(a_high.gsp_networks, recurrent=False)
    np.random.seed(0)
    a_low.learn_gsp_mse(a_low.gsp_networks, recurrent=False)

    sd_high_after = _state_dict_cpu(a_high.gsp_networks["actor"])
    sd_low_after = _state_dict_cpu(a_low.gsp_networks["actor"])

    delta_high = _total_abs_delta(sd_high_init, sd_high_after)
    delta_low = _total_abs_delta(sd_low_init, sd_low_after)

    assert delta_high > 0, "High-LR agent head parameters did not change at all"
    assert delta_low > 0, "Low-LR agent head parameters did not change at all"

    actual_ratio = delta_high / delta_low
    expected_ratio = _LR_RATIO
    tol = _HEAD_RATIO_TOLERANCE

    assert abs(actual_ratio - expected_ratio) / expected_ratio <= tol, (
        f"GSP head parameter change ratio is {actual_ratio:.2f}, expected "
        f"{expected_ratio:.1f} ± {tol*100:.0f}%. "
        f"High-LR delta: {delta_high:.4f}, Low-LR delta: {delta_low:.4f}. "
        f"GSP_HEAD_LR=3e-4 should produce ~30x larger head updates than 1e-5."
    )


def test_trunk_parameter_change_identical_for_both_agents():
    """After one DDQN step on identical data, trunk parameters change by the
    SAME amount in both agents (relative difference < 1%).

    This verifies that GSP_HEAD_LR does not bleed into the trunk/actor
    optimizer path. Both agents use config['LR']=1e-4 for the trunk.
    """
    config = _load_config()

    a_high = _make_agent(42, _GSP_HEAD_LR_HIGH, config)
    a_low = _make_agent(42, _GSP_HEAD_LR_LOW, config)

    # Fill both main replay buffers with identical transitions
    _fill_main_replay(a_high, np.random.default_rng(2024), 50)
    _fill_main_replay(a_low, np.random.default_rng(2024), 50)

    sd_high_before = _state_dict_cpu(a_high.networks["q_eval"])
    sd_low_before = _state_dict_cpu(a_low.networks["q_eval"])

    # Confirm initial trunk weights are identical
    for key in sd_high_before:
        assert torch.equal(sd_high_before[key], sd_low_before[key]), (
            f"Initial trunk weights differ for '{key}' despite same seed."
        )

    # One DDQN trunk step — same batch (same np.random seed for sampling)
    np.random.seed(7)
    a_high.learn_DDQN(a_high.networks)
    np.random.seed(7)
    a_low.learn_DDQN(a_low.networks)

    sd_high_after = _state_dict_cpu(a_high.networks["q_eval"])
    sd_low_after = _state_dict_cpu(a_low.networks["q_eval"])

    delta_high = _total_abs_delta(sd_high_before, sd_high_after)
    delta_low = _total_abs_delta(sd_low_before, sd_low_after)

    assert delta_high > 0, "High-GSP-LR agent trunk parameters did not change"
    assert delta_low > 0, "Low-GSP-LR agent trunk parameters did not change"

    max_delta = max(delta_high, delta_low)
    rel_diff = abs(delta_high - delta_low) / max_delta

    assert rel_diff < _TRUNK_REL_DIFF_TOLERANCE, (
        f"Trunk parameter change differs by {rel_diff*100:.3f}% between the two agents "
        f"(tolerance {_TRUNK_REL_DIFF_TOLERANCE*100:.0f}%). "
        f"High-GSP-LR agent trunk delta: {delta_high:.6f}, "
        f"Low-GSP-LR agent trunk delta: {delta_low:.6f}. "
        f"GSP_HEAD_LR must not affect trunk optimizer behavior."
    )

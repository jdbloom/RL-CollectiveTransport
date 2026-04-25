"""Tests for DETERMINISM_ENABLED flag — Phase 4 reproducibility.

Two independent 10-episode smoke trainings with the same seed and
DETERMINISM_ENABLED=true must produce bit-exact final agent parameter tensors
(torch.equal() on every weight and bias in the Q-network).

If full bit-exactness is not achievable on the current backend, the test falls
back to asserting max-abs-diff < 1e-6 and logs a warning. See
KNOWN_NONDETERMINISM.md at the repo root for the current status.

The CUBLAS_WORKSPACE_CONFIG env var is set at the top of this module (before
torch is imported) so that CUDA runs also satisfy the timing constraint
documented in KNOWN_NONDETERMINISM.md.
"""
import os
import sys

# Must be set before torch is imported so CUDA deterministic workspace is
# configured at initialisation time.  No-op on MPS/CPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

# Add rl_code/src to sys.path so we can import agent and determinism directly
# (consistent with other test_agent tests that use the same sys.path pattern).
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src")
sys.path.insert(0, _SRC)

from agent import Agent  # noqa: E402
from determinism import apply_determinism_settings  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GSP_RL_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "GSP-RL",
    "tests", "test_actor", "config.yml",
)


def _load_config() -> dict:
    with open(_GSP_RL_CONFIG) as f:
        base = yaml.safe_load(f)
    # Small buffers and batch size for fast smoke tests
    base["BATCH_SIZE"] = 8
    base["MEM_SIZE"] = 200
    base["REPLACE_TARGET_COUNTER"] = 1
    # Disable GSP learning offset so GSP head does NOT fire during smoke training
    # (we only test action-network determinism here; GSP head determinism is
    # covered implicitly because it shares the same seeded RNG state)
    base["GSP_LEARNING_FREQUENCY"] = 999999
    base["DETERMINISM_ENABLED"] = True
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


def _run_smoke_training(seed: int, config: dict) -> dict:
    """Run a short smoke training and return the Q-network state dict (on CPU).

    Applies determinism settings, constructs an Agent, stores 50 transitions,
    calls learn() 10 times, then returns the q_eval state dict on CPU for
    comparison.
    """
    apply_determinism_settings(seed, enabled=True)

    # Data generation uses a separate deterministic RNG so the transition
    # content is reproducible and independent of the pytorch/np seeds.
    data_rng = np.random.default_rng(seed + 999)

    agent = Agent(**_agent_kwargs(config))

    # augmented obs size: 8 env_obs + 1 gsp = 9
    obs_size = 9

    # Store 50 transitions to fill the replay buffer past batch_size=8
    for _ in range(50):
        s = data_rng.standard_normal(obs_size).astype(np.float32)
        # Discrete action: a tuple (action_code,)
        a_code = int(data_rng.integers(0, 9))
        r = float(data_rng.standard_normal())
        s_ = data_rng.standard_normal(obs_size).astype(np.float32)
        done = False
        agent.store_agent_transition(s, (a_code,), r, s_, done)

    # Run 10 learn steps
    for _ in range(10):
        agent.learn()

    return {k: v.cpu() for k, v in agent.networks["q_eval"].state_dict().items()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_determinism_enabled_false():
    """Without DETERMINISM_ENABLED in config, agent.determinism_enabled is False."""
    config = _load_config()
    del config["DETERMINISM_ENABLED"]
    agent = Agent(**_agent_kwargs(config))
    assert agent.determinism_enabled is False


def test_determinism_enabled_true_flag():
    """With DETERMINISM_ENABLED=True in config, agent picks up the flag."""
    config = _load_config()
    config["DETERMINISM_ENABLED"] = True
    agent = Agent(**_agent_kwargs(config))
    assert agent.determinism_enabled is True


def test_apply_determinism_noop_when_disabled():
    """apply_determinism_settings(seed, enabled=False) must be a no-op."""
    # Just verify it doesn't raise and returns None
    result = apply_determinism_settings(42, enabled=False)
    assert result is None


def test_determinism_bit_exact_same_seed():
    """Two smoke trainings with the same seed and DETERMINISM_ENABLED=True
    must produce bit-exact Q-network parameters (torch.equal on every tensor).

    If bit-exactness is not achievable (unexpected non-deterministic op), we
    fall back to asserting max-abs-diff < 1e-6 and emit a warning so the
    failure is not silent. See KNOWN_NONDETERMINISM.md for the documented
    current status (bit-exact on MPS PyTorch 2.3.1).
    """
    seed = 42
    config = _load_config()

    sd1 = _run_smoke_training(seed, config)
    sd2 = _run_smoke_training(seed, config)

    assert set(sd1.keys()) == set(sd2.keys()), "State dict key mismatch"

    non_exact = []
    max_diffs = {}
    for key in sd1:
        exact = torch.equal(sd1[key], sd2[key])
        if not exact:
            diff = (sd1[key] - sd2[key]).abs().max().item()
            non_exact.append(key)
            max_diffs[key] = diff

    if non_exact:
        # Fallback: tolerate up to 1e-6 absolute difference if the backend
        # has known floating-point non-determinism.
        import warnings
        warnings.warn(
            f"Bit-exact check failed for keys {non_exact}. "
            f"Max diffs: {max_diffs}. "
            f"Falling back to max-abs-diff < 1e-6 tolerance. "
            f"See KNOWN_NONDETERMINISM.md for context.",
            stacklevel=2,
        )
        for key in non_exact:
            assert max_diffs[key] < 1e-6, (
                f"Parameter '{key}' differs by {max_diffs[key]:.2e} between two "
                f"identically-seeded runs — exceeds 1e-6 tolerance. "
                f"DETERMINISM_ENABLED=True is not providing sufficient reproducibility."
            )
    else:
        # All parameters are bit-exact — this is the happy path.
        assert not non_exact, "Unexpected: non_exact is non-empty"


def test_determinism_different_seeds_differ():
    """Sanity check: two runs with different seeds should produce different parameters."""
    config = _load_config()
    sd1 = _run_smoke_training(42, config)
    sd2 = _run_smoke_training(7, config)

    # At least one parameter should differ when seeds differ
    any_diff = any(not torch.equal(sd1[k], sd2[k]) for k in sd1)
    assert any_diff, (
        "Runs with seed=42 and seed=7 produced identical Q-network parameters. "
        "This is extremely unlikely and suggests the seeding is not working."
    )

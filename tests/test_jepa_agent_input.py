"""Tests for JEPA path in agent.make_agent_state.

Verifies that when heading_gsp is a 32-d JEPA latent vector, make_agent_state
concatenates it raw (no degrees/10 scaling). This is the critical correctness
contract separating the JEPA path from the legacy scalar path.

Concretely:
- Legacy scalar:  heading_gsp = 0.5  → gsp_slot = [np.degrees(0.5 / 10)]
- JEPA latent:    heading_gsp = np.ones(32)  → gsp_slot = np.ones(32)  (raw)

The test checks that the values in the augmented state match the raw latent
(not a scaled version), and that the state length equals env_obs + 32.
"""

import numpy as np
import pytest

from src.agent import Agent


JEPA_DIM = 32
N_OBS = 31


@pytest.fixture
def agent_config():
    return {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
        # JEPA enabled — actor uses encoder_dim for network_input_size
        "GSP_JEPA_ENABLED": True,
        "GSP_ENCODER_DIM": JEPA_DIM,
        "GSP_ENCODER_EMA_TAU": 0.995,
    }


@pytest.fixture
def jepa_agent(agent_config):
    """Agent with GSP_JEPA_ENABLED=True, flat (non-neighbor) GSP variant."""
    return Agent(
        config=agent_config,
        network='DQN',
        n_agents=4,
        n_obs=N_OBS,
        n_actions=2,
        options_per_action=3,
        id=0,
        min_max_action=0.1,
        meta_param_size=1,
        gsp=True,
        recurrent=False,
        attention=False,
        neighbors=False,
        gsp_input_size=6,
        gsp_output_size=1,
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
    )


class TestMakeAgentStateJepa:
    """make_agent_state correctness when heading_gsp is a JEPA latent."""

    def test_jepa_latent_skips_scaling(self, jepa_agent):
        """A 32-d latent must be concatenated raw — no degrees/10 transformation."""
        env_obs = np.zeros(N_OBS, dtype=np.float32)
        # Use distinctive values so we can check they appear unmodified in the output
        latent = np.arange(JEPA_DIM, dtype=np.float32) + 1.0  # [1, 2, ..., 32]

        state = jepa_agent.make_agent_state(env_obs, heading_gsp=latent)

        # Total length must be N_OBS + JEPA_DIM
        assert len(state) == N_OBS + JEPA_DIM, (
            f"Expected state length {N_OBS + JEPA_DIM}, got {len(state)}"
        )

        # The JEPA slot must be the raw latent values — not degrees(x/10)
        gsp_slot = state[N_OBS:]
        np.testing.assert_array_almost_equal(
            gsp_slot, latent,
            decimal=5,
            err_msg=(
                "JEPA latent values were rescaled instead of concatenated raw. "
                f"Expected {latent[:4]}..., got {gsp_slot[:4]}..."
            ),
        )

    def test_jepa_latent_not_degrees_scaled(self, jepa_agent):
        """Explicitly verify the JEPA slot differs from the degrees/10 transform."""
        env_obs = np.zeros(N_OBS, dtype=np.float32)
        latent = np.ones(JEPA_DIM, dtype=np.float32) * 0.5

        state = jepa_agent.make_agent_state(env_obs, heading_gsp=latent)
        gsp_slot = state[N_OBS:]

        # Legacy scalar path would produce np.degrees(0.5 / 10) ≈ 2.865
        # JEPA raw path produces 0.5
        legacy_value = float(np.degrees(0.5 / 10))
        jepa_value = 0.5
        assert not np.allclose(gsp_slot[0], legacy_value, atol=1e-3), (
            f"JEPA slot value {gsp_slot[0]:.4f} matches legacy degrees/10 value "
            f"{legacy_value:.4f} — scaling was applied when it should not be."
        )
        assert np.allclose(gsp_slot[0], jepa_value, atol=1e-5), (
            f"JEPA slot value {gsp_slot[0]:.4f} does not match raw latent {jepa_value}"
        )

    def test_legacy_scalar_still_scaled(self, jepa_agent):
        """Scalar heading_gsp (size 1) must still apply degrees/10 even on JEPA agent.

        The size>5 check gates the JEPA path; a scalar should fall through to the
        legacy normalization regardless of the agent's JEPA flag.
        """
        env_obs = np.zeros(N_OBS, dtype=np.float32)
        scalar_val = 0.5

        state = jepa_agent.make_agent_state(env_obs, heading_gsp=scalar_val)
        gsp_slot = state[N_OBS:]

        expected = float(np.degrees(scalar_val / 10))
        assert len(gsp_slot) == 1
        assert np.isclose(gsp_slot[0], expected, atol=1e-5), (
            f"Legacy scalar path broken: expected {expected:.4f}, got {gsp_slot[0]:.4f}"
        )

    def test_env_obs_unchanged(self, jepa_agent):
        """The env_obs portion of the state must not be modified."""
        env_obs = np.arange(N_OBS, dtype=np.float32)
        latent = np.ones(JEPA_DIM, dtype=np.float32)

        state = jepa_agent.make_agent_state(env_obs.copy(), heading_gsp=latent)
        np.testing.assert_array_equal(
            state[:N_OBS], env_obs,
            err_msg="env_obs portion was modified by make_agent_state"
        )

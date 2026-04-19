"""Tests for GSP input enrichment flags (Change 2).

Verifies:
- GSP_INPUT_INCLUDE_GOAL, GSP_INPUT_INCLUDE_CYL_REL, GSP_INPUT_FULL_PROX flags
  produce the correct gsp_input_size and make_gsp_states output vector length.
- Backward compat: all flags False produces identical 6-dim input to legacy behavior.
- Multi-dim heading_gsp in make_agent_state concatenates the full vector.
"""
import math
import numpy as np
import pytest
from src.agent import Agent


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_agent(config_overrides: dict, n_hop_neighbors: int = 1) -> Agent:
    """Build a GSP-N agent with optional enrichment flags set in config."""
    config = {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
        **config_overrides,
    }
    return Agent(
        config=config,
        network="DDQN",
        n_agents=4,
        n_obs=31,
        n_actions=2,
        options_per_action=3,
        id=0,
        min_max_action=0.1,
        meta_param_size=1,
        gsp=True,
        recurrent=False,
        attention=False,
        neighbors=True,
        gsp_input_size=6,   # legacy default — recomputed by Agent.__init__
        gsp_output_size=1,
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
        n_hop_neighbors=n_hop_neighbors,
    )


def _fake_env_obs(n_agents: int = 4) -> list:
    """Return minimal env_observations with the indices used by enrichment."""
    # index 1: angle_to_goal (radians), 4: dist_to_cyl, 5: angle_to_cyl, 7:31: raw prox
    obs = []
    for i in range(n_agents):
        o = np.zeros(31, dtype=np.float32)
        o[1] = 0.5 + i * 0.1   # angle_to_goal
        o[4] = 0.3 + i * 0.05  # dist_to_cyl
        o[5] = -0.2 + i * 0.1  # angle_to_cyl
        o[7:31] = np.linspace(0.1, 0.5, 24)  # raw prox
        obs.append(o)
    return obs


# ── gsp_input_size tests ──────────────────────────────────────────────────────

class TestGspInputSizeComputed:
    """gsp_network_input reflects effective input size after flag computation."""

    def test_no_flags_default_6(self):
        """Backward compat: all flags False → base 6-dim input (GSP-N, 1-hop)."""
        agent = _make_agent({})
        # Base: 2 (self) + 2*2 (2 neighbors for 1 hop) = 6
        assert agent.gsp_network_input == 6

    def test_include_goal_adds_2(self):
        agent = _make_agent({"GSP_INPUT_INCLUDE_GOAL": True})
        # Base 6 + 2 (cos/sin goal) = 8
        assert agent.gsp_network_input == 8

    def test_include_cyl_rel_adds_2(self):
        agent = _make_agent({"GSP_INPUT_INCLUDE_CYL_REL": True})
        # Base 6 + 2 = 8
        assert agent.gsp_network_input == 8

    def test_both_include_flags(self):
        agent = _make_agent({
            "GSP_INPUT_INCLUDE_GOAL": True,
            "GSP_INPUT_INCLUDE_CYL_REL": True,
        })
        # Base 6 + 2 + 2 = 10
        assert agent.gsp_network_input == 10

    def test_full_prox_net_plus23(self):
        """GSP_INPUT_FULL_PROX replaces avg_prox(1) with raw_prox(24): net +23."""
        agent = _make_agent({"GSP_INPUT_FULL_PROX": True})
        # Base 6 + 23 (replace avg_prox with 24 raw) = 29
        assert agent.gsp_network_input == 29


# ── make_gsp_states output size tests ────────────────────────────────────────

class TestMakeGspStatesOutputSize:
    """make_gsp_states returns vectors of the correct length."""

    def _prox(self, n: int = 4) -> list:
        return [0.2] * n

    def _prev_gsp(self, n: int = 4) -> list:
        return [0.0] * n

    def test_no_flags_baseline_length(self):
        agent = _make_agent({})
        states = agent.make_gsp_states(self._prox(), self._prev_gsp())
        assert len(states) == 4
        assert len(states[0]) == 6

    def test_include_goal_length(self):
        agent = _make_agent({"GSP_INPUT_INCLUDE_GOAL": True})
        env_obs = _fake_env_obs()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), env_observations=env_obs
        )
        assert len(states[0]) == 8

    def test_include_cyl_rel_length(self):
        agent = _make_agent({"GSP_INPUT_INCLUDE_CYL_REL": True})
        env_obs = _fake_env_obs()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), env_observations=env_obs
        )
        assert len(states[0]) == 8

    def test_both_flags_length(self):
        agent = _make_agent({
            "GSP_INPUT_INCLUDE_GOAL": True,
            "GSP_INPUT_INCLUDE_CYL_REL": True,
        })
        env_obs = _fake_env_obs()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), env_observations=env_obs
        )
        assert len(states[0]) == 10

    def test_full_prox_length(self):
        agent = _make_agent({"GSP_INPUT_FULL_PROX": True})
        env_obs = _fake_env_obs()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), env_observations=env_obs
        )
        assert len(states[0]) == 29


# ── make_gsp_states content tests ────────────────────────────────────────────

class TestMakeGspStatesContent:
    """Enrichment fields land at the correct positions in the vector."""

    def _prox(self) -> list:
        return [0.25, 0.30, 0.20, 0.10]

    def _prev_gsp(self) -> list:
        return [0.1, 0.2, 0.3, 0.4]

    def test_include_goal_cos_sin_values(self):
        agent = _make_agent({"GSP_INPUT_INCLUDE_GOAL": True})
        env_obs = _fake_env_obs()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), env_observations=env_obs
        )
        # Self slot: [avg_prox(0), prev_gsp(1), cos_goal(2), sin_goal(3), nb0(4), nb1(5), nb0gsp(wait no)]
        # Layout: self_prox(0), self_prev_gsp(1), cos_goal(2), sin_goal(3), n0_prox(4), n0_gsp(5), n1_prox(6), n1_gsp(7)
        s0 = states[0]
        angle0 = float(env_obs[0][1])
        assert s0[2] == pytest.approx(math.cos(angle0), abs=1e-5)
        assert s0[3] == pytest.approx(math.sin(angle0), abs=1e-5)

    def test_include_cyl_rel_values(self):
        agent = _make_agent({"GSP_INPUT_INCLUDE_CYL_REL": True})
        env_obs = _fake_env_obs()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), env_observations=env_obs
        )
        # Layout: self_prox(0), self_prev_gsp(1), dist_cyl(2), ang_cyl(3), ...
        s0 = states[0]
        assert s0[2] == pytest.approx(float(env_obs[0][4]), abs=1e-5)
        assert s0[3] == pytest.approx(float(env_obs[0][5]), abs=1e-5)

    def test_full_prox_raw_values(self):
        """Full prox replaces position 0 with raw_prox[0:24]."""
        agent = _make_agent({"GSP_INPUT_FULL_PROX": True})
        env_obs = _fake_env_obs()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), env_observations=env_obs
        )
        s0 = states[0]
        expected_raw = np.linspace(0.1, 0.5, 24).astype(np.float32)
        np.testing.assert_allclose(s0[:24], expected_raw, atol=1e-5)

    def test_no_flags_self_prox_at_position_0(self):
        """Backward compat: position 0 is still avg_prox with no flags set."""
        agent = _make_agent({})
        states = agent.make_gsp_states([0.42, 0.0, 0.0, 0.0], [0.0] * 4)
        assert states[0][0] == pytest.approx(0.42, abs=1e-5)


# ── make_agent_state multi-dim heading_gsp tests ──────────────────────────────

class TestMakeAgentStateMultiDimGsp:
    """make_agent_state concatenates the full GSP vector for O>1 heads."""

    @pytest.fixture
    def agent(self, agent_config):
        return Agent(
            config=agent_config, network="DDQN", n_agents=4, n_obs=31,
            n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
            meta_param_size=1, gsp=True, recurrent=False, attention=False,
            neighbors=True, gsp_input_size=6, gsp_output_size=1,
            gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
        )

    def test_scalar_heading_gsp_produces_32_dim(self, agent):
        obs = np.ones(31)
        state = agent.make_agent_state(obs, heading_gsp=0.5)
        assert len(state) == 32

    def test_scalar_heading_gsp_applies_degrees_scaling(self, agent):
        obs = np.ones(31)
        heading_val = 0.5
        state = agent.make_agent_state(obs, heading_gsp=heading_val)
        expected = np.degrees(heading_val / 10)
        assert state[31] == pytest.approx(expected, abs=1e-5)

    def test_3d_vector_heading_gsp_produces_34_dim(self, agent):
        obs = np.ones(31)
        gsp_vec = np.array([0.1, 0.2, 0.3])
        state = agent.make_agent_state(obs, heading_gsp=gsp_vec)
        assert len(state) == 34

    def test_3d_vector_heading_gsp_values_concatenated_as_is(self, agent):
        """3D output: no degrees/10 scaling — values appended verbatim."""
        obs = np.zeros(31)
        gsp_vec = np.array([0.1, -0.2, 0.05])
        state = agent.make_agent_state(obs, heading_gsp=gsp_vec)
        np.testing.assert_allclose(state[31:34], gsp_vec, atol=1e-6)

    def test_4d_vector_heading_gsp_produces_35_dim(self, agent):
        obs = np.ones(31)
        gsp_vec = np.array([0.1, 0.2, 0.3, 0.4])
        state = agent.make_agent_state(obs, heading_gsp=gsp_vec)
        assert len(state) == 35

    def test_zero_out_signal_with_vector_gsp_produces_zeros(self, agent):
        """GSP_ZERO_OUT_SIGNAL=True zeros the slot regardless of vector width."""
        agent.gsp_zero_out_signal = True
        agent.gsp_network_output = 3
        obs = np.ones(31)
        gsp_vec = np.array([0.9, 0.8, 0.7])
        state = agent.make_agent_state(obs, heading_gsp=gsp_vec)
        assert len(state) == 34
        np.testing.assert_array_equal(state[31:34], np.zeros(3))

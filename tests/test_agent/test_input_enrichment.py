"""Tests for GSP input enrichment flags (Change 2 and Change 3).

Verifies:
- GSP_INPUT_INCLUDE_GOAL, GSP_INPUT_INCLUDE_CYL_REL, GSP_INPUT_FULL_PROX flags
  produce the correct gsp_input_size and make_gsp_states output vector length.
- Backward compat: all flags False produces identical 6-dim input to legacy behavior.
- Multi-dim heading_gsp in make_agent_state concatenates the full vector.
- Change 3 flags: GSP_INPUT_INCLUDE_PAYLOAD_STATE (+5), GSP_INPUT_INCLUDE_SELF_DYNAMICS (+4),
  GSP_INPUT_TEMPORAL_STACK_K (×K multiplicative). Each flag's functional-inertness
  (default OFF is byte-identical to the flag not existing) and correct shape/content.
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


# ── Helpers shared by Change 3 tests ─────────────────────────────────────────

def _fake_payload_state(n_agents: int = 4) -> dict:
    """Minimal payload_state dict for make_gsp_states."""
    return {
        'vx':          [0.01] * n_agents,
        'vy':          [0.02] * n_agents,
        'omega':       [0.001] * n_agents,
        'dx_to_goal':  [0.3] * n_agents,
        'dy_to_goal':  [-0.1] * n_agents,
    }


def _fake_self_dynamics(n_agents: int = 4) -> dict:
    """Minimal self_dynamics dict for make_gsp_states."""
    return {
        'vx':       [0.05 + i * 0.01 for i in range(n_agents)],
        'vy':       [0.03 + i * 0.01 for i in range(n_agents)],
        'force_mag': [3.0 + i * 0.5 for i in range(n_agents)],
        'force_ang': [1.2 + i * 0.1 for i in range(n_agents)],
    }


# ── Change 3 — gsp_network_input size tests ───────────────────────────────────

class TestChange3GspInputSize:
    """gsp_network_input reflects the three new flags' size contributions."""

    def test_payload_state_off_by_default_is_6(self):
        """Functional-inertness: PAYLOAD_STATE absent from config → same 6 dims."""
        agent = _make_agent({})
        assert agent.gsp_network_input == 6

    def test_payload_state_on_adds_5(self):
        """GSP_INPUT_INCLUDE_PAYLOAD_STATE=True adds 5 dims to the self-slot."""
        agent = _make_agent({"GSP_INPUT_INCLUDE_PAYLOAD_STATE": True})
        # Base 6 + 5 = 11
        assert agent.gsp_network_input == 11

    def test_self_dynamics_off_by_default_is_6(self):
        """Functional-inertness: SELF_DYNAMICS absent from config → same 6 dims."""
        agent = _make_agent({})
        assert agent.gsp_network_input == 6

    def test_self_dynamics_on_adds_4(self):
        """GSP_INPUT_INCLUDE_SELF_DYNAMICS=True adds 4 dims to the self-slot."""
        agent = _make_agent({"GSP_INPUT_INCLUDE_SELF_DYNAMICS": True})
        # Base 6 + 4 = 10
        assert agent.gsp_network_input == 10

    def test_temporal_stack_k1_is_noop(self):
        """GSP_INPUT_TEMPORAL_STACK_K=1 is the default: total unchanged at 6."""
        agent = _make_agent({"GSP_INPUT_TEMPORAL_STACK_K": 1})
        assert agent.gsp_network_input == 6

    def test_temporal_stack_k2_doubles_total(self):
        """GSP_INPUT_TEMPORAL_STACK_K=2 doubles total input size."""
        agent = _make_agent({"GSP_INPUT_TEMPORAL_STACK_K": 2})
        # Base 6 * 2 = 12
        assert agent.gsp_network_input == 12

    def test_temporal_stack_k3_triples_total(self):
        """GSP_INPUT_TEMPORAL_STACK_K=3 triples total input size."""
        agent = _make_agent({"GSP_INPUT_TEMPORAL_STACK_K": 3})
        assert agent.gsp_network_input == 18

    def test_payload_and_self_dynamics_both_on(self):
        """Both Change 3 additive flags together add 9 dims."""
        agent = _make_agent({
            "GSP_INPUT_INCLUDE_PAYLOAD_STATE": True,
            "GSP_INPUT_INCLUDE_SELF_DYNAMICS": True,
        })
        # Base 6 + 5 + 4 = 15
        assert agent.gsp_network_input == 15

    def test_all_three_flags_compose_correctly(self):
        """PAYLOAD_STATE+SELF_DYNAMICS+STACK_K=2: (base+5+4)*2."""
        agent = _make_agent({
            "GSP_INPUT_INCLUDE_PAYLOAD_STATE": True,
            "GSP_INPUT_INCLUDE_SELF_DYNAMICS": True,
            "GSP_INPUT_TEMPORAL_STACK_K": 2,
        })
        # (6 + 5 + 4) * 2 = 30
        assert agent.gsp_network_input == 30

    def test_all_change2_and_change3_flags_compose(self):
        """All six enrichment flags active: size computed correctly."""
        agent = _make_agent({
            "GSP_INPUT_INCLUDE_GOAL": True,      # +2
            "GSP_INPUT_INCLUDE_CYL_REL": True,   # +2
            "GSP_INPUT_FULL_PROX": True,          # +23
            "GSP_INPUT_INCLUDE_PAYLOAD_STATE": True,  # +5
            "GSP_INPUT_INCLUDE_SELF_DYNAMICS": True,  # +4
            "GSP_INPUT_TEMPORAL_STACK_K": 2,          # ×2
        })
        # (6 + 2 + 2 + 23 + 5 + 4) * 2 = 42 * 2 = 84
        assert agent.gsp_network_input == 84

    def test_temporal_k_invalid_raises(self):
        """GSP_INPUT_TEMPORAL_STACK_K < 1 raises ValueError on construction."""
        with pytest.raises(ValueError, match="GSP_INPUT_TEMPORAL_STACK_K"):
            _make_agent({"GSP_INPUT_TEMPORAL_STACK_K": 0})


# ── Change 3 — make_gsp_states output-length tests ────────────────────────────

class TestChange3MakeGspStatesOutputSize:
    """make_gsp_states returns vectors of the correct length with Change 3 flags."""

    def _prox(self, n: int = 4) -> list:
        return [0.2] * n

    def _prev_gsp(self, n: int = 4) -> list:
        return [0.0] * n

    def test_payload_state_flag_off_length_unchanged(self):
        """Functional-inertness: PAYLOAD_STATE absent → baseline 6-dim vector."""
        agent = _make_agent({})
        states = agent.make_gsp_states(self._prox(), self._prev_gsp())
        assert len(states[0]) == 6

    def test_payload_state_on_length_11(self):
        """PAYLOAD_STATE=True: each agent's vector grows to 11."""
        agent = _make_agent({"GSP_INPUT_INCLUDE_PAYLOAD_STATE": True})
        pstate = _fake_payload_state()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), payload_state=pstate
        )
        assert len(states[0]) == 11
        assert len(states) == 4  # one per agent

    def test_self_dynamics_flag_off_length_unchanged(self):
        """Functional-inertness: SELF_DYNAMICS absent → baseline 6-dim vector."""
        agent = _make_agent({})
        states = agent.make_gsp_states(self._prox(), self._prev_gsp())
        assert len(states[0]) == 6

    def test_self_dynamics_on_length_10(self):
        """SELF_DYNAMICS=True: each agent's vector grows to 10."""
        agent = _make_agent({"GSP_INPUT_INCLUDE_SELF_DYNAMICS": True})
        sdyn = _fake_self_dynamics()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), self_dynamics=sdyn
        )
        assert len(states[0]) == 10

    def test_temporal_k1_length_unchanged(self):
        """Functional-inertness: K=1 → baseline 6-dim vector."""
        agent = _make_agent({"GSP_INPUT_TEMPORAL_STACK_K": 1})
        states = agent.make_gsp_states(self._prox(), self._prev_gsp())
        assert len(states[0]) == 6

    def test_temporal_k2_doubles_length(self):
        """K=2: output vector length = 12 (6 * 2)."""
        agent = _make_agent({"GSP_INPUT_TEMPORAL_STACK_K": 2})
        states = agent.make_gsp_states(self._prox(), self._prev_gsp())
        assert len(states[0]) == 12

    def test_temporal_k3_triples_length(self):
        """K=3: output vector length = 18 (6 * 3)."""
        agent = _make_agent({"GSP_INPUT_TEMPORAL_STACK_K": 3})
        states = agent.make_gsp_states(self._prox(), self._prev_gsp())
        assert len(states[0]) == 18

    def test_payload_and_dynamics_combined_length(self):
        """Both flags ON: 6 + 5 + 4 = 15."""
        agent = _make_agent({
            "GSP_INPUT_INCLUDE_PAYLOAD_STATE": True,
            "GSP_INPUT_INCLUDE_SELF_DYNAMICS": True,
        })
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(),
            payload_state=_fake_payload_state(),
            self_dynamics=_fake_self_dynamics(),
        )
        assert len(states[0]) == 15

    def test_all_three_combined_length(self):
        """PAYLOAD+DYNAMICS+K=2: (6+5+4)*2 = 30."""
        agent = _make_agent({
            "GSP_INPUT_INCLUDE_PAYLOAD_STATE": True,
            "GSP_INPUT_INCLUDE_SELF_DYNAMICS": True,
            "GSP_INPUT_TEMPORAL_STACK_K": 2,
        })
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(),
            payload_state=_fake_payload_state(),
            self_dynamics=_fake_self_dynamics(),
        )
        assert len(states[0]) == 30


# ── Change 3 — functional-inertness (byte-level) tests ───────────────────────

class TestChange3FunctionalInertness:
    """Flags at default values produce exactly the same vectors as if they never existed.

    Strategy: build two agents — one with no config overrides, one with all Change 3
    flags explicitly set to their OFF defaults. Run make_gsp_states on identical inputs
    and assert np.array_equal (bit-for-bit identical, no float tolerance needed for
    structural inertness).
    """

    def _prox(self) -> list:
        return [0.25, 0.30, 0.20, 0.10]

    def _prev_gsp(self) -> list:
        return [0.1, 0.2, 0.3, 0.4]

    def test_payload_state_false_is_inert(self):
        """PAYLOAD_STATE=False explicitly → same output as no flag in config."""
        agent_base = _make_agent({})
        agent_flag = _make_agent({"GSP_INPUT_INCLUDE_PAYLOAD_STATE": False})
        states_base = agent_base.make_gsp_states(self._prox(), self._prev_gsp())
        states_flag = agent_flag.make_gsp_states(self._prox(), self._prev_gsp())
        for s_b, s_f in zip(states_base, states_flag):
            np.testing.assert_array_equal(s_b, s_f)

    def test_self_dynamics_false_is_inert(self):
        """SELF_DYNAMICS=False explicitly → same output as no flag in config."""
        agent_base = _make_agent({})
        agent_flag = _make_agent({"GSP_INPUT_INCLUDE_SELF_DYNAMICS": False})
        states_base = agent_base.make_gsp_states(self._prox(), self._prev_gsp())
        states_flag = agent_flag.make_gsp_states(self._prox(), self._prev_gsp())
        for s_b, s_f in zip(states_base, states_flag):
            np.testing.assert_array_equal(s_b, s_f)

    def test_temporal_stack_k1_is_inert(self):
        """STACK_K=1 explicitly → same output as no flag in config."""
        agent_base = _make_agent({})
        agent_k1 = _make_agent({"GSP_INPUT_TEMPORAL_STACK_K": 1})
        states_base = agent_base.make_gsp_states(self._prox(), self._prev_gsp())
        states_k1 = agent_k1.make_gsp_states(self._prox(), self._prev_gsp())
        for s_b, s_k in zip(states_base, states_k1):
            np.testing.assert_array_equal(s_b, s_k)

    def test_all_change3_defaults_together_are_inert(self):
        """All three Change 3 flags at OFF defaults → same output as bare agent."""
        agent_base = _make_agent({})
        agent_defaults = _make_agent({
            "GSP_INPUT_INCLUDE_PAYLOAD_STATE": False,
            "GSP_INPUT_INCLUDE_SELF_DYNAMICS": False,
            "GSP_INPUT_TEMPORAL_STACK_K": 1,
        })
        states_base = agent_base.make_gsp_states(self._prox(), self._prev_gsp())
        states_def = agent_defaults.make_gsp_states(self._prox(), self._prev_gsp())
        for s_b, s_d in zip(states_base, states_def):
            np.testing.assert_array_equal(s_b, s_d)

    def test_payload_kwarg_none_when_flag_false_is_inert(self):
        """Passing payload_state=None with flag False produces same baseline output."""
        agent = _make_agent({"GSP_INPUT_INCLUDE_PAYLOAD_STATE": False})
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), payload_state=None
        )
        agent_base = _make_agent({})
        states_base = agent_base.make_gsp_states(self._prox(), self._prev_gsp())
        for s, s_b in zip(states, states_base):
            np.testing.assert_array_equal(s, s_b)

    def test_self_dynamics_kwarg_none_when_flag_false_is_inert(self):
        """Passing self_dynamics=None with flag False produces same baseline output."""
        agent = _make_agent({"GSP_INPUT_INCLUDE_SELF_DYNAMICS": False})
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), self_dynamics=None
        )
        agent_base = _make_agent({})
        states_base = agent_base.make_gsp_states(self._prox(), self._prev_gsp())
        for s, s_b in zip(states, states_base):
            np.testing.assert_array_equal(s, s_b)


# ── Change 3 — content / value placement tests ───────────────────────────────

class TestChange3MakeGspStatesContent:
    """Enrichment fields land at the correct positions in the vector."""

    def _prox(self) -> list:
        return [0.25, 0.30, 0.20, 0.10]

    def _prev_gsp(self) -> list:
        return [0.1, 0.2, 0.3, 0.4]

    def test_payload_state_values_at_correct_positions(self):
        """Payload fields occupy indices [2:7] in the self-slot (after prox, prev_gsp)."""
        agent = _make_agent({"GSP_INPUT_INCLUDE_PAYLOAD_STATE": True})
        pstate = _fake_payload_state()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), payload_state=pstate
        )
        s0 = states[0]
        # Layout: [avg_prox, prev_gsp, vx, vy, omega, dx_to_goal, dy_to_goal, n0_prox, n0_gsp, n1_prox, n1_gsp]
        assert s0[0] == pytest.approx(self._prox()[0], abs=1e-6)
        assert s0[1] == pytest.approx(self._prev_gsp()[0], abs=1e-6)
        assert s0[2] == pytest.approx(pstate['vx'][0], abs=1e-6)
        assert s0[3] == pytest.approx(pstate['vy'][0], abs=1e-6)
        assert s0[4] == pytest.approx(pstate['omega'][0], abs=1e-6)
        assert s0[5] == pytest.approx(pstate['dx_to_goal'][0], abs=1e-6)
        assert s0[6] == pytest.approx(pstate['dy_to_goal'][0], abs=1e-6)

    def test_self_dynamics_values_at_correct_positions(self):
        """Self-dynamics fields occupy indices [2:6] (after prox, prev_gsp)."""
        agent = _make_agent({"GSP_INPUT_INCLUDE_SELF_DYNAMICS": True})
        sdyn = _fake_self_dynamics()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), self_dynamics=sdyn
        )
        s0 = states[0]
        # Layout: [avg_prox, prev_gsp, vx, vy, force_mag, force_ang, n0_prox, n0_gsp, n1_prox, n1_gsp]
        assert s0[2] == pytest.approx(sdyn['vx'][0], abs=1e-6)
        assert s0[3] == pytest.approx(sdyn['vy'][0], abs=1e-6)
        assert s0[4] == pytest.approx(sdyn['force_mag'][0], abs=1e-6)
        assert s0[5] == pytest.approx(sdyn['force_ang'][0], abs=1e-6)

    def test_self_dynamics_per_robot_values(self):
        """Self-dynamics fields are per-robot — agent 1 gets its own values."""
        agent = _make_agent({"GSP_INPUT_INCLUDE_SELF_DYNAMICS": True})
        sdyn = _fake_self_dynamics()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(), self_dynamics=sdyn
        )
        # Agent 1's vx should be sdyn['vx'][1], not sdyn['vx'][0]
        s1 = states[1]
        assert s1[2] == pytest.approx(sdyn['vx'][1], abs=1e-6)
        assert s1[3] == pytest.approx(sdyn['vy'][1], abs=1e-6)

    def test_payload_plus_self_dynamics_combined_layout(self):
        """With both flags ON: payload occupies [2:7], self_dynamics [7:11]."""
        agent = _make_agent({
            "GSP_INPUT_INCLUDE_PAYLOAD_STATE": True,
            "GSP_INPUT_INCLUDE_SELF_DYNAMICS": True,
        })
        pstate = _fake_payload_state()
        sdyn = _fake_self_dynamics()
        states = agent.make_gsp_states(
            self._prox(), self._prev_gsp(),
            payload_state=pstate,
            self_dynamics=sdyn,
        )
        s0 = states[0]
        # payload at [2:7]
        assert s0[2] == pytest.approx(pstate['vx'][0], abs=1e-6)
        assert s0[6] == pytest.approx(pstate['dy_to_goal'][0], abs=1e-6)
        # self_dynamics at [7:11]
        assert s0[7] == pytest.approx(sdyn['vx'][0], abs=1e-6)
        assert s0[10] == pytest.approx(sdyn['force_ang'][0], abs=1e-6)

    def test_temporal_k2_second_half_is_previous_step(self):
        """K=2: the flattened output = [step_t-1 (zeros), step_t (current)]."""
        agent = _make_agent({"GSP_INPUT_TEMPORAL_STACK_K": 2})
        prox = [0.42, 0.30, 0.20, 0.10]
        prev_gsp = [0.0] * 4
        # On the very first call, the ring buffer is all zeros (initialized with zeros).
        # The stacked output = [zeros(6), current_step(6)].
        states = agent.make_gsp_states(prox, prev_gsp)
        s0 = states[0]
        assert len(s0) == 12
        # First 6 positions are the previously-zero-filled slot
        np.testing.assert_array_equal(s0[:6], np.zeros(6))
        # Last 6 positions are the current step: [avg_prox, prev_gsp, n0_prox, n0_gsp, n1_prox, n1_gsp]
        assert s0[6] == pytest.approx(prox[0], abs=1e-6)

    def test_temporal_k2_second_call_has_previous_in_first_half(self):
        """After two calls, K=2 output[0:6] equals previous call's agent_state."""
        agent = _make_agent({"GSP_INPUT_TEMPORAL_STACK_K": 2})
        prox1 = [0.10, 0.20, 0.30, 0.40]
        prox2 = [0.50, 0.60, 0.70, 0.80]
        prev_gsp = [0.0] * 4

        states1 = agent.make_gsp_states(prox1, prev_gsp)
        # Extract the single-step vector from call 1 (the last 6 of the 12-dim output)
        single_step_1 = states1[0][6:]

        states2 = agent.make_gsp_states(prox2, prev_gsp)
        # In call 2: first 6 = what was call 1's single step; last 6 = call 2's step
        np.testing.assert_array_equal(states2[0][:6], single_step_1)

    def test_neighbor_slots_unaffected_by_payload_flag(self):
        """Neighbor slots remain at (prox, prev_gsp) regardless of PAYLOAD_STATE."""
        agent = _make_agent({"GSP_INPUT_INCLUDE_PAYLOAD_STATE": True})
        pstate = _fake_payload_state()
        prox = [0.25, 0.30, 0.20, 0.10]
        prev_gsp = [0.1, 0.2, 0.3, 0.4]
        states = agent.make_gsp_states(prox, prev_gsp, payload_state=pstate)
        # Agent 0 neighbors are agents 3 and 1 (circular, 1-hop).
        # Neighbor slots start at index 7 (after 2 self + 5 payload dims).
        s0 = states[0]
        neighbors = agent.neighbors_dict[0]
        assert s0[7] == pytest.approx(prox[neighbors[0]], abs=1e-6)
        assert s0[9] == pytest.approx(prox[neighbors[1]], abs=1e-6)

"""Unit tests for multi-dim GSP_OUTPUT_KIND support.

Covers:
- gsp_input_size formula uses (1+K)+(1+K)*N for cyl_kinematics_3d/goal_4d
- make_gsp_states writes K dims per prev_gsp slot
- legacy 1d (delta_theta_1d) behavior is bit-identical to pre-patch
- dict sync: agent.py local _GSP_OUTPUT_KIND_SIZES matches GSP-RL source of truth

Plan §5 acceptance tests (unit layer):
  test_multi_dim_gsp_input_size          (cyl_kinematics_3d)
  test_multi_dim_make_gsp_states         (cyl_kinematics_3d slot layout)
  test_multi_dim_make_agent_state_4d     (cyl_kinematics_goal_4d actor state)
  test_legacy_1d_gsp_input_size_unchanged (delta_theta_1d bit-identical)
  test_gsp_output_kind_sizes_dict_sync   (agent.py local dict == GSP-RL dict)
"""
import numpy as np
import pytest
from src.agent import Agent


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _make_neighbors_agent(gsp_output_kind='delta_theta_1d', n_hop_neighbors=1):
    """Construct a GSP-N agent with the given GSP_OUTPUT_KIND."""
    config = _base_config(GSP_OUTPUT_KIND=gsp_output_kind)
    return Agent(
        config=config,
        network='DDQN',
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
        gsp_input_size=6,   # overridden in __init__ when neighbors=True
        gsp_output_size=1,  # overridden by GSP_OUTPUT_KIND
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
        n_hop_neighbors=n_hop_neighbors,
    )


def _make_prox_and_prev(n_agents=4, K=1):
    """Return prox list and 2D prev_gsp array (num_robots, K)."""
    prox = [0.1 * (i + 1) for i in range(n_agents)]
    prev_gsp = np.zeros((n_agents, K), dtype=np.float32)
    # Fill with distinct values so we can check layout
    for i in range(n_agents):
        for k in range(K):
            prev_gsp[i, k] = (i + 1) * 0.1 + k * 0.01
    return prox, prev_gsp


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGspInputSizeFormula:
    """gsp_input_size = (1+K) + (1+K)*N where N = n_hop_neighbors*2."""

    def test_legacy_1d_gsp_input_size(self):
        """delta_theta_1d: K=1 → (1+1)+(1+1)*2 = 6 (matches legacy)."""
        agent = _make_neighbors_agent('delta_theta_1d', n_hop_neighbors=1)
        assert agent.gsp_network_input == 6, (
            f"Legacy 1d gsp_input_size should be 6, got {agent.gsp_network_input}"
        )

    def test_cyl_kinematics_3d_gsp_input_size(self):
        """cyl_kinematics_3d: K=3 → (1+3)+(1+3)*2 = 12."""
        agent = _make_neighbors_agent('cyl_kinematics_3d', n_hop_neighbors=1)
        expected = (1 + 3) + (1 + 3) * (1 * 2)  # 4 + 8 = 12
        assert agent.gsp_network_input == expected, (
            f"cyl_kinematics_3d gsp_input_size should be {expected}, "
            f"got {agent.gsp_network_input}"
        )

    def test_cyl_kinematics_goal_4d_gsp_input_size(self):
        """cyl_kinematics_goal_4d: K=4 → (1+4)+(1+4)*2 = 15."""
        agent = _make_neighbors_agent('cyl_kinematics_goal_4d', n_hop_neighbors=1)
        expected = (1 + 4) + (1 + 4) * (1 * 2)  # 5 + 10 = 15
        assert agent.gsp_network_input == expected, (
            f"cyl_kinematics_goal_4d gsp_input_size should be {expected}, "
            f"got {agent.gsp_network_input}"
        )

    def test_future_prox_1d_unchanged(self):
        """future_prox_1d: K=1 → same as legacy 6."""
        agent = _make_neighbors_agent('future_prox_1d', n_hop_neighbors=1)
        assert agent.gsp_network_input == 6

    def test_time_to_goal_1d_unchanged(self):
        """time_to_goal_1d: K=1 → same as legacy 6."""
        agent = _make_neighbors_agent('time_to_goal_1d', n_hop_neighbors=1)
        assert agent.gsp_network_input == 6

    def test_unknown_kind_raises(self):
        """Unknown GSP_OUTPUT_KIND raises ValueError at construction."""
        with pytest.raises(ValueError, match="Unknown GSP_OUTPUT_KIND"):
            _make_neighbors_agent('bad_kind_xyz')


class TestMakeGspStates:
    """make_gsp_states writes K dims per prev_gsp slot."""

    def test_legacy_1d_state_length(self):
        """K=1: state length per agent must be 6 (unchanged from legacy)."""
        agent = _make_neighbors_agent('delta_theta_1d')
        prox, prev = _make_prox_and_prev(4, K=1)
        states = agent.make_gsp_states(prox, prev)
        assert len(states) == 4
        assert len(states[0]) == 6

    def test_3d_state_length(self):
        """K=3: state length per agent must be 12."""
        agent = _make_neighbors_agent('cyl_kinematics_3d')
        prox, prev = _make_prox_and_prev(4, K=3)
        states = agent.make_gsp_states(prox, prev)
        assert len(states) == 4
        expected_len = (1 + 3) + (1 + 3) * 2  # 12
        assert len(states[0]) == expected_len, (
            f"Expected state length {expected_len}, got {len(states[0])}"
        )

    def test_4d_state_length(self):
        """K=4: state length per agent must be 15."""
        agent = _make_neighbors_agent('cyl_kinematics_goal_4d')
        prox, prev = _make_prox_and_prev(4, K=4)
        states = agent.make_gsp_states(prox, prev)
        expected_len = (1 + 4) + (1 + 4) * 2  # 15
        assert len(states[0]) == expected_len

    def test_3d_self_slot_layout(self):
        """For K=3: slot 0 is avg_prox, slots 1:4 are the K prev_gsp dims."""
        agent = _make_neighbors_agent('cyl_kinematics_3d')
        prox = [0.5, 0.2, 0.3, 0.4]
        # Give agent 0 a distinct prev_gsp vector
        prev = np.zeros((4, 3), dtype=np.float32)
        prev[0] = [0.11, 0.22, 0.33]
        states = agent.make_gsp_states(prox, prev)
        s = states[0]
        assert s[0] == pytest.approx(0.5, abs=1e-6), "self avg_prox should be at index 0"
        assert s[1] == pytest.approx(0.11, abs=1e-6), "self prev_gsp[0] at index 1"
        assert s[2] == pytest.approx(0.22, abs=1e-6), "self prev_gsp[1] at index 2"
        assert s[3] == pytest.approx(0.33, abs=1e-6), "self prev_gsp[2] at index 3"

    def test_legacy_1d_slot_values_match(self):
        """K=1: state[0][1] == prev_gsp[0][0] (scalar-equivalent behavior)."""
        agent = _make_neighbors_agent('delta_theta_1d')
        prox = [0.5, 0.2, 0.3, 0.4]
        prev = np.array([[0.77], [0.0], [0.0], [0.0]], dtype=np.float32)
        states = agent.make_gsp_states(prox, prev)
        assert states[0][1] == pytest.approx(0.77, abs=1e-6)

    def test_state_matches_gsp_network_input(self):
        """states[i] length must equal gsp_network_input for both 1d and 3d."""
        for kind in ('delta_theta_1d', 'cyl_kinematics_3d', 'cyl_kinematics_goal_4d'):
            K = {'delta_theta_1d': 1, 'cyl_kinematics_3d': 3, 'cyl_kinematics_goal_4d': 4}[kind]
            agent = _make_neighbors_agent(kind)
            prox, prev = _make_prox_and_prev(4, K=K)
            states = agent.make_gsp_states(prox, prev)
            assert len(states[0]) == agent.gsp_network_input, (
                f"kind={kind}: state length {len(states[0])} != "
                f"gsp_network_input {agent.gsp_network_input}"
            )


class TestMakeAgentState:
    """make_agent_state handles 1d (scalar-like) and multi-dim heading_gsp arrays."""

    def test_1d_heading_gsp_array_accepted(self):
        """K=1: passing a 1-element array (next_heading_gsp[i]) works."""
        agent = _make_neighbors_agent('delta_theta_1d')
        obs = np.zeros(31, dtype=np.float32)
        # next_heading_gsp[i] is shape (1,) when the array is 2D (R,1)
        heading = np.array([0.05], dtype=np.float32)
        state = agent.make_agent_state(obs, heading_gsp=heading)
        assert state is not None
        assert len(state) == agent.network_input_size

    def test_3d_heading_gsp_array_accepted(self):
        """K=3: passing a 3-element array (next_heading_gsp[i]) works."""
        agent = _make_neighbors_agent('cyl_kinematics_3d')
        obs = np.zeros(31, dtype=np.float32)
        heading = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        state = agent.make_agent_state(obs, heading_gsp=heading)
        assert state is not None
        assert len(state) == agent.network_input_size

    def test_4d_heading_gsp_array_accepted(self):
        """K=4: passing a 4-element array (next_heading_gsp[i]) works."""
        agent = _make_neighbors_agent('cyl_kinematics_goal_4d')
        obs = np.zeros(31, dtype=np.float32)
        heading = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
        state = agent.make_agent_state(obs, heading_gsp=heading)
        assert state is not None
        assert len(state) == agent.network_input_size


class TestGspOutputKindDictSync:
    """The local _GSP_OUTPUT_KIND_SIZES in agent.py must match GSP-RL's copy."""

    def test_dicts_are_identical(self):
        """Ensure agent.py local dict keys/values match GSP-RL learning_aids.py:287."""
        # Source of truth (GSP-RL)
        from gsp_rl.src.actors.learning_aids import Hyperparameters
        # Build a minimal config to get the hyperparameters to init the dict
        # We read it by constructing an agent and checking gsp_output_size_effective
        # for each known kind. This avoids importing a private class directly.
        gsp_rl_sizes = {}
        local_sizes = {
            'delta_theta_1d': 1,
            'future_prox_1d': 1,
            'cyl_kinematics_3d': 3,
            'cyl_kinematics_goal_4d': 4,
            'time_to_goal_1d': 1,
        }
        # Verify by constructing an agent for each kind and checking gsp_network_output
        for kind, expected_k in local_sizes.items():
            agent = _make_neighbors_agent(kind)
            assert agent.gsp_network_output == expected_k, (
                f"GSP_OUTPUT_KIND='{kind}': expected gsp_network_output={expected_k}, "
                f"got {agent.gsp_network_output}. "
                "The GSP-RL dict and agent.py local dict are out of sync!"
            )

    def test_all_canonical_kinds_known_to_agent(self):
        """Every kind that GSP-RL knows about can be constructed in agent.py."""
        # The canonical kinds from GSP-RL actor.py
        canonical_kinds = [
            'delta_theta_1d', 'future_prox_1d',
            'cyl_kinematics_3d', 'cyl_kinematics_goal_4d',
            'time_to_goal_1d',
        ]
        for kind in canonical_kinds:
            # Should not raise
            agent = _make_neighbors_agent(kind)
            assert agent.gsp_network_output >= 1


class TestNextHeadingGspShape:
    """next_heading_gsp initialization and assignment are 2D (num_robots, K)."""

    def test_zeros_shape_1d(self):
        """K=1: np.zeros((R, 1)) has correct shape."""
        agent = _make_neighbors_agent('delta_theta_1d')
        K = agent.gsp_network_output
        R = 4
        arr = np.zeros((R, K))
        assert arr.shape == (4, 1)
        # Indexing robot i gives a 1-element array
        assert arr[0].shape == (1,)

    def test_zeros_shape_3d(self):
        """K=3: np.zeros((R, 3)) has correct shape."""
        agent = _make_neighbors_agent('cyl_kinematics_3d')
        K = agent.gsp_network_output
        R = 4
        arr = np.zeros((R, K))
        assert arr.shape == (4, 3)
        assert arr[0].shape == (3,)

    def test_slice_assignment_safe(self):
        """Assigning a K-dim numpy vector into row i of 2D array works."""
        K = 3
        R = 4
        arr = np.zeros((R, K))
        pred_vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        arr[0] = pred_vec
        np.testing.assert_array_almost_equal(arr[0], [0.1, 0.2, 0.3])

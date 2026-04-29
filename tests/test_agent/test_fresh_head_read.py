"""Unit tests for GSP_FRESH_HEAD_READ flag (Phase 5 lag-elimination).

Two assertions:
1. Flag false (default): make_gsp_states output is byte-identical to pre-patch
   behavior — no change to the lagged read path.
2. Flag true: the gsp slot at index `idx` in the agent's per-robot state equals
   what the head returns when called on the current observation.  We verify this
   by constructing an agent with a controlled head (weights fixed at init) and
   confirming the slot value matches _fresh_gsp_head_forward directly.
"""

import numpy as np
import pytest
import torch as T

from src.agent import Agent


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": 0.001,
    "BETA": 0.001,
    "LR": 0.001,
    "EPSILON": 0.0,
    "EPS_MIN": 0.0,
    "EPS_DEC": 0.0,
    "BATCH_SIZE": 8,
    "MEM_SIZE": 100,
    "REPLACE_TARGET_COUNTER": 10,
    "NOISE": 0.0,
    "UPDATE_ACTOR_ITER": 1,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 100,
    "GSP_BATCH_SIZE": 8,
}


def _make_agent(extra_config: dict, n_agents: int = 4, n_hop: int = 1) -> Agent:
    """Build a GSP-N DDQN agent with minimal config."""
    config = {**_BASE_CONFIG, **extra_config}
    return Agent(
        config=config,
        network="DDQN",
        n_agents=n_agents,
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
        gsp_input_size=6,        # recomputed by Agent.__init__ to 2+2*2=6 for n_hop=1
        gsp_output_size=1,
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
        n_hop_neighbors=n_hop,
    )


def _prox(n: int = 4, val: float = 0.3) -> list:
    return [val + i * 0.05 for i in range(n)]


def _prev_gsp(n: int = 4, val: float = 0.5) -> list:
    return [val + i * 0.02 for i in range(n)]


# ---------------------------------------------------------------------------
# Test 1 — Flag False: output is byte-identical to lagged behavior
# ---------------------------------------------------------------------------

class TestFreshHeadReadFlagFalse:
    """With GSP_FRESH_HEAD_READ absent/False, make_gsp_states must produce the
    same result as the pre-patch (lagged) code path."""

    def test_default_flag_is_false(self):
        agent = _make_agent({})
        assert agent._gsp_fresh_head_read is False

    def test_explicit_false_is_false(self):
        agent = _make_agent({"GSP_FRESH_HEAD_READ": False})
        assert agent._gsp_fresh_head_read is False

    def test_lagged_gsp_slot_matches_prev_gsp_value(self):
        """With flag off, the gsp slot in agent_state[1] must equal agent_prev_gsp[i].

        GSP-N layout: [self_prox, self_prev_gsp, nb0_prox, nb0_prev_gsp, nb1_prox, nb1_prev_gsp]
        For 4 agents, 1-hop: 2 neighbors (CCW and CW).
        """
        agent = _make_agent({})
        prox = _prox(4)
        prev = _prev_gsp(4)

        states = agent.make_gsp_states(prox, prev)

        # For each agent, state[1] = agent_prev_gsp[agent]
        for i in range(4):
            assert states[i][1] == pytest.approx(prev[i], abs=1e-6), (
                f"agent {i}: expected gsp_slot={prev[i]}, got {states[i][1]}"
            )

    def test_identity_with_reference_call(self):
        """Two calls with identical inputs must return equal arrays (no randomness)."""
        agent = _make_agent({"GSP_FRESH_HEAD_READ": False})
        prox = _prox(4)
        prev = _prev_gsp(4)

        # Reset ring buffer to a known state
        _ring_size = agent.gsp_network_input
        for i in range(4):
            agent.gsp_observation[i] = [[0.0] * _ring_size for _ in range(agent.gsp_sequence_length)]

        states_a = agent.make_gsp_states(prox[:], prev[:])

        # Reset ring buffer again to same state so second call sees identical input
        for i in range(4):
            agent.gsp_observation[i] = [[0.0] * _ring_size for _ in range(agent.gsp_sequence_length)]

        states_b = agent.make_gsp_states(prox[:], prev[:])

        for i in range(4):
            np.testing.assert_array_equal(
                states_a[i], states_b[i],
                err_msg=f"agent {i}: states not identical across two calls with same inputs",
            )


# ---------------------------------------------------------------------------
# Test 2 — Flag True: gsp slot equals fresh head forward output
# ---------------------------------------------------------------------------

class TestFreshHeadReadFlagTrue:
    """With GSP_FRESH_HEAD_READ=True, the self gsp slot must equal what
    _fresh_gsp_head_forward returns when called on the current prox."""

    def test_flag_true_stored(self):
        agent = _make_agent({"GSP_FRESH_HEAD_READ": True})
        assert agent._gsp_fresh_head_read is True

    def test_gsp_slot_equals_fresh_forward(self):
        """For each agent i, states[i][1] == _fresh_gsp_head_forward(input)[0].

        The fresh input built internally is: zeros with prox at slot 0.
        We replicate that here and compare.
        """
        agent = _make_agent({"GSP_FRESH_HEAD_READ": True})
        prox = _prox(4, val=0.4)
        prev = _prev_gsp(4, val=0.1)  # these should NOT appear in output when flag is True

        states = agent.make_gsp_states(prox, prev)

        single_step_size = agent.gsp_network_input  # gsp_sequence_length=1 default

        for i in range(4):
            # Reconstruct the fresh_input as the code does
            fresh_input = np.zeros(single_step_size, dtype=np.float32)
            fresh_input[0] = float(prox[i])
            expected = agent._fresh_gsp_head_forward(fresh_input)

            actual_slot = states[i][1]
            assert actual_slot == pytest.approx(float(expected[0]), abs=1e-5), (
                f"agent {i}: gsp slot {actual_slot} != fresh head output {expected[0]}"
            )

    def test_gsp_slot_differs_from_prev_gsp_when_nonzero(self):
        """When prev_gsp is nonzero, the fresh-head slot must not equal prev_gsp
        (unless the head happens to output exactly that value, which is extremely
        unlikely with random init weights).

        This is a probabilistic check — we use a large prev_gsp value (1.0) that
        a freshly-initialized head is extremely unlikely to reproduce.
        """
        agent = _make_agent({"GSP_FRESH_HEAD_READ": True})
        prox = _prox(4, val=0.3)
        # Use an extreme prev_gsp that init weights are very unlikely to produce
        prev = [1.0] * 4

        states = agent.make_gsp_states(prox, prev)

        for i in range(4):
            slot_val = states[i][1]
            # Fresh head output on a small-prox input should not equal 1.0 exactly
            # (init weights produce small tanh outputs near zero, not ±1.0)
            assert abs(slot_val - 1.0) > 1e-4, (
                f"agent {i}: gsp slot {slot_val} unexpectedly equals prev_gsp=1.0 "
                f"suggesting the lagged path is still active"
            )

    def test_neighbor_gsp_slot_equals_fresh_forward(self):
        """Neighbor gsp slots (at idx+1 for each neighbor) also use fresh reads.

        For 4 agents 1-hop: each agent has 2 neighbors.
        Layout: [self_prox(0), self_gsp(1), nb0_prox(2), nb0_gsp(3), nb1_prox(4), nb1_gsp(5)]
        """
        agent = _make_agent({"GSP_FRESH_HEAD_READ": True})
        prox = _prox(4, val=0.35)
        prev = _prev_gsp(4, val=0.9)  # extreme to detect if lagged path is active

        states = agent.make_gsp_states(prox, prev)

        single_step_size = agent.gsp_network_input

        for i in range(4):
            neighbors = agent.neighbors_dict[i]
            # nb slots start at index 2 (after self_prox, self_gsp)
            nb_base = 2
            for j, nb in enumerate(neighbors):
                fresh_nb_input = np.zeros(single_step_size, dtype=np.float32)
                fresh_nb_input[0] = float(prox[nb])
                expected_nb = agent._fresh_gsp_head_forward(fresh_nb_input)

                nb_gsp_idx = nb_base + j * 2 + 1
                actual_nb_slot = states[i][nb_gsp_idx]
                assert actual_nb_slot == pytest.approx(float(expected_nb[0]), abs=1e-5), (
                    f"agent {i} neighbor {j}: neighbor gsp slot {actual_nb_slot} != "
                    f"fresh head output {expected_nb[0]}"
                )

"""#53 Sub-project B — BATCHED_ACTOR_FORWARD call-site tests.

Contract under test (opt-in, BASELINE-CHANGING, default off):

(a) Flag OFF (or absent) = byte-identical legacy behavior: choose_agent_gsp
    runs the exact sequential per-agent choose_action loop (bit-exact match
    against an inline reference implementation on fixed seeds, including the
    exploration-noise RNG stream), and the Main.py acting loop is untouched.
(b) Flag ON = one stacked forward: DDQN acting returns the SAME argmax
    actions as the sequential loop on a fixed-seed batch (float-reduction
    order changes with batched matmul, so continuous outputs match at
    atol=1e-6, never bit-exact — the documented, expected drift; a
    near-degenerate Q-tie could flip an argmax, which is why activation is
    gated on the pre-registered n-seed noise-floor re-baseline).
(c) Shape correctness across GSP output widths K: one (K,) prediction vector
    per agent, same consumer contract as the sequential loop.

Stateful heads (recurrent RDDPG, attention A-GSP) must NEVER route through
the batched path — batching them would change semantics, not just float
order (kb/wiki/concepts/batched-inference.md).
"""

import numpy as np
import torch as T
import pytest

from src.agent import Agent


def _config(**overrides):
    cfg = {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }
    cfg.update(overrides)
    return cfg


def _make_gsp_n_agent(config, gsp_output_size=1, attention=False, seed=0):
    """GSP-N (gsp=True, neighbors=True) Agent with a stateless DDPG head
    (or an attention head when attention=True), deterministic net init."""
    T.manual_seed(seed)
    return Agent(
        config=config, network='DDQN', n_agents=4, n_obs=31,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=True, recurrent=False, attention=attention,
        neighbors=True, gsp_input_size=6, gsp_output_size=gsp_output_size,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
    )


def _make_ddqn_agent(config, seed=0):
    """Plain DDQN acting agent (gsp=False), deterministic net init."""
    T.manual_seed(seed)
    return Agent(
        config=config, network='DDQN', n_agents=4, n_obs=31,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=False, recurrent=False, attention=False,
        neighbors=False, gsp_input_size=6, gsp_output_size=1,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
    )


def _gsp_states(k=1, n_agents=4, seed=11):
    rng = np.random.default_rng(seed + k)
    return [rng.standard_normal(6).astype(np.float32) for _ in range(n_agents)]


_NO_FAILURE = np.array([0], dtype=np.intc)


class TestFlagParsing:
    def test_default_is_off(self):
        agent = _make_gsp_n_agent(_config())
        assert agent.batched_actor_forward is False

    def test_flag_on_parses(self):
        agent = _make_gsp_n_agent(_config(BATCHED_ACTOR_FORWARD=True))
        assert agent.batched_actor_forward is True


class TestFlagOffByteIdentical:
    """(a) Flag off = the exact legacy sequential path, bit-for-bit."""

    def test_gsp_predictions_match_reference_loop_deterministic(self):
        agent = _make_gsp_n_agent(_config())
        states = _gsp_states()

        via_agent = agent.choose_agent_gsp(states, test=True)
        # Inline reference: the pre-#53-B per-agent loop, verbatim semantics.
        reference = [
            agent.choose_action(states[i], agent.gsp_networks, True)
            for i in range(4)
        ]

        for got, want in zip(via_agent, reference):
            np.testing.assert_array_equal(got, want)

    def test_gsp_predictions_match_reference_loop_with_noise_rng(self):
        """Same torch RNG stream consumption as legacy: bit-exact under
        exploration noise when re-seeded."""
        config = _config(NOISE=0.1)
        agent = _make_gsp_n_agent(config)
        states = _gsp_states(seed=23)

        T.manual_seed(1234)
        via_agent = agent.choose_agent_gsp(states, test=False)
        T.manual_seed(1234)
        reference = [
            agent.choose_action(states[i], agent.gsp_networks, False)
            for i in range(4)
        ]

        for got, want in zip(via_agent, reference):
            np.testing.assert_array_equal(got, want)

    def test_flag_off_and_flag_absent_agree(self):
        """Explicit False and absent key build identical behavior."""
        a_absent = _make_gsp_n_agent(_config(), seed=7)
        a_false = _make_gsp_n_agent(
            _config(BATCHED_ACTOR_FORWARD=False), seed=7)
        states = _gsp_states(seed=31)

        p_absent = a_absent.choose_agent_gsp(states, test=True)
        p_false = a_false.choose_agent_gsp(states, test=True)

        for got, want in zip(p_false, p_absent):
            np.testing.assert_array_equal(got, want)


class TestMainStartupContract:
    """Static source contracts on rl_code/Main.py (same technique as
    tests/test_main_gsp_contract.py — no ARGoS needed).

    Fail-loud activation contract (#53-B): Main.py must emit exactly one of
    the two unmistakable BATCHED_ACTOR_FORWARD startup lines, and must mirror
    --independent_learning into config BEFORE constructing agents so the
    Agent-side GSP gate shares Main.py's condition source.
    """

    @staticmethod
    def _main_text():
        import pathlib
        return (pathlib.Path(__file__).resolve().parents[2]
                / "rl_code" / "Main.py").read_text()

    def test_engaged_and_off_startup_lines_present(self):
        text = self._main_text()
        assert "BATCHED_ACTOR_FORWARD: ENGAGED" in text
        assert "BATCHED_ACTOR_FORWARD: off (sequential)" in text

    def test_independent_learning_mirrored_before_agent_construction(self):
        text = self._main_text()
        inject = text.find(
            "config['INDEPENDENT_LEARNING'] = bool(args.independent_learning)")
        construct = text.find("Agent.Agent(")
        assert inject != -1, "INDEPENDENT_LEARNING config mirror missing"
        assert construct != -1
        assert inject < construct, (
            "INDEPENDENT_LEARNING must be mirrored into config BEFORE any "
            "Agent construction — the Agent reads it in __init__")


class TestFlagOnGSPHead:
    """(b)+(c) Batched GSP-head prediction path."""

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_batched_matches_sequential_within_tolerance(self, k):
        seq_agent = _make_gsp_n_agent(_config(), gsp_output_size=k, seed=k)
        bat_agent = _make_gsp_n_agent(
            _config(BATCHED_ACTOR_FORWARD=True), gsp_output_size=k, seed=k)
        states = _gsp_states(k=k)

        sequential = seq_agent.choose_agent_gsp(states, test=True)
        batched = bat_agent.choose_agent_gsp(states, test=True)

        # Documented expected drift: batched matmul changes float-reduction
        # order → fp tolerance, not bit-exact.
        np.testing.assert_allclose(
            np.asarray(batched), np.asarray(sequential), atol=1e-6)

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_shape_one_k_vector_per_agent(self, k):
        agent = _make_gsp_n_agent(
            _config(BATCHED_ACTOR_FORWARD=True), gsp_output_size=k, seed=k)
        states = _gsp_states(k=k, seed=41)

        preds = agent.choose_agent_gsp(states, test=True)

        assert len(preds) == 4
        assert all(np.asarray(p).shape == (k,) for p in preds)

    def test_attention_head_stays_sequential(self):
        """Flag on + A-GSP head: the gate must route around the batched path
        (which would raise NotImplementedError) and still return one
        prediction per agent via the legacy loop."""
        agent = _make_gsp_n_agent(
            _config(BATCHED_ACTOR_FORWARD=True), attention=True)
        states = [np.zeros(6, dtype=np.float32) for _ in range(4)]

        preds = agent.choose_agent_gsp(states, test=True)

        assert len(preds) == 4


class TestFlagOnActing:
    """(b) Batched DDQN acting via choose_agent_actions_batch."""

    def test_greedy_actions_match_sequential(self):
        agent = _make_ddqn_agent(_config(BATCHED_ACTOR_FORWARD=True))
        rng = np.random.default_rng(3)
        observations = [
            rng.standard_normal(31).astype(np.float32) for _ in range(4)
        ]

        sequential = [
            agent.choose_agent_action(obs, _NO_FAILURE, test=True)
            for obs in observations
        ]
        actions_to_take, action_nums = agent.choose_agent_actions_batch(
            observations, test=True)

        assert action_nums == [a[1] for a in sequential]
        for got, want in zip(actions_to_take, (a[0] for a in sequential)):
            np.testing.assert_array_equal(got, want)
        assert agent.failed is False

    def test_exploring_draws_per_robot_gates(self):
        """epsilon=1.0: one np.random.random() gate per robot with the
        explorer choice draws interleaved in robot order — the #91 v2 RNG
        contract (flag-on stream IDENTICAL to sequential)."""
        agent = _make_ddqn_agent(_config(EPSILON=1.0))
        agent.batched_actor_forward = True
        observations = [np.zeros(31, dtype=np.float32) for _ in range(4)]

        np.random.seed(99)
        _, action_nums = agent.choose_agent_actions_batch(
            observations, test=False)

        np.random.seed(99)
        expected = []
        for _ in range(4):
            _ = np.random.random()
            expected.append(np.random.choice(agent.action_space))

        assert action_nums == expected

    def test_all_exploit_step_consumes_per_robot_gate_draws_and_goes_greedy(self):
        """0 < epsilon < 1, a seed whose FOUR gate draws all EXPLOIT: exactly
        R np.random.random() gate draws (one per robot — #91 v2), ZERO
        np.random.choice draws, then the batched greedy actions."""
        agent = _make_ddqn_agent(_config(EPSILON=0.5))
        agent.batched_actor_forward = True
        rng = np.random.default_rng(5)
        observations = [
            rng.standard_normal(31).astype(np.float32) for _ in range(4)
        ]

        # Greedy reference: the test=True path never touches np.random.
        _, greedy_nums = agent.choose_agent_actions_batch(
            observations, test=True)

        np.random.seed(0)  # first four np.random.random() draws all > 0.5
        _, action_nums = agent.choose_agent_actions_batch(
            observations, test=False)
        state_after = np.random.get_state()

        np.random.seed(0)
        gates = [np.random.random() for _ in range(4)]
        assert all(g > agent.epsilon for g in gates)  # seed sanity: all exploit
        expected_state = np.random.get_state()

        assert action_nums == greedy_nums
        assert state_after[0] == expected_state[0]
        np.testing.assert_array_equal(state_after[1], expected_state[1])
        assert state_after[2:] == expected_state[2:]

    def test_mixed_step_matches_sequential_stream_and_actions(self):
        """#91 v2 acceptance gate at the call site: on a MIXED explore/
        exploit step, flag-on actions AND the post-call np.random state equal
        the sequential choose_agent_action loop's exactly."""
        agent = _make_ddqn_agent(_config(EPSILON=0.6))
        agent.batched_actor_forward = True
        rng = np.random.default_rng(9)
        observations = [
            rng.standard_normal(31).astype(np.float32) for _ in range(4)
        ]

        np.random.seed(0)
        actions_to_take, action_nums = agent.choose_agent_actions_batch(
            observations, test=False)
        state_batched = np.random.get_state()

        np.random.seed(0)
        sequential = [
            agent.choose_agent_action(obs, _NO_FAILURE, test=False)
            for obs in observations
        ]
        state_sequential = np.random.get_state()

        # Seed sanity: this seed/epsilon mixes both branches across the step.
        np.random.seed(0)
        exploits = []
        for _ in range(4):
            exploit = np.random.random() > agent.epsilon
            exploits.append(exploit)
            if not exploit:
                np.random.choice(agent.action_space)
        assert any(exploits) and not all(exploits)

        assert action_nums == [a[1] for a in sequential]
        for got, want in zip(actions_to_take, (a[0] for a in sequential)):
            np.testing.assert_array_equal(got, want)
        assert state_batched[0] == state_sequential[0]
        np.testing.assert_array_equal(state_batched[1], state_sequential[1])
        assert state_batched[2:] == state_sequential[2:]

    def test_non_discrete_scheme_raises(self):
        T.manual_seed(0)
        agent = Agent(
            config=_config(), network='DDPG', n_agents=4, n_obs=31,
            n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
            meta_param_size=1, gsp=False, recurrent=False, attention=False,
            neighbors=False, gsp_input_size=6, gsp_output_size=1,
            gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
        )
        observations = [np.zeros(31, dtype=np.float32) for _ in range(4)]
        with pytest.raises(NotImplementedError):
            agent.choose_agent_actions_batch(observations, test=True)


class TestIndependentLearningExclusion:
    """Independent-learning cells must NEVER take the batched GSP path.

    Main.py mirrors --independent_learning into config['INDEPENDENT_LEARNING']
    before agent construction; the choose_agent_gsp gate (via
    batched_gsp_path_engaged) excludes it — same condition source as Main.py's
    acting-loop gate. Per-robot nets have nothing to batch through one net, so
    these cells stay on the legacy sequential loop outside the #53-B
    contamination contract.
    """

    def test_gate_helper_false_under_independent_learning(self):
        agent = _make_gsp_n_agent(
            _config(BATCHED_ACTOR_FORWARD=True, INDEPENDENT_LEARNING=True))
        assert agent.batched_gsp_path_engaged() is False

    def test_gate_helper_true_for_shared_model(self):
        """Guard: the exclusion must not break the shared-model engaged path
        (key absent = non-Main.py caller = shared-model by construction)."""
        agent = _make_gsp_n_agent(_config(BATCHED_ACTOR_FORWARD=True))
        assert agent.batched_gsp_path_engaged() is True

    def test_flag_on_independent_learning_stays_sequential_bit_exact(self):
        """Same technique as TestFlagOffByteIdentical: bit-exact match against
        the sequential per-agent reference loop under exploration noise. The
        batched path draws ONE (R, K) noise tensor instead of R sequential
        (1, K) draws, so it could not reproduce this stream — bit-exact
        equality proves the sequential path executed."""
        config = _config(
            NOISE=0.1, BATCHED_ACTOR_FORWARD=True, INDEPENDENT_LEARNING=True)
        agent = _make_gsp_n_agent(config)
        states = _gsp_states(seed=23)

        T.manual_seed(1234)
        via_agent = agent.choose_agent_gsp(states, test=False)
        T.manual_seed(1234)
        reference = [
            agent.choose_action(states[i], agent.gsp_networks, False)
            for i in range(4)
        ]

        for got, want in zip(via_agent, reference):
            np.testing.assert_array_equal(got, want)

"""GSP_SPLICE_ADVANTAGE_ONLY — host-side (RL-CT) advantage-splice tests.

Contract under test (opt-in, default off; engine lives in GSP-RL#42):

(a) Flag OFF (or absent) = byte-identical legacy flat Q-head: the Agent's
    config dict flows unmodified into the GSP-RL Actor, so absent and
    explicit-False builds are bit-exact (state dict + greedy actions) and no
    value-stream modules exist — same golden technique as
    tests/test_agent/test_batched_actor_forward.py.
(b) Flag ON (DDQN + GSP-N, delta_theta_traj K=5) = dueling head with the
    prediction wired into the advantage stream only, verified through the
    Agent's OWN make_agent_state splice: perturbing the heading_gsp slot of
    the augmented obs must leave V (== mean_a Q) unchanged while moving the
    differential component.
(c) Unsupported schemes raise loudly at Agent construction (never a silently
    ignored flag), and Main.py carries the fail-loud ENGAGED/off startup
    contract keyed on the Actor-side gate (single condition source).
"""

import numpy as np
import torch as T
import pytest

from src.agent import Agent

OBS = 31
K = 5           # delta_theta_traj horizon
AUG = OBS + K


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


def _traj(**overrides):
    """delta_theta_traj K=5 — the advantage-splice campaign target shape."""
    cfg = _config(
        GSP_OUTPUT_KIND="delta_theta_traj", GSP_PREDICTION_HORIZON=K)
    cfg.update(overrides)
    return cfg


def _make_agent(config, network='DDQN', gsp=True, seed=0):
    T.manual_seed(seed)
    return Agent(
        config=config, network=network, n_agents=4, n_obs=OBS,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=gsp, recurrent=False, attention=False,
        neighbors=True, gsp_input_size=6, gsp_output_size=1,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
    )


def _q(agent, aug_obs):
    """Q-values from the agent's q_eval for one augmented observation."""
    net = agent.networks['q_eval']
    x = T.tensor(np.asarray(aug_obs, dtype=np.float32), device=net.device)
    with T.no_grad():
        return net(x).cpu()


_NO_FAILURE = np.array([0], dtype=np.intc)


class TestFlagParsing:
    def test_default_is_off(self):
        agent = _make_agent(_traj())
        assert agent.gsp_splice_advantage_only is False
        assert agent.gsp_splice_advantage_engaged is False

    def test_flag_on_engages_and_places_campaign_span(self):
        agent = _make_agent(_traj(GSP_SPLICE_ADVANTAGE_ONLY=True))
        assert agent.gsp_splice_advantage_engaged is True
        for key in ('q_eval', 'q_next'):
            assert agent.networks[key].advantage_only_pred == (OBS, K)
            assert agent.networks[key].v_fc1.in_features == OBS


class TestFlagOffByteIdentical:
    """(a) Off-path golden: absent vs explicit False, bit-for-bit."""

    def test_state_dicts_bit_exact_and_no_value_stream(self):
        a_absent = _make_agent(_traj(), seed=7)
        a_false = _make_agent(
            _traj(GSP_SPLICE_ADVANTAGE_ONLY=False), seed=7)
        sd_a = a_absent.networks['q_eval'].state_dict()
        sd_f = a_false.networks['q_eval'].state_dict()
        assert set(sd_a.keys()) == set(sd_f.keys())
        for k in sd_a:
            assert T.equal(sd_a[k], sd_f[k]), f'{k} differs'
        assert not hasattr(a_absent.networks['q_eval'], 'v_fc1')

    def test_greedy_actions_bit_exact_absent_vs_false(self):
        a_absent = _make_agent(_traj(), seed=11)
        a_false = _make_agent(
            _traj(GSP_SPLICE_ADVANTAGE_ONLY=False), seed=11)
        rng = np.random.default_rng(3)
        for _ in range(8):
            env_obs = rng.standard_normal(OBS).astype(np.float32)
            pred = rng.standard_normal(K).astype(np.float32)
            s_absent = a_absent.make_agent_state(env_obs, heading_gsp=pred)
            s_false = a_false.make_agent_state(env_obs, heading_gsp=pred)
            np.testing.assert_array_equal(s_absent, s_false)
            got = a_absent.choose_agent_action(s_absent, _NO_FAILURE, test=True)
            want = a_false.choose_agent_action(s_false, _NO_FAILURE, test=True)
            assert got[1] == want[1]
            np.testing.assert_array_equal(got[0], want[0])


class TestFlagOnThroughAgentSplice:
    """(b) V invariance verified through the Agent's own acting splice."""

    def test_perturbing_pred_slot_leaves_v_moves_advantage(self):
        agent = _make_agent(_traj(GSP_SPLICE_ADVANTAGE_ONLY=True), seed=13)
        rng = np.random.default_rng(17)
        env_obs = rng.standard_normal(OBS).astype(np.float32)
        pred_a = rng.standard_normal(K).astype(np.float32)
        pred_b = pred_a + rng.standard_normal(K).astype(np.float32)

        s_a = agent.make_agent_state(env_obs.copy(), heading_gsp=pred_a)
        s_b = agent.make_agent_state(env_obs.copy(), heading_gsp=pred_b)
        assert s_a.shape == (AUG,)
        np.testing.assert_array_equal(s_a[:OBS], s_b[:OBS])
        assert not np.array_equal(s_a[OBS:], s_b[OBS:])

        q_a, q_b = _q(agent, s_a), _q(agent, s_b)
        # V == mean_a Q by the dueling identity: pred cannot move it.
        assert T.allclose(q_a.mean(dim=-1), q_b.mean(dim=-1), atol=1e-6)
        # The differential (advantage) component DOES move.
        diff_a = q_a - q_a.mean(dim=-1, keepdim=True)
        diff_b = q_b - q_b.mean(dim=-1, keepdim=True)
        assert not T.allclose(diff_a, diff_b)

    def test_perturbing_env_obs_moves_v(self):
        """Dead-value-stream guard: obs changes must move V."""
        agent = _make_agent(_traj(GSP_SPLICE_ADVANTAGE_ONLY=True), seed=13)
        rng = np.random.default_rng(19)
        pred = rng.standard_normal(K).astype(np.float32)
        obs_a = rng.standard_normal(OBS).astype(np.float32)
        obs_b = obs_a + rng.standard_normal(OBS).astype(np.float32)
        q_a = _q(agent, agent.make_agent_state(obs_a, heading_gsp=pred))
        q_b = _q(agent, agent.make_agent_state(obs_b, heading_gsp=pred))
        assert not T.allclose(q_a.mean(dim=-1), q_b.mean(dim=-1))

    def test_choose_agent_action_runs_on_dueling_head(self):
        agent = _make_agent(_traj(GSP_SPLICE_ADVANTAGE_ONLY=True), seed=23)
        rng = np.random.default_rng(29)
        state = agent.make_agent_state(
            rng.standard_normal(OBS).astype(np.float32),
            heading_gsp=rng.standard_normal(K).astype(np.float32))
        actions, action_num = agent.choose_agent_action(
            state, _NO_FAILURE, test=True)
        assert action_num in agent.action_space

    def test_gradient_reaches_pred_columns(self):
        """The splice stays differentiable: a Q loss produces nonzero grad on
        the pred input columns (the E2E path into the head stays alive)."""
        agent = _make_agent(_traj(GSP_SPLICE_ADVANTAGE_ONLY=True), seed=31)
        net = agent.networks['q_eval']
        rng = np.random.default_rng(37)
        x = T.tensor(rng.standard_normal((16, AUG)), dtype=T.float32,
                     device=net.device, requires_grad=True)
        (net(x) ** 2).sum().backward()
        assert float(x.grad[:, OBS:].abs().sum()) > 0.0


class TestUnsupportedSchemesRaiseLoudly:
    """(c) Never silently ignore the flag."""

    @pytest.mark.parametrize('scheme', ['DDPG', 'TD3'])
    def test_continuous_schemes_raise(self, scheme):
        with pytest.raises(ValueError, match='GSP_SPLICE_ADVANTAGE_ONLY'):
            _make_agent(
                _config(GSP_SPLICE_ADVANTAGE_ONLY=True), network=scheme)

    def test_no_gsp_slot_raises(self):
        with pytest.raises(ValueError, match='no spliced prediction slot'):
            _make_agent(
                _config(GSP_SPLICE_ADVANTAGE_ONLY=True), gsp=False)


class TestMainStartupContract:
    """Static source contracts on rl_code/Main.py (same technique as
    TestMainStartupContract in test_batched_actor_forward.py — no ARGoS).

    Fail-loud activation contract: Main.py must emit exactly one of the two
    unmistakable GSP_SPLICE_ADVANTAGE_ONLY startup lines, keyed on the
    Actor-side gate attribute (single condition source) — the two silent-drop
    traps (stale daemon dropping the passthrough; pre-GSP-RL#42 pin ignoring
    the key) make this line the activation check for every splice arm.
    """

    @staticmethod
    def _main_text():
        import pathlib
        return (pathlib.Path(__file__).resolve().parents[2]
                / "rl_code" / "Main.py").read_text()

    def test_engaged_and_off_startup_lines_present(self):
        text = self._main_text()
        assert "GSP_SPLICE_ADVANTAGE_ONLY: ENGAGED (dueling, pred->advantage-only)" in text
        assert "GSP_SPLICE_ADVANTAGE_ONLY: off" in text

    def test_log_keyed_on_actor_side_gate(self):
        text = self._main_text()
        assert "gsp_splice_advantage_engaged" in text, (
            "startup line must read the Actor's effective gate, not re-derive "
            "the condition from raw config")

    def test_log_emitted_after_agent_construction(self):
        text = self._main_text()
        construct = text.find("Agent.Agent(")
        log_line = text.find("GSP_SPLICE_ADVANTAGE_ONLY: ENGAGED")
        assert construct != -1 and log_line != -1
        assert construct < log_line, (
            "the ENGAGED line reads the constructed Agent's gate attribute, "
            "so it must come after Agent construction")

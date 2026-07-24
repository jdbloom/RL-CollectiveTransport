"""Threading tests for GSP_ACTION_CONDITIONED (Stage-2 Arm C diagonal-Q).

Covers plan 2f (docs/superpowers/plans/2026-07-24-actcond-rod-threading.md):
  1. Flag OFF -> make_agent_state / head-store paths byte-identical.
  2. Store-site augmentation: stored gsp_obs width = GSP_INPUT_SIZE + N and
     the one-hot index equals the FIFO-carried action.
  3. Guard matrix: every 2e violation raises ValueError naming the lever.
  4. Diagonal wrapper: failure=True mirrors choose_agent_action verbatim;
     non-failure mapping (parse_action) and the stored==scored obs equality.
  5. ENGAGED line emitted exactly once, plus static source contracts on
     Main.py (same technique as test_e2e_arith_startup_contract.py — the
     guards/act/store blocks live in Main.py's episode loop, which cannot be
     imported without ARGoS/ZMQ).

Real Agent instances throughout — no mocks.
"""
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src"))

from agent import Agent, validate_action_conditioned  # noqa: E402


N_ACTIONS = 9   # options_per_action ** n_actions = 3 ** 2
K = 5           # GSP_PREDICTION_HORIZON == head output width for delta_theta_traj
GSP_IN = 8      # neighbors head input: 2 (self) + 2 * 3 neighbors


def _base_config(**overrides):
    cfg = {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
        "GSP_OUTPUT_KIND": "delta_theta_traj",
        "GSP_PREDICTION_TARGET": "delta_theta_traj",
        "GSP_PREDICTION_HORIZON": K,
    }
    cfg.update(overrides)
    return cfg


def _actcond_overrides():
    return {
        "GSP": 1,
        "GSP_ACTION_CONDITIONED": True,
        "GSP_ACTION_COND_ENCODING": "onehot",
        "GSP_ACTION_COND_N": N_ACTIONS,
        "MAX_NUM_ROBOT_FAILURES": 0,
    }


def _make_agent(cfg, neighbors=True):
    return Agent(
        config=cfg,
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
        neighbors=neighbors,
        gsp_input_size=6,
        gsp_output_size=1,
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
    )


def _engaged_agent(**cfg_overrides):
    cfg = _base_config(**{**_actcond_overrides(), **cfg_overrides})
    return cfg, _make_agent(cfg)


def _main_text():
    return (pathlib.Path(__file__).resolve().parents[2]
            / "rl_code" / "Main.py").read_text()


# ---------------------------------------------------------------------------
# 1. Flag OFF -> byte-identical make_agent_state + head-store paths
# ---------------------------------------------------------------------------
class TestFlagOffByteIdentical:

    def test_make_agent_state_is_raw_concat_for_absent_and_false(self):
        rng = np.random.default_rng(7)
        agent_absent = _make_agent(_base_config())
        agent_false = _make_agent(_base_config(GSP_ACTION_CONDITIONED=False))
        for _ in range(10):
            obs = rng.normal(size=31).astype(np.float32)
            pred = rng.normal(size=K).astype(np.float32)
            expected = np.concatenate([obs, pred])
            out_a = agent_absent.make_agent_state(obs, heading_gsp=pred)
            out_b = agent_false.make_agent_state(obs, heading_gsp=pred)
            assert np.array_equal(out_a, expected)
            assert np.array_equal(out_a, out_b)
            assert out_a.tobytes() == out_b.tobytes()

    def test_head_store_rollout_hash_identical_and_unaugmented(self):
        """Same synthetic FIFO rollout on both agents -> byte-identical
        stored arrays, width == gsp_network_input (no one-hot appended)."""
        hashes = []
        for flag in (None, False):
            cfg = _base_config() if flag is None else _base_config(
                GSP_ACTION_CONDITIONED=False)
            agent = _make_agent(cfg)
            rng = np.random.default_rng(123)  # same stream for both agents
            n_r = 4
            for step in range(K + 3):
                states = [rng.normal(size=agent.gsp_network_input
                                     ).astype(np.float32) for _ in range(n_r)]
                agent.push_pending_gsp_obs(
                    states, states, payload_angle_deg=float(step),
                    action_per_robot=[int(rng.integers(N_ACTIONS))
                                      for _ in range(n_r)],
                )
                matured = agent.pop_matured_gsp_label(None)
                if matured is not None:
                    for i in range(n_r):
                        s = matured["state_per_robot"][i]
                        assert len(s) == agent.gsp_network_input
                        agent.store_gsp_transition(
                            s, np.zeros(K, dtype=np.float32), 0, s, 0)
            replay = agent.gsp_networks["replay"]
            assert replay.state_memory.shape[1] == agent.gsp_network_input
            n = min(replay.mem_ctr, replay.mem_size)
            assert n > 0, "rollout must mature and store at least one sample"
            hashes.append(replay.state_memory[:n].tobytes())
        assert hashes[0] == hashes[1]


# ---------------------------------------------------------------------------
# 2. Store-site augmentation (flag ON)
# ---------------------------------------------------------------------------
class TestStoreSiteAugmentation:

    def test_stored_width_and_onehot_index_match_fifo_action(self):
        cfg, agent = _engaged_agent()
        n_in = agent.gsp_network_input
        replay = agent.gsp_networks["replay"]
        # GSP-RL #46 contract: engaged replay is constructed augmented-width.
        assert replay.num_observations == n_in + N_ACTIONS
        assert replay.state_memory.shape[1] == n_in + N_ACTIONS

        rng = np.random.default_rng(5)
        n_r = 4
        pushed_actions = []
        stored = 0
        for step in range(K + 4):
            states = [rng.normal(size=n_in).astype(np.float32)
                      for _ in range(n_r)]
            acts = [int(rng.integers(N_ACTIONS)) for _ in range(n_r)]
            pushed_actions.append(acts)
            agent.push_pending_gsp_obs(
                states, states, payload_angle_deg=float(step),
                action_per_robot=acts,
            )
            matured = agent.pop_matured_gsp_label(None)
            if matured is not None:
                act_row = matured.get("action_per_robot")
                assert act_row is not None, (
                    "FIFO must carry the realized actions when the flag is on")
                for i in range(n_r):
                    # Mirror the Main.py 2d store block exactly.
                    act_i = int(act_row[i])
                    assert 0 <= act_i < N_ACTIONS
                    enc = np.zeros(N_ACTIONS, dtype=np.float32)
                    enc[act_i] = 1.0
                    s_to_store = np.concatenate([
                        np.asarray(matured["state_per_robot"][i],
                                   dtype=np.float32), enc])
                    agent.store_gsp_transition(
                        s_to_store, np.zeros(K, dtype=np.float32),
                        0, s_to_store, 0)
                    row = replay.state_memory[stored]
                    assert row.shape[0] == n_in + N_ACTIONS
                    onehot = row[n_in:]
                    assert onehot.sum() == 1.0
                    assert int(np.argmax(onehot)) == act_i
                    stored += 1
        assert stored > 0
        # Maturity pairing: matured actions are the ones pushed K steps ago.
        assert pushed_actions[0][0] == int(
            np.argmax(replay.state_memory[0][n_in:]))

    def test_main_store_site_source_contract(self):
        text = _main_text()
        assert "GSP_ACTION_CONDITIONED head-store" in text
        assert "action_per_robot" in text
        # The augmented state (reassigned s_to_store) must flow to BOTH the
        # replay store and the h5 record.
        idx = text.find("_actcond_enc[_act_i] = 1.0")
        assert idx != -1
        after = text[idx:idx + 1500]
        assert "model.store_gsp_transition(s_to_store" in after
        assert "record_stored_transition(label_to_store, s_to_store" in after


# ---------------------------------------------------------------------------
# 3. Guard matrix — every 2e violation raises, naming the lever
# ---------------------------------------------------------------------------
class TestGuardMatrix:

    def test_flag_off_returns_disengaged(self):
        cfg = _base_config()
        agent = _make_agent(cfg)
        assert validate_action_conditioned(cfg, agent) == (False, 0, [])

    def test_engaged_happy_path(self):
        cfg, agent = _engaged_agent()
        engaged, n, lines = validate_action_conditioned(cfg, agent)
        assert engaged is True
        assert n == N_ACTIONS
        assert len(lines) == 2

    @pytest.mark.parametrize("override,lever", [
        ({"INDEPENDENT_LEARNING": True}, "INDEPENDENT_LEARNING"),
        ({"GSP": 0}, "lever GSP"),
        ({"GSP_OUTPUT_KIND": "delta_theta_1d"}, "GSP_OUTPUT_KIND"),
        ({"GSP_E2E_ENABLED": 1}, "GSP_E2E_ENABLED"),
        ({"GSP_E2E_NORMALIZE_FEATURE": 1}, "GSP_E2E_NORMALIZE_FEATURE"),
        ({"GSP_ACTION_COND_ENCODING": "embedding"}, "GSP_ACTION_COND_ENCODING"),
        ({"GLOBAL_KNOWLEDGE": 1}, "GLOBAL_KNOWLEDGE"),
        ({"MAX_NUM_ROBOT_FAILURES": 2}, "MAX_NUM_ROBOT_FAILURES"),
        ({"GSP_ACTION_COND_N": None}, "GSP_ACTION_COND_N"),
        ({"GSP_ACTION_COND_N": 5}, "does not match the actor"),
    ])
    def test_config_lever_violations_raise(self, override, lever):
        cfg, agent = _engaged_agent()
        bad = dict(cfg)
        bad.update(override)
        with pytest.raises(ValueError, match=lever):
            validate_action_conditioned(bad, agent)

    def test_neighbors_off_raises(self):
        cfg = _base_config(**_actcond_overrides())
        agent = _make_agent(cfg, neighbors=False)
        with pytest.raises(ValueError, match="neighbors"):
            validate_action_conditioned(cfg, agent)

    def test_splice_gain_raises(self):
        cfg, agent = _engaged_agent()
        agent.gsp_e2e_splice_gain = 2.0
        with pytest.raises(ValueError, match="GSP_E2E_SPLICE_GAIN"):
            validate_action_conditioned(cfg, agent)

    def test_zero_out_raises(self):
        cfg, agent = _engaged_agent()
        agent.gsp_zero_out_signal = True
        with pytest.raises(ValueError, match="GSP_ZERO_OUT_SIGNAL"):
            validate_action_conditioned(cfg, agent)

    def test_boltzmann_raises(self):
        cfg, agent = _engaged_agent()
        agent.boltzmann_temperature = 0.5
        with pytest.raises(ValueError, match="BOLTZMANN_TEMPERATURE"):
            validate_action_conditioned(cfg, agent)

    def test_disengaged_actor_gate_raises(self):
        """Config says ON but the Actor was built flag-off (stale GSP-RL pin
        scenario) -> the actor-side gate check fires."""
        off_cfg = _base_config()
        agent = _make_agent(off_cfg)
        on_cfg = dict(off_cfg, **_actcond_overrides())
        with pytest.raises(ValueError, match="gsp_action_conditioned_engaged"):
            validate_action_conditioned(on_cfg, agent)


# ---------------------------------------------------------------------------
# 4. Diagonal wrapper — failure contract + mapping + stored==scored equality
# ---------------------------------------------------------------------------
class TestDiagonalWrapper:

    def test_failure_path_mirrors_choose_agent_action(self):
        cfg, agent = _engaged_agent()
        # choose_agent_action's failure branch returns self.failure_action,
        # which no constructor ever assigns (latent legacy trap) — set it the
        # way a failure-enabled host would, identically for both calls.
        agent.failure_action = np.array([0.0, 0.0, 0.0])
        preds = np.zeros((N_ACTIONS, K), dtype=np.float32)
        obs = np.zeros(31, dtype=np.float32)

        ref_action, ref_num = agent.choose_agent_action(
            np.zeros(31 + K, dtype=np.float32), True, False)
        assert agent.failed is True
        diag_action, diag_num = agent.choose_agent_action_diagonal(
            obs, preds, True, False)
        assert agent.failed is True
        assert np.array_equal(ref_action, diag_action)
        assert ref_num == diag_num == agent.failure_action_code

    def test_nonfailure_mapping_and_stored_equals_scored(self):
        cfg, agent = _engaged_agent()
        rng = np.random.default_rng(11)
        obs = rng.normal(size=31).astype(np.float32)
        gsp_state = rng.normal(size=agent.gsp_network_input).astype(np.float32)
        preds = agent.predict_gsp_actions(gsp_state, N_ACTIONS)
        preds_np = preds.detach().cpu().numpy().astype(np.float32)

        action, action_num = agent.choose_agent_action_diagonal(
            obs, preds_np, False, True)  # test=True, eval_epsilon 0 -> greedy
        assert agent.failed is False
        assert 0 <= action_num < N_ACTIONS
        assert np.array_equal(action, agent.parse_action(action_num))
        # Same inputs, same greedy call -> same action (determinism).
        assert action_num == agent.choose_action_diagonal(
            obs, preds_np, agent.networks, test=True)

        # 2c invariant: the agent_state the act site rebuilds from the CHOSEN
        # row equals the obs row the diagonal scored (raw concat under the 2e
        # transform guards).
        rebuilt = agent.make_agent_state(obs, heading_gsp=preds_np[action_num])
        scored = np.concatenate([obs, preds_np[action_num]])
        assert np.array_equal(rebuilt, scored)


# ---------------------------------------------------------------------------
# 5. ENGAGED line + Main.py source contracts
# ---------------------------------------------------------------------------
class TestStartupContract:

    def test_engaged_line_emitted_exactly_once(self, capsys):
        cfg, agent = _engaged_agent()
        engaged, n, lines = validate_action_conditioned(cfg, agent)
        # Emulate Main.py's consumption loop (log.info per returned line).
        for line in lines:
            print(line)
        out = capsys.readouterr().out
        assert out.count("[ACTCOND] ENGAGED:") == 1
        assert (f"[ACTCOND] ENGAGED: N={N_ACTIONS} encoding=onehot "
                f"target=delta_theta_traj K={K}") in out
        assert out.count(
            "[ACTCOND] batched actor forward disabled "
            "(per-robot diagonal path)") == 1

    def test_main_startup_source_contract(self):
        text = _main_text()
        # Exactly one consumption loop for the returned lines, plus the
        # exactly-one-of-two off line (repo startup-line convention).
        assert text.count("validate_action_conditioned(") == 1
        assert 'log.info("GSP_ACTION_CONDITIONED: off")' in text
        assert "_batched_actor_forward = False" in text

    def test_main_act_site_source_contract(self):
        text = _main_text()
        # Diagonal branch must be ordered BEFORE the batched branch and set
        # the chosen row + rebuild through make_agent_state.
        i_diag = text.find("choose_agent_action_diagonal(")
        i_batch = text.find("model.choose_agent_actions_batch(")
        assert i_diag != -1 and i_batch != -1 and i_diag < i_batch
        after = text[i_diag:i_diag + 1200]
        assert "next_heading_gsp[i] = _actcond_preds[i][action_num]" in after
        assert "agent_states[i] = model.make_agent_state(" in after

    def test_main_predict_site_source_contract(self):
        text = _main_text()
        assert text.count("model.predict_gsp_actions(") == 1
        i_pred = text.find("model.predict_gsp_actions(")
        window = text[i_pred - 2500:i_pred + 2500]
        # Row-wise eval ablation and the stash write both live at the site.
        assert "apply_pred_ablation" in window
        assert "_actcond_preds[i] = _p_rows" in window

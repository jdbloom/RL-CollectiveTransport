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
        window = text[i_pred - 2500:i_pred + 5000]
        # Row-wise eval ablation and the stash write both live at the site.
        assert "apply_pred_ablation" in window
        assert "_actcond_preds[i] = _p_rows" in window


# ---------------------------------------------------------------------------
# Review FIX 1 — eval-ablation row identity + baseline fold cardinality
# ---------------------------------------------------------------------------
class TestAblationRowIdentity:
    """The actcond eval ablation applies ONCE per robot per step (fold-stream
    = realized-action rows only, R folds/step like baseline arms) and
    broadcasts the single ablated vector to all N rows (bit-identical rows ->
    the diagonal collapses to plain Q)."""

    def _site_ablate(self, agent, p_rows, last_act_i, mode, rng, mean_state):
        """Mirror of the Main.py FIX 1 predict-site ablation block."""
        from pred_ablation import apply_pred_ablation
        abl_row = apply_pred_ablation(
            p_rows[int(last_act_i)], mode, rng, mean_state)
        return np.tile(np.asarray(abl_row, dtype=np.float32), (N_ACTIONS, 1))

    def test_rows_bit_identical_and_fold_count_is_R_per_step(self):
        from pred_ablation import RunningMeanState
        cfg, agent = _engaged_agent()
        rng_np = np.random.default_rng(3)
        abl_rng = np.random.default_rng(0)
        mean_state = RunningMeanState()
        n_r = 4
        last_act = np.zeros(n_r, dtype=int)
        n_steps = 3
        for step in range(n_steps):
            for i in range(n_r):
                gsp_state = rng_np.normal(
                    size=agent.gsp_network_input).astype(np.float32)
                p_rows = agent.predict_gsp_actions(
                    gsp_state, N_ACTIONS).detach().cpu().numpy().astype(np.float32)
                p_rows = self._site_ablate(
                    agent, p_rows, last_act[i], "frozen_mean",
                    abl_rng, mean_state)
                # All N rows bit-identical after ablation.
                for a in range(1, N_ACTIONS):
                    assert p_rows[a].tobytes() == p_rows[0].tobytes()
                last_act[i] = int(rng_np.integers(N_ACTIONS))
            # Accumulator ingests exactly R realized-prediction-shaped
            # vectors per step — NOT N*R.
            assert mean_state.count == n_r * (step + 1)

    def test_zero_mode_rows_identical_zero(self):
        from pred_ablation import RunningMeanState
        cfg, agent = _engaged_agent()
        p_rows = agent.predict_gsp_actions(
            np.random.default_rng(5).normal(
                size=agent.gsp_network_input).astype(np.float32),
            N_ACTIONS).detach().cpu().numpy().astype(np.float32)
        out = self._site_ablate(
            agent, p_rows, 0, "zero",
            np.random.default_rng(0), RunningMeanState())
        assert out.shape == (N_ACTIONS, K)
        assert not out.any()

    def test_main_predict_site_single_fold_source_contract(self):
        text = _main_text()
        i_pred = text.find("model.predict_gsp_actions(")
        window = text[i_pred:i_pred + 5000]
        # Single apply_pred_ablation call in the actcond branch (no per-row
        # loop), fed by the realized-action index, broadcast via np.tile.
        assert window.count("apply_pred_ablation(") == 1
        assert "_actcond_last_act[i]" in window
        assert "np.tile(" in window
        # The old per-row fold pattern must be gone from the actcond branch.
        assert "for _a in range(_actcond_n):" not in window


# ---------------------------------------------------------------------------
# Review FIX 2 — one-iteration delayed store: next_state == acted state @ t+1
# ---------------------------------------------------------------------------
class TestDelayedNextStateStore:
    """Baseline next-state contract in actcond mode: the stored next_state is
    the exact (bit-equal) obs acted on at t+1 — completed one iteration late
    via the pending-store flush; terminal step flushes the held
    new_agent_states fallback (done-masked, inert for learning)."""

    def _trace(self, agent, n_steps=3):
        """Synthetic per-robot=1 trace mirroring the Main.py actcond flow:
        act(+rebuild) -> flush pending -> step -> predict -> build new_state
        -> push pending. Returns (acted_states, pushed fallbacks)."""
        rng = np.random.default_rng(17)
        pending = []
        acted_states = []
        fallbacks = []
        preds = np.zeros((N_ACTIONS, K), dtype=np.float32)
        obs = rng.normal(size=31).astype(np.float32)
        for t in range(n_steps + 1):  # one extra act to flush step n-1
            # --- act site: choose + rebuild ---
            action, action_num = agent.choose_agent_action_diagonal(
                obs, preds, False, True)
            acted = agent.make_agent_state(
                obs, heading_gsp=preds[action_num]).astype(np.float32)
            acted_states.append(acted.copy())
            # --- act site: flush pending (FIX 2) ---
            for (p_s, p_a, p_r, p_i, p_ns_fb, p_done) in pending:
                agent.store_agent_transition(
                    p_s, p_a, p_r, acted.copy(), p_done)
            pending = []
            if t == n_steps:
                break
            # --- env step: fresh obs; predict: fresh forecast stack ---
            obs = rng.normal(size=31).astype(np.float32)
            gsp_state = rng.normal(
                size=agent.gsp_network_input).astype(np.float32)
            preds = agent.predict_gsp_actions(
                gsp_state, N_ACTIONS).detach().cpu().numpy().astype(np.float32)
            # --- store site: build fallback next_state, push pending ---
            ns_fallback = agent.make_agent_state(
                obs, heading_gsp=preds[action_num]).astype(np.float32)
            fallbacks.append(ns_fallback.copy())
            pending.append((
                acted.copy(),
                (action_num, action),
                np.array([0.5], dtype=np.float32),
                0,
                ns_fallback.copy(),
                False,
            ))
        return acted_states, fallbacks, pending

    def test_stored_next_state_equals_acted_state_at_t_plus_1(self):
        cfg, agent = _engaged_agent()
        acted_states, _, leftover = self._trace(agent, n_steps=3)
        assert leftover == []  # act-site flush leaves nothing pending
        replay = agent.networks["replay"]
        n = min(replay.mem_ctr, replay.mem_size)
        assert n == 3  # exactly one transition per pushed step (no drop/dup)
        for t in range(3):
            assert (replay.state_memory[t].astype(np.float32).tobytes()
                    == acted_states[t].tobytes())
            assert (replay.new_state_memory[t].astype(np.float32).tobytes()
                    == acted_states[t + 1].tobytes()), (
                f"stored next_state[{t}] must be the exact obs acted on at "
                f"t+1 (fresh splice), bit-equal")

    def test_terminal_flush_uses_fallback_and_keeps_count(self):
        cfg, agent = _engaged_agent()
        rng = np.random.default_rng(23)
        # 2 normal steps + 1 terminal step whose pending is flushed with the
        # held new_agent_states fallback (no t+1 act).
        acted_states, fallbacks, _ = self._trace(agent, n_steps=2)
        base = min(agent.networks["replay"].mem_ctr,
                   agent.networks["replay"].mem_size)
        assert base == 2
        # Terminal pending: push then flush with fallback (Main.py teardown).
        obs = rng.normal(size=31).astype(np.float32)
        preds = agent.predict_gsp_actions(
            rng.normal(size=agent.gsp_network_input).astype(np.float32),
            N_ACTIONS).detach().cpu().numpy().astype(np.float32)
        action, action_num = agent.choose_agent_action_diagonal(
            obs, preds, False, True)
        s_t = agent.make_agent_state(
            obs, heading_gsp=preds[action_num]).astype(np.float32)
        ns_fb = rng.normal(size=31 + K).astype(np.float32)
        agent.store_agent_transition(
            s_t, (action_num, action), np.array([0.0], dtype=np.float32),
            ns_fb, True)
        replay = agent.networks["replay"]
        n = min(replay.mem_ctr, replay.mem_size)
        assert n == 3  # exactly one transition per step overall
        assert (replay.new_state_memory[2].astype(np.float32).tobytes()
                == ns_fb.tobytes())
        assert bool(replay.terminal_memory[2]) is True  # done-masked -> inert

    def test_main_delayed_store_source_contract(self):
        text = _main_text()
        # Pending push replaces the immediate store ONLY in the actcond arm;
        # the legacy immediate store is untouched (OFF path byte-identical).
        i_push = text.find("_actcond_pending_store.append((")
        assert i_push != -1
        assert "elif _actcond:" in text[i_push - 2500:i_push]
        # The legacy store still exists right after the actcond arm.
        after_push = text[i_push:i_push + 3500]
        assert "model.store_agent_transition(agent_states[i]," in after_push
        # Act-site flush completes with the rebuilt agent_states.
        i_flush = text.find("if _actcond_pending_store:")
        assert i_flush != -1
        flush_win = text[i_flush:i_flush + 900]
        assert "np.asarray(agent_states[_p_i]" in flush_win
        # Terminal flush lives in the episode_done teardown, uses the
        # fallback, and documents the done-mask inertness.
        i_term = text.find("FIX 2 terminal flush")
        assert i_term != -1
        term_win = text[i_term:i_term + 1600]
        assert "_p_ns_fb, _p_done" in term_win
        assert "done-masked" in term_win
        # Episode-boundary invariant raises loudly.
        assert "survived the episode boundary" in text


# ---------------------------------------------------------------------------
# Review FIX 3 — K=1 (scalar-slot rescale) guard
# ---------------------------------------------------------------------------
class TestScalarSlotGuard:

    def test_horizon_one_raises(self):
        cfg = _base_config(**_actcond_overrides())
        cfg["GSP_PREDICTION_HORIZON"] = 1  # dtraj K=1 -> gsp_network_output=1
        agent = _make_agent(cfg)
        assert agent.gsp_network_output == 1
        with pytest.raises(ValueError, match="GSP_PREDICTION_HORIZON"):
            validate_action_conditioned(cfg, agent)

    def test_vector_slot_still_passes(self):
        cfg, agent = _engaged_agent()  # K=5
        engaged, n, lines = validate_action_conditioned(cfg, agent)
        assert engaged is True and n == N_ACTIONS


# ---------------------------------------------------------------------------
# Review FIX 4 — additive actcond_pred_matrix h5 dataset
# ---------------------------------------------------------------------------
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestActcondPredMatrixDataset:

    def _writerow_kwargs(self):
        return dict(
            rewards=[0.1] * 4, epsilons=0.5, terminations=False, losses=0.0,
            force_magnitudes=[0.0] * 4, force_angles=[0.0] * 4,
            average_force_vectors=[0.0, 0.0], cyl_x_poses=0.0,
            cyl_y_poses=0.0, cyl_angles=0.0, gate_stats=0, obstacle_stats=0,
            gsp_rewards=[0.0] * 4, gsp_headings=[0.0] * 4, run_times=0.0,
            robots_x_poses=[0.0] * 4, robots_y_poses=[0.0] * 4,
            robot_angles=[0.0] * 4, robot_failure=[False] * 4,
        )

    def test_dataset_present_and_shaped_when_flag_on(self, tmp_path):
        from hdf5_logger import HDF5Logger
        path = str(tmp_path / "ep.h5")
        logger = HDF5Logger(path, count_episodes=False)
        T, R = 3, 4
        rng = np.random.default_rng(9)
        mats = [rng.normal(size=(R, N_ACTIONS, K)).astype(np.float32)
                for _ in range(T)]
        for t in range(T):
            logger.writerow(**self._writerow_kwargs())
            logger.record_actcond_pred_matrix(mats[t])
        logger.write_episode(0)
        with h5py.File(path) as f:
            grp = f["episode_0000"]
            assert "actcond_pred_matrix" in grp
            assert grp["actcond_pred_matrix"].shape == (T, R, N_ACTIONS, K)
            np.testing.assert_array_equal(
                grp["actcond_pred_matrix"][1], mats[1])

    def test_dataset_absent_when_flag_off_and_resets_per_episode(self, tmp_path):
        from hdf5_logger import HDF5Logger
        path = str(tmp_path / "ep.h5")
        logger = HDF5Logger(path, count_episodes=False)
        # Episode 0: engaged-style (dataset present).
        logger.writerow(**self._writerow_kwargs())
        logger.record_actcond_pred_matrix(
            np.zeros((4, N_ACTIONS, K), dtype=np.float32))
        logger.write_episode(0)
        # Episode 1: no record calls -> buffer reset -> dataset absent.
        logger.writerow(**self._writerow_kwargs())
        logger.write_episode(1)
        with h5py.File(path) as f:
            assert "actcond_pred_matrix" in f["episode_0000"]
            assert "actcond_pred_matrix" not in f["episode_0001"]

    def test_main_records_at_predict_site_and_documents_basis(self):
        text = _main_text()
        i_pred = text.find("model.predict_gsp_actions(")
        window = text[i_pred:i_pred + 5000]
        assert "hdf5_writer.record_actcond_pred_matrix(" in window
        # Basis-offset comment at the writerow site (existing columns
        # unchanged; analyses redirected to the additive dataset).
        i_row = text.find("hdf5_writer.writerow(")
        before_row = text[i_row - 1500:i_row]
        assert "actcond_pred_matrix" in before_row
        assert "basis t-1" in before_row

"""Tests for the K-step GLOBAL trajectory GSP targets
(goal_progress_traj, cyl_displacement_traj).

Motivation (2026-07-09, force-causal-use campaign): the payload-rotation
trajectory (delta_theta_traj) is learnable + coupled but value-null. The
objective variables of the task are GLOBAL — the payload's translation and its
progress-to-goal — so these two targets mirror the delta_theta_traj machinery
exactly (same delayed K-step FIFO, same auto-derive/consistency behavior) but
predict:

  goal_progress_traj     O = K   per-step payload progress-to-goal delta
                                 (prev_cyl_dist2goal - curr, positive = toward
                                 goal — the exact quantity from the
                                 cyl_kinematics_goal_4d kind's 4th component)
                                 over the NEXT K steps.
  cyl_displacement_traj  O = 2K  per-step payload (dx, dy) over the next K
                                 steps, flattened [dx1,dy1,...,dxK,dyK].

Labels are RAW physical units (meters) — no magic scaling (the F15
loss-balance lesson: lambda is tuned per measured label std, not baked into
the label). K = GSP_PREDICTION_HORIZON.

The FIFO carries a per-step `payload_track` dict ({'dist2goal','cyl_x',
'cyl_y'}) the same way delta_theta_traj carries payload_angle_deg; at maturity
pop returns the ordered K+1-entry `payload_track_window` and the driver
(Main.py) differences consecutive entries.
"""
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rl_code", "src"))

from agent import Agent  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_agent(gsp_output_kind="delta_theta_1d", prediction_target=None, K=5,
                n_agents=4, n_hop_neighbors=1, **extra):
    cfg = _base_config(
        GSP_OUTPUT_KIND=gsp_output_kind,
        GSP_PREDICTION_HORIZON=K,
        **extra,
    )
    if prediction_target is not None:
        cfg["GSP_PREDICTION_TARGET"] = prediction_target
    return Agent(
        config=cfg,
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
        gsp_input_size=6,   # overridden in __init__ when neighbors=True
        gsp_output_size=1,  # overridden by GSP_OUTPUT_KIND
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
        n_hop_neighbors=n_hop_neighbors,
    )


def _progress_traj(dist_window):
    """Reference size-(len-1) per-step goal-progress trajectory:
    prev_dist2goal - curr_dist2goal, positive = toward goal."""
    return np.array([
        float(dist_window[k]) - float(dist_window[k + 1])
        for k in range(len(dist_window) - 1)
    ], dtype=np.float32)


def _displacement_traj(x_window, y_window):
    """Reference size-2*(len-1) flattened per-step (dx, dy) trajectory."""
    out = []
    for k in range(len(x_window) - 1):
        out.append(float(x_window[k + 1]) - float(x_window[k]))
        out.append(float(y_window[k + 1]) - float(y_window[k]))
    return np.array(out, dtype=np.float32)


def _push_track(agent, states, dist2goal, cyl_x, cyl_y):
    agent.push_pending_gsp_obs(
        states, states,
        payload_track={
            "dist2goal": float(dist2goal),
            "cyl_x": float(cyl_x),
            "cyl_y": float(cyl_y),
        },
    )


# (target/kind name, horizon multiplier)
_CASES = [("goal_progress_traj", 1), ("cyl_displacement_traj", 2)]


# ---------------------------------------------------------------------------
# (a) accepted kinds + delayed-label registration + size == mult*K
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,mult", _CASES)
def test_output_size_equals_mult_times_horizon(kind, mult):
    for K in (1, 2, 3, 5):
        agent = _make_agent(gsp_output_kind=kind, K=K)
        assert agent.gsp_output_kind == kind
        assert agent.gsp_output_size_effective == mult * K, f"K={K}"
        assert agent.gsp_network_output == mult * K, f"K={K}"


@pytest.mark.parametrize("kind,mult", _CASES)
def test_is_delayed_label_target(kind, mult):
    assert kind in Agent._DELAYED_LABEL_TARGETS
    agent = _make_agent(gsp_output_kind=kind, prediction_target=kind, K=3)
    assert agent._is_delayed_label_target() is True


@pytest.mark.parametrize("kind,mult", _CASES)
def test_horizon_zero_rejected(kind, mult):
    with pytest.raises(ValueError, match="GSP_PREDICTION_HORIZON"):
        _make_agent(gsp_output_kind=kind, K=0)


# ---------------------------------------------------------------------------
# (b) size wired through actor input width + neighbor-sharing width
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,mult", _CASES)
def test_input_size_scales_with_slot_width(kind, mult):
    """GSP-N per-agent layout: input = (1 + mult*K) per slot x 3 slots
    (self + 2 neighbors at n_hop_neighbors=1)."""
    for K in (1, 2, 3, 5):
        agent = _make_agent(gsp_output_kind=kind, K=K, n_hop_neighbors=1)
        expected = (1 + mult * K) * 3
        assert agent.gsp_network_input == expected, (
            f"K={K}: expected input {expected}, got {agent.gsp_network_input}"
        )


@pytest.mark.parametrize("kind,mult", _CASES)
def test_target_autoderives_kind_when_output_kind_default(kind, mult):
    """GSP_PREDICTION_TARGET alone (kind left at scalar default) must size the
    head input/output — the same auto-derive as delta_theta_traj."""
    for K in (1, 3, 5):
        agent = _make_agent(
            gsp_output_kind="delta_theta_1d",
            prediction_target=kind,
            K=K,
        )
        assert agent.gsp_output_kind == kind
        assert agent.gsp_network_output == mult * K
        assert agent.gsp_network_input == (1 + mult * K) * 3


@pytest.mark.parametrize("kind,mult", _CASES)
def test_actor_state_width_matches_net(kind, mult):
    """The augmented actor state from make_agent_state (fed a size-mult*K
    prediction) must equal the actor Q-net's network_input_size — for 2K=10
    this exercises the former `size > 5` JEPA-skip misfire path."""
    for K in (1, 3, 5):
        agent = _make_agent(gsp_output_kind=kind, prediction_target=kind, K=K)
        pred = np.zeros(agent.gsp_network_output, dtype=np.float32)
        env_obs = np.zeros(agent.input_size, dtype=np.float32)
        state = agent.make_agent_state(env_obs, heading_gsp=pred)
        assert state.shape[0] == agent.network_input_size, (
            f"K={K}: actor state width {state.shape[0]} != net input "
            f"{agent.network_input_size}"
        )


@pytest.mark.parametrize("kind,mult", _CASES)
def test_explicit_contradiction_rejected(kind, mult):
    with pytest.raises(ValueError, match=kind):
        _make_agent(
            gsp_output_kind="future_prox_1d",
            prediction_target=kind,
            K=5,
        )


# ---------------------------------------------------------------------------
# (c) FIFO payload_track carry + matured label reconstruction
# ---------------------------------------------------------------------------

def test_fifo_returns_none_until_K_plus_one_pushes():
    agent = _make_agent(
        gsp_output_kind="goal_progress_traj",
        prediction_target="goal_progress_traj",
        K=3,
    )
    n = agent.n_agents
    for t in range(3):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        _push_track(agent, states, dist2goal=1.0, cyl_x=0.0, cyl_y=0.0)
        assert agent.pop_matured_gsp_label(None) is None, f"t={t}"


def test_matured_goal_progress_label_is_size_K():
    """Drive a synthetic dist2goal sequence; the matured window paired with the
    state pushed at t must reconstruct the size-K per-step progress trajectory
    over [t..t+K] (positive = toward goal), in RAW distance units."""
    K = 3
    n = 4
    agent = _make_agent(
        gsp_output_kind="goal_progress_traj",
        prediction_target="goal_progress_traj",
        K=K, n_agents=n,
    )
    dist_seq = [5.0, 4.6, 4.5, 3.9, 3.95, 3.2]
    matured_by_state = {}
    for t in range(len(dist_seq)):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        _push_track(agent, states, dist2goal=dist_seq[t], cyl_x=0.0, cyl_y=0.0)
        matured = agent.pop_matured_gsp_label(None)
        if matured is not None:
            win = matured["payload_track_window"]
            traj = _progress_traj([e["dist2goal"] for e in win])
            matured_by_state[float(matured["state_per_robot"][0][0])] = traj
    for src_t in (0, 1, 2):
        assert src_t in matured_by_state
        expected = _progress_traj(dist_seq[src_t:src_t + K + 1])
        assert expected.shape == (K,)
        np.testing.assert_allclose(matured_by_state[src_t], expected, atol=1e-6)
    # Sign convention: distance DROP (toward goal) is POSITIVE.
    assert matured_by_state[0][0] == pytest.approx(0.4, abs=1e-6)
    # Away-from-goal step is negative (dist 3.9 -> 3.95 at k=3..4 in window of t=1).
    assert matured_by_state[1][2] == pytest.approx(-0.05, abs=1e-6)


def test_matured_displacement_label_is_size_2K_flattened():
    """Synthetic (x, y) payload track; the matured window must reconstruct the
    flattened [dx1,dy1,...,dxK,dyK] trajectory in RAW meters."""
    K = 3
    n = 4
    agent = _make_agent(
        gsp_output_kind="cyl_displacement_traj",
        prediction_target="cyl_displacement_traj",
        K=K, n_agents=n,
    )
    x_seq = [0.0, 0.1, 0.25, 0.3, 0.28, 0.5]
    y_seq = [1.0, 0.95, 0.9, 1.0, 1.2, 1.1]
    matured_by_state = {}
    for t in range(len(x_seq)):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        _push_track(agent, states, dist2goal=0.0, cyl_x=x_seq[t], cyl_y=y_seq[t])
        matured = agent.pop_matured_gsp_label(None)
        if matured is not None:
            win = matured["payload_track_window"]
            traj = _displacement_traj([e["cyl_x"] for e in win],
                                      [e["cyl_y"] for e in win])
            matured_by_state[float(matured["state_per_robot"][0][0])] = traj
    for src_t in (0, 1, 2):
        assert src_t in matured_by_state
        expected = _displacement_traj(x_seq[src_t:src_t + K + 1],
                                      y_seq[src_t:src_t + K + 1])
        assert expected.shape == (2 * K,)
        np.testing.assert_allclose(matured_by_state[src_t], expected, atol=1e-6)
    # Flattening order: [dx1, dy1, dx2, dy2, ...].
    np.testing.assert_allclose(
        matured_by_state[0][:2], [0.1, -0.05], atol=1e-6
    )


def test_K1_reduces_to_single_step():
    agent = _make_agent(
        gsp_output_kind="goal_progress_traj",
        prediction_target="goal_progress_traj",
        K=1,
    )
    n = agent.n_agents
    dist_seq = [2.0, 1.7, 1.6]
    got = {}
    for t in range(len(dist_seq)):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        _push_track(agent, states, dist2goal=dist_seq[t], cyl_x=0.0, cyl_y=0.0)
        matured = agent.pop_matured_gsp_label(None)
        if matured is not None:
            win = matured["payload_track_window"]
            got[float(matured["state_per_robot"][0][0])] = _progress_traj(
                [e["dist2goal"] for e in win]
            )
    assert got[0.0].shape == (1,)
    assert got[0.0][0] == pytest.approx(0.3, abs=1e-6)


def test_same_label_for_every_robot():
    """The trajectory label is GLOBAL — the same vector for every robot (the
    driver replicates it per robot exactly like delta_theta_traj)."""
    K = 2
    n = 3
    agent = _make_agent(
        gsp_output_kind="cyl_displacement_traj",
        prediction_target="cyl_displacement_traj",
        K=K, n_agents=n,
    )
    for t in range(K + 1):
        states = [np.full(8, float(t) + 0.1 * i, dtype=np.float32)
                  for i in range(n)]
        _push_track(agent, states, dist2goal=0.0, cyl_x=float(t), cyl_y=-float(t))
    matured = agent.pop_matured_gsp_label(None)
    assert matured is not None
    win = matured["payload_track_window"]
    traj = _displacement_traj([e["cyl_x"] for e in win],
                              [e["cyl_y"] for e in win])
    # One shared window -> one shared trajectory; per-robot states are distinct.
    assert traj.shape == (2 * K,)
    assert len(matured["state_per_robot"]) == n


# ---------------------------------------------------------------------------
# (d) legacy behavior unchanged
# ---------------------------------------------------------------------------

def test_fifo_noop_for_default_target():
    agent = _make_agent(gsp_output_kind="goal_progress_traj", K=3)
    assert agent.gsp_prediction_target == "delta_theta"
    assert agent._is_delayed_label_target() is False
    n = agent.n_agents
    for _ in range(10):
        states = [np.zeros(8, dtype=np.float32) for _ in range(n)]
        _push_track(agent, states, dist2goal=1.0, cyl_x=0.0, cyl_y=0.0)
        assert agent.pop_matured_gsp_label(None) is None


def test_push_without_track_window_is_all_none():
    """payload_track is optional: dtraj/future_prox/neighbor_force pushes (which
    do not pass it) yield an all-None payload_track_window — byte-identical
    behavior for the existing targets."""
    K = 2
    n = 3
    agent = _make_agent(
        gsp_output_kind="delta_theta_traj",
        prediction_target="delta_theta_traj",
        K=K, n_agents=n,
    )
    for t in range(K + 1):
        states = [np.full(8, float(t), dtype=np.float32) for _ in range(n)]
        agent.push_pending_gsp_obs(states, states, payload_angle_deg=float(t))
    matured = agent.pop_matured_gsp_label(None)
    assert matured is not None
    assert all(e is None for e in matured["payload_track_window"])
    # And the angle window still works as before.
    assert matured["payload_angle_window"] == [0.0, 1.0, 2.0]


def test_delta_theta_traj_sizes_unchanged():
    agent = _make_agent(gsp_output_kind="delta_theta_traj",
                        prediction_target="delta_theta_traj", K=4)
    assert agent.gsp_network_output == 4          # K, not 2K
    assert agent.gsp_network_input == (1 + 4) * 3


# ---------------------------------------------------------------------------
# (e) make_agent_state slot discrimination: JEPA flag, not a size heuristic
# ---------------------------------------------------------------------------

def test_2K_vector_slot_concatenated_raw():
    """A 2K=10 prediction (non-JEPA) must be concatenated RAW — no degrees/10
    scalar scaling, no truncation. Before the fix the `size > 5` heuristic
    classified any width>5 slot as a JEPA latent; the raveled result was the
    same, but the discrimination is now explicit (gsp_jepa_enabled flag)."""
    agent = _make_agent(
        gsp_output_kind="cyl_displacement_traj",
        prediction_target="cyl_displacement_traj",
        K=5,
    )
    assert agent.gsp_network_output == 10
    pred = np.linspace(-0.01, 0.01, 10).astype(np.float32)
    env_obs = np.zeros(agent.input_size, dtype=np.float32)
    state = agent.make_agent_state(env_obs, heading_gsp=pred)
    np.testing.assert_allclose(
        state[agent.input_size:agent.input_size + 10], pred, atol=1e-7,
        err_msg="vector GSP slot must be raw physical units, no rescaling",
    )


def test_scalar_slot_still_degrees_over_10():
    """Legacy scalar slot keeps the historical degrees(x/10) scaling."""
    agent = _make_agent(gsp_output_kind="delta_theta_1d", K=1)
    val = 0.02
    env_obs = np.zeros(agent.input_size, dtype=np.float32)
    state = agent.make_agent_state(env_obs, heading_gsp=np.float32(val))
    assert state[agent.input_size] == pytest.approx(np.degrees(val / 10), rel=1e-5)


def test_jepa_latent_slot_raw_via_flag():
    """With gsp_jepa_enabled the slot is the encoder latent, concatenated raw —
    keyed on the FLAG, for any latent width (including widths <= 5, which the
    old size heuristic would have misrouted through the scalar/vector paths)."""
    agent = _make_agent(gsp_output_kind="delta_theta_1d", K=1)
    agent.gsp_jepa_enabled = True  # host-side flag, as parsed from GSP_JEPA_ENABLED
    for width in (4, 32):
        latent = np.linspace(0.5, 1.5, width).astype(np.float32)
        env_obs = np.zeros(agent.input_size, dtype=np.float32)
        state = agent.make_agent_state(env_obs, heading_gsp=latent)
        np.testing.assert_allclose(
            state[agent.input_size:agent.input_size + width], latent, atol=1e-7,
            err_msg=f"JEPA latent (width {width}) must be concatenated raw",
        )


_AGENT_PY = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "rl_code" / "src" / "agent.py"
)


def test_agent_no_size_heuristic_for_jepa():
    """The JEPA-latent discrimination must use the gsp_jepa_enabled flag, not
    the `size > 5` width heuristic (which misfires for 2K=10)."""
    text = _AGENT_PY.read_text()
    assert "size > 5" not in text, (
        "agent.py must not discriminate the JEPA latent by width; use the "
        "gsp_jepa_enabled flag"
    )


# ---------------------------------------------------------------------------
# (f) Main.py wiring contract (static) — raw labels, deferred E2E store
# ---------------------------------------------------------------------------

_MAIN_PY = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "rl_code" / "Main.py"
)


def test_main_traj_target_set_contains_global_targets():
    text = _MAIN_PY.read_text()
    assert "_GSP_TRAJ_TARGETS" in text
    assert "'goal_progress_traj'" in text
    assert "'cyl_displacement_traj'" in text


def test_main_defers_immediate_e2e_store_for_traj_set():
    """The immediate main-replay E2E store must be deferred for ALL trajectory
    targets (the delayed FIFO owns those transitions)."""
    text = _MAIN_PY.read_text()
    idx = text.index("_defer_immediate_e2e_store = bool(")
    block = text[idx:idx + 400]
    assert "_GSP_TRAJ_TARGETS" in block, (
        "the defer gate must cover the whole trajectory-target set, "
        "not just delta_theta_traj"
    )


def test_main_global_traj_labels_single_scale_inside_builder():
    """Global-target label scaling contract (2026-07-10): the ONLY scale on
    goal_progress/cyl_displacement labels is GSP_TRAJ_LABEL_SCALE (default 1.0
    = raw meters), applied INSIDE _build_traj_label_from_windows so head-store,
    E2E, and h5 logging share one target definition. The dtraj-specific
    _delta_theta_traj_label_scale must not touch them, no per-target scale
    constants may appear, and no caller may apply a second scale outside the
    builder."""
    text = _MAIN_PY.read_text()
    # The single scale constant is read once with a raw default.
    assert "GSP_TRAJ_LABEL_SCALE" in text
    assert "config.get('GSP_TRAJ_LABEL_SCALE', 1.0)" in text
    # Applied inside the builder, on both metric kinds.
    b_idx = text.index("def _build_traj_label_from_windows")
    builder = text[b_idx:text.index("raise ValueError", b_idx)]
    assert builder.count("* _gsp_traj_label_scale") == 2, (
        "expected the scale MULTIPLIED exactly once per metric kind inside the builder"
    )
    # The E2E caller applies NO second scale to the global targets: its
    # else-branch (non-dtraj) is a bare astype.
    e_idx = text.index("E2E delayed main-replay store")
    block = text[e_idx:e_idx + 6000]
    assert "_traj_label = _traj_e2e.astype(np.float32)" in block
    # No per-target magic scale constants.
    assert "GSP_GOAL_PROGRESS_TRAJ_LABEL_SCALE" not in text
    assert "GSP_CYL_DISPLACEMENT_TRAJ_LABEL_SCALE" not in text

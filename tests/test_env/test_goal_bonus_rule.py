"""GOAL-ENTRY bonus rule tests (Option A, operator-approved 2026-07-13).

Contract under test (flag-gated, default OFF = byte-identical) — the
contact-rule suite twin (tests/test_env/test_obstacle_contact_rule.py):

(a) GOAL_BONUS absent/0.0 = byte-identical legacy behavior: identical reward
    stream, identical stored transitions (values AND done flags), no RNG
    consumption — proven against an inline pre-change reference loop on a
    fixed fake-stats episode (Main.py itself is not importable in a unit
    test, so the wiring is mirrored by a mini-simulator and pinned to
    Main.py by static source contracts, the accepted pattern).
(b) Threshold arithmetic: GOAL_TERMINAL_DIST defaults to the env success
    radius (GOAL_RADIUS, else 2.0) + GOAL_ENTRY_MARGIN_M 0.25; explicit
    value wins; values at or inside the success radius raise (the
    learner-invisibility trap: the ARGoS-terminal transition is never
    stored, so a radius-equal detector pays a bonus the learner cannot see).
(c) GOAL_BONUS is added to EVERY robot's reward exactly once on the
    goal-entry step (shared consequence; per-robot convention — never
    divided by num_robots).
(d) Goal entry stores the entry-step transition with done=True (the
    bootstrap cut that grounds the bonus: q_next[dones]=0 in GSP-RL
    learning_aids) and stores nothing afterwards (logical termination,
    zombie phase to the ARGoS success).
(e) First-event-wins vs the contact rule, BOTH orders: contact-before-goal
    keeps contact semantics (goal detector suppressed); goal-before-contact
    books the bonus terminal (contact detector suppressed); a same-step tie
    resolves to contact.
(f) Startup log line: exactly one of ENGAGED/off, keyed on the effective
    gate (bonus != 0).
(g) h5 per-episode attrs (goal_terminal/goal_step/goal_store_dropped)
    written when engaged, absent (byte-identical h5) when off.
(h) goal_store_dropped edges: a goal entry within K steps of the physical
    episode end leaves its done=True E2E transition unmatured in the
    delayed FIFO (flagged by Agent.unmatured_done_e2e_transitions before
    the buffer reset deletes it); a goal entry at time_steps<=2 is dropped
    by the legacy store guard. Both are LOUD (INFO log + attr).
"""

import numpy as np
import pytest

try:
    from rl_code.src.goal_rule import (
        GoalBonusRule, DEFAULT_GOAL_RADIUS_M, GOAL_ENTRY_MARGIN_M,
    )
    from rl_code.src.contact_rule import ContactRule
except ImportError:  # pragma: no cover - path-dependent import
    from src.goal_rule import (  # type: ignore
        GoalBonusRule, DEFAULT_GOAL_RADIUS_M, GOAL_ENTRY_MARGIN_M,
    )
    from src.contact_rule import ContactRule  # type: ignore

try:
    import h5py
    _HAS_H5PY = True
except ImportError:  # pragma: no cover
    _HAS_H5PY = False


R = 4  # robots


def _robot_stats(xy_list):
    """Per-robot [x, y, z, x_deg, y_deg, z_deg] arrays (parse_robot_stats shape)."""
    return [np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for x, y in xy_list]


def _obstacles(xy_list):
    """Flat [x0, y0, x1, y1, ...] array (parse_obstacle_stats shape)."""
    return np.asarray([c for xy in xy_list for c in xy], dtype=np.float32)


# ---------------------------------------------------------------------------
# Flag parsing / effective gate / fail-loud
# ---------------------------------------------------------------------------

class TestFlagParsing:
    def test_default_is_off(self):
        rule = GoalBonusRule({})
        assert rule.enabled is False
        assert rule.bonus == 0.0

    def test_flags_absent_and_explicit_off_agree(self):
        a = GoalBonusRule({})
        b = GoalBonusRule({"GOAL_BONUS": 0.0})
        assert (a.enabled, a.bonus, a.terminal_dist) == \
               (b.enabled, b.bonus, b.terminal_dist)

    def test_bonus_engages(self):
        rule = GoalBonusRule({"GOAL_BONUS": 10000})
        assert rule.enabled is True
        assert rule.bonus == 10000.0
        assert isinstance(rule.bonus, float)  # YAML int arrives as float

    def test_default_dist_derived_from_success_radius_plus_margin(self):
        """Default threshold = goal_radius default 2.0 (generate_argos
        --goal_radius -> template threshold/min_threshold -> cpp
        m_fThreshold) + the 0.25 m entry margin."""
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0})
        expected = DEFAULT_GOAL_RADIUS_M + GOAL_ENTRY_MARGIN_M
        assert rule.terminal_dist == pytest.approx(expected)
        assert rule.terminal_dist == pytest.approx(2.25)

    def test_dist_respects_goal_radius_override(self):
        """Per-cell GOAL_RADIUS shifts the derived default."""
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0, "GOAL_RADIUS": 1.0})
        assert rule.terminal_dist == pytest.approx(1.0 + GOAL_ENTRY_MARGIN_M)

    def test_explicit_dist_wins(self):
        rule = GoalBonusRule(
            {"GOAL_BONUS": 10000.0, "GOAL_TERMINAL_DIST": 3.0})
        assert rule.terminal_dist == 3.0

    def test_engaged_dist_at_success_radius_raises(self):
        """Fail loud on the learner-invisibility trap: a threshold at (or
        inside) the success radius first fires on the ARGoS terminal step,
        whose transition the legacy episode_done store guard drops."""
        with pytest.raises(ValueError, match="learner-invisible"):
            GoalBonusRule({"GOAL_BONUS": 10000.0, "GOAL_TERMINAL_DIST": 2.0})

    def test_engaged_dist_inside_overridden_radius_raises(self):
        with pytest.raises(ValueError, match="learner-invisible"):
            GoalBonusRule({
                "GOAL_BONUS": 10000.0, "GOAL_RADIUS": 3.0,
                "GOAL_TERMINAL_DIST": 2.9,
            })

    def test_off_rule_never_raises_on_weird_dist(self):
        """The off path must construct for every legacy config."""
        rule = GoalBonusRule({"GOAL_TERMINAL_DIST": 0.0})
        assert rule.enabled is False


# ---------------------------------------------------------------------------
# (f) Startup log line — both states
# ---------------------------------------------------------------------------

class TestStartupLine:
    def test_engaged_line(self):
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0})
        line = rule.startup_line()
        assert line == "GOAL_BONUS rule: ENGAGED (bonus=10000.0 dist=2.25)"

    def test_off_line(self):
        rule = GoalBonusRule({})
        assert rule.startup_line() == "GOAL_BONUS rule: off"


# ---------------------------------------------------------------------------
# (b) Detection — strict `<` on cyl_dist2goal
# ---------------------------------------------------------------------------

class TestDetect:
    def test_below_threshold_is_entry(self):
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0})
        assert rule.detect(2.2499) is True

    def test_above_threshold_is_not_entry(self):
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0})
        assert rule.detect(2.26) is False

    def test_exactly_at_threshold_is_not_entry(self):
        """Strict `<` — the boundary itself is not an entry (mirrors both
        ObjectAtTarget()'s `< m_fThreshold` and the contact predicate)."""
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0})
        assert rule.detect(2.25) is False


# ---------------------------------------------------------------------------
# Mini-simulator: the POST-change Main.py wiring for BOTH rules, mirrored
# step-for-step (detection gates -> shared penalty/bonus -> done-flag
# override -> zombie-phase store suppression -> episode attrs). Main.py is
# not importable in a unit test; the real wiring is pinned by
# TestMainSourceContract below.
# ---------------------------------------------------------------------------

def _run_fake_episode(goal_rule, contact_rule, cyl_traj, robot_traj,
                      obstacles, episode_len, base_reward=-2.0):
    """Mirror of the Main.py step loop for both rules' touch points.

    cyl_traj: dict {step (1-based) -> cyl_dist2goal} — steps not present
    reuse the previous value (default far, 8.0).
    robot_traj: dict {step -> list of R (x, y)} — same hold-previous rule.
    Returns (stored, state, per_step_rewards).
    """
    stored = []
    per_step_rewards = {}
    contact_logical_done = False
    contact_first_step = -1
    contact_events = 0
    goal_logical_done = False
    goal_first_step = -1
    goal_store_dropped = False
    cyl_dist = 8.0
    positions = robot_traj[min(robot_traj)] if robot_traj else \
        [(3.0, 3.0), (3.0, -3.0), (-3.0, 3.0), (-3.0, -3.0)]

    for t in range(1, episode_len + 1):
        episode_done = (t == episode_len)
        positions = robot_traj.get(t, positions)
        cyl_dist = cyl_traj.get(t, cyl_dist)
        robot_stats = _robot_stats(positions)

        # --- contact detection (gated on BOTH zombie phases) ---
        contact_now = False
        if (contact_rule.enabled and not contact_logical_done
                and not goal_logical_done):
            hit, _ri, _oi, _d = contact_rule.detect(robot_stats, obstacles)
            if hit:
                contact_now = True
                contact_events += 1
                if contact_first_step < 0:
                    contact_first_step = t

        # --- goal detection (same-step tie -> contact wins) ---
        goal_now = False
        if (goal_rule.enabled and not goal_logical_done
                and not contact_logical_done and not contact_now
                and goal_rule.detect(cyl_dist)):
            goal_now = True
            goal_first_step = t

        step_store_done = bool(episode_done) or (
            contact_now and contact_rule.terminate) or goal_now

        # --- reward loop (penalty then bonus, every robot) ---
        rewards = [np.array([base_reward], dtype=np.float32) for _ in range(R)]
        for i in range(R):
            if contact_now:
                rewards[i] += contact_rule.penalty
            if goal_now:
                rewards[i] += goal_rule.bonus
            # --- immediate store site (guard chain mirrored) ---
            if t > 2 and not goal_logical_done and not contact_logical_done:
                if not episode_done:
                    stored.append((t, i, float(rewards[i][0]), step_store_done))
        per_step_rewards[t] = [float(rw[0]) for rw in rewards]

        # --- logical terminations flip AFTER the step's stores ---
        if contact_now and contact_rule.terminate:
            contact_logical_done = True
        if goal_now:
            # Drop edges mirrored from the Main.py flip site: the
            # terminal-coincident entry (the `if not episode_done` guard
            # drops the bonus-bearing transition — the section-2.5 trap)
            # and the legacy t<=2 store-guard edge.
            if episode_done or t <= 2:
                goal_store_dropped = True
            goal_logical_done = True

    state = {
        "goal_terminal": goal_logical_done,
        "goal_step": goal_first_step,
        "goal_store_dropped": goal_store_dropped,
        "contact_terminated": contact_logical_done,
        "contact_step": contact_first_step,
        "contact_count": contact_events,
    }
    return stored, state, per_step_rewards


_FAR = [(3.0, 3.0), (3.0, -3.0), (-3.0, 3.0), (-3.0, -3.0)]
_OBSTACLES = _obstacles([(0.0, 0.0), (6.0, 6.0)])
_CONTACT_OFF = ContactRule({}, num_obstacles=2)


def _touching():
    """Robot 0 inside the default contact threshold of obstacle 0."""
    pos = list(_FAR)
    pos[0] = (0.3, 0.0)
    return pos


# ---------------------------------------------------------------------------
# (a) Flag-off byte-identical golden — inline pre-change reference
# ---------------------------------------------------------------------------

class TestFlagOffByteIdentical:
    def _reference_episode(self, episode_len, base_reward=-2.0):
        """Verbatim pre-change Main.py semantics: no detection, no bonus,
        done flag = episode_done, stores on every non-terminal step > 2."""
        stored = []
        per_step_rewards = {}
        for t in range(1, episode_len + 1):
            episode_done = (t == episode_len)
            rewards = [np.array([base_reward], dtype=np.float32)
                       for _ in range(R)]
            for i in range(R):
                if t > 2:
                    if not episode_done:
                        stored.append((t, i, float(rewards[i][0]), episode_done))
            per_step_rewards[t] = [float(rw[0]) for rw in rewards]
        return stored, per_step_rewards

    def test_rewards_transitions_and_rng_identical(self):
        """Off path on a trajectory that WOULD enter the goal region:
        bit-identical rewards and stored transitions vs the pre-change
        reference, and the global np RNG stream is never consumed."""
        rule = GoalBonusRule({})
        cyl_traj = {1: 8.0, 5: 2.0, 8: 1.0}  # enters the would-be threshold

        np.random.seed(4242)
        rng_probe_before = np.random.get_state()
        stored, state, rewards = _run_fake_episode(
            rule, _CONTACT_OFF, cyl_traj, {}, _OBSTACLES, episode_len=12)
        rng_probe_after = np.random.get_state()

        ref_stored, ref_rewards = self._reference_episode(episode_len=12)

        assert stored == ref_stored
        assert rewards == ref_rewards
        assert all(done is False for (_, _, _, done) in stored)
        assert state["goal_terminal"] is False
        assert state["goal_step"] == -1
        # RNG untouched (the rule consumes no randomness on any path).
        assert rng_probe_before[0] == rng_probe_after[0]
        np.testing.assert_array_equal(rng_probe_before[1], rng_probe_after[1])
        assert rng_probe_before[2:] == rng_probe_after[2:]


# ---------------------------------------------------------------------------
# (c) Shared bonus — every robot, exactly once, on the entry step
# ---------------------------------------------------------------------------

class TestSharedBonus:
    def test_bonus_applied_to_all_robots_exactly_once(self):
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0})
        cyl_traj = {1: 8.0, 6: 2.2}  # entry at step 6, stays inside
        stored, state, rewards = _run_fake_episode(
            rule, _CONTACT_OFF, cyl_traj, {}, _OBSTACLES, episode_len=20)

        # Entry step 6: EVERY robot's reward = base + bonus (shared
        # consequence, per-robot convention).
        assert rewards[6] == [pytest.approx(9998.0)] * R
        # Exactly once: no other step carries the bonus (the zombie phase
        # suppresses re-detection after the entry).
        for t, vals in rewards.items():
            if t != 6:
                assert vals == [pytest.approx(-2.0)] * R
        assert state["goal_terminal"] is True
        assert state["goal_step"] == 6

    def test_bonus_not_divided_by_num_robots(self):
        """10000 means 10000 PER ROBOT (per-robot reward convention), total
        team-level consequence R * bonus."""
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0})
        cyl_traj = {1: 2.2, 2: 8.0}
        stored, _, rewards = _run_fake_episode(
            rule, _CONTACT_OFF, cyl_traj, {}, _OBSTACLES, episode_len=5)
        assert sum(rewards[1]) == pytest.approx(R * (-2.0 + 10000.0))


# ---------------------------------------------------------------------------
# (d) Goal terminal — entry transition stored done=True, nothing after
# ---------------------------------------------------------------------------

class TestGoalTerminal:
    def test_entry_step_stored_done_true_then_nothing(self):
        """The done=True store is the bootstrap cut that grounds the bonus
        (q_next[dones]=0 in GSP-RL learning_aids: the entry-step Q-target is
        exactly the bonus-bearing reward)."""
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0})
        cyl_traj = {1: 8.0, 7: 2.2}  # entry at step 7, stays inside
        stored, state, _ = _run_fake_episode(
            rule, _CONTACT_OFF, cyl_traj, {}, _OBSTACLES, episode_len=20)

        # The entry step is the LAST stored step (zombie to physical end).
        assert max(t for (t, _, _, _) in stored) == 7
        entry_rows = [row for row in stored if row[0] == 7]
        assert len(entry_rows) == R
        for (_, _, reward, done) in entry_rows:
            assert done is True                        # bootstrap cut
            assert reward == pytest.approx(9998.0)     # bonus in the terminal
        # Every pre-entry stored transition keeps done=False.
        for (t, _, _, done) in stored:
            if t < 7:
                assert done is False
        assert state["goal_terminal"] is True
        assert state["goal_step"] == 7

    def test_entry_on_argos_terminal_step_is_not_stored(self):
        """Entry coinciding with the ARGoS terminal step: the legacy
        `if not episode_done` guard drops the transition (the documented
        section-2.5 trap the >radius default margin exists to dodge) —
        attrs still record it, and the drop is LOUD: the flip site sets
        goal_store_dropped so the BONUS-VISIBILITY denominator can
        subtract the episode."""
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0})
        cyl_traj = {1: 8.0, 20: 2.2}
        stored, state, _ = _run_fake_episode(
            rule, _CONTACT_OFF, cyl_traj, {}, _OBSTACLES, episode_len=20)
        assert all(t < 20 for (t, _, _, _) in stored)
        assert state["goal_terminal"] is True
        assert state["goal_step"] == 20
        assert state["goal_store_dropped"] is True


# ---------------------------------------------------------------------------
# (e) First-event-wins vs the contact rule — BOTH orders + the tie
# ---------------------------------------------------------------------------

def _both_rules():
    goal = GoalBonusRule({"GOAL_BONUS": 10000.0})
    contact = ContactRule(
        {"OBSTACLE_CONTACT_TERMINATE": True, "OBSTACLE_CONTACT_PENALTY": -10.0},
        num_obstacles=2,
    )
    return goal, contact


class TestFirstEventWins:
    def test_contact_before_goal_keeps_contact_semantics(self):
        """Contact at 5, would-be goal entry at 9: contact owns the episode;
        the goal detector is suppressed in the contact zombie phase — no
        bonus anywhere, no goal terminal."""
        goal, contact = _both_rules()
        robot_traj = {1: _FAR, 5: _touching()}
        cyl_traj = {1: 8.0, 9: 2.2}
        stored, state, rewards = _run_fake_episode(
            goal, contact, cyl_traj, robot_traj, _OBSTACLES, episode_len=20)

        assert state["contact_terminated"] is True
        assert state["contact_step"] == 5
        assert state["goal_terminal"] is False
        assert state["goal_step"] == -1
        # Contact-step semantics intact (shared penalty, done=True, last store).
        assert rewards[5] == [pytest.approx(-12.0)] * R
        assert max(t for (t, _, _, _) in stored) == 5
        assert [row[3] for row in stored if row[0] == 5] == [True] * R
        # The bonus never lands.
        for vals in rewards.values():
            assert all(v < 0 for v in vals)

    def test_goal_before_contact_books_the_bonus_terminal(self):
        """Goal entry at 5, would-be contact at 9: the bonus terminal is
        booked; the contact detector is suppressed in the goal zombie phase
        — no penalty anywhere, no contact termination."""
        goal, contact = _both_rules()
        robot_traj = {1: _FAR, 9: _touching()}
        cyl_traj = {1: 8.0, 5: 2.2}
        stored, state, rewards = _run_fake_episode(
            goal, contact, cyl_traj, robot_traj, _OBSTACLES, episode_len=20)

        assert state["goal_terminal"] is True
        assert state["goal_step"] == 5
        assert state["contact_terminated"] is False
        assert state["contact_step"] == -1
        assert state["contact_count"] == 0
        assert rewards[5] == [pytest.approx(9998.0)] * R
        assert max(t for (t, _, _, _) in stored) == 5
        assert [row[3] for row in stored if row[0] == 5] == [True] * R
        # The penalty never lands.
        assert rewards[9] == [pytest.approx(-2.0)] * R

    def test_same_step_tie_resolves_to_contact(self):
        """Degenerate same-step tie: contact wins (the goal detector is
        gated on `not _contact_now` — the catastrophic event is the
        conservative call)."""
        goal, contact = _both_rules()
        robot_traj = {1: _FAR, 6: _touching()}
        cyl_traj = {1: 8.0, 6: 2.2}
        stored, state, rewards = _run_fake_episode(
            goal, contact, cyl_traj, robot_traj, _OBSTACLES, episode_len=20)

        assert state["contact_terminated"] is True
        assert state["goal_terminal"] is False
        assert rewards[6] == [pytest.approx(-12.0)] * R  # penalty, no bonus


# ---------------------------------------------------------------------------
# (g) h5 per-episode attrs — engaged adds exactly three attrs; off is
#     byte-identical (mirrors the contact-attr golden guard)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed")
class TestH5GoalAttrs:
    @staticmethod
    def _import_logger():
        try:
            from rl_code.src.hdf5_logger import HDF5Logger
        except ImportError:  # pragma: no cover
            from src.hdf5_logger import HDF5Logger  # type: ignore
        return HDF5Logger

    @staticmethod
    def _writerow_kwargs(t):
        return dict(
            rewards=[0.1 * t, 0.2 * t, 0.3 * t, 0.4 * t],
            epsilons=0.5 - 0.01 * t, terminations=(t == 5), losses=0.01 * t,
            force_magnitudes=[0.11 * t] * 4, force_angles=[0.22 * t] * 4,
            average_force_vectors=[0.3 * t, -0.3 * t],
            cyl_x_poses=1.0 * t, cyl_y_poses=-1.0 * t, cyl_angles=2.0 * t,
            gate_stats=0, obstacle_stats=0,
            gsp_rewards=[-0.05 * t] * 4, gsp_headings=[0.07 * t] * 4,
            run_times=0.001 * t,
            robots_x_poses=[0.5 * t] * 4, robots_y_poses=[-0.5 * t] * 4,
            robot_angles=[0.9 * t] * 4, robot_failure=[False] * 4,
            gsp_target=[0.15 * t] * 4,
        )

    def _run_episode(self, path, goal_state):
        HDF5Logger = self._import_logger()
        logger = HDF5Logger(path)
        for t in range(6):
            logger.writerow(**self._writerow_kwargs(t))
        if goal_state is not None:
            logger.record_goal_state(**goal_state)
        logger.write_episode(0)

    def test_engaged_episode_writes_the_three_attrs(self, tmp_path):
        p = str(tmp_path / "goal.h5")
        self._run_episode(p, dict(terminal=True, goal_step=42))
        with h5py.File(p) as f:
            attrs = f["episode_0000"].attrs
            assert bool(attrs["goal_terminal"]) is True
            assert int(attrs["goal_step"]) == 42
            # store_dropped defaults False and is ALWAYS written on engaged
            # episodes (the BONUS-VISIBILITY denominator must be computable).
            assert bool(attrs["goal_store_dropped"]) is False

    def test_store_dropped_attr_written_when_flagged(self, tmp_path):
        p = str(tmp_path / "dropped.h5")
        self._run_episode(p, dict(terminal=True, goal_step=3,
                                  store_dropped=True))
        with h5py.File(p) as f:
            attrs = f["episode_0000"].attrs
            assert bool(attrs["goal_terminal"]) is True
            assert bool(attrs["goal_store_dropped"]) is True

    def test_no_entry_episode_of_engaged_run_records_denominator(self, tmp_path):
        p = str(tmp_path / "noentry.h5")
        self._run_episode(p, dict(terminal=False, goal_step=-1))
        with h5py.File(p) as f:
            attrs = f["episode_0000"].attrs
            assert bool(attrs["goal_terminal"]) is False
            assert int(attrs["goal_step"]) == -1
            assert bool(attrs["goal_store_dropped"]) is False

    def test_off_path_h5_byte_identical_and_attr_free(self, tmp_path):
        """Rule off (record_goal_state never called): no goal attrs, and
        every dataset byte-identical to an un-instrumented baseline."""
        p_base = str(tmp_path / "base.h5")
        p_off = str(tmp_path / "off.h5")
        self._run_episode(p_base, None)
        self._run_episode(p_off, None)
        with h5py.File(p_off) as f:
            attrs = f["episode_0000"].attrs
            assert "goal_terminal" not in attrs
            assert "goal_step" not in attrs
            assert "goal_store_dropped" not in attrs

        def dataset_bytes(path):
            out = {}
            with h5py.File(path) as f:
                grp = f["episode_0000"]
                for key in grp.keys():
                    out[key] = np.asarray(grp[key][()]).tobytes()
            return out
        base, off = dataset_bytes(p_base), dataset_bytes(p_off)
        assert set(base.keys()) == set(off.keys())
        for key in base:
            assert base[key] == off[key]

    def test_goal_state_resets_between_episodes(self, tmp_path):
        """An engaged episode followed by a no-record episode must not leak
        the previous episode's attrs (the _reset contract)."""
        HDF5Logger = self._import_logger()
        p = str(tmp_path / "reset.h5")
        logger = HDF5Logger(p)
        for t in range(3):
            logger.writerow(**self._writerow_kwargs(t))
        logger.record_goal_state(terminal=True, goal_step=2)
        logger.write_episode(0)
        for t in range(3):
            logger.writerow(**self._writerow_kwargs(t))
        logger.write_episode(1)
        with h5py.File(p) as f:
            assert "goal_terminal" in f["episode_0000"].attrs
            assert "goal_terminal" not in f["episode_0001"].attrs


# ---------------------------------------------------------------------------
# Static source contracts on rl_code/Main.py — pins the mini-simulator's
# mirrored wiring to the real file (contact-suite technique)
# ---------------------------------------------------------------------------

class TestMainSourceContract:
    @staticmethod
    def _main_text():
        import pathlib
        return (pathlib.Path(__file__).resolve().parents[2]
                / "rl_code" / "Main.py").read_text()

    def test_startup_lines_present(self):
        """Main.py logs exactly the one line startup_line() returns; the two
        literal ENGAGED/off strings live in src/goal_rule.py."""
        text = self._main_text()
        assert "log.info(_goal_rule.startup_line())" in text
        import pathlib
        rule_src = (pathlib.Path(__file__).resolve().parents[2]
                    / "rl_code" / "src" / "goal_rule.py").read_text()
        assert "GOAL_BONUS rule: ENGAGED" in rule_src
        assert "GOAL_BONUS rule: off" in rule_src

    def test_rule_constructed_from_config(self):
        text = self._main_text()
        assert "_goal_rule = GoalBonusRule(config)" in text

    def test_detection_after_contact_before_reward_loop_and_store(self):
        """Goal detection must run after contact detection (first-event
        ordering, same-step tie -> contact) and before the reward loop, so
        the bonus lands in the stored entry transition."""
        text = self._main_text()
        contact_detect = text.find("_contact_rule.detect(")
        goal_detect = text.find("_goal_rule.detect(")
        bonus = text.find("rewards[i] += _goal_rule.bonus")
        store = text.find("_step_store_done,")
        assert -1 not in (contact_detect, goal_detect, bonus, store)
        assert contact_detect < goal_detect < bonus < store

    def test_done_flag_includes_goal_entry(self):
        text = self._main_text()
        assert "_contact_now and _contact_rule.terminate) or _goal_now" in text

    def test_same_step_tie_gated_on_not_contact_now(self):
        text = self._main_text()
        assert "not _contact_logical_done and not _contact_now" in text

    def test_zombie_phase_suppresses_stores(self):
        text = self._main_text()
        # Immediate store guard, GSP head-store gate, E2E FIFO — all three
        # suppression sites must read the goal zombie flag.
        assert "and not _goal_logical_done" in text
        assert "if _goal_logical_done:" in text
        assert "None if _goal_logical_done else" in text

    def test_contact_detector_suppressed_in_goal_zombie(self):
        """First-event-wins, goal-first order: the contact detection gate
        must also read the goal zombie flag."""
        text = self._main_text()
        assert ("if (_contact_rule.enabled and not _contact_logical_done\n"
                "                            and not _goal_logical_done):") in text

    def test_goal_event_log_line_present(self):
        text = self._main_text()
        assert "GOAL_BONUS event: episode=%d step=%d dist=%.4f " in text

    def test_h5_goal_state_recorded_when_engaged(self):
        text = self._main_text()
        assert "hdf5_writer.record_goal_state(" in text
        assert "if _goal_rule.enabled:" in text

    def test_outcome_gate_unchanged(self):
        """The success counter stays keyed on the PHYSICAL outcome and the
        contact rule only — goal termination is not a failure (physical
        success normally follows in the zombie phase; analyses read the
        goal_terminal attr)."""
        text = self._main_text()
        assert "_ep_outcome_success = reached_goal and not _contact_logical_done" in text

    def test_store_dropped_edges_are_loud_and_recorded(self):
        """All three goal drop edges (terminal-coincident entry, t<=2
        legacy guard, unmatured E2E FIFO) log the same greppable INFO
        prefix, and the flag reaches the h5 attrs through
        record_goal_state."""
        text = self._main_text()
        assert text.count("GOAL_BONUS store dropped:") == 3
        assert "store_dropped=_goal_store_dropped," in text
        assert "_goal_store_dropped = False" in text  # per-episode reset
        # The contact edges are untouched (still exactly two).
        assert text.count("OBSTACLE_CONTACT store dropped:") == 2

    def test_unmatured_fifo_scan_covers_both_rules_and_attributes(self):
        """The episode-end unmatured-done scan runs when EITHER rule is
        engaged and attributes the drop by which logical termination fired
        (first-event-wins makes that unambiguous)."""
        text = self._main_text()
        assert "(_contact_rule.enabled or _goal_rule.enabled)" in text
        assert "_unmatured_done and _goal_logical_done" in text
        scan = text.find("model.unmatured_done_e2e_transitions()")
        reset = text.find("model.reset_gsp_label_buffer()")
        assert -1 not in (scan, reset)
        assert scan < reset

    def test_timesteps_leq_2_edge_flagged_at_the_goal_flip_site(self):
        """The goal flip block carries its own <=2 edge, after the contact
        flip block (text order pins the first-occurrence contract of the
        contact suite)."""
        text = self._main_text()
        contact_flip = text.find("_contact_logical_done = True")
        goal_flip = text.find("_goal_logical_done = True")
        assert -1 not in (contact_flip, goal_flip)
        assert contact_flip < goal_flip


# ---------------------------------------------------------------------------
# (h) K-step FIFO edge — the real Agent FIFO, boundary-exact (the goal twin
# of the contact suite's TestUnmaturedDoneFifoEdge; the scan machinery is
# shared, the goal-entry tx has the identical done=True /
# guard_episode_done=False shape)
# ---------------------------------------------------------------------------

def _make_traj_agent(K):
    """Real Agent with a delayed-label trajectory target and horizon K
    (the E2E FIFO configuration the goal rule threads done=True through)."""
    import os
    import sys
    import yaml
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "rl_code", "src"))
    from agent import Agent  # noqa: E402
    cfg_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "GSP-RL",
        "tests", "test_actor", "config.yml",
    )
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    config["GSP_PREDICTION_TARGET"] = "delta_theta_traj"
    config["GSP_PREDICTION_HORIZON"] = K
    return Agent(
        config=config, network="DDQN", n_agents=4, n_obs=8, n_actions=2,
        options_per_action=3, id=1, min_max_action=1.0, meta_param_size=2,
        gsp=True, recurrent=False, attention=False, neighbors=True,
        gsp_input_size=6, gsp_output_size=1, gsp_min_max_action=1.0,
        gsp_look_back=2, gsp_sequence_length=5,
    )


def _push(agent, e2e_transition, angle=0.0):
    """One Main.py-shaped FIFO push (state/gsp_obs content irrelevant here)."""
    agent.push_pending_gsp_obs(
        [np.zeros(8, dtype=np.float32)] * 4,
        [np.zeros(6, dtype=np.float32)] * 4,
        payload_angle_deg=angle,
        payload_track={"dist2goal": 0.0, "cyl_x": 0.0, "cyl_y": 0.0},
        e2e_transition=e2e_transition,
    )


def _goal_tx():
    """Minimal Main.py _e2e_tx shape for a GOAL-ENTRY step: the goal-aware
    done=True with ARGoS still running (guard_episode_done False) — the
    identical shape the contact rule threads through, so the shared
    unmatured_done_e2e_transitions scan covers both rules."""
    return {"done": True, "guard_episode_done": False, "guard_time_steps": 50}


class TestUnmaturedDoneFifoEdge:
    def test_goal_entry_within_k_of_physical_end_is_flagged(self):
        """Goal entry K-1 pushes before the physical (ARGoS success) end:
        the done=True entry has NOT matured when the episode ends — the scan
        must surface it (Main.py then logs + sets goal_store_dropped before
        the reset deletes it)."""
        K = 5
        agent = _make_traj_agent(K)
        for _ in range(3):  # pre-entry steps (mature + drain normally)
            _push(agent, {"done": False, "guard_episode_done": False,
                          "guard_time_steps": 10})
            agent.pop_matured_gsp_label(None)
        _push(agent, _goal_tx())  # the goal-entry step
        agent.pop_matured_gsp_label(None)
        for _ in range(K - 1):  # zombie pushes, one short of maturity
            _push(agent, None)
            agent.pop_matured_gsp_label(None)

        unmatured = agent.unmatured_done_e2e_transitions()
        assert len(unmatured) == 1
        assert unmatured[0]["done"] is True
        # ... and the reset (what Main.py calls right after) silently
        # deletes it — the reason the scan exists.
        agent.reset_gsp_label_buffer()
        assert agent.unmatured_done_e2e_transitions() == []

    def test_goal_entry_k_or_more_before_end_matures_and_is_not_flagged(self):
        """Goal entry >= K pushes before the end (the margin's whole job):
        the transition matures out of the FIFO (stored by Main.py's maturity
        block, done=True + bonus-bearing reward) — nothing to flag."""
        K = 5
        agent = _make_traj_agent(K)
        _push(agent, _goal_tx())
        matured_done = []
        for _ in range(K + 1):  # full maturation window of zombie pushes
            _push(agent, None)
            m = agent.pop_matured_gsp_label(None)
            if m is not None and m.get("e2e_transition") is not None:
                matured_done.append(m["e2e_transition"])
        assert len(matured_done) == 1
        assert matured_done[0]["done"] is True
        assert agent.unmatured_done_e2e_transitions() == []


# ---------------------------------------------------------------------------
# (h) time_steps<=2 edge — the legacy guard drops the terminal transition
# ---------------------------------------------------------------------------

class TestTimestepsLeq2Edge:
    def test_goal_entry_at_step_2_stores_nothing(self):
        """Goal entry at t=2: the episode logically terminates (attrs count
        it) but the legacy t>2 store guard drops the bonus-bearing done=True
        transition — no behavior change, the edge is only made visible
        (INFO + goal_store_dropped, wiring pinned by
        TestMainSourceContract)."""
        rule = GoalBonusRule({"GOAL_BONUS": 10000.0})
        cyl_traj = {1: 8.0, 2: 2.2}
        stored, state, _ = _run_fake_episode(
            rule, _CONTACT_OFF, cyl_traj, {}, _OBSTACLES, episode_len=20)
        assert state["goal_terminal"] is True
        assert state["goal_step"] == 2
        # NOTHING was stored: the entry step fails the t>2 guard and the
        # zombie phase suppresses every later store.
        assert stored == []

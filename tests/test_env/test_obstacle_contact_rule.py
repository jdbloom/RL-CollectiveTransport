"""OBSTACLE-CONTACT rule tests (operator-directed 2026-07-10).

Contract under test (flag-gated, default OFF = byte-identical):

(a) Flags off/absent = byte-identical legacy behavior: identical reward
    stream, identical stored transitions (values AND done flags), no RNG
    consumption — proven against an inline pre-change reference loop on a
    fixed fake-stats episode (test_golden_t7 technique; Main.py itself is not
    importable in a unit test, so the wiring is mirrored by a mini-simulator
    and pinned to Main.py by static source contracts, the accepted pattern
    from tests/test_agent/test_batched_actor_forward.py).
(b) Contact detection geometry: min robot-obstacle CENTER distance vs the
    OBSTACLE_CONTACT_DIST threshold (default derived from the argos geometry:
    obstacle_radius 0.5 + footbot_radius 0.085036758 + eps 0.01).
(c) OBSTACLE_CONTACT_PENALTY is added to EVERY robot's reward exactly once on
    the contact step (shared consequence; per-robot convention — never
    divided by num_robots).
(d) OBSTACLE_CONTACT_TERMINATE stores the contact-step transition with
    done=True and stores nothing afterwards (logical termination).
(e) Penalty-only (TERMINATE=false, PENALTY<0) never terminates.
(f) Startup log line: exactly one of ENGAGED/off, keyed on the effective gate.
(g) h5 per-episode attrs (contact_terminated/contact_step/contact_count/
    contact_store_dropped) written when engaged, absent (byte-identical h5)
    when off.
(h) contact_store_dropped edges (2026-07-10 review findings): a terminating
    contact within K steps of the physical episode end leaves its done=True
    E2E transition unmatured in the delayed FIFO (flagged by
    Agent.unmatured_done_e2e_transitions before the buffer reset deletes it);
    a terminating contact at time_steps<=2 is dropped by the legacy store
    guard. Both are LOUD (INFO log + attr), neither changes store behavior.
"""

import numpy as np
import pytest

try:
    from rl_code.src.contact_rule import (
        ContactRule, FOOTBOT_RADIUS_M, DEFAULT_OBSTACLE_RADIUS_M, CONTACT_EPS_M,
    )
except ImportError:  # pragma: no cover - path-dependent import
    from src.contact_rule import (  # type: ignore
        ContactRule, FOOTBOT_RADIUS_M, DEFAULT_OBSTACLE_RADIUS_M, CONTACT_EPS_M,
    )

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
        rule = ContactRule({}, num_obstacles=4)
        assert rule.enabled is False
        assert rule.terminate is False
        assert rule.penalty == 0.0

    def test_flags_absent_and_explicit_off_agree(self):
        a = ContactRule({}, num_obstacles=4)
        b = ContactRule(
            {"OBSTACLE_CONTACT_TERMINATE": False, "OBSTACLE_CONTACT_PENALTY": 0.0},
            num_obstacles=4,
        )
        assert (a.enabled, a.terminate, a.penalty, a.contact_dist) == \
               (b.enabled, b.terminate, b.penalty, b.contact_dist)

    def test_terminate_only_engages(self):
        rule = ContactRule({"OBSTACLE_CONTACT_TERMINATE": True}, num_obstacles=4)
        assert rule.enabled is True and rule.terminate is True
        assert rule.penalty == 0.0

    def test_penalty_only_engages(self):
        rule = ContactRule({"OBSTACLE_CONTACT_PENALTY": -10.0}, num_obstacles=4)
        assert rule.enabled is True and rule.terminate is False
        assert rule.penalty == -10.0

    def test_default_dist_derived_from_argos_geometry(self):
        """Default threshold = obstacle_radius + footbot_radius + eps, from the
        documented argos sources (generate_argos.py 0.5 default,
        collectiveRlTransport.cpp FOOTBOT_RADIUS)."""
        rule = ContactRule({"OBSTACLE_CONTACT_TERMINATE": True}, num_obstacles=4)
        expected = DEFAULT_OBSTACLE_RADIUS_M + FOOTBOT_RADIUS_M + CONTACT_EPS_M
        assert rule.contact_dist == pytest.approx(expected)
        assert rule.contact_dist == pytest.approx(0.595036758)

    def test_dist_respects_obstacle_radius_override(self):
        """Per-cell OBSTACLE_RADIUS (scale-geom flag) shifts the derived default."""
        rule = ContactRule(
            {"OBSTACLE_CONTACT_TERMINATE": True, "OBSTACLE_RADIUS": 0.61},
            num_obstacles=4,
        )
        assert rule.contact_dist == pytest.approx(0.61 + FOOTBOT_RADIUS_M + CONTACT_EPS_M)

    def test_explicit_dist_wins(self):
        rule = ContactRule(
            {"OBSTACLE_CONTACT_TERMINATE": True, "OBSTACLE_CONTACT_DIST": 0.7},
            num_obstacles=4,
        )
        assert rule.contact_dist == 0.7

    def test_engaged_without_obstacles_raises(self):
        """Fail loud: an engaged rule in an obstacle-free env can never fire."""
        with pytest.raises(ValueError, match="num_obstacles=0"):
            ContactRule({"OBSTACLE_CONTACT_TERMINATE": True}, num_obstacles=0)

    def test_engaged_with_nonpositive_dist_raises(self):
        with pytest.raises(ValueError, match="OBSTACLE_CONTACT_DIST"):
            ContactRule(
                {"OBSTACLE_CONTACT_PENALTY": -10.0, "OBSTACLE_CONTACT_DIST": 0.0},
                num_obstacles=4,
            )

    def test_off_without_obstacles_is_fine(self):
        """Default-off cells in obstacle-free envs must construct (legacy runs)."""
        rule = ContactRule({}, num_obstacles=0)
        assert rule.enabled is False


# ---------------------------------------------------------------------------
# (f) Startup log line — both states
# ---------------------------------------------------------------------------

class TestStartupLine:
    def test_engaged_line(self):
        rule = ContactRule(
            {"OBSTACLE_CONTACT_TERMINATE": True, "OBSTACLE_CONTACT_PENALTY": -10.0},
            num_obstacles=4,
        )
        line = rule.startup_line()
        assert line.startswith("OBSTACLE_CONTACT rule: ENGAGED")
        assert "terminate=True" in line
        assert "penalty=-10.0" in line
        assert "dist=" in line

    def test_off_line(self):
        rule = ContactRule({}, num_obstacles=4)
        assert rule.startup_line() == "OBSTACLE_CONTACT rule: off"


# ---------------------------------------------------------------------------
# (b) Contact detection geometry
# ---------------------------------------------------------------------------

class TestDetectGeometry:
    def _rule(self, **over):
        cfg = {"OBSTACLE_CONTACT_TERMINATE": True}
        cfg.update(over)
        return ContactRule(cfg, num_obstacles=2)

    def test_below_threshold_is_contact(self):
        rule = self._rule()
        robots = _robot_stats([(0.0, 0.0), (3.0, 3.0), (-3.0, 3.0), (-3.0, -3.0)])
        obstacles = _obstacles([(0.59, 0.0), (5.0, 5.0)])  # 0.59 < 0.595036758
        hit, ri, oi, d = rule.detect(robots, obstacles)
        assert hit is True
        assert (ri, oi) == (0, 0)
        assert d == pytest.approx(0.59)

    def test_above_threshold_is_not_contact(self):
        rule = self._rule()
        robots = _robot_stats([(0.0, 0.0), (3.0, 3.0), (-3.0, 3.0), (-3.0, -3.0)])
        obstacles = _obstacles([(0.60, 0.0), (5.0, 5.0)])  # 0.60 > 0.595036758
        hit, _, _, d = rule.detect(robots, obstacles)
        assert hit is False
        assert d == pytest.approx(0.60)

    def test_exactly_at_threshold_is_not_contact(self):
        """Strict `<` — the boundary itself is not a contact."""
        rule = self._rule(OBSTACLE_CONTACT_DIST=0.5)
        robots = _robot_stats([(0.5, 0.0), (3.0, 3.0), (-3.0, 3.0), (-3.0, -3.0)])
        obstacles = _obstacles([(0.0, 0.0), (5.0, 5.0)])
        hit, _, _, d = rule.detect(robots, obstacles)
        assert hit is False and d == pytest.approx(0.5)

    def test_min_over_all_pairs(self):
        """The closest (robot, obstacle) pair is reported, not the first."""
        rule = self._rule()
        robots = _robot_stats([(9.0, 9.0), (9.0, -9.0), (-9.0, 9.0), (2.0, 2.0)])
        obstacles = _obstacles([(-5.0, -5.0), (2.0, 2.3)])  # robot 3 <-> obstacle 1: 0.3
        hit, ri, oi, d = rule.detect(robots, obstacles)
        assert hit is True
        assert (ri, oi) == (3, 1)
        assert d == pytest.approx(0.3)

    def test_diagonal_distance(self):
        """Euclidean, not per-axis: (0.3, 0.4) offset = 0.5 center distance."""
        rule = self._rule(OBSTACLE_CONTACT_DIST=0.51)
        robots = _robot_stats([(0.3, 0.4), (9.0, 9.0), (-9.0, 9.0), (9.0, -9.0)])
        obstacles = _obstacles([(0.0, 0.0), (5.0, 5.0)])
        hit, _, _, d = rule.detect(robots, obstacles)
        assert hit is True and d == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Mini-simulator: the Main.py wiring, mirrored step-for-step.
#
# Main.py cannot be imported in a unit test (module-level argparse + ZMQ), so
# the acceptance-level semantics — detection gate -> shared penalty ->
# done-flag override -> zombie-phase store suppression -> episode attrs —
# are exercised on a fake-stats episode here, and the presence/ordering of
# the real wiring in Main.py is pinned by TestMainSourceContract below.
# ---------------------------------------------------------------------------

def _run_fake_episode(rule, robot_traj, obstacles, episode_len, base_reward=-2.0):
    """Mirror of the Main.py step loop for the contact rule's touch points.

    robot_traj: dict {step (1-based) -> list of R (x, y)} — steps not present
    reuse the previous positions (static robots between waypoints).
    Returns (stored, contact_state, per_step_rewards).
      stored: list of (step, robot, reward, done) — mirrors the
              store_agent_transition guard chain (train_mode, no failures,
              learning scheme active).
      contact_state: dict with terminated/contact_step/contact_count (the
              h5-attr values Main.py records when the rule is engaged).
      per_step_rewards: {step: [reward per robot]} AFTER all shaping.
    """
    stored = []
    per_step_rewards = {}
    contact_logical_done = False
    contact_first_step = -1
    contact_events = 0
    positions = robot_traj[min(robot_traj)]

    for t in range(1, episode_len + 1):
        episode_done = (t == episode_len)
        positions = robot_traj.get(t, positions)
        robot_stats = _robot_stats(positions)

        # --- detection (Main.py: after parse_obstacle_stats) ---
        contact_now = False
        if rule.enabled and not contact_logical_done:
            hit, _ri, _oi, _d = rule.detect(robot_stats, obstacles)
            if hit:
                contact_now = True
                contact_events += 1
                if contact_first_step < 0:
                    contact_first_step = t
        step_store_done = bool(episode_done) or (contact_now and rule.terminate)

        # --- reward loop (Main.py: rewards[i] += -prox, then the penalty) ---
        rewards = [np.array([base_reward], dtype=np.float32) for _ in range(R)]
        for i in range(R):
            if contact_now:
                rewards[i] += rule.penalty
            # --- immediate store site (guard chain mirrored) ---
            if t > 2 and not contact_logical_done:
                if not episode_done:
                    stored.append((t, i, float(rewards[i][0]), step_store_done))
        per_step_rewards[t] = [float(rw[0]) for rw in rewards]

        # --- logical termination flips AFTER the contact step's stores ---
        if contact_now and rule.terminate:
            contact_logical_done = True

    contact_state = {
        "contact_terminated": contact_logical_done,
        "contact_step": contact_first_step,
        "contact_count": contact_events,
    }
    return stored, contact_state, per_step_rewards


_FAR = [(3.0, 3.0), (3.0, -3.0), (-3.0, 3.0), (-3.0, -3.0)]
_OBSTACLES = _obstacles([(0.0, 0.0), (6.0, 6.0)])


def _touching(step_positions=None):
    """Robot 0 inside the default threshold of obstacle 0; others far."""
    pos = list(_FAR)
    pos[0] = (0.3, 0.0)
    return pos


# ---------------------------------------------------------------------------
# (a) Flag-off byte-identical golden — inline pre-change reference
# ---------------------------------------------------------------------------

class TestFlagOffByteIdentical:
    def _reference_episode(self, robot_traj, episode_len, base_reward=-2.0):
        """Verbatim pre-change Main.py semantics: no detection, no penalty,
        done flag = episode_done, stores on every non-terminal step > 2."""
        stored = []
        per_step_rewards = {}
        for t in range(1, episode_len + 1):
            episode_done = (t == episode_len)
            rewards = [np.array([base_reward], dtype=np.float32) for _ in range(R)]
            for i in range(R):
                if t > 2:
                    if not episode_done:
                        stored.append((t, i, float(rewards[i][0]), episode_done))
            per_step_rewards[t] = [float(rw[0]) for rw in rewards]
        return stored, per_step_rewards

    def test_rewards_transitions_and_rng_identical(self):
        """Off path on a trajectory that WOULD contact: bit-identical rewards
        and stored transitions vs the pre-change reference, and the global
        np RNG stream is never consumed."""
        rule = ContactRule({}, num_obstacles=2)
        traj = {1: _FAR, 5: _touching(), 8: _FAR}

        np.random.seed(4242)
        rng_probe_before = np.random.get_state()
        stored, contact_state, rewards = _run_fake_episode(
            rule, traj, _OBSTACLES, episode_len=12)
        rng_probe_after = np.random.get_state()

        ref_stored, ref_rewards = self._reference_episode(traj, episode_len=12)

        assert stored == ref_stored
        assert rewards == ref_rewards
        # Not a single done=True transition and no contact bookkeeping.
        assert all(done is False for (_, _, _, done) in stored)
        assert contact_state == {
            "contact_terminated": False, "contact_step": -1, "contact_count": 0,
        }
        # RNG untouched (the rule consumes no randomness on any path).
        assert rng_probe_before[0] == rng_probe_after[0]
        np.testing.assert_array_equal(rng_probe_before[1], rng_probe_after[1])
        assert rng_probe_before[2:] == rng_probe_after[2:]


# ---------------------------------------------------------------------------
# (c) Shared penalty — every robot, exactly once, on the contact step
# ---------------------------------------------------------------------------

class TestSharedPenalty:
    def test_penalty_applied_to_all_robots_exactly_once(self):
        rule = ContactRule(
            {"OBSTACLE_CONTACT_TERMINATE": True, "OBSTACLE_CONTACT_PENALTY": -10.0},
            num_obstacles=2,
        )
        traj = {1: _FAR, 6: _touching()}
        stored, contact_state, rewards = _run_fake_episode(
            rule, traj, _OBSTACLES, episode_len=20)

        # Contact step 6: EVERY robot's reward = base + penalty — the far-side
        # robots (1..3, nowhere near the obstacle) feel the full consequence.
        assert rewards[6] == [pytest.approx(-12.0)] * R
        # Exactly once: no other step carries the penalty (zombie phase
        # suppresses re-detection after the terminating contact).
        for t, vals in rewards.items():
            if t != 6:
                assert vals == [pytest.approx(-2.0)] * R
        assert contact_state["contact_count"] == 1
        assert contact_state["contact_step"] == 6

    def test_penalty_not_divided_by_num_robots(self):
        """-10 means -10 PER ROBOT (per-robot reward convention), total
        team-level consequence R * penalty."""
        rule = ContactRule({"OBSTACLE_CONTACT_PENALTY": -10.0}, num_obstacles=2)
        traj = {1: _touching(), 2: _FAR}
        _, _, rewards = _run_fake_episode(rule, traj, _OBSTACLES, episode_len=5)
        assert sum(rewards[1]) == pytest.approx(R * (-2.0 - 10.0))


# ---------------------------------------------------------------------------
# (d) Termination — final transition stored done=True, nothing after
# ---------------------------------------------------------------------------

class TestTerminate:
    def test_contact_step_stored_done_true_then_nothing(self):
        rule = ContactRule(
            {"OBSTACLE_CONTACT_TERMINATE": True, "OBSTACLE_CONTACT_PENALTY": -10.0},
            num_obstacles=2,
        )
        traj = {1: _FAR, 7: _touching()}  # stays touching from step 7 on
        stored, contact_state, _ = _run_fake_episode(
            rule, traj, _OBSTACLES, episode_len=20)

        # The contact step is the LAST stored step.
        assert max(t for (t, _, _, _) in stored) == 7
        contact_rows = [row for row in stored if row[0] == 7]
        assert len(contact_rows) == R
        for (_, _, reward, done) in contact_rows:
            assert done is True                     # bootstrap cut
            assert reward == pytest.approx(-12.0)   # penalty in the terminal reward
        # Every pre-contact stored transition keeps done=False.
        for (t, _, _, done) in stored:
            if t < 7:
                assert done is False
        assert contact_state["contact_terminated"] is True
        assert contact_state["contact_step"] == 7

    def test_terminate_only_composes_without_penalty(self):
        """TERMINATE=true, PENALTY=0: done=True cut with unmodified reward."""
        rule = ContactRule({"OBSTACLE_CONTACT_TERMINATE": True}, num_obstacles=2)
        traj = {1: _FAR, 7: _touching()}
        stored, contact_state, rewards = _run_fake_episode(
            rule, traj, _OBSTACLES, episode_len=20)
        assert rewards[7] == [pytest.approx(-2.0)] * R
        assert [row[3] for row in stored if row[0] == 7] == [True] * R
        assert max(t for (t, _, _, _) in stored) == 7
        assert contact_state["contact_terminated"] is True

    def test_contact_on_argos_terminal_step_is_not_stored(self):
        """Contact coinciding with the ARGoS terminal step: the legacy
        `if not episode_done` guard drops the transition (same as every
        terminal transition today) — documented edge, attrs still record it."""
        rule = ContactRule(
            {"OBSTACLE_CONTACT_TERMINATE": True, "OBSTACLE_CONTACT_PENALTY": -10.0},
            num_obstacles=2,
        )
        traj = {1: _FAR, 20: _touching()}
        stored, contact_state, _ = _run_fake_episode(
            rule, traj, _OBSTACLES, episode_len=20)
        assert all(t < 20 for (t, _, _, _) in stored)
        assert contact_state["contact_terminated"] is True
        assert contact_state["contact_step"] == 20


# ---------------------------------------------------------------------------
# (e) Penalty-only — dense re-application, never terminates
# ---------------------------------------------------------------------------

class TestPenaltyOnly:
    def test_penalty_only_does_not_terminate(self):
        rule = ContactRule({"OBSTACLE_CONTACT_PENALTY": -10.0}, num_obstacles=2)
        # In contact for steps 5..7, clear afterwards.
        traj = {1: _FAR, 5: _touching(), 8: _FAR}
        stored, contact_state, rewards = _run_fake_episode(
            rule, traj, _OBSTACLES, episode_len=15)

        # Episode runs to its natural end; stores continue after the contact.
        assert max(t for (t, _, _, _) in stored) == 14  # last non-terminal step
        # No done=True anywhere (no ARGoS terminal store, no contact cut).
        assert all(done is False for (_, _, _, done) in stored)
        # Dense re-application: each in-contact step penalized, all robots.
        for t in (5, 6, 7):
            assert rewards[t] == [pytest.approx(-12.0)] * R
        for t in (4, 8, 9):
            assert rewards[t] == [pytest.approx(-2.0)] * R
        assert contact_state["contact_terminated"] is False
        assert contact_state["contact_step"] == 5
        assert contact_state["contact_count"] == 3


# ---------------------------------------------------------------------------
# (g) h5 per-episode attrs — engaged adds exactly three attrs; off is
#     byte-identical (mirrors the M4 additive-only golden guard)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed")
class TestH5ContactAttrs:
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

    def _run_episode(self, path, contact_state):
        HDF5Logger = self._import_logger()
        logger = HDF5Logger(path)
        for t in range(6):
            logger.writerow(**self._writerow_kwargs(t))
        if contact_state is not None:
            logger.record_contact_state(**contact_state)
        logger.write_episode(0)

    def test_engaged_episode_writes_the_four_attrs(self, tmp_path):
        p = str(tmp_path / "contact.h5")
        self._run_episode(p, dict(terminated=True, contact_step=42, contact_count=1))
        with h5py.File(p) as f:
            attrs = f["episode_0000"].attrs
            assert bool(attrs["contact_terminated"]) is True
            assert int(attrs["contact_step"]) == 42
            assert int(attrs["contact_count"]) == 1
            # store_dropped defaults False and is ALWAYS written on engaged
            # episodes (the denominator must be computable, not inferred).
            assert bool(attrs["contact_store_dropped"]) is False

    def test_store_dropped_attr_written_when_flagged(self, tmp_path):
        p = str(tmp_path / "dropped.h5")
        self._run_episode(p, dict(terminated=True, contact_step=3,
                                  contact_count=1, store_dropped=True))
        with h5py.File(p) as f:
            attrs = f["episode_0000"].attrs
            assert bool(attrs["contact_terminated"]) is True
            assert bool(attrs["contact_store_dropped"]) is True

    def test_no_contact_episode_of_engaged_run_records_denominator(self, tmp_path):
        p = str(tmp_path / "nocontact.h5")
        self._run_episode(p, dict(terminated=False, contact_step=-1, contact_count=0))
        with h5py.File(p) as f:
            attrs = f["episode_0000"].attrs
            assert bool(attrs["contact_terminated"]) is False
            assert int(attrs["contact_step"]) == -1
            assert int(attrs["contact_count"]) == 0
            assert bool(attrs["contact_store_dropped"]) is False

    def test_off_path_h5_byte_identical_and_attr_free(self, tmp_path):
        """Rule off (record_contact_state never called): no contact attrs, and
        every dataset byte-identical to an un-instrumented baseline episode."""
        p_base = str(tmp_path / "base.h5")
        p_off = str(tmp_path / "off.h5")
        self._run_episode(p_base, None)
        self._run_episode(p_off, None)
        with h5py.File(p_off) as f:
            attrs = f["episode_0000"].attrs
            assert "contact_terminated" not in attrs
            assert "contact_step" not in attrs
            assert "contact_count" not in attrs
            assert "contact_store_dropped" not in attrs
        # Dataset bytes identical (additive-only guard, M4 pattern).
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

    def test_contact_state_resets_between_episodes(self, tmp_path):
        """An engaged episode followed by a no-record episode must not leak
        the previous episode's attrs (the _reset contract)."""
        HDF5Logger = self._import_logger()
        p = str(tmp_path / "reset.h5")
        logger = HDF5Logger(p)
        for t in range(3):
            logger.writerow(**self._writerow_kwargs(t))
        logger.record_contact_state(terminated=True, contact_step=2, contact_count=1)
        logger.write_episode(0)
        for t in range(3):
            logger.writerow(**self._writerow_kwargs(t))
        logger.write_episode(1)
        with h5py.File(p) as f:
            assert "contact_terminated" in f["episode_0000"].attrs
            assert "contact_terminated" not in f["episode_0001"].attrs


# ---------------------------------------------------------------------------
# Static source contracts on rl_code/Main.py (test_batched_actor_forward
# technique — pins the mini-simulator's mirrored wiring to the real file)
# ---------------------------------------------------------------------------

class TestMainSourceContract:
    @staticmethod
    def _main_text():
        import pathlib
        return (pathlib.Path(__file__).resolve().parents[2]
                / "rl_code" / "Main.py").read_text()

    def test_startup_lines_present(self):
        """Main.py logs exactly the one line startup_line() returns; the two
        literal ENGAGED/off strings live in src/contact_rule.py (asserted
        against the runtime values in TestStartupLine above)."""
        text = self._main_text()
        assert "log.info(_contact_rule.startup_line())" in text
        import pathlib
        rule_src = (pathlib.Path(__file__).resolve().parents[2]
                    / "rl_code" / "src" / "contact_rule.py").read_text()
        assert "OBSTACLE_CONTACT rule: ENGAGED" in rule_src
        assert "OBSTACLE_CONTACT rule: off" in rule_src

    def test_rule_constructed_from_config_and_handshake_obstacle_count(self):
        text = self._main_text()
        assert "ContactRule(config, Utility.params['num_obstacles'])" in text

    def test_detection_precedes_reward_loop_and_store(self):
        """Detection must run after the post-step stats parse and before the
        reward loop, so the penalty lands in the stored contact transition."""
        text = self._main_text()
        detect = text.find("_contact_rule.detect(")
        penalty = text.find("rewards[i] += _contact_rule.penalty")
        store = text.find("_step_store_done,")
        assert -1 not in (detect, penalty, store)
        assert detect < penalty < store

    def test_store_sites_use_step_store_done(self):
        """Both store_agent_transition call sites must pass the contact-aware
        done flag (== episode_done when the rule is off)."""
        text = self._main_text()
        # 2 store_agent_transition call sites + the E2E delayed-FIFO tx dict
        # + the [ACTCOND] one-iteration delayed-store pending tuple (review
        # FIX 2), which holds the flag at push time and stores it verbatim at
        # the act-site / terminal flush.
        assert text.count("_step_store_done,") == 4
        assert "'done': _step_store_done," in text
        assert "_step_store_done = bool(episode_done) or (" in text

    def test_zombie_phase_suppresses_stores(self):
        text = self._main_text()
        assert "and not _contact_logical_done):" in text  # immediate store guard
        assert "if _contact_logical_done:" in text        # GSP head-store gate
        assert "None if _contact_logical_done else _e2e_tx" in text  # E2E FIFO

    def test_contact_event_log_line_present(self):
        text = self._main_text()
        assert "OBSTACLE_CONTACT event: episode=%d step=%d " in text

    def test_h5_contact_state_recorded_when_engaged(self):
        text = self._main_text()
        assert "hdf5_writer.record_contact_state(" in text
        assert "if _contact_rule.enabled:" in text

    def test_outcome_gate_counts_contact_termination_as_failure(self):
        text = self._main_text()
        assert "_ep_outcome_success = reached_goal and not _contact_logical_done" in text

    # --- (h) contact_store_dropped wiring (2026-07-10 review findings) ----

    def test_store_dropped_edges_are_loud_and_recorded(self):
        """Both drop edges log the same greppable INFO prefix, and the flag
        reaches the h5 attrs through record_contact_state."""
        text = self._main_text()
        # Two INFO sites: the time_steps<=2 legacy-guard edge and the
        # unmatured-FIFO edge at episode end.
        assert text.count("OBSTACLE_CONTACT store dropped:") == 2
        assert "store_dropped=_contact_store_dropped," in text
        assert "_contact_store_dropped = False" in text  # per-episode reset

    def test_fifo_flush_check_runs_before_buffer_reset(self):
        """The unmatured-done scan must run BEFORE reset_gsp_label_buffer
        deletes the evidence."""
        text = self._main_text()
        scan = text.find("model.unmatured_done_e2e_transitions()")
        reset = text.find("model.reset_gsp_label_buffer()")
        assert -1 not in (scan, reset)
        assert scan < reset

    def test_timesteps_leq_2_edge_flagged_at_the_flip_site(self):
        """The <=2 edge is detected where the logical termination flips (so
        it covers BOTH the immediate-store guard and the E2E maturity
        guard_time_steps>2 drop of the same transition)."""
        text = self._main_text()
        flip = text.find("_contact_logical_done = True")
        edge = text.find("if time_steps <= 2:")
        assert -1 not in (flip, edge)
        assert edge < flip  # inside the same terminate block, before the flip


# ---------------------------------------------------------------------------
# (h) K-step FIFO edge — the real Agent FIFO, boundary-exact
# ---------------------------------------------------------------------------

def _make_traj_agent(K):
    """Real Agent with a delayed-label trajectory target and horizon K
    (the E2E FIFO configuration the contact rule threads done=True through)."""
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


def _contact_tx():
    """Minimal Main.py _e2e_tx shape for a TERMINATING-contact step: the
    contact-aware done=True with ARGoS still running (guard_episode_done
    False) — exactly what distinguishes it from a legacy terminal push."""
    return {"done": True, "guard_episode_done": False, "guard_time_steps": 50}


class TestUnmaturedDoneFifoEdge:
    def test_contact_within_k_of_physical_end_is_flagged(self):
        """Contact at K-1 pushes before the physical end: the done=True entry
        has NOT matured when the episode ends — the scan must surface it
        (Main.py then logs + sets contact_store_dropped before the reset)."""
        K = 5
        agent = _make_traj_agent(K)
        for _ in range(3):  # some pre-contact steps (mature + drain normally)
            _push(agent, {"done": False, "guard_episode_done": False,
                          "guard_time_steps": 10})
            agent.pop_matured_gsp_label(None)
        _push(agent, _contact_tx())  # the terminating contact step
        agent.pop_matured_gsp_label(None)
        for _ in range(K - 1):  # zombie pushes, one short of maturity
            _push(agent, None)
            agent.pop_matured_gsp_label(None)

        unmatured = agent.unmatured_done_e2e_transitions()
        assert len(unmatured) == 1
        assert unmatured[0]["done"] is True
        # ... and the reset (what Main.py calls right after) silently deletes
        # it — the reason the scan exists.
        agent.reset_gsp_label_buffer()
        assert agent.unmatured_done_e2e_transitions() == []

    def test_contact_k_or_more_before_end_matures_and_is_not_flagged(self):
        """Contact ≥ K pushes before the end: the transition matures out of
        the FIFO (stored by Main.py's maturity block) — nothing to flag."""
        K = 5
        agent = _make_traj_agent(K)
        _push(agent, _contact_tx())
        matured_done = []
        for _ in range(K + 1):  # full maturation window of zombie pushes
            _push(agent, None)
            m = agent.pop_matured_gsp_label(None)
            if m is not None and m.get("e2e_transition") is not None:
                matured_done.append(m["e2e_transition"])
        assert len(matured_done) == 1  # it matured (Main.py stores it here)
        assert matured_done[0]["done"] is True
        assert agent.unmatured_done_e2e_transitions() == []

    def test_argos_terminal_push_is_not_counted(self):
        """A legacy ARGoS-terminal push (done=True via _step_store_done but
        guard_episode_done=True) is the never-stored legacy terminal, not a
        contact drop — must not be flagged."""
        agent = _make_traj_agent(5)
        _push(agent, {"done": True, "guard_episode_done": True,
                      "guard_time_steps": 4500})
        assert agent.unmatured_done_e2e_transitions() == []

    def test_no_op_for_non_delayed_targets(self):
        """delta_theta (immediate-label) agents never populate the FIFO; the
        scan is an empty no-op."""
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
        agent = Agent(
            config=config, network="DDQN", n_agents=4, n_obs=8, n_actions=2,
            options_per_action=3, id=1, min_max_action=1.0, meta_param_size=2,
            gsp=True, recurrent=False, attention=False, neighbors=True,
            gsp_input_size=6, gsp_output_size=1, gsp_min_max_action=1.0,
            gsp_look_back=2, gsp_sequence_length=5,
        )
        _push(agent, _contact_tx())  # no-op push (not a delayed-label target)
        assert agent.unmatured_done_e2e_transitions() == []


# ---------------------------------------------------------------------------
# (h) time_steps<=2 edge — the legacy guard drops the terminal transition
# ---------------------------------------------------------------------------

class TestTimestepsLeq2Edge:
    def test_contact_at_step_2_stores_nothing(self):
        """Terminating contact at t=2: the episode logically terminates
        (attrs count it) but the legacy t>2 store guard drops the
        penalty-bearing done=True transition — no behavior change, the edge
        is only made visible (INFO + contact_store_dropped, wiring pinned by
        TestMainSourceContract.test_timesteps_leq_2_edge_flagged_at_the_flip_site)."""
        rule = ContactRule(
            {"OBSTACLE_CONTACT_TERMINATE": True, "OBSTACLE_CONTACT_PENALTY": -10.0},
            num_obstacles=2,
        )
        traj = {1: _FAR, 2: _touching()}
        stored, contact_state, _ = _run_fake_episode(
            rule, traj, _OBSTACLES, episode_len=20)
        # Logically terminated at step 2 ...
        assert contact_state["contact_terminated"] is True
        assert contact_state["contact_step"] == 2
        # ... but NOTHING was stored: the contact step fails the t>2 guard
        # and the zombie phase suppresses every later store.
        assert stored == []

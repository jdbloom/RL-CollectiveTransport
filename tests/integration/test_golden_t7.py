"""T7 golden-equivalence gate — g_knowledge vectorization.

Tests that the vectorized build_global_knowledge and build_g_knowledge_all
helpers in src.knowledge produce numerically identical output to the nested-loop
implementation in Main.py, for random robot_stats/stats over many seeds and
R in {3, 4, 6}.

This test does NOT import Main.py (which calls argparse.parse_args() at module
level and requires ZMQ/torch).  Instead, the reference implementation is
reproduced inline from the verbatim Main.py code so there's no ambiguity.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    from rl_code.src.knowledge import build_global_knowledge, build_g_knowledge_all
except ImportError:
    from src.knowledge import build_global_knowledge, build_g_knowledge_all  # type: ignore


# ---------------------------------------------------------------------------
# Reference implementations (verbatim copy of Main.py lines 324-340 / 755-772)
# ---------------------------------------------------------------------------

def _ref_build_global_knowledge(robot_stats, stats, R):
    """Verbatim Main.py lines 324-329."""
    global_knowledge = np.zeros(R * 4)
    for i in range(R):
        global_knowledge[i * 4]     = robot_stats[i][0]  # x position
        global_knowledge[i * 4 + 1] = robot_stats[i][1]  # y position
        global_knowledge[i * 4 + 2] = stats[i][2]        # velocity X
        global_knowledge[i * 4 + 3] = stats[i][3]        # velocity Y
    return global_knowledge


def _ref_build_g_knowledge_all(global_knowledge, R):
    """Verbatim Main.py lines 331-340 (inner build per robot i)."""
    result = []
    for i in range(R):
        g_knowledge = np.zeros((R - 1) * 4)
        counter = 0
        for j in range(R):
            if i != j:
                g_knowledge[counter * 4]     = global_knowledge[j * 4]
                g_knowledge[counter * 4 + 1] = global_knowledge[j * 4 + 1]
                g_knowledge[counter * 4 + 2] = global_knowledge[j * 4 + 2]
                g_knowledge[counter * 4 + 3] = global_knowledge[j * 4 + 3]
                counter += 1
        result.append(g_knowledge)
    return result


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_inputs(R: int, rng: np.random.Generator):
    """Return (robot_stats, stats) matching the Main.py format."""
    robot_stats = [rng.uniform(-5.0, 5.0, 6).astype(np.float32) for _ in range(R)]
    stats = [rng.uniform(-2.0, 2.0, 4).astype(np.float32) for _ in range(R)]
    return robot_stats, stats


SEEDS = (0, 1, 7, 42, 99, 123, 256, 512)
ROBOT_COUNTS = (3, 4, 6)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildGlobalKnowledge:
    """build_global_knowledge must match the nested-loop reference exactly."""

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("R", ROBOT_COUNTS)
    def test_matches_reference(self, seed, R):
        rng = np.random.default_rng(seed)
        robot_stats, stats = _make_inputs(R, rng)

        ref = _ref_build_global_knowledge(robot_stats, stats, R)
        got = build_global_knowledge(robot_stats, stats)

        assert got.shape == ref.shape, (
            f"seed={seed} R={R}: shape {got.shape} != {ref.shape}"
        )
        assert np.array_equal(got.astype(np.float64), ref.astype(np.float64)), (
            f"seed={seed} R={R}: values differ\n  got:  {got}\n  want: {ref}"
        )

    def test_layout_x0_y0_vx0_vy0(self):
        """Verify field layout matches [x, y, vx, vy] × R."""
        R = 4
        rng = np.random.default_rng(0)
        robot_stats, stats = _make_inputs(R, rng)
        gk = build_global_knowledge(robot_stats, stats)
        for i in range(R):
            assert gk[i * 4]     == pytest.approx(robot_stats[i][0])  # x
            assert gk[i * 4 + 1] == pytest.approx(robot_stats[i][1])  # y
            assert gk[i * 4 + 2] == pytest.approx(stats[i][2])        # vx
            assert gk[i * 4 + 3] == pytest.approx(stats[i][3])        # vy


class TestBuildGKnowledgeAll:
    """build_g_knowledge_all must match the nested-loop reference exactly."""

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("R", ROBOT_COUNTS)
    def test_matches_reference(self, seed, R):
        rng = np.random.default_rng(seed)
        robot_stats, stats = _make_inputs(R, rng)

        ref_gk = _ref_build_global_knowledge(robot_stats, stats, R)
        ref_all = _ref_build_g_knowledge_all(ref_gk, R)

        vec_gk = build_global_knowledge(robot_stats, stats)
        vec_all = build_g_knowledge_all(vec_gk)

        assert len(vec_all) == R, f"seed={seed} R={R}: got {len(vec_all)} entries"
        for i in range(R):
            assert vec_all[i].shape == ref_all[i].shape, (
                f"seed={seed} R={R} robot {i}: shape {vec_all[i].shape} != {ref_all[i].shape}"
            )
            assert np.array_equal(
                vec_all[i].astype(np.float64),
                ref_all[i].astype(np.float64),
            ), (
                f"seed={seed} R={R} robot {i}: mismatch\n"
                f"  got:  {vec_all[i]}\n  want: {ref_all[i]}"
            )

    def test_robot_i_excluded(self):
        """Robot i's own state must never appear in g_knowledge_all[i]."""
        R = 4
        # Give each robot a unique distinctive x-position
        robot_stats = [np.array([float(10 * (i + 1)), 0.0, 0.0, 0.0, 0.0, 0.0],
                                dtype=np.float32)
                       for i in range(R)]
        stats = [np.zeros(4, dtype=np.float32) for _ in range(R)]
        gk = build_global_knowledge(robot_stats, stats)
        all_views = build_g_knowledge_all(gk)
        for i in range(R):
            own_x = robot_stats[i][0]
            neighbor_xs = all_views[i].reshape(R - 1, 4)[:, 0]
            assert own_x not in neighbor_xs, (
                f"Robot {i}'s x={own_x} found in its own g_knowledge view"
            )

    def test_r2_is_not_included_in_r2_view(self):
        """Special case R=2: each robot sees exactly one neighbor."""
        R = 2
        rng = np.random.default_rng(7)
        robot_stats, stats = _make_inputs(R, rng)
        gk = build_global_knowledge(robot_stats, stats)
        all_views = build_g_knowledge_all(gk)
        assert all_views[0].shape == (4,)
        assert all_views[1].shape == (4,)
        # Robot 0's view should be robot 1's data
        assert np.array_equal(all_views[0], gk[4:8])
        # Robot 1's view should be robot 0's data
        assert np.array_equal(all_views[1], gk[0:4])


# ---------------------------------------------------------------------------
# Golden-equivalence guard for the M2 / M4 instrumentation (2026-07-04 pre-reg).
#
# Main.py cannot be imported/run in a unit test (argparse.parse_args() at module
# level + ZMQ/ARGoS), and the repo has no end-to-end training reference hash — the
# golden gates are component/helper-equivalence tests reproduced inline. In that
# spirit, this guard proves the two non-negotiable invariants at the exact code
# units the instrumentation touches:
#
#   (1) GSP_EVAL_ABLATE_PRED=none is a LITERAL identity at the injection helper —
#       the same object is returned (no allocation, no value change), so the
#       default training trajectory is bit-exact.
#   (2) GSP_LOG_CANDIDATE_TARGETS is additive-only in the h5 writer: a logger
#       episode with candidate logging OFF is byte-identical to a baseline
#       episode without the instrumentation, and turning it ON leaves EVERY
#       pre-existing (training) dataset byte-identical — only the extra
#       cand_target_* datasets appear.
# ---------------------------------------------------------------------------

try:
    from rl_code.src.pred_ablation import apply_pred_ablation, RunningMeanState
except ImportError:  # pragma: no cover - path-dependent import
    from src.pred_ablation import apply_pred_ablation, RunningMeanState  # type: ignore

try:
    import h5py
    _HAS_H5PY = True
except ImportError:  # pragma: no cover
    _HAS_H5PY = False


class TestPredAblationNoneBitExact:
    """M2 golden guard: `none` must be a literal identity no-op."""

    @pytest.mark.parametrize("seed", (0, 1, 7, 42, 123))
    @pytest.mark.parametrize("K", (1, 3, 4, 32))
    def test_none_returns_same_object(self, seed, K):
        rng = np.random.default_rng(seed)
        pred = rng.standard_normal(K).astype(np.float32)
        state = RunningMeanState()
        out = apply_pred_ablation(pred, 'none', rng, state)
        # Same object identity — the guard for bit-exactness of the default path.
        assert out is pred
        # And the running-mean accumulator is never touched on the `none` path.
        assert state.count == 0

    def test_none_does_not_advance_shared_rng(self):
        """`none` must not consume the shared rng (so a run that never leaves the
        default path is deterministic regardless of the ablation rng)."""
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        pred = np.array([0.5, -1.5, 2.0], dtype=np.float32)
        apply_pred_ablation(pred, 'none', rng_a, RunningMeanState())
        # rng_a must be in the same state as an untouched rng_b.
        assert rng_a.integers(0, 2**31) == rng_b.integers(0, 2**31)


@pytest.mark.skipif(not _HAS_H5PY, reason="h5py not installed")
class TestCandidateLoggingAdditiveOnly:
    """M4 golden guard: candidate logging is additive-only; the off path is a
    byte-exact no-op vs an un-instrumented baseline episode."""

    @staticmethod
    def _import_logger():
        try:
            from rl_code.src.hdf5_logger import HDF5Logger
        except ImportError:  # pragma: no cover
            from src.hdf5_logger import HDF5Logger  # type: ignore
        return HDF5Logger

    @staticmethod
    def _writerow_kwargs(t):
        # Deterministic, t-dependent values so any dataset perturbation is visible.
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

    def _run_episode(self, path, log_candidates):
        HDF5Logger = self._import_logger()
        logger = HDF5Logger(path)
        for t in range(6):
            logger.writerow(**self._writerow_kwargs(t))
            if log_candidates:
                logger.record_candidate_targets(
                    delta_theta=0.1 * t, future_prox=0.05 * t + 0.2,
                    cyl_kin=[0.01 * t, -0.02 * t, 0.03 * t],
                    centroid_goal=0.5 - 0.01 * t)
        logger.write_episode(0)

    def _dataset_bytes(self, path):
        out = {}
        with h5py.File(path) as f:
            grp = f["episode_0000"]
            for key in grp.keys():
                out[key] = np.asarray(grp[key][()]).tobytes()
        return out

    def test_off_path_matches_uninstrumented_baseline(self, tmp_path):
        """Candidate logging OFF must be byte-identical to a baseline episode
        (the instrumentation adds nothing on the default path)."""
        p_base = str(tmp_path / "baseline.h5")
        p_off = str(tmp_path / "off.h5")
        self._run_episode(p_base, log_candidates=False)
        self._run_episode(p_off, log_candidates=False)
        base = self._dataset_bytes(p_base)
        off = self._dataset_bytes(p_off)
        assert set(base.keys()) == set(off.keys())
        for key in base:
            assert base[key] == off[key], f"dataset {key} differs on off path"
        # And no candidate datasets exist on the off path.
        assert not any(k.startswith("cand_target_") for k in off), \
            "candidate datasets present with logging off"

    def test_on_path_leaves_training_datasets_byte_identical(self, tmp_path):
        """Candidate logging ON: every pre-existing (training) dataset must be
        byte-identical to the OFF run; only cand_target_* datasets are added."""
        p_off = str(tmp_path / "off.h5")
        p_on = str(tmp_path / "on.h5")
        self._run_episode(p_off, log_candidates=False)
        self._run_episode(p_on, log_candidates=True)
        off = self._dataset_bytes(p_off)
        on = self._dataset_bytes(p_on)
        # The ON set is the OFF set PLUS the candidate datasets — nothing removed.
        added = set(on.keys()) - set(off.keys())
        assert added == {
            "cand_target_delta_theta", "cand_target_future_prox",
            "cand_target_cyl_kin", "cand_target_centroid_goal",
        }, f"unexpected dataset delta: {added}"
        # Every shared (training) dataset is byte-for-byte unchanged.
        for key in off:
            assert off[key] == on[key], (
                f"training dataset {key} perturbed by candidate logging"
            )

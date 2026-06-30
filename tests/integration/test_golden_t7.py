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

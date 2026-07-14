"""Registration + semantics tests for the force_disagreement GSP target.

force_disagreement is a GLOBAL directional-disagreement scalar of the applied
forces — 1 - |Σf_i|/Σ|f_i| (0 = all aligned, ~1 = forces cancel / robots
fighting) — broadcast to every robot. It is a delayed-label scalar target
(mirrors neighbor_force), size 1, output kind 'force_disagreement_1d'. The label
itself is computed in Main.py at maturity; here we test the agent-side
registration and pin the disagreement formula's semantics.
"""
import sys

import numpy as np
import pytest

sys.path.insert(0, "tests/test_agent")
from test_delta_theta_traj_target import _make_agent  # noqa: E402


def test_force_disagreement_is_delayed_label_target():
    agent = _make_agent(gsp_output_kind="force_disagreement_1d",
                        prediction_target="force_disagreement", n_agents=4)
    assert agent._is_delayed_label_target() is True


def test_force_disagreement_output_size_is_one():
    agent = _make_agent(gsp_output_kind="force_disagreement_1d",
                        prediction_target="force_disagreement", n_agents=4)
    assert agent.gsp_network_output == 1


def _disagreement(mags, angs_deg):
    """Reference impl of the Main.py label math — pins the intended semantics."""
    fmag = np.asarray(mags, dtype=np.float64)
    fang = np.deg2rad(np.asarray(angs_deg, dtype=np.float64))
    netx = float((fmag * np.cos(fang)).sum())
    nety = float((fmag * np.sin(fang)).sum())
    summag = float(fmag.sum())
    return 1.0 - float(np.hypot(netx, nety)) / summag if summag > 1e-9 else 0.0


def test_disagreement_zero_when_aligned():
    # All forces same direction → net magnitude == sum of magnitudes → disagreement 0.
    assert _disagreement([5.0, 6.0, 7.0, 5.5], [30.0, 30.0, 30.0, 30.0]) == pytest.approx(0.0, abs=1e-9)


def test_disagreement_one_when_canceling():
    # Two opposing equal pairs → net force zero → disagreement 1.
    assert _disagreement([5.0, 5.0, 5.0, 5.0], [0.0, 180.0, 90.0, -90.0]) == pytest.approx(1.0, abs=1e-9)


def test_disagreement_partial_is_between():
    d = _disagreement([5.0, 5.0, 5.0, 5.0], [0.0, 90.0, 0.0, 90.0])
    assert 0.0 < d < 1.0

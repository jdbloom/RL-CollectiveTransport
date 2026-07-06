"""Performance-gated gate curriculum (gate_curriculum==2) decision logic.

The ARGoS C++ loop function is not built with a unit-test harness and a full
mode-2 run requires the ZMQ/PyTorch server, so this test follows the repo
convention (see test_golden_t7.py): the decision + advancement logic is
reproduced inline from the verbatim C++ so there is no ambiguity, and the
truth table + a simulated episode stream are asserted against it.

Verbatim source: argos/collectiveRlTransport.cpp
  - ShouldAdvanceGate(...)  (static decision function)
  - PostStep() mode-2 block (outcome window + offset advance + floor)
"""
from __future__ import annotations

from collections import deque


# ---------------------------------------------------------------------------
# Reference logic (verbatim mirror of collectiveRlTransport.cpp)
# ---------------------------------------------------------------------------

def should_advance_gate(window_size, threshold, outcomes_in_window,
                        outcomes_observed, episodes_since_advance):
    """Mirror of CCollectiveRLTransport::ShouldAdvanceGate."""
    if window_size == 0:
        return False
    if outcomes_observed < window_size:
        return False
    if episodes_since_advance < window_size:
        return False
    success_rate = outcomes_in_window / window_size
    return success_rate >= threshold


def simulate_mode2(outcomes, window_size, threshold, gate_update,
                   start_offset, gate_minimum):
    """Mirror of the PostStep() mode-2 block driven over a list of episode
    outcomes (1=goal reached, 0=timeout). Returns the offset trajectory
    (offset recorded for the episode ABOUT to run, one entry per input outcome
    plus the initial episode-0 offset).
    """
    floor = gate_minimum / 2.0
    offset = start_offset
    dq: deque = deque()
    episodes_since_advance = 0
    # Episode 0 geometry is built in Init() at start_offset.
    trajectory = [offset]
    advances = []  # (episode_index_of_next_run, new_offset)
    for ep_finished, outcome in enumerate(outcomes):
        dq.append(1 if outcome else 0)
        if len(dq) > window_size:
            dq.popleft()
        episodes_since_advance += 1
        outcomes_in_window = sum(dq)
        if should_advance_gate(window_size, threshold, outcomes_in_window,
                               len(dq), episodes_since_advance):
            offset = offset - gate_update
            if offset <= floor:
                offset = floor
            episodes_since_advance = 0
            advances.append((ep_finished + 1, offset))
        # geometry for the next episode (ep_finished + 1) uses current offset
        trajectory.append(offset)
    return trajectory, advances


# ---------------------------------------------------------------------------
# Truth table required by the spec
# ---------------------------------------------------------------------------

WINDOW = 20
THRESH = 0.8


def test_below_threshold_no_advance():
    # Full window, window elapsed, but success rate below threshold.
    assert should_advance_gate(WINDOW, THRESH, 15, WINDOW, WINDOW) is False  # 0.75 < 0.8


def test_at_threshold_but_window_not_elapsed():
    # Success rate clears threshold and window is full, but consolidation
    # guard: not enough episodes since the last advance.
    assert should_advance_gate(WINDOW, THRESH, 20, WINDOW, WINDOW - 1) is False


def test_threshold_and_window_elapsed_advances():
    # Full window, elapsed, success rate == threshold -> advance.
    assert should_advance_gate(WINDOW, THRESH, 16, WINDOW, WINDOW) is True  # 0.80 == 0.8
    assert should_advance_gate(WINDOW, THRESH, 20, WINDOW, WINDOW) is True  # 1.00


def test_window_not_full_no_advance():
    # Fewer outcomes observed than the window size.
    assert should_advance_gate(WINDOW, THRESH, 10, 10, 10) is False


def test_zero_window_never_advances():
    assert should_advance_gate(0, THRESH, 0, 0, 0) is False


# ---------------------------------------------------------------------------
# End-to-end behavioural mirror
# ---------------------------------------------------------------------------

def test_always_success_advances_after_window():
    """Always-succeed stream: offset narrows once per window and floors."""
    start = 5.0
    gmin = 4.0        # floor = 2.0
    gupd = 0.25
    n = 300
    traj, advances = simulate_mode2([1] * n, WINDOW, THRESH, gupd, start, gmin)
    # First advance occurs after exactly WINDOW successful episodes.
    assert advances[0][0] == WINDOW
    # Offset strictly narrows across successive advances until it floors.
    offs = [a[1] for a in advances]
    for a, b in zip(offs, offs[1:]):
        assert b <= a
    # It reaches and stays at the floor (gate_minimum / 2).
    assert offs[-1] == gmin / 2.0
    assert traj[-1] == gmin / 2.0
    # Advances are spaced at least WINDOW episodes apart (consolidation guard).
    idxs = [a[0] for a in advances]
    for a, b in zip(idxs, idxs[1:]):
        assert b - a >= WINDOW


def test_always_fail_keeps_offset_wide():
    """Always-fail stream: offset never moves off the initial wide value."""
    start = 5.0
    gmin = 4.0
    gupd = 0.25
    traj, advances = simulate_mode2([0] * 300, WINDOW, THRESH, gupd, start, gmin)
    assert advances == []
    assert all(o == start for o in traj)


def test_floor_not_exceeded():
    """Offset never drops below gate_minimum/2 even with many advances."""
    start = 5.0
    gmin = 4.0        # floor 2.0
    gupd = 1.0        # large step so it would overshoot without the clamp
    traj, advances = simulate_mode2([1] * 200, WINDOW, THRESH, gupd, start, gmin)
    assert min(traj) == gmin / 2.0
    assert all(o >= gmin / 2.0 for o in traj)

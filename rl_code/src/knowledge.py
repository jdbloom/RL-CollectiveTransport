"""Global-knowledge helpers for collective-transport training.

Extracted from Main.py so they can be unit-tested and vectorized without
touching Main.py's argparse/zmq boilerplate.

Public API
----------
build_global_knowledge(robot_stats, stats)
    Build the (R*4,) flat global-knowledge array from per-robot position and
    velocity data.

build_g_knowledge_all(global_knowledge)
    Return all R per-robot leave-one-out neighbor views from the global
    knowledge array.  Each view is a (R-1)*4 array.

Both helpers are pure functions (no side effects, no globals) and are
guaranteed inert — output is numerically identical to the nested-loop
implementation in Main.py.
"""
from __future__ import annotations

import numpy as np


def build_global_knowledge(robot_stats: list, stats: list) -> np.ndarray:
    """Vectorized replacement for Main.py lines 324-329 / 755-760.

    Parameters
    ----------
    robot_stats : list of ndarray, shape (6,) each
        Per-robot pose arrays (x_pos=idx0, y_pos=idx1, …).
    stats : list of ndarray, shape (num_stats,) each
        Per-robot stats arrays (magnitude=idx0, angle=idx1,
        deltaX=idx2, deltaY=idx3, …).

    Returns
    -------
    global_knowledge : np.ndarray, shape (R*4,)
        Flat layout: [x0, y0, vx0, vy0, x1, y1, vx1, vy1, …]
    """
    R = len(robot_stats)
    # Stack positions and velocities in a single numpy call.
    rs = np.stack(robot_stats, axis=0)  # (R, >=2)
    st = np.stack(stats, axis=0)        # (R, >=4)
    gk = np.empty(R * 4, dtype=np.float32)
    gk[0::4] = rs[:, 0]  # x_pos
    gk[1::4] = rs[:, 1]  # y_pos
    gk[2::4] = st[:, 2]  # deltaX (velocity X)
    gk[3::4] = st[:, 3]  # deltaY (velocity Y)
    return gk


def build_g_knowledge_all(global_knowledge: np.ndarray) -> list[np.ndarray]:
    """Return all R per-robot leave-one-out neighbor views.

    Parameters
    ----------
    global_knowledge : np.ndarray, shape (R*4,)
        Flat array from :func:`build_global_knowledge`.

    Returns
    -------
    g_knowledge_all : list of np.ndarray
        R arrays each of shape ((R-1)*4,).  ``g_knowledge_all[i]`` is the
        concatenated state of all robots except robot i, in the same order
        as the original nested-loop implementation (skip i, others in
        ascending j order).

    Notes
    -----
    At R=4 (16 inner iterations) the original nested-loop is faster than the
    np.delete approach because np.delete allocates a new array on every call.
    This is the verbatim Main.py logic (lines 331-340 / 763-772), kept in
    this module for testability.
    """
    R = global_knowledge.shape[0] // 4
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

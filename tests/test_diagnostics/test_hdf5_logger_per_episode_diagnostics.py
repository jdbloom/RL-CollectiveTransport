"""Tests for per-episode diagnostic attrs in HDF5Logger.

The diagnostics library (gsp_rl/src/actors/diagnostics.py) returns dicts of
namespaced scalars (keys prefixed with diag_*). The HDF5Logger needs a way to
accept those dicts once per diagnostic episode and persist them as per-episode
HDF5 attrs so the analyzer + reanalysis scripts can read them back.

It also needs to optionally persist the fixed eval-batch states (the 1024 states
used for all diagnostic computations) as a dataset on the episode where the
batch is first frozen, so later reanalysis can reconstruct exactly which states
were measured.

See docs/specs/2026-04-17-diagnostics-instrumentation.md for the spec.
"""
import os
import numpy as np
import pytest

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

pytestmark = pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")


if HAS_H5PY:
    from src.hdf5_logger import HDF5Logger


def _base_writerow_kwargs():
    return dict(
        rewards=[0.1] * 4,
        epsilons=0.5,
        terminations=False,
        losses=0.0,
        force_magnitudes=[0.0] * 4,
        force_angles=[0.0] * 4,
        average_force_vectors=[0.0, 0.0],
        cyl_x_poses=0.0,
        cyl_y_poses=0.0,
        cyl_angles=0.0,
        gate_stats=0,
        obstacle_stats=0,
        gsp_rewards=[0.0] * 4,
        gsp_headings=[0.0] * 4,
        run_times=0.0,
        robots_x_poses=[0.0] * 4,
        robots_y_poses=[0.0] * 4,
        robot_angles=[0.0] * 4,
        robot_failure=[False] * 4,
    )


def test_record_episode_diagnostics_writes_scalar_attrs(tmp_path):
    """A single diagnostic dict passed before write_episode shows up as h5 attrs
    on the episode group."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)
    for _ in range(3):
        logger.writerow(**_base_writerow_kwargs())
    diag = {
        "diag_actor_fau_fc1": 0.12,
        "diag_actor_fau_fc2": 0.08,
        "diag_actor_wnorm_fc1": 2.5,
        "diag_q_action_gap_mean": 1.75,
        "diag_gsp_erank_penult": 4.0,
    }
    logger.record_episode_diagnostics(diag)
    logger.write_episode(7)

    with h5py.File(path) as f:
        grp = f["episode_0007"]
        for k, expected in diag.items():
            assert k in grp.attrs, f"missing diag attr {k}"
            assert abs(float(grp.attrs[k]) - expected) < 1e-6, k


def test_diagnostics_optional_backward_compat(tmp_path):
    """Episodes without a record_episode_diagnostics call should have NO diag_* attrs
    and the rest of the episode should write normally."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)
    for _ in range(2):
        logger.writerow(**_base_writerow_kwargs())
    logger.write_episode(0)

    with h5py.File(path) as f:
        grp = f["episode_0000"]
        diag_attrs = [k for k in grp.attrs.keys() if k.startswith("diag_")]
        assert diag_attrs == [], f"unexpected diagnostic attrs on non-diag episode: {diag_attrs}"


def test_record_eval_batch_states_creates_dataset(tmp_path):
    """When the eval batch is frozen, its states are stored as a dataset on that
    episode so reanalysis can reconstruct exactly what was measured."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)
    for _ in range(2):
        logger.writerow(**_base_writerow_kwargs())
    eval_batch = np.random.randn(1024, 6).astype(np.float32)
    logger.record_eval_batch_states(eval_batch)
    logger.write_episode(50)

    with h5py.File(path) as f:
        grp = f["episode_0050"]
        assert "diag_eval_batch_states" in grp, "eval batch states dataset missing"
        assert grp["diag_eval_batch_states"].shape == (1024, 6)
        np.testing.assert_allclose(grp["diag_eval_batch_states"][:], eval_batch, rtol=1e-5)


def test_eval_batch_states_optional(tmp_path):
    """Episodes without a record_eval_batch_states call should not have that dataset."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)
    for _ in range(2):
        logger.writerow(**_base_writerow_kwargs())
    logger.record_episode_diagnostics({"diag_actor_fau_fc1": 0.1})
    logger.write_episode(50)

    with h5py.File(path) as f:
        grp = f["episode_0050"]
        assert "diag_eval_batch_states" not in grp


def test_diagnostics_reset_between_episodes(tmp_path):
    """record_episode_diagnostics() state is per-episode: calling it for ep 0 must
    not leak diag attrs onto ep 1."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)

    # Episode 0: with diagnostics
    for _ in range(2):
        logger.writerow(**_base_writerow_kwargs())
    logger.record_episode_diagnostics({"diag_actor_fau_fc1": 0.42})
    logger.write_episode(0)

    # Episode 1: no diagnostics recorded — should have none on the group
    for _ in range(2):
        logger.writerow(**_base_writerow_kwargs())
    logger.write_episode(1)

    with h5py.File(path) as f:
        ep0 = f["episode_0000"]
        ep1 = f["episode_0001"]
        assert "diag_actor_fau_fc1" in ep0.attrs
        assert "diag_actor_fau_fc1" not in ep1.attrs


def test_multiple_diagnostic_calls_merge(tmp_path):
    """Multiple record_episode_diagnostics calls before write_episode should merge
    their keys (later calls overwrite earlier on conflict)."""
    path = str(tmp_path / "ep.h5")
    logger = HDF5Logger(path)
    for _ in range(2):
        logger.writerow(**_base_writerow_kwargs())
    logger.record_episode_diagnostics({"diag_actor_fau_fc1": 0.1, "diag_q_action_gap_mean": 2.0})
    logger.record_episode_diagnostics({"diag_actor_wnorm_fc1": 5.5})
    # Conflict — later wins
    logger.record_episode_diagnostics({"diag_q_action_gap_mean": 3.0})
    logger.write_episode(10)

    with h5py.File(path) as f:
        grp = f["episode_0010"]
        assert abs(float(grp.attrs["diag_actor_fau_fc1"]) - 0.1) < 1e-6
        assert abs(float(grp.attrs["diag_actor_wnorm_fc1"]) - 5.5) < 1e-6
        assert abs(float(grp.attrs["diag_q_action_gap_mean"]) - 3.0) < 1e-6

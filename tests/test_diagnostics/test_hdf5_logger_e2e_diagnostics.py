"""Tests for e2e per-learn-step diagnostic fields added to HDF5Logger (schema v3).

These fields are written when GSP_E2E_ENABLED is set and the agent's
learn_DDQN_e2e method populates last_e2e_diagnostics. The logger records
11 scalars per learning step.
"""

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


def _sample_diag(step: int) -> dict:
    """Return a realistic diagnostics dict for learn step `step`."""
    return {
        'ddqn_loss': 0.1 + step * 0.01,
        'gsp_mse_loss': 0.05 + step * 0.005,
        'total_loss': 0.15 + step * 0.015,
        'gsp_grad_norm': 1.2 + step * 0.1,
        'gsp_grad_norm_pre_clip': 2.0 + step * 0.2,
        'ddqn_grad_norm': 0.8 + step * 0.05,
        'gsp_input_grad': 0.3 + step * 0.03,
        'gsp_pred_mean': 0.0 + step * 0.01,
        'gsp_pred_std': 0.5 + step * 0.02,
        'gsp_label_mean': 0.01 + step * 0.01,
        'gsp_label_std': 0.6 + step * 0.01,
    }


class TestRecordE2eDiagnostics:
    def test_all_11_datasets_written(self, tmp_path):
        path = str(tmp_path / "ep.h5")
        logger = HDF5Logger(path)
        for t in range(3):
            logger.writerow(**_base_writerow_kwargs())
        for step in range(3):
            logger.record_e2e_diagnostics(_sample_diag(step))
        logger.write_episode(0)

        expected_keys = [
            'e2e_ddqn_loss', 'e2e_gsp_mse_loss', 'e2e_total_loss',
            'e2e_gsp_grad_norm', 'e2e_gsp_grad_norm_pre_clip', 'e2e_ddqn_grad_norm',
            'e2e_gsp_input_grad', 'e2e_gsp_pred_mean', 'e2e_gsp_pred_std',
            'e2e_gsp_label_mean', 'e2e_gsp_label_std',
        ]
        with h5py.File(path) as f:
            grp = f["episode_0000"]
            for key in expected_keys:
                assert key in grp, f"Missing dataset: {key}"
                assert grp[key].shape == (3,), f"Wrong shape for {key}: {grp[key].shape}"

    def test_values_stored_correctly(self, tmp_path):
        path = str(tmp_path / "ep.h5")
        logger = HDF5Logger(path)
        logger.writerow(**_base_writerow_kwargs())
        logger.record_e2e_diagnostics({
            'ddqn_loss': 0.25,
            'gsp_mse_loss': 0.10,
            'total_loss': 0.35,
            'gsp_grad_norm': 1.5,
            'gsp_grad_norm_pre_clip': 3.0,
            'ddqn_grad_norm': 0.9,
            'gsp_input_grad': 0.4,
            'gsp_pred_mean': 0.01,
            'gsp_pred_std': 0.55,
            'gsp_label_mean': 0.02,
            'gsp_label_std': 0.65,
        })
        logger.write_episode(0)

        with h5py.File(path) as f:
            grp = f["episode_0000"]
            np.testing.assert_allclose(float(grp['e2e_ddqn_loss'][0]), 0.25, rtol=1e-5)
            np.testing.assert_allclose(float(grp['e2e_total_loss'][0]), 0.35, rtol=1e-5)
            np.testing.assert_allclose(float(grp['e2e_gsp_grad_norm'][0]), 1.5, rtol=1e-5)

    def test_summary_attrs_written(self, tmp_path):
        path = str(tmp_path / "ep.h5")
        logger = HDF5Logger(path)
        logger.writerow(**_base_writerow_kwargs())
        for step in range(3):
            logger.record_e2e_diagnostics(_sample_diag(step))
        logger.write_episode(0)

        with h5py.File(path) as f:
            grp = f["episode_0000"]
            assert "e2e_gsp_grad_norm_mean" in grp.attrs
            assert "e2e_gsp_pred_std_mean" in grp.attrs
            # Verify the mean is in reasonable range
            assert float(grp.attrs["e2e_gsp_grad_norm_mean"]) > 0.0
            assert float(grp.attrs["e2e_gsp_pred_std_mean"]) > 0.0

    def test_missing_keys_default_to_zero(self, tmp_path):
        """Partial diagnostics dict — missing keys default to 0, not KeyError."""
        path = str(tmp_path / "ep.h5")
        logger = HDF5Logger(path)
        logger.writerow(**_base_writerow_kwargs())
        logger.record_e2e_diagnostics({'ddqn_loss': 0.5})  # only one key present
        logger.write_episode(0)

        with h5py.File(path) as f:
            grp = f["episode_0000"]
            assert 'e2e_ddqn_loss' in grp
            np.testing.assert_allclose(float(grp['e2e_ddqn_loss'][0]), 0.5, rtol=1e-5)
            np.testing.assert_allclose(float(grp['e2e_gsp_mse_loss'][0]), 0.0, atol=1e-6)

    def test_none_gsp_input_grad_defaults_to_zero(self, tmp_path):
        """gsp_input_grad can be None when input grads are not tracked; coerce to 0."""
        path = str(tmp_path / "ep.h5")
        logger = HDF5Logger(path)
        logger.writerow(**_base_writerow_kwargs())
        diag = _sample_diag(0)
        diag['gsp_input_grad'] = None
        logger.record_e2e_diagnostics(diag)
        logger.write_episode(0)

        with h5py.File(path) as f:
            grp = f["episode_0000"]
            np.testing.assert_allclose(float(grp['e2e_gsp_input_grad'][0]), 0.0, atol=1e-6)

    def test_no_e2e_diagnostics_no_datasets_written(self, tmp_path):
        """When record_e2e_diagnostics is never called, no e2e_ datasets appear."""
        path = str(tmp_path / "ep.h5")
        logger = HDF5Logger(path)
        logger.writerow(**_base_writerow_kwargs())
        logger.write_episode(0)

        with h5py.File(path) as f:
            grp = f["episode_0000"]
            e2e_keys = [k for k in grp.keys() if k.startswith('e2e_')]
            assert e2e_keys == [], f"Expected no e2e_ datasets, found: {e2e_keys}"
            assert "e2e_gsp_grad_norm_mean" not in grp.attrs
            assert "e2e_gsp_pred_std_mean" not in grp.attrs

    def test_buffers_reset_between_episodes(self, tmp_path):
        """After write_episode, e2e buffers are cleared for the next episode."""
        path = str(tmp_path / "ep.h5")
        logger = HDF5Logger(path)

        logger.writerow(**_base_writerow_kwargs())
        for step in range(4):
            logger.record_e2e_diagnostics(_sample_diag(step))
        logger.write_episode(0)

        # Episode 1: only 2 learn steps
        logger.writerow(**_base_writerow_kwargs())
        for step in range(2):
            logger.record_e2e_diagnostics(_sample_diag(step))
        logger.write_episode(1)

        with h5py.File(path) as f:
            assert f["episode_0000"]["e2e_ddqn_loss"].shape == (4,)
            assert f["episode_0001"]["e2e_ddqn_loss"].shape == (2,)

    def test_schema_version_bumped_to_3(self, tmp_path):
        """Schema version must be 3 after adding e2e fields."""
        path = str(tmp_path / "ep.h5")
        logger = HDF5Logger(path)
        logger.writerow(**_base_writerow_kwargs())
        logger.write_episode(0)

        with h5py.File(path) as f:
            assert f["episode_0000"].attrs["log_schema_version"] == 3

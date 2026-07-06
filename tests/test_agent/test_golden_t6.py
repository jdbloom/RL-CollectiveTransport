"""T6 golden gate — prox + GSP-reward vectorization.

Self-contained: uses only numpy + pickle (no aios import). Golden refs are
frozen from the UNMODIFIED baseline via tests/test_agent/freeze_t6.py and
committed alongside this file. The optimized code must reproduce them exactly.

Pass criteria:
  - filter_prox_values: filtered_values list EXACTLY equal (no fp rounding),
    indices list EXACTLY equal.
  - calculate_gsp_reward: rewards + squared_errors within rtol=1e-6 atol=1e-6;
    label + raw_diff_rad within rtol=1e-6 atol=1e-6.

If any assertion fails, the optimization introduced a behaviour change and MUST
NOT be merged.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------
# Path setup — works both from repo root (pytest) and direct invocation.
# --------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "rl_code"))
_GSP_RL = _REPO_ROOT.parent / "GSP-RL"
if _GSP_RL.exists():
    sys.path.insert(0, str(_GSP_RL))

from src.agent import Agent
from src.env import calculate_gsp_reward

GOLDEN_DIR = Path(__file__).parent / "golden_refs"


# --------------------------------------------------------------------------
# Shared agent fixture — same config as freeze_t6.py.
# --------------------------------------------------------------------------

def _make_agent(prox_filter_angle_deg: float = 45.0) -> Agent:
    config = {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }
    return Agent(
        config=config, network="DQN", n_agents=4, n_obs=31,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=False, recurrent=False, attention=False,
        neighbors=False, gsp_input_size=6, gsp_output_size=1,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
        prox_filter_angle_deg=prox_filter_angle_deg,
    )


# --------------------------------------------------------------------------
# T6a: filter_prox_values golden gate
# --------------------------------------------------------------------------

class TestFilterProxGoldenT6:
    """filter_prox_values must return bit-identical results after vectorization."""

    @pytest.fixture(scope="class")
    def agent(self):
        return _make_agent(prox_filter_angle_deg=45.0)

    @pytest.fixture(scope="class")
    def golden_cases(self):
        path = GOLDEN_DIR / "t6_filter_prox.pkl"
        assert path.exists(), (
            f"Golden ref not found: {path}\n"
            "Run: PYTHONPATH=rl_code:<GSP-RL> python tests/test_agent/freeze_t6.py"
        )
        return pickle.loads(path.read_bytes())

    def test_filtered_values_exact(self, agent, golden_cases):
        """Filtered sensor values must be EXACTLY equal (no fp rounding tolerance)."""
        mismatches = []
        for i, case in enumerate(golden_cases):
            fv, _ = agent.filter_prox_values(case["prox"], case["angle"])
            ref = case["filtered_values"]
            if fv != ref:
                mismatches.append(
                    f"case {i} angle={case['angle']:.2f}: got {fv}, expected {ref}"
                )
        assert not mismatches, (
            f"{len(mismatches)} filter_prox_values mismatches:\n" + "\n".join(mismatches[:5])
        )

    def test_indices_exact(self, agent, golden_cases):
        """Filtered-out indices must be EXACTLY equal (set membership is boolean)."""
        mismatches = []
        for i, case in enumerate(golden_cases):
            _, idx = agent.filter_prox_values(case["prox"], case["angle"])
            ref = case["indices"]
            if idx != ref:
                mismatches.append(
                    f"case {i} angle={case['angle']:.2f}: got {idx}, expected {ref}"
                )
        assert not mismatches, (
            f"{len(mismatches)} index mismatches:\n" + "\n".join(mismatches[:5])
        )

    def test_partition_invariant(self, agent, golden_cases):
        """filtered + indices must always partition all 24 sensors."""
        for i, case in enumerate(golden_cases):
            fv, idx = agent.filter_prox_values(case["prox"], case["angle"])
            assert len(fv) + len(idx) == 24, (
                f"case {i}: partition broken, got {len(fv)} filtered + {len(idx)} indices"
            )


# --------------------------------------------------------------------------
# T6b: calculate_gsp_reward golden gate
# --------------------------------------------------------------------------

class TestGSPRewardGoldenT6:
    """calculate_gsp_reward must return fp-identical results after vectorization."""

    @pytest.fixture(scope="class")
    def golden_cases(self):
        path = GOLDEN_DIR / "t6_gsp_reward.pkl"
        assert path.exists(), (
            f"Golden ref not found: {path}\n"
            "Run: PYTHONPATH=rl_code:<GSP-RL> python tests/test_agent/freeze_t6.py"
        )
        return pickle.loads(path.read_bytes())

    def test_rewards_allclose(self, golden_cases):
        """Per-robot rewards within rtol=1e-6 atol=1e-6."""
        for i, case in enumerate(golden_cases):
            rewards, _, _, _ = calculate_gsp_reward(
                True, case["old_cyl"], case["cyl"], case["preds"], case["num_robots"]
            )
            np.testing.assert_allclose(
                rewards, case["rewards"],
                rtol=1e-6, atol=1e-6,
                err_msg=f"rewards mismatch at case {i} old_cyl={case['old_cyl']:.1f} cyl={case['cyl']:.1f}",
            )

    def test_squared_errors_allclose(self, golden_cases):
        """Per-robot squared errors within rtol=1e-6 atol=1e-6."""
        for i, case in enumerate(golden_cases):
            _, _, sq, _ = calculate_gsp_reward(
                True, case["old_cyl"], case["cyl"], case["preds"], case["num_robots"]
            )
            np.testing.assert_allclose(
                sq, case["squared_errors"],
                rtol=1e-6, atol=1e-6,
                err_msg=f"squared_errors mismatch at case {i}",
            )

    def test_label_allclose(self, golden_cases):
        """Scalar label within rtol=1e-6 atol=1e-6."""
        for i, case in enumerate(golden_cases):
            _, label, _, _ = calculate_gsp_reward(
                True, case["old_cyl"], case["cyl"], case["preds"], case["num_robots"]
            )
            np.testing.assert_allclose(
                label, case["label"],
                rtol=1e-6, atol=1e-6,
                err_msg=f"label mismatch at case {i}",
            )

    def test_raw_diff_rad_allclose(self, golden_cases):
        """raw_diff_rad within rtol=1e-6 atol=1e-6."""
        for i, case in enumerate(golden_cases):
            _, _, _, raw = calculate_gsp_reward(
                True, case["old_cyl"], case["cyl"], case["preds"], case["num_robots"]
            )
            np.testing.assert_allclose(
                raw, case["raw_diff_rad"],
                rtol=1e-6, atol=1e-6,
                err_msg=f"raw_diff_rad mismatch at case {i}",
            )

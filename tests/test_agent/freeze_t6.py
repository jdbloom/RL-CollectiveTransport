"""Freeze script for T6 golden gate.

Run this ONCE on the UNMODIFIED baseline to capture reference outputs:

    PYTHONPATH=rl_code:<GSP-RL-path> python tests/test_agent/freeze_t6.py

Saves golden_refs/t6_filter_prox.pkl and golden_refs/t6_gsp_reward.pkl.
Commit both .pkl files alongside tests/test_agent/test_golden_t6.py.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

# Add rl_code to path so src.agent / src.env are importable
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "rl_code"))

gsp_rl_path = repo_root.parent / "GSP-RL"
if gsp_rl_path.exists():
    sys.path.insert(0, str(gsp_rl_path))

from src.agent import Agent
from src.env import calculate_gsp_reward

GOLDEN_DIR = Path(__file__).parent / "golden_refs"
GOLDEN_DIR.mkdir(exist_ok=True)

N_SEEDS = 50
N_PROX = 24  # _ROBOT_PROXIMITY_ANGLES has 24 entries


def make_agent(prox_filter_angle_deg=45.0):
    config = {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }
    return Agent(
        config=config, network='DQN', n_agents=4, n_obs=31,
        n_actions=2, options_per_action=3, id=0, min_max_action=0.1,
        meta_param_size=1, gsp=False, recurrent=False, attention=False,
        neighbors=False, gsp_input_size=6, gsp_output_size=1,
        gsp_min_max_action=1.0, gsp_look_back=2, gsp_sequence_length=5,
        prox_filter_angle_deg=prox_filter_angle_deg,
    )


def freeze_filter_prox():
    """Capture filter_prox_values outputs for 50 random (prox, angle) pairs."""
    agent = make_agent(prox_filter_angle_deg=45.0)
    rng = np.random.default_rng(0)

    results = []
    # Include angles covering all three branches: normal, near +180, near -180
    angles = []
    angles += list(rng.uniform(-179.0, 179.0, 40))   # generic angles
    angles += list(rng.uniform(135.0, 179.9, 5))      # near +180 wrap branch
    angles += list(rng.uniform(-179.9, -135.0, 5))    # near -180 wrap branch

    for angle in angles[:N_SEEDS]:
        prox = rng.uniform(0.0, 1.0, N_PROX)
        filtered, indices = agent.filter_prox_values(prox, float(angle))
        results.append({
            "prox": prox.copy(),
            "angle": float(angle),
            "filtered_values": list(filtered),
            "indices": list(indices),
        })

    path = GOLDEN_DIR / "t6_filter_prox.pkl"
    path.write_bytes(pickle.dumps(results))
    print(f"Saved {len(results)} filter_prox cases → {path}")
    return results


def freeze_gsp_reward():
    """Capture calculate_gsp_reward outputs for 50 random (old_cyl, cyl, preds) inputs."""
    rng = np.random.default_rng(1)

    results = []
    num_robots = 4

    for _ in range(N_SEEDS):
        old_cyl = float(rng.uniform(0.0, 360.0))
        cyl = float(rng.uniform(0.0, 360.0))
        preds = rng.uniform(-1.5, 1.5, num_robots).tolist()
        rewards, label, sq_errors, raw_diff = calculate_gsp_reward(
            True, old_cyl, cyl, preds, num_robots
        )
        results.append({
            "old_cyl": old_cyl,
            "cyl": cyl,
            "preds": preds,
            "num_robots": num_robots,
            "rewards": list(rewards),
            "label": float(label),
            "squared_errors": list(sq_errors),
            "raw_diff_rad": float(raw_diff),
        })

    path = GOLDEN_DIR / "t6_gsp_reward.pkl"
    path.write_bytes(pickle.dumps(results))
    print(f"Saved {len(results)} gsp_reward cases → {path}")
    return results


if __name__ == "__main__":
    freeze_filter_prox()
    freeze_gsp_reward()
    print("Done. Commit golden_refs/t6_filter_prox.pkl and golden_refs/t6_gsp_reward.pkl.")

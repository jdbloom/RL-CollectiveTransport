"""Integration learning regression tests.

Runs actual ARGoS + Python training for each algorithm × GSP variant combination.
Verifies that training produces data, doesn't crash, and shows a learning signal
(later episodes have better rewards than early episodes).

These tests require ARGoS3, Buzz, and the nonuniform-objects plugin installed.
They are slow (~5 min each, ~25 min total) and should run on PRs only.

Usage:
    pytest tests/integration/test_learning_regression.py -v -s --timeout=600

    Or standalone:
    python tests/integration/test_learning_regression.py
"""

import os
import sys
import time
import shutil
import signal
import pickle
import subprocess
import tempfile
import yaml
import numpy as np
import pytest

# Mark all tests in this module as slow + integration
pytestmark = [pytest.mark.slow, pytest.mark.integration]

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARGOS_BIN = shutil.which("argos3")
NUM_EPISODES = 10
NUM_EPISODES_RECURRENT = 5  # RDDPG is much slower (per-sample LSTM loop on CPU)
SEED = 42


def _check_argos_installed():
    """Skip all tests if ARGoS is not installed."""
    if ARGOS_BIN is None:
        pytest.skip("ARGoS3 not installed — skipping integration tests")


def _make_config(exp_name, learning_scheme, port, gsp=False, neighbors=False,
                 recurrent=False, attention=False):
    """Generate an experiment config dict."""
    return {
        "TEST": False,
        "MODEL_NUM": 980,
        "EXP_NAME": exp_name,
        "ARGOS_FILE_NAME": f"collectiveRlTransport_{exp_name}.argos",
        "NUM_EPISODES": NUM_EPISODES,
        "NUM_OBSTACLES": 0,
        "USE_GATE": 0,
        "GATE_MIN": 4,
        "GATE_CURRICULUM": 0,
        "NUM_ROBOTS": 4,
        "MAX_NUM_ROBOT_FAILURES": 0,
        "CHANCE_FAILURE": 0.25,
        "PORT": port,
        "SEED": SEED,
        "USE_PRISMS": 0,
        "RANDOM_OBJECTS": 0,
        "TEST_PRISM": 0,
        "LEARNING_SCHEME": learning_scheme,
        "OPTIONS_PER_ACTION": 3,
        "MIN_MAX_ACTION": 1.0 if learning_scheme in ["DDPG", "TD3"] else 0.1,
        "META_PARAM_SIZE": 1,
        "PROX_FILTER_ANGLE_DEG": 60.0,
        "GLOBAL_KNOWLEDGE": False,
        "GSP": gsp,
        "RECURRENT": recurrent,
        "ATTENTION": attention,
        "NEIGHBORS": neighbors,
        "GSP_INPUT_SIZE": 4,
        "GSP_OUTPUT_SIZE": 1,
        "GSP_MIN_MAX_ACTION": 1.0,
        "GSP_LOOK_BACK": 2,
        "GSP_SEQUENCE_LENGTH": 5,
        "RECURRENT_HIDDEN_SIZE": 256,
        "RECURRENT_EMBEDDING_SIZE": 256,
        "RECURRENT_NUM_LAYERS": 5,
        "GAMMA": 0.99997,
        "TAU": 0.005,
        "ALPHA": 0.001,
        "BETA": 0.001,
        "LR": 0.0001,
        "EPSILON": 1.0,
        "EPS_MIN": 0.01,
        "EPS_DEC": 0.0001,
        "BATCH_SIZE": 64,
        "MEM_SIZE": 100000,
        "REPLACE_TARGET_COUNTER": 1000,
        "NOISE": 0.1,
        "UPDATE_ACTOR_ITER": 2,
        "WARMUP": 1000,
        "GSP_LEARNING_FREQUENCY": 500,
        "GSP_BATCH_SIZE": 128,
    }


def _write_yaml_config(config, path):
    """Write config dict as YAML file matching the awk-parseable format."""
    with open(path, "w") as f:
        for key, value in config.items():
            if isinstance(value, bool):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")


def _generate_argos_xml(config):
    """Run generate_argos.py to create the .argos file."""
    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "argos", "generate_argos.py"),
        "--num_obstacles", str(config["NUM_OBSTACLES"]),
        "--num_robots", str(config["NUM_ROBOTS"]),
        "--max_num_robot_failures", str(config["MAX_NUM_ROBOT_FAILURES"]),
        "--chance_failure", str(config["CHANCE_FAILURE"]),
        "--num_episodes", str(config["NUM_EPISODES"]),
        "--pytorch_port", str(config["PORT"]),
        "--use_gate", str(config["USE_GATE"]),
        "--gate_curriculum", str(config["GATE_CURRICULUM"]),
        "--seed", str(config["SEED"]),
        "--argos_filename", config["ARGOS_FILE_NAME"],
        "--gate_minimum", str(config["GATE_MIN"]),
        "--use_prisms", str(config["USE_PRISMS"]),
        "--random_objs", str(config["RANDOM_OBJECTS"]),
        "--test_prism", str(config["TEST_PRISM"]),
    ]
    result = subprocess.run(cmd, cwd=os.path.join(PROJECT_ROOT, "argos"),
                           capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"generate_argos.py failed: {result.stderr}")


def _run_experiment(config):
    """Run a full training experiment: ARGoS + Main.py.

    Returns:
        data_dir: path to the Data/ directory with pkl files
    """
    exp_name = config["EXP_NAME"]
    data_root = os.path.join(PROJECT_ROOT, "rl_code", "Data", exp_name)

    # Create output directories
    os.makedirs(os.path.join(data_root, "Data"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "Models"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "plots"), exist_ok=True)

    # Write config
    config_path = os.path.join(data_root, "agent_config.yml")
    _write_yaml_config(config, config_path)

    # Generate ARGoS XML
    _generate_argos_xml(config)
    argos_file = os.path.join(PROJECT_ROOT, "argos", config["ARGOS_FILE_NAME"])

    # Copy argos file
    shutil.copy(argos_file, os.path.join(data_root, config["ARGOS_FILE_NAME"]))

    # Launch ARGoS in background
    argos_proc = subprocess.Popen(
        [ARGOS_BIN, "-c", argos_file],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Launch Python RL server
    main_proc = subprocess.Popen(
        [sys.executable, "Main.py", f"Data/{exp_name}"],
        cwd=os.path.join(PROJECT_ROOT, "rl_code"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for completion (20 min default, R-GSP-N may need longer on CPU)
        main_proc.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        main_proc.kill()
        argos_proc.kill()
        raise RuntimeError(f"Experiment {exp_name} timed out after 3600s")
    finally:
        # Ensure ARGoS is cleaned up
        if argos_proc.poll() is None:
            argos_proc.kill()
            argos_proc.wait(timeout=10)

    if main_proc.returncode != 0:
        stderr = main_proc.stderr.read().decode() if main_proc.stderr else ""
        raise RuntimeError(f"Main.py failed (rc={main_proc.returncode}): {stderr[-500:]}")

    return os.path.join(data_root, "Data")


def _check_learning_signal(data_dir, num_episodes):
    """Verify that training shows improvement: later episodes have better rewards.

    Returns:
        (early_avg, late_avg, improved): average rewards and whether learning improved
    """
    pkl_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pkl")],
        key=lambda x: int(x.replace("Data_Episode_", "").replace(".pkl", ""))
    )

    if len(pkl_files) < num_episodes:
        raise RuntimeError(f"Expected {num_episodes} pkl files, found {len(pkl_files)}")

    # Load early and late episode rewards
    early_rewards = []
    late_rewards = []
    n_early = min(3, num_episodes // 3)
    n_late = min(3, num_episodes // 3)

    for f in pkl_files[:n_early]:
        with open(os.path.join(data_dir, f), "rb") as fh:
            data = pickle.load(fh)
            early_rewards.append(np.sum(data.get("reward", data.get("rewards", [0]))))

    for f in pkl_files[-n_late:]:
        with open(os.path.join(data_dir, f), "rb") as fh:
            data = pickle.load(fh)
            late_rewards.append(np.sum(data.get("reward", data.get("rewards", [0]))))

    early_avg = np.mean(early_rewards)
    late_avg = np.mean(late_rewards)
    improved = late_avg > early_avg

    return early_avg, late_avg, improved


def _cleanup_experiment(exp_name):
    """Remove experiment data directory."""
    data_root = os.path.join(PROJECT_ROOT, "rl_code", "Data", exp_name)
    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    # Clean up argos file
    for f in os.listdir(os.path.join(PROJECT_ROOT, "argos")):
        if exp_name in f and f.endswith(".argos"):
            os.remove(os.path.join(PROJECT_ROOT, "argos", f))


class TestDQN_IC:
    """DQN with no communication — baseline discrete learning."""

    exp_name = "regression_DQN_IC"
    port = 55580

    def setup_method(self):
        _check_argos_installed()
        _cleanup_experiment(self.exp_name)

    def teardown_method(self):
        _cleanup_experiment(self.exp_name)

    def test_dqn_ic_trains_and_learns(self):
        config = _make_config(
            exp_name=self.exp_name,
            learning_scheme="DQN",
            port=self.port,
            gsp=False,
        )
        data_dir = _run_experiment(config)

        pkl_count = len([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
        assert pkl_count == NUM_EPISODES, f"Expected {NUM_EPISODES} episodes, got {pkl_count}"

        early_avg, late_avg, improved = _check_learning_signal(data_dir, NUM_EPISODES)
        assert improved, (
            f"DQN+IC failed to show learning signal: "
            f"early avg={early_avg:.1f}, late avg={late_avg:.1f}"
        )


class TestDQN_GSP:
    """DQN with broadcast GSP — tests GSP prediction pipeline."""

    exp_name = "regression_DQN_GSP"
    port = 55581

    def setup_method(self):
        _check_argos_installed()
        _cleanup_experiment(self.exp_name)

    def teardown_method(self):
        _cleanup_experiment(self.exp_name)

    def test_dqn_gsp_trains_and_learns(self):
        config = _make_config(
            exp_name=self.exp_name,
            learning_scheme="DQN",
            port=self.port,
            gsp=True,
            neighbors=False,
        )
        data_dir = _run_experiment(config)

        pkl_count = len([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
        assert pkl_count == NUM_EPISODES, f"Expected {NUM_EPISODES} episodes, got {pkl_count}"

        early_avg, late_avg, improved = _check_learning_signal(data_dir, NUM_EPISODES)
        assert improved, (
            f"DQN+GSP failed to show learning signal: "
            f"early avg={early_avg:.1f}, late avg={late_avg:.1f}"
        )


class TestDDQN_GSP_N:
    """DDQN with GSP-N — tests neighbor topology + double-Q."""

    exp_name = "regression_DDQN_GSP_N"
    port = 55582

    def setup_method(self):
        _check_argos_installed()
        _cleanup_experiment(self.exp_name)

    def teardown_method(self):
        _cleanup_experiment(self.exp_name)

    def test_ddqn_gsp_n_trains_and_learns(self):
        config = _make_config(
            exp_name=self.exp_name,
            learning_scheme="DDQN",
            port=self.port,
            gsp=True,
            neighbors=True,
        )
        data_dir = _run_experiment(config)

        pkl_count = len([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
        assert pkl_count == NUM_EPISODES, f"Expected {NUM_EPISODES} episodes, got {pkl_count}"

        early_avg, late_avg, improved = _check_learning_signal(data_dir, NUM_EPISODES)
        assert improved, (
            f"DDQN+GSP-N failed to show learning signal: "
            f"early avg={early_avg:.1f}, late avg={late_avg:.1f}"
        )


class TestDDPG_R_GSP_N:
    """DDPG with R-GSP-N — tests continuous actions + LSTM memory."""

    exp_name = "regression_DDPG_R_GSP_N"
    port = 55583

    def setup_method(self):
        _check_argos_installed()
        _cleanup_experiment(self.exp_name)

    def teardown_method(self):
        _cleanup_experiment(self.exp_name)

    def test_ddpg_r_gsp_n_trains_and_learns(self):
        config = _make_config(
            exp_name=self.exp_name,
            learning_scheme="DDPG",
            port=self.port,
            gsp=True,
            neighbors=True,
            recurrent=True,
        )
        # R-GSP-N is slow on CPU (MPS LSTM fallback) — use fewer episodes + smaller batch
        config["NUM_EPISODES"] = NUM_EPISODES_RECURRENT
        config["BATCH_SIZE"] = 16
        config["GSP_BATCH_SIZE"] = 16
        data_dir = _run_experiment(config)

        pkl_count = len([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
        assert pkl_count == NUM_EPISODES_RECURRENT, f"Expected {NUM_EPISODES_RECURRENT} episodes, got {pkl_count}"

        early_avg, late_avg, improved = _check_learning_signal(data_dir, NUM_EPISODES_RECURRENT)
        assert improved, (
            f"DDPG+R-GSP-N failed to show learning signal: "
            f"early avg={early_avg:.1f}, late avg={late_avg:.1f}"
        )


class TestTD3_A_GSP_N:
    """TD3 with A-GSP-N — tests twin critics + transformer attention."""

    exp_name = "regression_TD3_A_GSP_N"
    port = 55584

    def setup_method(self):
        _check_argos_installed()
        _cleanup_experiment(self.exp_name)

    def teardown_method(self):
        _cleanup_experiment(self.exp_name)

    def test_td3_a_gsp_n_trains_and_learns(self):
        config = _make_config(
            exp_name=self.exp_name,
            learning_scheme="TD3",
            port=self.port,
            gsp=True,
            neighbors=True,
            attention=True,
        )
        data_dir = _run_experiment(config)

        pkl_count = len([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
        assert pkl_count == NUM_EPISODES, f"Expected {NUM_EPISODES} episodes, got {pkl_count}"

        early_avg, late_avg, improved = _check_learning_signal(data_dir, NUM_EPISODES)
        assert improved, (
            f"TD3+A-GSP-N failed to show learning signal: "
            f"early avg={early_avg:.1f}, late avg={late_avg:.1f}"
        )


if __name__ == "__main__":
    """Run standalone without pytest for quick testing."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                        help="Which test to run: dqn_ic, dqn_gsp, ddqn_gsp_n, ddpg_r_gsp_n, td3_a_gsp_n, all")
    args = parser.parse_args()

    tests = {
        "dqn_ic": ("DQN", False, False, False, False, 55580),
        "dqn_gsp": ("DQN", True, False, False, False, 55581),
        "ddqn_gsp_n": ("DDQN", True, True, False, False, 55582),
        "ddpg_r_gsp_n": ("DDPG", True, True, True, False, 55583),
        "td3_a_gsp_n": ("TD3", True, True, False, True, 55584),
    }

    to_run = tests.keys() if args.test == "all" else [args.test]

    for name in to_run:
        scheme, gsp, neighbors, recurrent, attention, port = tests[name]
        exp_name = f"regression_{name}"
        print(f"\n{'='*60}")
        print(f"Running: {name} ({scheme} + {'GSP' if gsp else 'IC'}"
              f"{'_N' if neighbors else ''}{'_R' if recurrent else ''}{'_A' if attention else ''})")
        print(f"{'='*60}")

        _cleanup_experiment(exp_name)
        config = _make_config(exp_name, scheme, port, gsp, neighbors, recurrent, attention)
        try:
            data_dir = _run_experiment(config)
            pkl_count = len([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
            early, late, improved = _check_learning_signal(data_dir, NUM_EPISODES)
            status = "PASS" if improved else "FAIL"
            print(f"  Episodes: {pkl_count}/{NUM_EPISODES}")
            print(f"  Early avg reward: {early:.1f}")
            print(f"  Late avg reward:  {late:.1f}")
            print(f"  Learning signal:  {status}")
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            _cleanup_experiment(exp_name)

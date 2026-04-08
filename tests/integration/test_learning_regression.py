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
NUM_EPISODES_RECURRENT = 10  # Was 5 when RDDPG was 200x slower; now vectorized
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


def _linear_regression(values):
    """Fit linear regression and return (slope, r_squared)."""
    if len(values) < 2:
        return 0.0, 0.0
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)
    slope = coeffs[0]
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((np.array(values) - y_pred) ** 2)
    ss_tot = np.sum((np.array(values) - np.mean(values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return slope, r_squared


def _check_learning_signal(data_dir, num_episodes):
    """Verify training improvement via linear regression on rewards, loss, and GSP reward.

    Returns dict with:
        reward_slope, reward_r2: reward trend (positive = improving)
        loss_slope, loss_r2: action network loss trend (negative = converging)
        gsp_slope, gsp_r2: GSP prediction reward trend (positive = better predictions)
        improved: bool, whether reward slope is positive
    """
    pkl_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pkl")],
        key=lambda x: int(x.replace("Data_Episode_", "").replace(".pkl", ""))
    )

    if len(pkl_files) < num_episodes:
        raise RuntimeError(f"Expected {num_episodes} pkl files, found {len(pkl_files)}")

    episode_rewards = []
    episode_losses = []
    episode_gsp_rewards = []

    for f in pkl_files:
        with open(os.path.join(data_dir, f), "rb") as fh:
            data = pickle.load(fh)

            # Total reward (sum across all robots and steps)
            rewards = data.get("reward", data.get("rewards", [0]))
            if isinstance(rewards, list) and rewards and isinstance(rewards[0], (list, np.ndarray)):
                total_reward = sum(sum(r) for r in rewards)
            else:
                total_reward = np.sum(rewards)
            episode_rewards.append(total_reward)

            # Average action network loss (exclude None entries from pre-warmup)
            try:
                losses = data.get("loss", [])
                valid_losses = [float(l) for l in losses if l is not None and np.isfinite(float(l))]
                avg_loss = np.mean(valid_losses) if valid_losses else 0.0
            except (TypeError, ValueError):
                avg_loss = 0.0
            episode_losses.append(avg_loss)

            # Average GSP reward (sum across robots per step, then average)
            try:
                gsp_rewards = data.get("gsp_reward", [])
                if gsp_rewards and isinstance(gsp_rewards[0], (list, np.ndarray)):
                    step_sums = [sum(float(v) for v in r) for r in gsp_rewards]
                    avg_gsp = np.mean(step_sums)
                elif gsp_rewards:
                    avg_gsp = np.mean([float(r) for r in gsp_rewards])
                else:
                    avg_gsp = 0.0
            except (TypeError, ValueError):
                avg_gsp = 0.0
            episode_gsp_rewards.append(avg_gsp)

    reward_slope, reward_r2 = _linear_regression(episode_rewards)
    loss_slope, loss_r2 = _linear_regression(episode_losses)
    gsp_slope, gsp_r2 = _linear_regression(episode_gsp_rewards)

    return {
        "reward_slope": reward_slope, "reward_r2": reward_r2,
        "loss_slope": loss_slope, "loss_r2": loss_r2,
        "gsp_slope": gsp_slope, "gsp_r2": gsp_r2,
        "improved": reward_slope > 0,
    }


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

        result = _check_learning_signal(data_dir, NUM_EPISODES)
        assert result["improved"], (
            f"DQN+IC failed: reward={result['reward_slope']:.0f}, loss={result['loss_slope']:.4f}, gsp={result['gsp_slope']:.4f}"
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

        result = _check_learning_signal(data_dir, NUM_EPISODES)
        assert result["improved"], (
            f"DQN+GSP failed: reward={result['reward_slope']:.0f}, loss={result['loss_slope']:.4f}, gsp={result['gsp_slope']:.4f}"
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

        result = _check_learning_signal(data_dir, NUM_EPISODES)
        assert result["improved"], (
            f"DDQN+GSP-N failed: reward={result['reward_slope']:.0f}, loss={result['loss_slope']:.4f}, gsp={result['gsp_slope']:.4f}"
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
        config["NUM_EPISODES"] = NUM_EPISODES_RECURRENT
        data_dir = _run_experiment(config)

        pkl_count = len([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
        assert pkl_count == NUM_EPISODES_RECURRENT, f"Expected {NUM_EPISODES_RECURRENT} episodes, got {pkl_count}"

        result = _check_learning_signal(data_dir, NUM_EPISODES_RECURRENT)
        assert result["improved"], (
            f"DDPG+R-GSP-N failed: reward={result['reward_slope']:.0f}, loss={result['loss_slope']:.4f}, gsp={result['gsp_slope']:.4f}"
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

        result = _check_learning_signal(data_dir, NUM_EPISODES)
        assert result["improved"], (
            f"TD3+A-GSP-N failed: reward={result['reward_slope']:.0f}, loss={result['loss_slope']:.4f}, gsp={result['gsp_slope']:.4f}"
        )


def _run_single_test(name, scheme, gsp, neighbors, recurrent, attention, port):
    """Run a single test and return results dict."""
    exp_name = f"regression_{name}"
    _cleanup_experiment(exp_name)
    config = _make_config(exp_name, scheme, port, gsp, neighbors, recurrent, attention)
    if recurrent:
        config["NUM_EPISODES"] = NUM_EPISODES_RECURRENT
    target_eps = NUM_EPISODES_RECURRENT if recurrent else NUM_EPISODES
    start = time.time()
    try:
        data_dir = _run_experiment(config)
        pkl_count = len([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
        result = _check_learning_signal(data_dir, pkl_count)
        return {
            "name": name, "status": "PASS" if result["improved"] else "FAIL",
            "episodes": pkl_count, "target": target_eps,
            "slope": result["reward_slope"], "r_squared": result["reward_r2"], "loss_slope": result["loss_slope"], "gsp_slope": result["gsp_slope"], "duration": time.time() - start,
        }
    except Exception as e:
        return {"name": name, "status": "ERROR", "error": str(e),
                "duration": time.time() - start}
    finally:
        _cleanup_experiment(exp_name)


def _run_parallel(tests_to_run, tests_dict):
    """Launch all tests in parallel with live progress display."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    start = time.time()
    results = []
    completed = {}
    lock = threading.Lock()

    # Track targets per test
    targets = {}
    for name in tests_to_run:
        _, _, _, recurrent, _, _ = tests_dict[name]
        targets[name] = NUM_EPISODES_RECURRENT if recurrent else NUM_EPISODES

    def _progress_display():
        """Background thread that prints progress bars every 10 seconds."""
        while len(completed) < len(tests_to_run):
            time.sleep(10)
            elapsed = time.time() - start
            lines = []
            for name in tests_to_run:
                target = targets[name]
                if name in completed:
                    r = completed[name]
                    status = r["status"]
                    bar = "█" * target
                    lines.append(f"  {name:<22s} [{bar}] {target}/{target} {status} ({r['duration']:.0f}s)")
                else:
                    exp_name = f"regression_{name}"
                    data_dir = os.path.join(PROJECT_ROOT, "rl_code", "Data", exp_name, "Data")
                    try:
                        ep_count = len([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
                    except FileNotFoundError:
                        ep_count = 0
                    filled = "█" * ep_count
                    empty = "░" * (target - ep_count)
                    lines.append(f"  {name:<22s} [{filled}{empty}] {ep_count}/{target}")

            # Clear and redraw
            print(f"\n  [{elapsed:.0f}s elapsed]", flush=True)
            for line in lines:
                print(line, flush=True)

    # Start progress thread
    progress_thread = threading.Thread(target=_progress_display, daemon=True)
    progress_thread.start()

    with ThreadPoolExecutor(max_workers=len(tests_to_run)) as executor:
        futures = {}
        for name in tests_to_run:
            scheme, gsp, neighbors, recurrent, attention, port = tests_dict[name]
            f = executor.submit(_run_single_test, name, scheme, gsp, neighbors, recurrent, attention, port)
            futures[f] = name

        for future in as_completed(futures):
            result = future.result()
            name = result["name"]
            with lock:
                completed[name] = result
            if result["status"] == "ERROR":
                print(f"\n  ✗ {name} — ERROR: {result['error']}", flush=True)
            else:
                print(f"\n  ✓ {name} — {result['status']} in {result['duration']:.0f}s "
                      f"(slope={result['slope']:.1f}, R²={result['r_squared']:.3f})", flush=True)
            results.append(result)

    total = time.time() - start
    return results, total


if __name__ == "__main__":
    """Run standalone without pytest.

    Usage:
        python test_learning_regression.py                    # all sequential
        python test_learning_regression.py --parallel         # all parallel
        python test_learning_regression.py --test dqn_ic      # single test
        python test_learning_regression.py --parallel --test dqn_ic,td3_a_gsp_n  # subset parallel
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="all",
                        help="Comma-separated test names, or 'all'")
    parser.add_argument("--parallel", action="store_true",
                        help="Run tests in parallel (each on its own port)")
    args = parser.parse_args()

    tests = {
        "dqn_ic": ("DQN", False, False, False, False, 55580),
        "dqn_gsp": ("DQN", True, False, False, False, 55581),
        "ddqn_gsp_n": ("DDQN", True, True, False, False, 55582),
        "ddpg_r_gsp_n": ("DDPG", True, True, True, False, 55583),
        "td3_a_gsp_n": ("TD3", True, True, False, True, 55584),
    }

    to_run = list(tests.keys()) if args.test == "all" else args.test.split(",")

    if args.parallel:
        print(f"Running {len(to_run)} tests in PARALLEL...", flush=True)
        results, total = _run_parallel(to_run, tests)

        print(f"\n{'='*70}")
        print(f"PARALLEL RESULTS: {total:.0f}s ({total/60:.1f} min)")
        print(f"{'='*70}")
        print(f"{'Test':<22s} {'Status':<8s} {'Episodes':<10s} {'Duration':<10s} {'Rwd Slope':<11s} {'Loss Slope':<12s} {'GSP Slope':<12s}")
        print(f"{'-'*74}")
        for r in sorted(results, key=lambda x: x["name"]):
            if r["status"] == "ERROR":
                print(f"{r['name']:<22s} {'ERROR':<8s} {'—':10s} {r['duration']:<10.0f}s {r['error']}")
            else:
                print(f"{r['name']:<22s} {r['status']:<8s} {r['episodes']:<10d} {r['duration']:<10.0f}s {r['slope']:<11.0f} {r.get('loss_slope',0):<12.4f} {r.get('gsp_slope',0):<12.4f}")

        passed = sum(1 for r in results if r["status"] == "PASS")
        print(f"\n{passed}/{len(results)} PASSED")
    else:
        for name in to_run:
            scheme, gsp, neighbors, recurrent, attention, port = tests[name]
            print(f"\n{'='*60}")
            print(f"Running: {name} ({scheme} + {'GSP' if gsp else 'IC'}"
                  f"{'_N' if neighbors else ''}{'_R' if recurrent else ''}{'_A' if attention else ''})")
            print(f"{'='*60}")

            result = _run_single_test(name, scheme, gsp, neighbors, recurrent, attention, port)
            if result["status"] == "ERROR":
                print(f"  ERROR: {result['error']}")
            else:
                print(f"  Episodes: {result['episodes']}/{result.get('target', NUM_EPISODES)}")
                print(f"  Slope: {result['slope']:.1f} (positive = learning)")
                print(f"  R²: {result['r_squared']:.3f}")
                print(f"  Learning signal:  {result['status']}")
                print(f"  Duration:         {result['duration']:.0f}s")

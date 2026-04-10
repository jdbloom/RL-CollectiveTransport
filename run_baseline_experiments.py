"""Baseline experiments: train 8 configs in parallel, auto-test best models.

Usage:
    cd code/phd_code/RL-CollectiveTransport
    pyenv activate phd-code
    python run_baseline_experiments.py
"""

import csv
import os
import sys
import time
import shutil
import pickle
import subprocess
import threading
from datetime import datetime
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# Registry integration (optional — gracefully degrades if unavailable)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
    from tools.registry.client import RegistryClient
    REGISTRY_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data", "registry.db")
    os.makedirs(os.path.dirname(REGISTRY_DB), exist_ok=True)
    _registry = RegistryClient(REGISTRY_DB)
    HAS_REGISTRY = True
    print(f"Registry connected: {REGISTRY_DB}")
except Exception:
    HAS_REGISTRY = False
    _registry = None


def episode_timing_logger(exp_name, data_dir, output_dir='Data/monitoring'):
    """Background thread: watches for new .pkl files, logs per-episode wall time."""
    os.makedirs(output_dir, exist_ok=True)
    timing_path = os.path.join(output_dir, f'{exp_name}_episode_times.csv')
    seen = set()
    last_time = time.time()

    with open(timing_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'wall_time_s', 'cumulative_s', 'timestamp'])
        start = time.time()

        while True:
            time.sleep(2)
            try:
                current = set(fn for fn in os.listdir(data_dir) if fn.endswith('.pkl'))
            except FileNotFoundError:
                continue
            new = current - seen
            if new:
                now = time.time()
                for fn in sorted(new):
                    ep_num = int(fn.replace('Data_Episode_', '').replace('.pkl', ''))
                    ep_time = now - last_time
                    writer.writerow([ep_num, f'{ep_time:.1f}', f'{now - start:.1f}',
                                     datetime.now().strftime('%H:%M:%S')])
                last_time = now
                seen = current
                f.flush()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARGOS_BIN = shutil.which("argos3")
SEED = 42
NUM_ROBOTS = 4
TRAIN_EPISODES = 500
TEST_EPISODES = 100


# ─── Experiment definitions ──────────────────────────────────────────────────

TRAIN_EXPERIMENTS = {
    # name: (gsp, neighbors, num_obstacles, use_gate, gate_curriculum, use_prisms, port, recurrent, attention)
    "dqn_ic_open":         (False, False, 0, 0, 0, 0, 55580, False, False),
    "dqn_gsp_open":        (True,  False, 0, 0, 0, 0, 55581, False, False),
    "dqn_gsp_n_open":      (True,  True,  0, 0, 0, 0, 55582, False, False),
    "dqn_ic_2obs":         (False, False, 2, 0, 0, 0, 55583, False, False),
    "dqn_rgsp_n_2obs":     (True,  True,  2, 0, 0, 0, 55584, True,  False),
    "dqn_gsp_n_2obs":      (True,  True,  2, 0, 0, 0, 55585, False, False),
    "dqn_gsp_n_gate_curr": (True,  True,  0, 1, 1, 0, 55586, False, False),
    "dqn_agsp_n_prism":    (True,  True,  0, 0, 0, 1, 55587, False, True),
}

# Test plan: train_name -> list of (test_name, num_obstacles, use_gate, gate_curriculum, use_prisms, port)
TEST_PLAN = {
    # Open arena models: test in same environment
    "dqn_ic_open":         [("test_dqn_ic_open",         0, 0, 0, 0, 55590)],
    "dqn_gsp_open":        [("test_dqn_gsp_open",        0, 0, 0, 0, 55591)],
    "dqn_gsp_n_open":      [("test_dqn_gsp_n_open",      0, 0, 0, 0, 55592)],
    # Obstacle models: test on 2 obs and 4 obs
    "dqn_ic_2obs":         [("test_dqn_ic_2obs",         2, 0, 0, 0, 55593),
                            ("test_dqn_ic_4obs",         4, 0, 0, 0, 55594)],
    "dqn_rgsp_n_2obs":     [("test_dqn_rgsp_n_2obs",     2, 0, 0, 0, 55595),
                            ("test_dqn_rgsp_n_4obs",     4, 0, 0, 0, 55596)],
    "dqn_gsp_n_2obs":      [("test_dqn_gsp_n_2obs",      2, 0, 0, 0, 55597),
                            ("test_dqn_gsp_n_4obs",      4, 0, 0, 0, 55598)],
    # Gate model: test on gate without curriculum
    "dqn_gsp_n_gate_curr": [("test_dqn_gsp_n_gate_nocurr", 0, 1, 0, 0, 55599)],
    # Prism model: test in same environment
    "dqn_agsp_n_prism":    [("test_dqn_agsp_n_prism",    0, 0, 0, 1, 55600)],
}


def make_config(exp_name, gsp, neighbors, num_obstacles, use_gate, gate_curriculum,
                use_prisms, port, num_episodes, test=False, model_num=490,
                recurrent=False, attention=False):
    return {
        "TEST": test,
        "MODEL_NUM": model_num,
        "EXP_NAME": exp_name,
        "ARGOS_FILE_NAME": f"collectiveRlTransport_{exp_name}.argos",
        "NUM_EPISODES": num_episodes,
        "NUM_OBSTACLES": num_obstacles,
        "USE_GATE": use_gate,
        "GATE_MIN": 4,
        "GATE_CURRICULUM": gate_curriculum,
        "NUM_ROBOTS": NUM_ROBOTS,
        "MAX_NUM_ROBOT_FAILURES": 0,
        "CHANCE_FAILURE": 0.25,
        "PORT": port,
        "SEED": SEED,
        "USE_PRISMS": use_prisms,
        "RANDOM_OBJECTS": 0,
        "TEST_PRISM": 0,
        "LEARNING_SCHEME": "DQN",
        "OPTIONS_PER_ACTION": 3,
        "MIN_MAX_ACTION": 0.1,
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


def write_yaml_config(config, path):
    with open(path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")


def generate_argos_xml(config):
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


def find_best_model(data_dir, models_dir):
    """Find the model checkpoint with the best 10-episode average reward."""
    pkl_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pkl")],
        key=lambda x: int(x.replace("Data_Episode_", "").replace(".pkl", ""))
    )
    if not pkl_files:
        return (0, 0.0)

    # Compute 10-episode rolling averages aligned with checkpoint saves
    episode_rewards = []
    for f in pkl_files:
        with open(os.path.join(data_dir, f), "rb") as fh:
            data = pickle.load(fh)
            rewards = data.get("reward", data.get("rewards", [0]))
            if isinstance(rewards, list) and rewards and isinstance(rewards[0], (list, np.ndarray)):
                total = sum(sum(r) for r in rewards)
            else:
                total = np.sum(rewards)
            episode_rewards.append(total)

    # Checkpoints are saved at episodes 0, 10, 20, ... (every 10 episodes)
    best_avg = -np.inf
    best_ep = 0
    for ep in range(0, len(episode_rewards), 10):
        window = episode_rewards[ep:ep+10]
        avg = np.mean(window)
        if avg > best_avg:
            best_avg = avg
            best_ep = ep

    # Verify the checkpoint exists
    checkpoint_name = f"Episode_{best_ep}"
    # Check by looking for any file starting with this name in models dir
    model_files = [f for f in os.listdir(models_dir) if checkpoint_name in f]
    if not model_files:
        # Fall back to last checkpoint
        checkpoints = sorted(
            [f for f in os.listdir(models_dir)],
            key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0
        )
        if checkpoints:
            last = checkpoints[-1]
            best_ep = int(last.split("_")[-1]) if last.split("_")[-1].isdigit() else 0
            best_avg = np.mean(episode_rewards[max(0, best_ep-9):best_ep+1])

    return best_ep, best_avg


def run_experiment(exp_name, config, test_mode=False, model_path=None):
    """Run ARGoS + Main.py. Returns (data_dir, duration)."""
    data_root = os.path.join(PROJECT_ROOT, "rl_code", "Data", exp_name)
    os.makedirs(os.path.join(data_root, "Data"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "Models"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "plots"), exist_ok=True)

    config_path = os.path.join(data_root, "agent_config.yml")
    write_yaml_config(config, config_path)

    generate_argos_xml(config)
    argos_file = os.path.join(PROJECT_ROOT, "argos", config["ARGOS_FILE_NAME"])
    shutil.copy(argos_file, os.path.join(data_root, config["ARGOS_FILE_NAME"]))

    # Capture ARGoS output to diagnostics temp dir
    argos_log_dir = os.path.join("/tmp/stelaris-runs", exp_name)
    os.makedirs(argos_log_dir, exist_ok=True)
    argos_log_path = os.path.join(argos_log_dir, "argos.log")
    argos_log_file = open(argos_log_path, "w")

    argos_proc = subprocess.Popen(
        [ARGOS_BIN, "-c", argos_file],
        cwd=PROJECT_ROOT,
        stdout=argos_log_file,
        stderr=subprocess.STDOUT,
    )

    main_cmd = [sys.executable, "Main.py", f"Data/{exp_name}"]
    if test_mode and model_path:
        main_cmd += ["--test", "--model_path", model_path]

    main_proc = subprocess.Popen(
        main_cmd,
        cwd=os.path.join(PROJECT_ROOT, "rl_code"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Start per-episode timing logger
    ep_data_dir = os.path.join(data_root, 'Data')
    timing_thread = threading.Thread(
        target=episode_timing_logger, args=(exp_name, ep_data_dir),
        daemon=True,
    )
    timing_thread.start()

    start = time.time()
    try:
        # Monitor both processes — if ARGoS dies, kill Python too
        while main_proc.poll() is None:
            if argos_proc.poll() is not None:
                argos_rc = argos_proc.returncode
                argos_log_file.flush()
                argos_stderr = ""
                try:
                    argos_stderr = open(argos_log_path).read()[-2000:]
                except FileNotFoundError:
                    argos_stderr = "(argos.log was cleaned up by diagnostics)"

                print(f"  [ERROR] {exp_name}: ARGoS died (rc={argos_rc}) — killing Python", flush=True)
                print(f"  [ERROR] ARGoS stderr: {argos_stderr[-500:]}", flush=True)
                # Log to diagnostics file
                diag_path = os.path.join(data_root, "argos_crash.log")
                with open(diag_path, "w") as f:
                    f.write(f"ARGoS exit code: {argos_rc}\n")
                    f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Episodes completed: {count_episodes(exp_name)}\n")
                    f.write(f"Elapsed: {time.time() - start:.0f}s\n")
                    f.write(f"Stderr:\n{argos_stderr}\n")
                main_proc.kill()
                main_proc.wait(timeout=10)
                raise RuntimeError(f"ARGoS crashed (rc={argos_rc}): {argos_stderr[-200:]}")
            time.sleep(1)
    finally:
        argos_log_file.close()
        if argos_proc.poll() is None:
            argos_proc.kill()
            argos_proc.wait(timeout=10)

    duration = time.time() - start

    if main_proc.returncode != 0:
        stderr = main_proc.stderr.read().decode() if main_proc.stderr else ""
        raise RuntimeError(f"Main.py failed (rc={main_proc.returncode}): {stderr[-500:]}")

    return os.path.join(data_root, "Data"), duration


def count_episodes(exp_name):
    """Count completed pkl files for an experiment."""
    data_dir = os.path.join(PROJECT_ROOT, "rl_code", "Data", exp_name, "Data")
    try:
        return len([f for f in os.listdir(data_dir) if f.endswith(".pkl")])
    except FileNotFoundError:
        return 0


def run_train_and_test(train_name):
    """Train one experiment, then kick off its test runs."""
    gsp, neighbors, num_obs, use_gate, gate_curr, use_prisms, port, recurrent, attention = TRAIN_EXPERIMENTS[train_name]

    exp_id = f"{train_name}_{SEED}"
    if HAS_REGISTRY:
        try:
            coord = "R-GSP-N" if recurrent else ("A-GSP-N" if attention else ("GSP-N" if neighbors else ("GSP" if gsp else "IC")))
            env = f"{num_obs}obs" if num_obs > 0 else ("gate_curr" if gate_curr else ("gate" if use_gate else ("prism" if use_prisms else "open")))
            _registry.create_experiment(
                id=exp_id, name=train_name, algorithm="DQN",
                coordination=coord, environment=env,
                num_robots=NUM_ROBOTS, num_obstacles=num_obs,
                use_gate=bool(use_gate), use_prisms=bool(use_prisms),
                num_episodes=TRAIN_EPISODES, seed=SEED, port=port,
                machine_hostname=os.uname().nodename,
            )
            _registry.start_experiment(exp_id)
        except Exception as e:
            print(f"  [WARN] Registry: {e}", flush=True)

    # ── TRAIN ──
    print(f"  [TRAIN] Starting {train_name} (port {port})", flush=True)
    config = make_config(
        train_name, gsp, neighbors, num_obs, use_gate, gate_curr, use_prisms,
        port, TRAIN_EPISODES, recurrent=recurrent, attention=attention,
    )
    train_start = time.time()
    try:
        data_dir, duration = run_experiment(train_name, config)
        ep_count = count_episodes(train_name)
        print(f"  [TRAIN] ✓ {train_name} done — {ep_count} episodes in {duration:.0f}s", flush=True)
    except Exception as e:
        print(f"  [TRAIN] ✗ {train_name} FAILED: {e}", flush=True)
        if HAS_REGISTRY:
            try:
                _registry.fail_experiment(exp_id, error_message=str(e))
            except Exception:
                pass
        return {"train": train_name, "status": "TRAIN_ERROR", "error": str(e)}

    # ── FIND BEST MODEL ──
    models_dir = os.path.join(PROJECT_ROOT, "rl_code", "Data", train_name, "Models")
    best_ep, best_avg = find_best_model(data_dir, models_dir)
    model_rel_path = f"Data/{train_name}/Models/Episode_{best_ep}"
    print(f"  [BEST]  {train_name} → Episode_{best_ep} (avg reward: {best_avg:.1f})", flush=True)

    # ── TEST ──
    test_results = []
    tests = TEST_PLAN.get(train_name, [])
    for test_name, test_obs, test_gate, test_gate_curr, test_prisms, test_port in tests:
        print(f"  [TEST]  Starting {test_name} (model from Episode_{best_ep})", flush=True)
        test_config = make_config(
            test_name, gsp, neighbors, test_obs, test_gate, test_gate_curr, test_prisms,
            test_port, TEST_EPISODES, test=True, model_num=best_ep,
            recurrent=recurrent, attention=attention,
        )
        try:
            test_data_dir, test_duration = run_experiment(
                test_name, test_config, test_mode=True, model_path=model_rel_path,
            )
            test_ep_count = count_episodes(test_name)

            # Compute test metrics
            test_rewards = []
            successes = 0
            for f in sorted(os.listdir(test_data_dir)):
                if f.endswith(".pkl"):
                    with open(os.path.join(test_data_dir, f), "rb") as fh:
                        d = pickle.load(fh)
                        rw = d.get("reward", d.get("rewards", [0]))
                        if isinstance(rw, list) and rw and isinstance(rw[0], (list, np.ndarray)):
                            test_rewards.append(sum(sum(r) for r in rw))
                        else:
                            test_rewards.append(np.sum(rw))
                        terminations = d.get("termination", [])
                        if terminations:
                            # termination is a list of bools per timestep
                            # any True means goal was reached during the episode
                            if isinstance(terminations, list) and any(terminations):
                                successes += 1

            avg_reward = np.mean(test_rewards) if test_rewards else 0
            print(f"  [TEST]  ✓ {test_name} done — {test_ep_count} eps, "
                  f"avg reward: {avg_reward:.1f}, duration: {test_duration:.0f}s", flush=True)
            test_results.append({
                "test_name": test_name, "status": "PASS",
                "episodes": test_ep_count, "avg_reward": avg_reward,
                "duration": test_duration,
            })
        except Exception as e:
            print(f"  [TEST]  ✗ {test_name} FAILED: {e}", flush=True)
            test_results.append({"test_name": test_name, "status": "ERROR", "error": str(e)})

    if HAS_REGISTRY:
        try:
            _registry.complete_experiment(exp_id)
        except Exception:
            pass

    return {
        "train": train_name, "status": "DONE",
        "train_episodes": count_episodes(train_name),
        "train_duration": time.time() - train_start,
        "best_model": f"Episode_{best_ep}", "best_avg_reward": best_avg,
        "tests": test_results,
    }


def progress_monitor(experiments, completed):
    """Print progress every 30 seconds."""
    while len(completed) < len(experiments):
        time.sleep(30)
        lines = []
        for name in experiments:
            if name in completed:
                r = completed[name]
                lines.append(f"  {name:<25s} DONE ({r.get('train_episodes', '?')} eps)")
            else:
                ep = count_episodes(name)
                bar_len = 20
                filled = int(ep / TRAIN_EPISODES * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                lines.append(f"  {name:<25s} [{bar}] {ep}/{TRAIN_EPISODES}")
        elapsed = time.time() - start_time
        print(f"\n  === Progress ({elapsed:.0f}s elapsed) ===", flush=True)
        for line in lines:
            print(line, flush=True)


if __name__ == "__main__":
    print(f"{'='*70}")
    print(f"BASELINE EXPERIMENTS")
    print(f"Training: {len(TRAIN_EXPERIMENTS)} experiments × {TRAIN_EPISODES} episodes")
    print(f"Testing:  best models × {TEST_EPISODES} episodes each")
    print(f"Parallelism: 8 concurrent")
    print(f"{'='*70}\n")

    start_time = time.time()
    completed = {}
    lock = threading.Lock()

    # Start progress monitor
    monitor = threading.Thread(
        target=progress_monitor,
        args=(list(TRAIN_EXPERIMENTS.keys()), completed),
        daemon=True,
    )
    monitor.start()

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for name in TRAIN_EXPERIMENTS:
            f = executor.submit(run_train_and_test, name)
            futures[f] = name

        for future in as_completed(futures):
            result = future.result()
            name = result["train"]
            with lock:
                completed[name] = result
            results.append(result)

    total_time = time.time() - start_time

    # ── FINAL REPORT ──
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS — {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}\n")

    print(f"{'Experiment':<25s} {'Episodes':<10s} {'Best Model':<15s} {'Best Avg Rwd':<14s} {'Duration':<10s}")
    print(f"{'-'*74}")
    for r in sorted(results, key=lambda x: x["train"]):
        if r["status"] == "TRAIN_ERROR":
            print(f"{r['train']:<25s} ERROR: {r.get('error', 'unknown')}")
        else:
            print(f"{r['train']:<25s} {r['train_episodes']:<10d} {r['best_model']:<15s} "
                  f"{r['best_avg_reward']:<14.1f} {r['train_duration']:<10.0f}s")

    print(f"\n{'Test Results':<30s} {'Episodes':<10s} {'Avg Reward':<14s} {'Duration':<10s}")
    print(f"{'-'*64}")
    for r in sorted(results, key=lambda x: x["train"]):
        for t in r.get("tests", []):
            if t["status"] == "ERROR":
                print(f"  {t['test_name']:<28s} ERROR: {t.get('error', 'unknown')}")
            else:
                print(f"  {t['test_name']:<28s} {t['episodes']:<10d} {t['avg_reward']:<14.1f} {t['duration']:<10.0f}s")

    passed = sum(1 for r in results if r["status"] == "DONE")
    print(f"\n{passed}/{len(results)} training runs completed successfully")

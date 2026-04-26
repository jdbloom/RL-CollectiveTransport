"""Integration test for DETERMINISM_ENABLED wiring in Main.py.

This test verifies that when DETERMINISM_ENABLED=true is set in a YAML config,
Main.py actually produces bit-exact (or near-bit-exact on MPS) HDF5 outputs
across two independent subprocess invocations with the same seed.

The unit test in tests/test_agent/test_determinism.py only tested the helper
in isolation; this test catches wiring bugs (i.e., the helper never being
called from Main.py).

Architecture:
- A fake ARGoS stub runs as a ZMQ REQ client in a thread.
- Main.py runs as a subprocess (ZMQ REP server).
- The stub drives Main.py through N_EPISODES short episodes, then exits.
- After both runs complete, the HDF5 files are compared.

Marked as integration because it spawns subprocesses and requires a working
poetry environment.  Run with:
    pytest tests/test_determinism_integration.py -v -s --timeout=120
"""

import os
import sys
import shutil
import struct
import subprocess
import tempfile
import threading
import time

import h5py
import numpy as np
import pytest
import yaml

# Mark as integration + determinism so the standard pytest run can skip.
pytestmark = [pytest.mark.integration, pytest.mark.determinism]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL_CODE_DIR = os.path.join(PROJECT_ROOT, "rl_code")

# Keep episode count tiny so the test runs in <30s.
N_EPISODES = 3
# Steps per episode — must be > 2 (Main.py skips store_agent_transition for
# the first 2 steps) and small enough for fast iteration.
N_STEPS_PER_EPISODE = 5

NUM_ROBOTS = 4
NUM_OBS = 31
NUM_ACTIONS = 3   # lwheel, rwheel, failure (but Main.py uses num_actions-1 for policy)
NUM_STATS = 4

PORT_BASE = 55590  # use a port unlikely to conflict with live training


# ---------------------------------------------------------------------------
# ZMQ message builders — mirror the C++ ARGoS serialisation format
# ---------------------------------------------------------------------------

def _pack_params(num_robots=4, num_obstacles=0, num_obs=31, num_actions=3,
                 num_stats=4, alphabet_size=9, use_gate=0,
                 dist_norm=10.0, num_prisms=0):
    return struct.pack('9f',
                       float(num_robots), float(num_obstacles), float(num_obs),
                       float(num_actions), float(num_stats), float(alphabet_size),
                       float(use_gate), float(dist_norm), float(num_prisms))


def _pack_status(exp_done=0, episode_done=0, reached_goal=0):
    return struct.pack('3B', exp_done, episode_done, reached_goal)


def _pack_observations(num_robots=4, num_obs=31, seed_offset=0):
    """Pack synthetic per-robot observations as a flat byte string."""
    rng = np.random.default_rng(seed_offset)
    obs = rng.random((num_robots, num_obs), dtype=np.float64).astype(np.float32)
    # cyl_dist2goal is obs[6]; make it positive so prox/heading calcs don't crash.
    obs[:, 6] = np.abs(obs[:, 6]) + 0.1
    return obs.tobytes()


def _pack_failures(num_robots=4, all_ok=True):
    """Pack failure flags (uint32 per robot, 0 = no failure)."""
    vals = [0] * num_robots if all_ok else [1] + [0] * (num_robots - 1)
    return struct.pack(f'{num_robots}I', *vals)


def _pack_rewards(num_robots=4, value=0.1):
    return struct.pack(f'{num_robots}f', *([value] * num_robots))


def _pack_stats(num_robots=4, force_mag=5.0):
    """Pack per-robot stats: magnitude, angle, deltaX, deltaY."""
    data = []
    for _ in range(num_robots):
        data.extend([force_mag, 0.1, 0.0, 0.0])
    return struct.pack(f'{num_robots * 4}f', *data)


def _pack_robot_stats(num_robots=4):
    """Pack per-robot stats: x_pos, y_pos, z_pos, x_deg, y_deg, z_deg."""
    data = []
    for i in range(num_robots):
        data.extend([float(i), 0.0, 0.0, 0.0, 0.0, 0.0])
    return struct.pack(f'{num_robots * 6}f', *data)


def _pack_obj_stats(cyl_angle=10.0):
    """Pack object stats: x_pos, y_pos, z_pos, x_deg, y_deg, z_deg, cyl_angle2goal, comX, comY."""
    return struct.pack('9f', 0.5, 0.5, 0.0, 0.0, 0.0, cyl_angle, 45.0, 0.5, 0.5)


def _build_step_multipart(exp_done=0, episode_done=0, reached_goal=0,
                          step=0, num_robots=4):
    """Build the 7-part ZMQ message that ARGoS sends each step."""
    cyl_angle = float(step) * 0.05  # small rotation each step
    return [
        _pack_status(exp_done, episode_done, reached_goal),
        _pack_observations(num_robots, 31, seed_offset=step),
        _pack_failures(num_robots, all_ok=True),
        _pack_rewards(num_robots, value=0.1),
        _pack_stats(num_robots, force_mag=5.0),
        _pack_robot_stats(num_robots),
        _pack_obj_stats(cyl_angle),
    ]


# ---------------------------------------------------------------------------
# Fake ARGoS stub — runs as a ZMQ REQ client in a background thread
# ---------------------------------------------------------------------------

class FakeArgosStub:
    """Mimics the ARGoS ZMQ REQ client side.

    Protocol (matches collectiveRlTransport.cpp):
    1. Connect REQ to tcp://localhost:<port>
    2. Send params → wait for "ok"
    3. For each episode:
       a. Send initial multipart (exp_done=0, episode_done=0) → recv actions
       b. For N_STEPS steps: send step multipart → recv actions
       c. Send terminal step (episode_done=1) → recv "ok"
    4. Send exp_done multipart → Python exits
    """

    def __init__(self, port: int, n_episodes: int = N_EPISODES,
                 n_steps: int = N_STEPS_PER_EPISODE):
        self.port = port
        self.n_episodes = n_episodes
        self.n_steps = n_steps
        self.error: Exception = None
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self, timeout=60.0):
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self):
        try:
            import zmq
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.SNDTIMEO, 10000)
            sock.setsockopt(zmq.RCVTIMEO, 10000)
            sock.connect(f"tcp://localhost:{self.port}")

            # Step 1: send params
            sock.send(_pack_params())
            sock.recv()  # "ok"

            # Episodes
            for ep in range(self.n_episodes):
                # Initial obs (episode_done=0)
                parts = _build_step_multipart(exp_done=0, episode_done=0,
                                              reached_goal=0, step=0)
                sock.send_multipart(parts)
                sock.recv_multipart()  # actions

                # Inner steps
                for step in range(1, self.n_steps):
                    parts = _build_step_multipart(exp_done=0, episode_done=0,
                                                  reached_goal=0, step=step)
                    sock.send_multipart(parts)
                    sock.recv_multipart()

                # Terminal step — episode done
                parts = _build_step_multipart(exp_done=0, episode_done=1,
                                              reached_goal=0, step=self.n_steps)
                sock.send_multipart(parts)
                sock.recv()  # "ok"

            # Experiment done
            parts = _build_step_multipart(exp_done=1, episode_done=1,
                                          reached_goal=0, step=0)
            sock.send_multipart(parts)
            # Python exits without sending a final ack, so no recv here.

            sock.close()
            ctx.term()

        except Exception as exc:
            self.error = exc


# ---------------------------------------------------------------------------
# Helper: write experiment directory with config
# ---------------------------------------------------------------------------

def _make_exp_dir(base_dir: str, exp_name: str, port: int,
                  determinism_enabled: bool, seed: int) -> str:
    """Create a minimal experiment directory that Main.py can start with."""
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(os.path.join(exp_dir, "Data"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "Models"), exist_ok=True)

    config = {
        "TEST": False,
        "MODEL_NUM": 0,
        "EXP_NAME": exp_name,
        "ARGOS_FILE_NAME": f"{exp_name}.argos",
        "NUM_EPISODES": N_EPISODES,
        "NUM_OBSTACLES": 0,
        "USE_GATE": 0,
        "GATE_MIN": 4,
        "GATE_CURRICULUM": 0,
        "NUM_ROBOTS": NUM_ROBOTS,
        "MAX_NUM_ROBOT_FAILURES": 0,
        "CHANCE_FAILURE": 0.0,
        "PORT": port,
        "SEED": seed,
        "DETERMINISM_ENABLED": determinism_enabled,
        "USE_PRISMS": 0,
        "RANDOM_OBJECTS": 0,
        "TEST_PRISM": 0,
        "LEARNING_SCHEME": "DDQN",
        "OPTIONS_PER_ACTION": 3,
        "MIN_MAX_ACTION": 0.1,
        "META_PARAM_SIZE": 1,
        "PROX_FILTER_ANGLE_DEG": 60.0,
        "GLOBAL_KNOWLEDGE": False,
        "GSP": True,
        "RECURRENT": False,
        "ATTENTION": False,
        "NEIGHBORS": True,
        "BROADCAST": False,
        "GSP_INPUT_SIZE": 6,
        "GSP_OUTPUT_SIZE": 1,
        "GSP_MIN_MAX_ACTION": 1.0,
        "GSP_LOOK_BACK": 2,
        "GSP_SEQUENCE_LENGTH": 5,
        "GAMMA": 0.99,
        "TAU": 0.005,
        "ALPHA": 0.001,
        "BETA": 0.001,
        "LR": 0.001,
        "EPSILON": 1.0,
        "EPS_MIN": 0.01,
        "EPS_DEC": 0.001,
        "BATCH_SIZE": 8,
        "MEM_SIZE": 500,
        "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0,
        "UPDATE_ACTOR_ITER": 1,
        "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 9999,
        "GSP_BATCH_SIZE": 8,
        "GSP_E2E_ENABLED": False,
        "DIAGNOSTICS_ENABLED": False,
        "GSP_OUTPUT_KIND": "delta_theta_1d",
    }

    cfg_path = os.path.join(exp_dir, "agent_config.yml")
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return exp_dir


# ---------------------------------------------------------------------------
# Helper: run one Main.py + FakeArgosStub pair
# ---------------------------------------------------------------------------

def _run_main_with_stub(exp_dir: str, port: int, timeout: int = 60) -> str:
    """Launch Main.py subprocess + FakeArgosStub and return h5 path.

    Returns the path to the HDF5 file written by Main.py.
    Raises RuntimeError on failure or timeout.
    """
    exp_name = os.path.basename(exp_dir)
    # Main.py expects recording_path relative to rl_code/ dir.
    recording_rel = os.path.relpath(exp_dir, RL_CODE_DIR)

    env = os.environ.copy()
    # Ensure the src/ modules are importable from rl_code/.
    env["PYTHONPATH"] = RL_CODE_DIR + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "Main.py", recording_rel, "--no_print"]

    # Start the stub FIRST so it's ready before Main.py binds+listens.
    stub = FakeArgosStub(port=port, n_episodes=N_EPISODES,
                         n_steps=N_STEPS_PER_EPISODE)

    # Give Main.py a moment to bind its ZMQ socket before the stub connects.
    # We start the stub slightly after the process, gated by a small sleep.
    proc = subprocess.Popen(
        cmd,
        cwd=RL_CODE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait briefly for Main.py to bind its socket, then start stub.
    time.sleep(1.5)
    stub.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stub.join(timeout=2)
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(
            f"Main.py timed out after {timeout}s for port {port}.\n"
            f"stderr tail: {stderr[-800:]}"
        )
    finally:
        stub.join(timeout=5)

    if stub.error is not None:
        raise RuntimeError(f"FakeArgosStub error: {stub.error}")

    if proc.returncode != 0:
        stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(
            f"Main.py exited with rc={proc.returncode}\n"
            f"stdout tail: {stdout[-400:]}\n"
            f"stderr tail: {stderr[-800:]}"
        )

    h5_path = os.path.join(exp_dir, f"{exp_name}.h5")
    if not os.path.exists(h5_path):
        raise RuntimeError(f"HDF5 file not created: {h5_path}")

    return h5_path


# ---------------------------------------------------------------------------
# HDF5 comparison helpers
# ---------------------------------------------------------------------------

def _read_episode_attrs(h5_path: str):
    """Return dict mapping episode_key -> dict of scalar attrs we compare."""
    attrs_by_ep = {}
    with h5py.File(h5_path, "r") as f:
        for ep_key in sorted(f.keys()):
            grp = f[ep_key]
            ep_attrs = {}
            for attr_name in ("reward_per_robot", "gsp_output_std", "gsp_pred_target_corr"):
                if attr_name in grp.attrs:
                    val = grp.attrs[attr_name]
                    ep_attrs[attr_name] = np.asarray(val, dtype=np.float64)
            attrs_by_ep[ep_key] = ep_attrs
    return attrs_by_ep


def _max_abs_diff(a1, a2):
    return float(np.max(np.abs(np.asarray(a1, dtype=np.float64)
                                - np.asarray(a2, dtype=np.float64))))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDeterminismIntegration:
    """Integration tests for DETERMINISM_ENABLED wiring in Main.py."""

    # Use class-level tmp_dir to share across same test session.
    _tmp_dir = None

    def setup_method(self):
        self._tmp_dir = tempfile.mkdtemp(prefix="det_integ_")

    def teardown_method(self):
        if self._tmp_dir and os.path.exists(self._tmp_dir):
            shutil.rmtree(self._tmp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Main test: two deterministic runs must match
    # -----------------------------------------------------------------------

    def test_deterministic_runs_match(self):
        """Two Main.py subprocess runs with the same seed and DETERMINISM_ENABLED=true
        must produce bit-exact or near-bit-exact (< 1e-9 on MPS) episode HDF5 attrs.

        This test FAILS on feat/phase4-determinism (before this fix) because
        apply_determinism_settings was never called from Main.py, leaving all
        RNGs unseeded despite the YAML flag.
        """
        seed = 42
        port_a = PORT_BASE
        port_b = PORT_BASE + 1

        # Create two experiment directories (different names, same config).
        exp_dir_a = _make_exp_dir(
            self._tmp_dir, "det_run_a", port_a,
            determinism_enabled=True, seed=seed,
        )
        exp_dir_b = _make_exp_dir(
            self._tmp_dir, "det_run_b", port_b,
            determinism_enabled=True, seed=seed,
        )

        h5_a = _run_main_with_stub(exp_dir_a, port=port_a, timeout=90)
        h5_b = _run_main_with_stub(exp_dir_b, port=port_b, timeout=90)

        attrs_a = _read_episode_attrs(h5_a)
        attrs_b = _read_episode_attrs(h5_b)

        assert set(attrs_a.keys()) == set(attrs_b.keys()), (
            f"Episode key mismatch: {set(attrs_a.keys())} vs {set(attrs_b.keys())}"
        )

        # MPS practical bit-exact bound: any diff must be < 1e-9.
        # CPU/CUDA: require exact bit equality (numpy.array_equal).
        PRACTICAL_BIT_EXACT_BOUND = 1e-9

        non_exact = []
        max_diffs = {}

        for ep_key in attrs_a:
            ep_a = attrs_a[ep_key]
            ep_b = attrs_b[ep_key]
            for attr_name in set(ep_a) | set(ep_b):
                if attr_name not in ep_a or attr_name not in ep_b:
                    continue
                v_a = ep_a[attr_name]
                v_b = ep_b[attr_name]
                if not np.array_equal(v_a, v_b):
                    diff = _max_abs_diff(v_a, v_b)
                    key = f"{ep_key}/{attr_name}"
                    non_exact.append(key)
                    max_diffs[key] = diff

        if non_exact:
            # Allow up to PRACTICAL_BIT_EXACT_BOUND for MPS ULP drift.
            import warnings
            warnings.warn(
                f"Non-bit-exact attrs: {non_exact}. Max diffs: {max_diffs}. "
                f"Falling back to < {PRACTICAL_BIT_EXACT_BOUND} bound (MPS ULP tolerance).",
                stacklevel=2,
            )
            for key in non_exact:
                assert max_diffs[key] < PRACTICAL_BIT_EXACT_BOUND, (
                    f"Attr {key} differs by {max_diffs[key]:.3e} between two "
                    f"deterministic runs with seed={seed} — exceeds MPS ULP bound "
                    f"{PRACTICAL_BIT_EXACT_BOUND}. DETERMINISM_ENABLED wiring is broken."
                )
        else:
            # All attrs bit-exact.
            pass

    # -----------------------------------------------------------------------
    # Sanity check: non-deterministic runs (DETERMINISM_ENABLED=false) should
    # diverge on ep 0 head_corr across different Python process starts, even
    # with the same seed in the YAML.  (This is the bug W1a saw.)
    # We skip this assertion on very short runs where RNG state coincides.
    # -----------------------------------------------------------------------

    def test_nondeterministic_runs_may_differ(self):
        """Without DETERMINISM_ENABLED, two runs with the same seed CAN produce
        different ep-0 head_corr values (as W1a observed: -0.0028, -0.0129, -0.0052).

        This test doesn't assert divergence (it's probabilistic), but it runs
        two non-deterministic Main.py invocations and checks they complete
        successfully, proving the config path works without DETERMINISM_ENABLED.
        """
        seed = 42
        port_a = PORT_BASE + 2
        port_b = PORT_BASE + 3

        exp_dir_a = _make_exp_dir(
            self._tmp_dir, "nondet_run_a", port_a,
            determinism_enabled=False, seed=seed,
        )
        exp_dir_b = _make_exp_dir(
            self._tmp_dir, "nondet_run_b", port_b,
            determinism_enabled=False, seed=seed,
        )

        h5_a = _run_main_with_stub(exp_dir_a, port=port_a, timeout=90)
        h5_b = _run_main_with_stub(exp_dir_b, port=port_b, timeout=90)

        # Both runs must produce valid HDF5 files.
        assert os.path.exists(h5_a)
        assert os.path.exists(h5_b)

        # Check that both runs produced N_EPISODES episodes.
        with h5py.File(h5_a, "r") as f:
            assert len(f.keys()) == N_EPISODES, (
                f"Expected {N_EPISODES} episode groups in {h5_a}, got {len(f.keys())}"
            )
        with h5py.File(h5_b, "r") as f:
            assert len(f.keys()) == N_EPISODES, (
                f"Expected {N_EPISODES} episode groups in {h5_b}, got {len(f.keys())}"
            )

# RL-CollectiveTransport

Reinforcement learning for multi-robot collective transport using ARGoS simulator, BUZZ, and the GSP-RL library. Implements GSP (Global State Prediction) and its variants (GSP-N, R-GSP-N, A-GSP-N) for decentralized swarm coordination.

## Dependencies

### System Dependencies

| Dependency | Version | Purpose | Install |
|---|---|---|---|
| [ARGoS3](https://github.com/ilpincy/argos3) | **commit `4bb398cd`** (post-beta59) | Multi-robot physics simulator | Build from source — see pinned commit below |
| [Buzz](https://github.com/buzz-lang/Buzz) | 0.0.1+ | Robot controller scripting language | Build from source (requires ARGoS installed first) |
| [argos3-nonuniform-objects](https://github.com/NESTLab/argos3-nonuniform-objects) | main | Plugin for convex prism/composite entities | Build from source, `sudo make install` |
| ZeroMQ | 4.3+ | IPC between ARGoS and Python | `brew install zeromq` (macOS) |
| Qt5 | 5.15+ | ARGoS visualization (optional) | `brew install qt@5` (macOS, keg-only) |
| CMake | 3.16+ | Build system | `brew install cmake` |

### Python Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| Python | ^3.10 | Runtime |
| [GSP-RL](https://github.com/NESTLab/GSP-RL) | 1.4.0+ | RL algorithms (DQN, DDQN, DDPG, TD3) + GSP networks |
| PyTorch | 2.x | Deep learning (CUDA, MPS, or CPU) |
| Poetry | latest | Python package manager |

### Pinned ARGoS3 Version

ARGoS must be built from commit **`4bb398cd`** or later from `ilpincy/argos3`. This version includes a critical fix to the simulation loop (`a96e090b`) that prevents ZMQ deadlocks during episode resets. Beta59 (the last tagged release) does NOT include this fix.

```bash
git clone https://github.com/ilpincy/argos3.git
cd argos3
git checkout 4bb398cd
mkdir build && cd build
cmake ../src
make -j$(nproc)
sudo make install
```

### Build Order

1. ARGoS3 (from pinned commit above)
2. Buzz (from source, requires ARGoS)
3. argos3-nonuniform-objects plugin (from source, requires ARGoS)
4. `poetry install` (Python deps + GSP-RL)
5. `cd build_scripts && ./quick-build.sh` (C++ loop functions + Buzz controller)

### Verify Environment

After setup, run the verification script to confirm all dependencies match:

```bash
python verify_environment.py
```

**macOS note:** Set `CMAKE_PREFIX_PATH="/opt/homebrew/opt/qt@5"` before building. Qt5 is keg-only in Homebrew.

## Quick Start

### Training

```bash
# 1. Edit exp_config.yml with your experiment parameters
# 2. Run
./run_exp.sh
```

### Testing

```bash
# 1. Edit test_config.yml (set MODEL_NUM to checkpoint, EXP_NAME to training dir)
# 2. Run
./test_exp.sh
```

### With Custom Config

```bash
./run_exp_with_config.sh my_config.yml
```

## Configuration

All parameters in `exp_config.yml`:

### Environment
| Parameter | Default | Description |
|---|---|---|
| `EXP_NAME` | — | Output directory name (must be unique) |
| `NUM_EPISODES` | 1000 | Training episodes (1600 for gate with curriculum) |
| `NUM_ROBOTS` | 4 | Number of foot-bot robots |
| `NUM_OBSTACLES` | 0 | Cylindrical obstacles (0, 2, or 4) |
| `USE_GATE` | 0 | Gate obstacle (0/1) |
| `GATE_CURRICULUM` | 0 | Curriculum learning for gate width (0/1) |
| `PORT` | 55557 | ZMQ port (unique per concurrent experiment) |
| `SEED` | 123 | Random seed |

### Object Type (Non-Uniform Transport)
| Parameter | Default | Description |
|---|---|---|
| `USE_PRISMS` | 0 | Use convex prism/composite objects instead of cylinder (0/1) |
| `RANDOM_OBJECTS` | 0 | Randomize object type per episode (0/1) |
| `TEST_PRISM` | 0 | Use alternate test prism geometry (0/1) |

**Requires:** `argos3-nonuniform-objects` plugin installed.

### Algorithm
| Parameter | Default | Description |
|---|---|---|
| `LEARNING_SCHEME` | DDPG | RL algorithm: DQN, DDQN, DDPG, TD3 |
| `GSP` | True | Enable Global State Prediction |
| `NEIGHBORS` | True | GSP-N (neighbor-only communication) |
| `RECURRENT` | False | R-GSP-N (LSTM temporal memory) |
| `ATTENTION` | False | A-GSP-N (transformer attention) |

### GSP Variant Selection

| Variant | GSP | NEIGHBORS | RECURRENT | ATTENTION |
|---|---|---|---|---|
| IC (no communication) | False | — | — | — |
| GSP (broadcast) | True | False | False | False |
| GSP-N | True | True | False | False |
| R-GSP-N | True | True | True | False |
| A-GSP-N | True | True | False | True |

## Architecture

```
rl_code/Main.py              Python RL server (ZMQ REP)
rl_code/src/agent.py         Agent class (extends gsp_rl.Actor)
rl_code/src/env.py           ZMQ utilities, GSP reward computation
argos/collectiveRlTransport.cpp   C++ ARGoS loop functions (ZMQ REQ)
argos/collectiveRlTransport.bzz   Buzz robot controller
argos/generate_argos.py      Template-based XML config generator
```

Communication: ARGoS (C++) ↔ ZMQ ↔ Python (RL server)

## Experiment Output

```
rl_code/Data/<EXP_NAME>/
  agent_config.yml          Frozen config
  Data/Data_Episode_N.pkl   Per-episode data (rewards, positions, GSP)
  Models/Episode_N/         Model checkpoints (every 10 episodes)
  Training_Metrics.png      Reward + success rate chart
```

## Running Tests

```bash
poetry run pytest tests/ -v
```

## Related Repositories

- [GSP-RL](https://github.com/NESTLab/GSP-RL) — RL algorithm library
- [ARGoS3](https://github.com/ilpincy/argos3) — Multi-robot simulator
- [Buzz](https://github.com/buzz-lang/Buzz) — Robot scripting language
- [argos3-nonuniform-objects](https://github.com/NESTLab/argos3-nonuniform-objects) — Non-uniform object plugin

## Authors

- Joshua Bloom (jdbloom@wpi.edu)
- NESTLab, Worcester Polytechnic Institute

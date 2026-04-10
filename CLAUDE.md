# RL-CollectiveTransport

Reinforcement learning for multi-robot collective transport using ARGoS simulator and BUZZ.

## Context Loading Guide

Load documentation based on what you are working on:

| Task | Load These Docs |
|------|----------------|
| **Any task** | This file (auto-loaded) |
| **System orientation** | `docs/architecture/TIER1_SYSTEM_DAG.md` |
| **ZMQ / message bugs** | `docs/protocols/ZMQ_PROTOCOL.md` |
| **Config changes** | `docs/protocols/CONFIG_SCHEMA.md` |
| **Modifying Main.py** | `docs/architecture/TIER2_MAIN.md` |
| **Modifying agent.py** | `docs/architecture/TIER2_AGENT.md` |
| **Modifying env.py** | `docs/architecture/TIER2_ENV.md` |
| **Modifying C++ loop functions** | `docs/architecture/TIER2_ARGOS_CPP.md` |
| **Modifying BUZZ controller** | `docs/architecture/TIER2_ARGOS_BUZZ.md` |
| **Config / experiment setup** | `docs/architecture/TIER2_CONFIG.md` |
| **Data logging / pickle format** | `docs/architecture/TIER2_DATA.md` |
| **Plotting / visualization** | `docs/architecture/TIER2_PLOTTING.md` |
| **Domain terminology** | `docs/GLOSSARY.md` |

**Loading order:** Read CLAUDE.md (this file) → TIER1 for orientation → relevant TIER2 + protocol docs for specific work. Never load all TIER2 files at once.

## Architecture

- `rl_code/Main.py` — Entry point. ZMQ REP server coordinating with ARGoS simulator.
- `rl_code/src/agent.py` — Agent class (extends gsp_rl.Actor): action selection, GSP, state building.
- `rl_code/src/env.py` — ZMQ_Utility (message serialization), calculate_gsp_reward, angle normalization.
- `rl_code/src/exp_data_structures.py` — data_logger: per-episode pickle serialization.
- `rl_code/src/plotting/` — Visualization scripts (trajectories, training curves).
- `argos/collectiveRlTransport.cpp` — C++ ARGoS loop functions (entity management, observations, ZMQ sends).
- `argos/collectiveRlTransport.bzz` — BUZZ robot controller (gripping, wheel control, failure).
- `argos/generate_argos.py` — Template-based ARGoS XML config generator.

## Languages

- **Python 3.10** (RL code, managed by Poetry)
- **C++** (ARGoS loop functions, built via CMake)
- **BUZZ** (robot controller scripts)

## Key Dependencies

- `gsp-rl` — Custom RL library (Git dependency from NESTLab/GSP-RL). Provides Actor base class with neural networks, replay buffers, learn(), save/load_model().
- `pytest` — Testing (dev dependency)

## Running

```bash
# Training/testing (terminal 1)
python Main.py recording_folder -flags

# ARGoS simulator (terminal 2)
argos3 -c argos/collectiveRlTransport0.argos
```

Recording folder must contain `agent_config.yml`, `Data/`, and `Models/` subdirectories.

## RL Algorithms

Selectable via config `LEARNING_SCHEME`: DQN, DDQN, DDPG, TD3

## Key Flags

- `--independent_learning` — Each robot gets its own neural network (vs shared CTDE)
- `--global_knowledge` — Augment observations with other robots' positions + velocities
- `--share_prox_values` — Share averaged proximity readings between robots
- `--test` — Test mode (no learning, loads saved model)

## Testing

```bash
poetry run pytest
```

## Conventions

- Feature branches only — never commit to main
- All new Python code must have pytest tests
- Preserve existing experiment config formats (`exp_config.yml`, `test_config.yml`)

## Debugging Protocol

When investigating a stall, crash, or unexpected behavior:

1. **Check diagnostics first** — read `/tmp/stelaris-runs/<exp>/python.log` before theorizing
2. **Use py-spy** — `py-spy dump --nonblocking --pid <PID>` to see WHERE the code is stuck
3. **Check ALL process states** — both Python AND ARGoS. Are they in recv? send? idle? print?
4. **Test with smallest reproduction** — 10 ticks/episode, 1 experiment, before scaling up
5. **When adding validation, test against all environment types** — open, obstacles, gate, prism
6. **Verify the fix works at scale** — run multi-seed stress test before committing to long runs

## Hard Rules (from learnings)

These rules are derived from bugs we found. See `kb/wiki/learnings/` for full context.

- **Never use `subprocess.PIPE` without consuming stdout** — use `DEVNULL` or read in a thread. Pipe buffer fills at ~65KB and blocks the child process. (`learnings/stdout-pipe-deadlock`)
- **Never use `except: pass` for code that needs debugging** — at minimum `except Exception as e: log.debug(e)`. Silent swallowing masks real errors. 
- **Never hardcode expected message counts from external systems** — validate against a range or schema, not exact values. ARGoS message structure varies by config. (`learnings/zmq-message-parts-mismatch`)
- **Always guard numerical functions against NaN/Inf** — `if not np.isfinite(x): return default`. NaN propagates silently; Inf causes infinite loops. (`learnings/nan-infinite-loop`)
- **Never create classes/types inside hot loops** — cache `namedtuple()`, compiled regexes, etc. (`learnings/namedtuple-class-creation-performance`)
- **Pin exact dependency versions** — use commit hashes, not "latest" or version ranges. Different machines must run identical code. See `verify_environment.py`.
- **After every bug fix, write a learnings entry** in `kb/wiki/learnings/` with symptom, root cause, fix, detection, and prevention.

"""Wrap Main.py with timing instrumentation.

Usage (from rl_code/):
    python profile_wrapper.py Data/profile_run [--test] [other Main.py flags]

Patches key functions with timers, runs Main.py normally, prints profile at exit.
"""
import time
import atexit
import sys
import functools

# Global timers
_timers = {}
_counts = {}

def _timed(name):
    """Decorator that accumulates call time under `name`."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            _timers[name] = _timers.get(name, 0) + elapsed
            _counts[name] = _counts.get(name, 0) + 1
            return result
        return wrapper
    return decorator

def _print_profile():
    if not _timers:
        return
    total = sum(_timers.values())
    print(f"\n{'='*70}")
    print(f"RL-CT PROFILING RESULTS")
    print(f"{'='*70}")
    print(f"{'Operation':<30s} {'Calls':<10s} {'Total (s)':<12s} {'Per-call (ms)':<14s} {'% total':<10s}")
    print(f"{'-'*76}")
    for key in sorted(_timers.keys(), key=lambda k: _timers[k], reverse=True):
        t = _timers[key]
        c = _counts[key]
        per_call = t / c * 1000 if c > 0 else 0
        pct = t / total * 100 if total > 0 else 0
        print(f"{key:<30s} {c:<10d} {t:<12.3f} {per_call:<14.3f} {pct:<10.1f}%")
    print(f"{'TOTAL':<30s} {'':10s} {total:<12.3f}")

atexit.register(_print_profile)

# Patch key functions before importing Main.py
import src.agent as agent_module
import src.env as env_module

# Patch Agent methods
_orig_filter_prox = agent_module.Agent.filter_prox_values
agent_module.Agent.filter_prox_values = _timed("filter_prox_values")(_orig_filter_prox)

_orig_make_agent_state = agent_module.Agent.make_agent_state
agent_module.Agent.make_agent_state = _timed("make_agent_state")(_orig_make_agent_state)

_orig_make_gsp_states = agent_module.Agent.make_gsp_states
agent_module.Agent.make_gsp_states = _timed("make_gsp_states")(_orig_make_gsp_states)

_orig_choose_agent_action = agent_module.Agent.choose_agent_action
agent_module.Agent.choose_agent_action = _timed("choose_agent_action")(_orig_choose_agent_action)

_orig_choose_agent_gsp = agent_module.Agent.choose_agent_gsp
agent_module.Agent.choose_agent_gsp = _timed("choose_agent_gsp")(_orig_choose_agent_gsp)

_orig_store_agent_transition = agent_module.Agent.store_agent_transition
agent_module.Agent.store_agent_transition = _timed("store_agent_transition")(_orig_store_agent_transition)

# Patch Actor.learn (from gsp_rl)
from gsp_rl.src.actors.actor import Actor
_orig_learn = Actor.learn
Actor.learn = _timed("actor.learn")(_orig_learn)

_orig_learn_gsp = Actor.learn_gsp
Actor.learn_gsp = _timed("actor.learn_gsp")(_orig_learn_gsp)

# Patch ZMQ utility
_orig_parse_msgs = env_module.ZMQ_Utility.parse_msgs
env_module.ZMQ_Utility.parse_msgs = _timed("zmq_parse_msgs")(_orig_parse_msgs)

_orig_serialize = env_module.ZMQ_Utility.serialize_actions
env_module.ZMQ_Utility.serialize_actions = _timed("zmq_serialize")(_orig_serialize)

_orig_gsp_reward = env_module.calculate_gsp_reward
env_module.calculate_gsp_reward = _timed("calculate_gsp_reward")(_orig_gsp_reward)

# Now run Main.py with the patched functions
exec(compile(open("Main.py").read(), "Main.py", "exec"))

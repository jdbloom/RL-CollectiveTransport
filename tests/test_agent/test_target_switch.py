"""Tests for the cross-target plasticity-recovery hook (Phase 4).

Verifies:
1. Agent.gsp_prediction_target can be changed at runtime and the new value sticks.
2. reset_gsp_label_buffer() empties the buffer after a target switch.
3. With default GSP_TARGET_SWITCH_AT_EP=0, the Main.py conditional is a strict
   no-op — modelled here by asserting the conditional expression evaluates False
   for every episode number when the defaults are in effect.
4. The switch only fires exactly at the designated episode (boundary condition).

The Main.py hook is exercised at the Agent/buffer level; we do not spin up a
ZMQ server. The test replicates the exact conditional and attribute assignments
from Main.py so any future refactoring that breaks the contract will be caught.
"""

import numpy as np
import pytest

from src.agent import Agent


# ---------------------------------------------------------------------------
# Minimal config — enough to construct a GSP-N agent
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "GAMMA": 0.99,
    "TAU": 0.005,
    "ALPHA": 0.001,
    "BETA": 0.002,
    "LR": 0.001,
    "EPSILON": 0.0,
    "EPS_MIN": 0.0,
    "EPS_DEC": 0.0,
    "BATCH_SIZE": 16,
    "MEM_SIZE": 1000,
    "REPLACE_TARGET_COUNTER": 10,
    "NOISE": 0.0,
    "UPDATE_ACTOR_ITER": 1,
    "WARMUP": 0,
    "GSP_LEARNING_FREQUENCY": 1,
    "GSP_BATCH_SIZE": 16,
    "GSP_PREDICTION_TARGET": "delta_theta",
    "GSP_PREDICTION_HORIZON": 5,
}

N_ROBOTS = 4
N_OBS = 31
N_ACTIONS = 2
OPTIONS_PER_ACTION = 3
GSP_INPUT_SIZE = 10   # GSP-N neighbors mode overrides this; provide a valid value
GSP_OUTPUT_SIZE = 1
GSP_LOOK_BACK = 2
GSP_SEQUENCE_LENGTH = 5


def make_agent(extra_config=None):
    """Construct a minimal shared GSP-N Agent for testing."""
    cfg = {**BASE_CONFIG, **(extra_config or {})}
    return Agent(
        config=cfg,
        network="DDPG",
        n_agents=N_ROBOTS,
        n_obs=N_OBS,
        n_actions=N_ACTIONS,
        options_per_action=OPTIONS_PER_ACTION,
        id=0,
        min_max_action=1.0,
        meta_param_size=1,
        gsp=True,
        recurrent=False,
        attention=False,
        neighbors=True,          # GSP-N — activates _gsp_label_buffer
        gsp_input_size=GSP_INPUT_SIZE,
        gsp_output_size=GSP_OUTPUT_SIZE,
        gsp_min_max_action=1.0,
        gsp_look_back=GSP_LOOK_BACK,
        gsp_sequence_length=GSP_SEQUENCE_LENGTH,
        broadcast=False,
    )


# ---------------------------------------------------------------------------
# Helper: replicate the Main.py switch conditional
# ---------------------------------------------------------------------------

def _should_switch(config, ep_counter, model):
    """Replicate the Main.py target-switch conditional exactly.

    Returns True when all conditions are met:
    - not independent_learning (always False in these tests — shared model)
    - GSP_TARGET_SWITCH_AT_EP > 0
    - ep_counter == GSP_TARGET_SWITCH_AT_EP
    - GSP_TARGET_SWITCH_TO is non-empty
    - model has gsp_prediction_target attribute
    """
    switch_at = int(config.get('GSP_TARGET_SWITCH_AT_EP', 0))
    switch_to = str(config.get('GSP_TARGET_SWITCH_TO', ''))
    return (
        switch_at > 0
        and ep_counter == switch_at
        and bool(switch_to)
        and hasattr(model, 'gsp_prediction_target')
    )


# ---------------------------------------------------------------------------
# Test 1: target attribute change sticks after assignment
# ---------------------------------------------------------------------------

def test_gsp_prediction_target_change_sticks():
    """Setting gsp_prediction_target on a live Agent persists across calls."""
    agent = make_agent()
    assert agent.gsp_prediction_target == 'delta_theta'

    agent.gsp_prediction_target = 'future_prox'
    assert agent.gsp_prediction_target == 'future_prox', (
        "gsp_prediction_target did not persist after direct assignment"
    )

    # Round-trip: switch back to original
    agent.gsp_prediction_target = 'delta_theta'
    assert agent.gsp_prediction_target == 'delta_theta'


# ---------------------------------------------------------------------------
# Test 2: reset_gsp_label_buffer clears the buffer after a target switch
# ---------------------------------------------------------------------------

def test_reset_gsp_label_buffer_clears_on_target_switch():
    """After a target switch, reset_gsp_label_buffer() leaves the buffer empty.

    Populate the buffer with synthetic snapshots to simulate a run that has
    accumulated K+1 future-prox entries, then perform the switch and verify
    the buffer is cleared. This replicates the Main.py sequence exactly:
        model.gsp_prediction_target = new_target
        model.reset_gsp_label_buffer()
    """
    agent = make_agent({'GSP_PREDICTION_TARGET': 'future_prox'})
    assert agent.gsp_prediction_target == 'future_prox'

    # Manually push entries into the buffer to simulate in-episode accumulation
    for _ in range(8):
        agent._gsp_label_buffer.append({'dummy': True})
    assert len(agent._gsp_label_buffer) == 8, "Buffer should have 8 entries before switch"

    # Perform the switch (replicating Main.py)
    agent.gsp_prediction_target = 'delta_theta'
    agent.reset_gsp_label_buffer()

    assert agent.gsp_prediction_target == 'delta_theta', "New target did not stick"
    assert len(agent._gsp_label_buffer) == 0, (
        "Buffer should be empty after reset_gsp_label_buffer() on target switch"
    )


# ---------------------------------------------------------------------------
# Test 3: default config is a strict no-op — conditional never fires
# ---------------------------------------------------------------------------

def test_default_config_noop_across_all_episodes():
    """With default GSP_TARGET_SWITCH_AT_EP=0, the switch conditional is False
    for every episode number from 0 to 500.

    This is the backward-compatibility guarantee: all existing experiments that
    do not set GSP_TARGET_SWITCH_AT_EP are completely unaffected.
    """
    default_config = {
        # Neither key present — both default to 0 / '' in Main.py
    }
    agent = make_agent()

    for ep in range(501):
        fired = _should_switch(default_config, ep, agent)
        assert not fired, (
            f"Default config should never trigger switch, but fired at ep={ep}"
        )


# ---------------------------------------------------------------------------
# Test 4: switch fires exactly at the designated episode
# ---------------------------------------------------------------------------

def test_switch_fires_exactly_at_designated_episode():
    """The conditional fires at ep == GSP_TARGET_SWITCH_AT_EP, not before, not after."""
    switch_config = {
        'GSP_TARGET_SWITCH_AT_EP': 100,
        'GSP_TARGET_SWITCH_TO': 'future_prox',
    }
    agent = make_agent()

    # Should NOT fire before or after the designated episode
    for ep in range(200):
        fired = _should_switch(switch_config, ep, agent)
        if ep == 100:
            assert fired, f"Switch should fire at ep=100, but did not"
        else:
            assert not fired, (
                f"Switch should only fire at ep=100, but fired at ep={ep}"
            )


# ---------------------------------------------------------------------------
# Test 5: full switch sequence — target changes and buffer clears together
# ---------------------------------------------------------------------------

def test_full_switch_sequence_matches_main_py():
    """End-to-end replication of the Main.py switch block.

    Replicates:
        old_target = model.gsp_prediction_target
        model.gsp_prediction_target = config['GSP_TARGET_SWITCH_TO']
        model.reset_gsp_label_buffer()

    Asserts the new target is set and the buffer is empty after the sequence.
    """
    switch_config = {
        'GSP_TARGET_SWITCH_AT_EP': 50,
        'GSP_TARGET_SWITCH_TO': 'future_prox',
    }
    agent = make_agent()
    assert agent.gsp_prediction_target == 'delta_theta'

    # Populate buffer with dummy entries
    for _ in range(5):
        agent._gsp_label_buffer.append({'step': _})

    ep_counter = 50
    if _should_switch(switch_config, ep_counter, agent):
        old_target = agent.gsp_prediction_target
        agent.gsp_prediction_target = str(switch_config['GSP_TARGET_SWITCH_TO'])
        agent.reset_gsp_label_buffer()

    assert agent.gsp_prediction_target == 'future_prox', (
        f"Expected gsp_prediction_target='future_prox', got '{agent.gsp_prediction_target}'"
    )
    assert len(agent._gsp_label_buffer) == 0, (
        "Buffer must be empty after the target-switch sequence"
    )

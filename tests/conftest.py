"""Shared fixtures for RL-CollectiveTransport tests."""

import pytest
import numpy as np
from struct import pack

from src.env import ZMQ_Utility
from src.agent import Agent


@pytest.fixture
def zmq_util():
    """ZMQ_Utility with 4-robot params pre-loaded."""
    util = ZMQ_Utility()
    params_bytes = pack('8f', 4.0, 2.0, 31.0, 3.0, 4.0, 9.0, 0.0, 10.0)
    util.get_params(params_bytes)
    util.set_obstacles_fields()
    return util


@pytest.fixture
def agent_config():
    """Minimal config for Agent construction."""
    return {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }

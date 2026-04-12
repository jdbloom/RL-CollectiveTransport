"""Per-algorithm config field defaults.

Used by run_baseline_experiments.make_config() to parameterize LEARNING_SCHEME
across DQN, DDQN, DDPG, TD3.
"""

ALGORITHM_DEFAULTS = {
    "DQN": {
        "LEARNING_SCHEME": "DQN",
        "LEARNING_RATE": 1e-4,
        "GAMMA": 0.99,
        "EPSILON_START": 1.0,
        "EPSILON_END": 0.05,
        "EPSILON_DECAY": 0.995,
        "REPLAY_BUFFER_SIZE": 100_000,
        "BATCH_SIZE": 64,
        "TARGET_UPDATE_FREQ": 1000,
    },
    "DDQN": {
        "LEARNING_SCHEME": "DDQN",
        "LEARNING_RATE": 1e-4,
        "GAMMA": 0.99,
        "EPSILON_START": 1.0,
        "EPSILON_END": 0.05,
        "EPSILON_DECAY": 0.995,
        "REPLAY_BUFFER_SIZE": 100_000,
        "BATCH_SIZE": 64,
        "TARGET_UPDATE_FREQ": 1000,
    },
    "DDPG": {
        "LEARNING_SCHEME": "DDPG",
        "ACTOR_LR": 1e-4,
        "CRITIC_LR": 1e-3,
        "GAMMA": 0.99,
        "TAU": 0.005,
        "NOISE_STD": 0.1,
        "REPLAY_BUFFER_SIZE": 100_000,
        "BATCH_SIZE": 64,
        "WARMUP_EPISODES": 300,
    },
    "TD3": {
        "LEARNING_SCHEME": "TD3",
        "ACTOR_LR": 1e-4,
        "CRITIC_LR": 1e-3,
        "GAMMA": 0.99,
        "TAU": 0.005,
        "NOISE_STD": 0.1,
        "TARGET_NOISE_STD": 0.2,
        "NOISE_CLIP": 0.5,
        "POLICY_DELAY": 2,
        "REPLAY_BUFFER_SIZE": 100_000,
        "BATCH_SIZE": 64,
        "WARMUP_EPISODES": 300,
    },
}


def merge_algorithm_defaults(base_config: dict, algorithm: str) -> dict:
    if algorithm not in ALGORITHM_DEFAULTS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Valid: {list(ALGORITHM_DEFAULTS)}")
    out = {**base_config}
    for k, v in ALGORITHM_DEFAULTS[algorithm].items():
        out.setdefault(k, v)
    return out

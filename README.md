# RL-CollectiveTransport
Collective transport done with reinforcement learning

# Running Python:
Python 3 is needed to run this project.

Run `$python Main.py recording_folder -flags`

`Main.py` Houses the main loop which works in tandom with Argos.

The `recording_folder` must contain two empty sub directories: `Data` and `Models`. The `recording_folder` is required for both testing and training.
For example: `Test_Run/Data` and `Test_Run/Models`

There are many optional Flags. Here are all of the options with their implementations and defaults.
- `--learning_scheme` is the desired algorithm learning. Currently implemented are:
    - DQN -> Deep Q-Network (Discrete Actions)
    - DDQN -> Double Deep Q-Network (Discrete Actions)
    - DDPG -> Deep Deterministic Policy Gradient (Continuous Actions)
    - TD3 -> Twin Delayed Deep Deterministic Policy Gradient (Continuous Actions)
    - default: None -> Will return actions equivilent to no action. Assumes an alternate controller similar to the one implemented in BUZZ (See Argos section)
    - To Do: PPO, AC3, ...

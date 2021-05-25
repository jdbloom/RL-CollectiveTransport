# RL-CollectiveTransport
Collective transport done with reinforcement learning

# Running Python:
Python 3 is needed to run this project.

Run `$python Main.py recording_folder -flags`

`Main.py` Houses the main loop which works in tandom with Argos.

The `recording_folder` must contain two empty sub directories: `Data` and `Models`. The `recording_folder` is required for both testing and training.
For example: `Test_Run/Data` and `Test_Run/Models`

There are many optional Flags. Here are all of the options with their implementations and defaults.
- `--learning_scheme` is the desired algorithm learning.
    - DQN -> Deep Q-Network (Discrete Actions)
    - DDQN -> Double Deep Q-Network (Discrete Actions)
    - DDPG -> Deep Deterministic Policy Gradient (Continuous Actions)
    - TD3 -> Twin Delayed Deep Deterministic Policy Gradient (Continuous Actions)
    - None (default) -> Will return actions equivilent to no action. Assumes an alternate controller similar to the one implemented in BUZZ (See Argos section)
    - To Do: PPO, AC3, ...
- `--comms_scheme` is the desired communications scheme for the robots.
    - None (default) -> No communication will occure between agents. No learning model will be implemented and thus no messages will be generated.
    - Right -> This method will direct messages to the robot directly next to the sending robot in the Counter Clockwise (CCW) direction
    - Left -> This method will direct messages to the robot directly next to the sending robot in the Clockwise (CW) direction
    - Neighbors -> This method will direct messages to the robots directly next to the sending robot in both the CW and CCW directions. 
    - Broadcast -> This method will direct messages from the sending robot to all other robots in the swarm.
- `--comms_mem` triggers a memory for each robot for past messages. The memory length is defaulted to $num_robots$

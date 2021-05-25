# RL-CollectiveTransport
Collective transport done with reinforcement learning

# Running Python:
Python 3 is needed to run this project.

Run `$python Main.py recording_folder -flags`

`Main.py` houses the main loop which works in tandom with Argos.

The `recording_folder` must contain two empty sub directories: `Data` and `Models`. The `recording_folder` is required for both testing and training.
For example: `Test_Run/Data` and `Test_Run/Models`

There are many optional Flags. Here are all of the options with their implementations and defaults.
- `--learning_scheme` is the desired algorithm learning.
    - None (default) -> Will return actions equivilent to no action. Assumes an alternate controller similar to the one implemented in BUZZ (See Argos section)
    - DQN -> Deep Q-Network (Discrete Actions)
    - DDQN -> Double Deep Q-Network (Discrete Actions)
    - DDPG -> Deep Deterministic Policy Gradient (Continuous Actions)
    - TD3 -> Twin Delayed Deep Deterministic Policy Gradient (Continuous Actions)
    - To Do: PPO, AC3, ...
- `--comms_scheme` is the desired communications scheme for the robots.
    - None (default) -> No communication will occure between agents. No learning model will be implemented and thus no messages will be generated.
    - Right -> This method will direct messages to the robot directly next to the sending robot in the Counter Clockwise (CCW) direction
    - Left -> This method will direct messages to the robot directly next to the sending robot in the Clockwise (CW) direction
    - Neighbors -> This method will direct messages to the robots directly next to the sending robot in both the CW and CCW directions. 
    - Broadcast -> This method will direct messages from the sending robot to all other robots in the swarm.
- `--comms_mem` (default: False) triggers a memory for each robot for past messages. The memory length is defaulted to `num_robots` and must be changed in the code.
- `--no_buffer` (default: False) triggers both the action network and the comms network to learn from current experiences and not from a replay buffer.
- `--use_horizon` (default: False) triggers horizon based learning within the replay buffer. Instead of randomly selecting a batch size, this flag tells the buffer to randomly select a starting index and then sample sequentialy untill the batch size is met.
- `--use_entropy` (default: False) triggers a custom entropy based loss function to be combined with the L2 Norm loss during learning. This loss is only applicable when using a communication network and is used to influence meaningful emergent communication.
- `--plot_comms` (default: False) triggers a "real-time" plot of the current messages being sent by the robots. This function should only be used during testing as it significantly slows run-time.
-  `--test` (default: False) triggers the loop to run in test mode. This will bypass all learning functionality and will load a model. The loaded model will only be asked to output actions.
-  `--model_path` takes a path to a specific saved model to be loaded in. For example: `Test_Run/Models/Episode_100` this will load both the action network and the communication network (if a communication network was trained and the `--comms_scheme` flag has an input.
- `--port` This argument is defaulted to `55555` which is the same as Argos. If you change this you must also change it in Argos. 

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
- `--port` This argument is defaulted to `55555` which is the same as Argos. If you change this you must also change it in `collectiveRlTransport0.argos`. 


# Runing ARGoS + BUZZ
Run `argos3 -z -c argos/collectiveRlTransport0.argos` for no visualization or `argos3 -c argos/collectiveRlTransport0.argos` for visualization

There are several flags and parameters to set in the `loop_functions` section:
- `data_file` location of the file to be written to. Currently no implentation for this.
- `num_robots` This is where you set the number of robots in the swarm for the experiment
- `max_robot_failures` This is the maximum number of allowed failures during an episode. 0 = no failures
- `latest_failure_time` indicates the latest time step in which a failure can occure. This is done for training purposes to ensure all robots that fail will fail before reaching the goal
- `chance_failure` This is the probability a single robot has of failing. Once `max_robot_failures` is met then this probability drops to 0 ensuring no more robots will fail in the episode
- `goal` This is the location of the goal (x, y)
- `threshold` creates a circle around the goal point where the cylinder is considered at the goal (m)
- `threshold_freq` This is used for "curriculum learning." The frequency is given in episodes and dictates how often the goal `threshold` should shrink
- `threshold_dec` is used in conjunction with `threshold_freq` and signifies the amount the threshold should shrink by (m)
- `min_threshold` indicated the minimum threshold from the goal position and will prevent the threshold from shrinking past it (m)
- `num_episodes` is the total number of episodes for an experiment
- `episode_time` is the maximum number of time steps an episode can have. If the cylinder reaches the goal then the episode is premeturely terminated
- `time_out_reward` The reward the robots get for a "failed" episode. Defaulted to 0
- `goal_reward` The reward the robots get for reaching the goal. Defaulted to 0
- `pytorch_url` This is the port for cpp to exchange information with pytorch. Defaulted to 55555. If changed you must also change in `Main.py`
- `alphabet_size` is the number of characters available for the communications network to send.
- `proximity_range` is the range for the proximity sensors (m)
- `num_obstacles` is the number of stationary cylindrical obstacels randomly generated in the environment
- `use_gate` (0/1) this flag indicates whether to use the gate obstacle
- `gate_curriculum` (0/1) this flag indicates whether to use curriculum learning
- `gate_update_frequency` indicates how often to shrink the gate (episodes)
- `gate_update_amount` indicates by how much to increase the wall sizes (shrink the opening). This is a per wall measure so multiply by 2 to get the opening shrinkage (m)
- `gate_minimum` is the minimum opening length (m)
- `use_base_model` (0/1) indicates to BUZZ to use a hand coded controller for the robots. This should be used in conjunction with setting both `--learning_scheme` and `--comms_scheme` to None and setting the flag `--test` when running `Main.py`

**Note on Gate Obstacle**
To test using the gate, set `use_gate = 1` and `gate_curriculum = 0` this will set the gate to the distance corresponding to `gate_minimum`

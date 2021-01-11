from collections import namedtuple
from struct import pack, unpack, Struct


class Parser:
    # Byte size of float in C++
    FLOAT_SIZE = 4
    # Byte size of int in C++
    INT_SIZE = 4

    #
    # Parse the fields from a message
    # Returns a dictionary
    #
    def parse_msg(msg, msgtype, fields, fmt):
        # Make an empty named tuple with the given fields
        Tx = namedtuple(msgtype, fields)
        # Fill the tuple with the contents of the message
        x = Tx._make(unpack(fmt, msg))
        # Return the tuple as a dictionary
        return x._asdict()

    #
    # Parse the experiment status
    # Returns a tuple (exp_done, episode_done, reached_goal)
    #
    def parse_status(msg):
        data = parse_msg(msgs[0], 'status', EXPERIMENT_FIELDS, EXPERIMENT_FMT)
        exp_done = (data['exp_done'] == 1)
        episode_done = (data['episode_done'] == 1)
        reached_goal = (data['reached_goal'] == 1)
        return exp_done, episode_done, reached_goal

    #
    # Parse the observations
    # Returns a list of numpy arrays, one per robot
    #
    def parse_obs(msg):
        obs = []
        # For each robot
        for r in range(0, params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * params['num_obs'] * FLOAT_SIZE:(r+1) * params['num_obs'] * FLOAT_SIZE]
            # Parse the bytes into a dictionary
            data = parse_msg(m, 'obs', OBS_FIELDS, OBS_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.float32, count=len(data))
            # Append it to the observations
            obs.append(nparr)
        return obs

    def parse_failures(msg):
        failures = []
        for r in range(0, params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * INT_SIZE:(r+1)*INT_SIZE]
            # Parse the bytes into a dictionary
            data = parse_msg(m, 'failure', FAILURE_FIELDS, FAILURE_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.intc, count = len(data))
            # Append it to the rewards
            failures.append(nparr)
        return failures

    def parse_rewards(msg):
        rewards = []
        for r in range(0, params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * FLOAT_SIZE:(r+1)*FLOAT_SIZE]
            # Parse the bytes into a dictionary
            data = parse_msg(m, 'reward', REWARDS_FIELDS, REWARDS_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.float32, count = len(data))
            # Append it to the rewards
            rewards.append(nparr)
        return rewards

    def parse_stats(msg):
        stats = []
        for r in range(0, params['num_robots']):
            # Get message bytes for this robot
            m = msg[r * params['num_stats'] * FLOAT_SIZE:(r+1) * params['num_stats'] * FLOAT_SIZE]
            # Parse the bytes into a dictionary
            data = parse_msg(m, 'stats', STATS_FIELDS, STATS_FMT)
            # Make a numpy array
            nparr = np.fromiter(data.values(), dtype=np.float32, count = len(data))
            # Append it to the rewards
            stats.append(nparr)
        return stats

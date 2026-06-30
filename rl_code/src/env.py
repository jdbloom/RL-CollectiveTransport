import math
import struct as _struct
import numpy as np
from collections import namedtuple
from struct import pack, unpack, Struct, calcsize

def angle_normalize_unsigned_deg(a):
  if not np.isfinite(a):
      return 0.0
  while a < 0: a += 360
  while a >= 360: a -= 360
  return a

def angle_normalize_signed_deg(a):
  if not np.isfinite(a):
      return 0.0
  while a < -180: a += 360
  while a >= 180: a -= 360
  return a

def calculate_gsp_reward(GSP, old_cyl_ang, cyl_ang, next_heading_gsp, num_robots):
    """Return (clipped_rewards, label, squared_errors, raw_diff_rad) per robot.

    The clipped reward saturates at -2 and hides the magnitude of large prediction errors.
    squared_errors carries the raw (diff - prediction)^2 per robot — needed for the
    information-collapse diagnostic (paper outline: "Revamped Reward structure for GSP
    to prevent information collapse").

    raw_diff_rad is the signed radian rotation BEFORE the ×100 / clip step; it's the
    quantity the supervised MSE is trying to predict. Returned so Main.py can log the
    per-step distribution for calibration diagnostics (2026-04-20 audit — the current
    ×100 scaling with clip to [-1, 1] likely degenerates the regression task into
    near-binary classification, but we need the actual distribution to confirm).
    """
    gsp_reward = []
    squared_errors = []
    label = 0
    raw_diff_rad = 0.0
    if GSP:
        old_cyl_ang = angle_normalize_unsigned_deg(old_cyl_ang)
        new_cyl_ang = angle_normalize_unsigned_deg(cyl_ang)
        diff = angle_normalize_signed_deg(new_cyl_ang-old_cyl_ang)
        diff = math.radians(diff)
        raw_diff_rad = float(diff)  # capture BEFORE scaling/clipping
        # Max rotation is 0.09 rad/step so we can multiply by 10 to get within range of -1, 1
        diff = np.clip(diff*100, -1, 1)
        label=diff
        # Vectorised per-robot computation — fp-identical to the loop above.
        # Extract the last element of each robot's (possibly multi-dim) prediction.
        preds = np.array(
            [float(np.asarray(next_heading_gsp[i]).ravel()[-1]) for i in range(num_robots)]
        )
        abs_reward_arr = (diff - preds) ** 2          # shape (num_robots,)
        squared_errors = [float(v) for v in abs_reward_arr]
        gsp_reward = list(np.clip(-abs_reward_arr, -2, 0))
    else:
        gsp_reward = [0 for i in range(num_robots)]
        squared_errors = [0 for i in range(num_robots)]

    return gsp_reward, label, squared_errors, raw_diff_rad


class ZMQ_Utility:
    def __init__(self):
        self.PARAMS_FIELDS = ['num_robots','num_obstacles', 'num_obs','num_actions', 'num_stats', 'alphabet_size',
                               'use_gate', 'distance_to_goal_normalization_factor', 'num_prisms']
        self.PARAMS_FMT = '9f'
        self.EXPERIMENT_FIELDS = ['exp_done', 'episode_done', 'reached_goal']
        self.EXPERIMENT_FMT = '3B'
        self.OBS_FIELDS = ['robot_dist2goal', 'robot_angle2goal', 'robot_lwheel',
                           'robot_rwheel', 'cyl_dist2robot', 'cyl_angle2robot', 'cyl_dist2goal',
                           'ProxVal_0',  'ProxVal_1',  'ProxVal_2',  'ProxVal_3',
                           'ProxVal_4',  'ProxVal_5',  'ProxVal_6',  'ProxVal_7',
                           'ProxVal_8',  'ProxVal_9',  'ProxVal_10', 'ProxVal_11',
                           'ProxVal_12', 'ProxVal_13', 'ProxVal_14', 'ProxVal_15',
                           'ProxVal_16', 'ProxVal_17', 'ProxVal_18', 'ProxVal_19',
                           'ProxVal_20', 'ProxVal_21', 'ProxVal_22', 'ProxVal_23']
        self.OBS_FMT = '31f'
        self.FAILURE_FIELDS = ['failure']
        self.FAILURE_FMT = '1I'
        self.REWARDS_FIELDS = ['reward']
        self.REWARDS_FMT = '1f'
        self.STATS_FIELDS = ['magnitude', 'angle', 'deltaX', 'deltaY']
        self.STATS_FMT = '4f'
        self.ROBOT_STATS_FIELDS = ['x_pos', 'y_pos', 'z_pos', 'x_deg', 'y_deg', 'z_deg']
        self.ROBOT_STATS_FMT = '6f'
        self.OBJ_STATS_FIELDS = ['x_pos', 'y_pos', 'z_pos', 'x_deg', 'y_deg', 'z_deg', 'cyl_angle2goal', 'comX', 'comY']
        self.OBJ_STATS_FMT = '9f'
        self.GATE_STATS_FIELDS = ['neg_wall_1_x', 'neg_wall_1_length_y',
                                  'pos_wall_2_x', 'pos_wall_2_length_y']
        self.GATE_STATS_FMT = '4f'
        # Need to account for differing numbers of obstacles
        self.OBSTACLE_STATS_FIELDS = []

        self.PRISM_SIZE_FIELDS = []
        self.PRISM_SIZE_FMT = ''
        self.PRISM_POINT_FIELDS = []
        self.PRISM_POINT_FMT = ''


        self.ACTIONS_FIELDS = ['lwheel', 'rwheel', 'failure']
        self.ACTIONS_FMT = '3f'

        # Byte size of float in C++
        self.FLOAT_SIZE = 4
        # Byte size of int in C++
        self.INT_SIZE = 4
        self._namedtuple_cache = {}

        # T5: precomputed Struct for serialize_actions (cached at __init__, re-created
        # only if ACTIONS_FMT changes — which it never does at runtime).
        self._actions_packer: Struct = Struct(self.ACTIONS_FMT)
        # T5: precomputed byte sizes for each per-robot message type.
        self._obs_stride: int = calcsize(self.OBS_FMT)         # 31*4 = 124
        self._fail_stride: int = self.INT_SIZE                  # 1*4  =   4
        self._rew_stride: int = self.FLOAT_SIZE                 # 1*4  =   4
        self._stat_stride: int = calcsize(self.STATS_FMT)      # 4*4  =  16
        self._rs_stride: int = calcsize(self.ROBOT_STATS_FMT)  # 6*4  =  24


    def get_params(self, msg):
        self.params = self.parse_msg(msg, 'params', self.PARAMS_FIELDS, self.PARAMS_FMT)
        self.params['num_robots'] = int(self.params['num_robots'])
        self.params['num_obstacles'] = int(self.params['num_obstacles'])
        self.params['num_obs'] = int(self.params['num_obs'])
        self.params['num_actions'] = int(self.params['num_actions'])
        self.params['num_stats'] = int(self.params['num_stats'])
        self.params['alphabet_size'] = int(self.params['alphabet_size'])
        self.params['use_gate'] = int(self.params['use_gate'])
        self.params['num_prisms'] = int(self.params['num_prisms'])

    def set_obstacles_fields(self):
        for i in range(self.params['num_obstacles']):
            self.OBSTACLE_STATS_FIELDS.append('obs_'+str(i)+'_x')
            self.OBSTACLE_STATS_FIELDS.append('obs_'+str(i)+'_y')
        self.OBSTACLE_STATS_FMT = str(self.params['num_obstacles']*2)+'f'

    def set_prism_sizes(self):
        """Build format for prism vertex count array."""
        for i in range(self.params['num_prisms']):
            self.PRISM_SIZE_FIELDS.append('prism_' + str(i) + '_size')
        self.PRISM_SIZE_FMT = str(self.params['num_prisms']) + 'I'

    def set_prism_points(self, prism_sizes):
        """Build format for prism vertex coordinate array."""
        total_points = 0
        for i, size in enumerate(prism_sizes):
            for j in range(size):
                self.PRISM_POINT_FIELDS.append('prism_' + str(i) + '_point_' + str(j) + '_x')
                self.PRISM_POINT_FIELDS.append('prism_' + str(i) + '_point_' + str(j) + '_y')
                total_points += 1
        self.PRISM_POINT_FMT = str(total_points * 2) + 'f'

    def parse_prism_sizes(self, msg):
        """Parse per-prism vertex counts from C++ handshake."""
        data = self.parse_msg(msg, 'prism_sizes', self.PRISM_SIZE_FIELDS, self.PRISM_SIZE_FMT)
        return list(data.values())

    def parse_prism_points(self, msg):
        """Parse prism vertex coordinates from C++ handshake."""
        data = self.parse_msg(msg, 'prism_points', self.PRISM_POINT_FIELDS, self.PRISM_POINT_FMT)
        return list(data.values())

    #
    # Parse the fields from a message
    # Returns a dictionary
    #
    def parse_msg(self, msg, msgtype, fields, fmt):
        import struct
        expected_size = struct.calcsize(fmt)
        if len(msg) != expected_size:
            raise ValueError(
                f"Message size mismatch for '{msgtype}': "
                f"expected {expected_size} bytes, got {len(msg)}"
            )
        cache_key = msgtype
        if cache_key not in self._namedtuple_cache:
            self._namedtuple_cache[cache_key] = namedtuple(msgtype, fields)
        Tx = self._namedtuple_cache[cache_key]
        x = Tx._make(unpack(fmt, msg))
        return x._asdict()

    #
    # Parse the experiment status
    # Returns a tuple (exp_done, episode_done, reached_goal)
    #
    def parse_status(self, msg):
        data = self.parse_msg(msg, 'status', self.EXPERIMENT_FIELDS, self.EXPERIMENT_FMT)
        exp_done = (data['exp_done'] == 1)
        episode_done = (data['episode_done'] == 1)
        reached_goal = (data['reached_goal'] == 1)
        return exp_done, episode_done, reached_goal

    #
    # Parse the observations
    # Returns a list of numpy arrays, one per robot
    #
    def parse_obs(self, msg):
        R = self.params['num_robots']
        n = self.params['num_obs']
        # T5: single frombuffer + reshape replaces R per-robot unpack+fromiter loops.
        # Safe because ARGoS observation messages are contiguous LE float32 (T0.3).
        matrix = np.frombuffer(msg, dtype='<f4').reshape(R, n)
        return [matrix[r].copy() for r in range(R)]

    def parse_failures(self, msg):
        R = self.params['num_robots']
        # T5: single frombuffer replaces R per-robot unpack+fromiter loops.
        # FAILURE_FMT is '1I' (unsigned int) — np.intc matches the legacy dtype.
        raw = np.frombuffer(msg, dtype='<u4').reshape(R, 1)
        return [raw[r].astype(np.intc) for r in range(R)]

    def parse_rewards(self, msg):
        R = self.params['num_robots']
        # T5: single frombuffer replaces R per-robot unpack+fromiter loops.
        raw = np.frombuffer(msg, dtype='<f4').reshape(R, 1)
        return [raw[r].copy() for r in range(R)]

    def parse_stats(self, msg):
        R = self.params['num_robots']
        n = self.params['num_stats']
        # T5: single frombuffer replaces R per-robot unpack+fromiter loops.
        raw = np.frombuffer(msg, dtype='<f4').reshape(R, n)
        return [raw[r].copy() for r in range(R)]
    
    def parse_robot_stats(self, msg):
        R = self.params['num_robots']
        n = len(self.ROBOT_STATS_FIELDS)   # always 6
        # T5: single frombuffer replaces R per-robot unpack+fromiter loops.
        raw = np.frombuffer(msg, dtype='<f4').reshape(R, n)
        return [raw[r].copy() for r in range(R)]

    def parse_obj_stats(self, msg):
        # Parse the bytes into a dictionary
        data = self.parse_msg(msg, 'obj_stats', self.OBJ_STATS_FIELDS, self.OBJ_STATS_FMT)
        # Make a numpy array
        obj_stats = np.fromiter(data.values(), dtype=np.float32, count = len(data))
        return obj_stats

    def parse_obstacle_stats(self, msg):
        # Parse the bytes into a dictionary
        data = self.parse_msg(msg, 'obstacle_stats', self.OBSTACLE_STATS_FIELDS, self.OBSTACLE_STATS_FMT)
        # Make a numpy array
        obj_stats = np.fromiter(data.values(), dtype=np.float32, count = len(data))
        return obj_stats

    def parse_gate_stats(self, msg):
        # Parse the bytes into a dictionary
        data = self.parse_msg(msg, 'gate_stats', self.GATE_STATS_FIELDS, self.GATE_STATS_FMT)
        # Make a numpy array
        obj_stats = np.fromiter(data.values(), dtype=np.float32, count = len(data))
        return obj_stats


    def serialize_actions(self, actions):
        # T5: use the Struct cached at __init__ rather than creating a new one per call.
        packer = self._actions_packer
        msg = bytearray(self.FLOAT_SIZE * self.params['num_actions'] * self.params['num_robots'])
        # For each robot
        for r in range(0, self.params['num_robots']):
            offset = self.FLOAT_SIZE * self.params['num_actions'] * r
            packer.pack_into(msg, offset, *(actions[r]))
        return msg

    def parse_msgs(self, msgs):
        if len(msgs) not in (7, 8):
            raise ValueError(
                f"Expected 7-8 message parts, got {len(msgs)}. "
                f"Part sizes: {[len(m) for m in msgs]}"
            )
        env_observations = self.parse_obs(msgs[1])
        failures = self.parse_failures(msgs[2])
        rewards = self.parse_rewards(msgs[3])
        stats = self.parse_stats(msgs[4])
        robot_stats = self.parse_robot_stats(msgs[5])
        obj_stats = self.parse_obj_stats(msgs[6])
        return env_observations, failures, rewards, stats, robot_stats, obj_stats
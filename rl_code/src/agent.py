from gsp_rl.src.actors import Actor

import math
import numpy as np
import statistics
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from collections import deque, namedtuple
from torch.optim import Adam


class Agent(Actor):
    def __init__(
            self,
            config: dict,
            network: str,
            n_agents: int,
            n_obs: int,
            n_actions: int,
            options_per_action: int,
            id: int,
            min_max_action: float,
            meta_param_size: int,
            gsp: bool,
            recurrent: bool,
            attention: bool,
            neighbors: bool,
            gsp_input_size: int,
            gsp_output_size: int,
            gsp_min_max_action: float,
            gsp_look_back: int,
            gsp_sequence_length: int,
            broadcast: bool = False,
            prox_filter_angle_deg: float = 45.0,
            n_hop_neighbors: int = 1,
    ):
        if neighbors and broadcast:
            raise ValueError(
                "GSP variants neighbors=True and broadcast=True are mutually exclusive — "
                "they overload gsp_input_size differently. Pick one."
            )
        if neighbors:
            # 2 inputs from ownship (prev_gsp, avg_prox)
            # 2 inputs from each neighbor (prev_gsp, avg_prox)
            # 2*n_hop_neighbors for symmetry in both CW and CCW
            gsp_input_size = 2+2*(n_hop_neighbors*2)
        if broadcast:
            # GSP-B: each agent's view is (self_prox, self_prev_gsp) + (other_prox, other_prev_gsp)
            # for all (n_agents - 1) other agents. Total 2*n_agents. Known limitation:
            # coupled to team size, not transferable across num_robots.
            gsp_input_size = 2 * n_agents

        # Input enrichment flags (Change 2). Computed before super().__init__ so
        # the effective gsp_input_size can be passed to the parent Actor constructor.
        # Each flag adds extra dimensions to the per-agent slice in make_gsp_states.
        _include_goal = bool(config.get('GSP_INPUT_INCLUDE_GOAL', False))
        _include_cyl_rel = bool(config.get('GSP_INPUT_INCLUDE_CYL_REL', False))
        _full_prox = bool(config.get('GSP_INPUT_FULL_PROX', False))
        # Change 3 enrichment flags (self-slot additions, GSP-N only):
        #   GSP_INPUT_INCLUDE_PAYLOAD_STATE: +5 dims (payload vx/vy/omega + payload-to-goal dx/dy)
        #   GSP_INPUT_INCLUDE_SELF_DYNAMICS: +4 dims (robot vx/vy + force magnitude/angle)
        #   GSP_INPUT_TEMPORAL_STACK_K:      int ≥1; K>1 stacks last K obs, multiplies total size
        _include_payload_state = bool(config.get('GSP_INPUT_INCLUDE_PAYLOAD_STATE', False))
        _include_self_dynamics = bool(config.get('GSP_INPUT_INCLUDE_SELF_DYNAMICS', False))
        _temporal_stack_k = int(config.get('GSP_INPUT_TEMPORAL_STACK_K', 1))
        if _temporal_stack_k < 1:
            raise ValueError(
                f"GSP_INPUT_TEMPORAL_STACK_K must be >= 1, got {_temporal_stack_k}"
            )
        # Base slot: self_avg_prox (1) + self_prev_gsp (1) + 2 per neighbor pair
        # When neighbors=True, gsp_input_size is already the base neighbor layout.
        # Enrichment flags are additive on top of the base per-agent layout:
        #   GSP_INPUT_INCLUDE_GOAL:           +2 per agent (cos/sin of angle_to_goal)
        #   GSP_INPUT_INCLUDE_CYL_REL:        +2 per agent (dist_to_cyl, angle_to_cyl)
        #   GSP_INPUT_FULL_PROX:              replace avg_prox(1) with raw_prox(24) → net +23
        #   GSP_INPUT_INCLUDE_PAYLOAD_STATE:  +5 per agent (self-slot only)
        #   GSP_INPUT_INCLUDE_SELF_DYNAMICS:  +4 per agent (self-slot only)
        #   GSP_INPUT_TEMPORAL_STACK_K:       multiplicative — total × K after all additive flags
        #
        # For the GSP-N layout each agent's slot is 2 (self_prox, self_prev_gsp).
        # The additions are per-slot, not per-neighbor. We compute the enrichment
        # delta per agent slot and multiply by the number of slots (1 self + N neighbors).
        if neighbors and (gsp_input_size > 0):
            n_slots = 1 + n_hop_neighbors * 2  # self + neighbors
        else:
            n_slots = 1  # non-neighbors: single shared state vector (not per-agent slots)
        _extra_per_slot = (2 if _include_goal else 0) + (2 if _include_cyl_rel else 0)
        _prox_delta = 23 if _full_prox else 0  # replace 1 avg_prox with 24 raw_prox
        _self_slot_extra = (5 if _include_payload_state else 0) + (4 if _include_self_dynamics else 0)
        if neighbors:
            # Only self-slot gets enrichment; neighbor slots keep their (prox, gsp) layout.
            gsp_input_size = gsp_input_size + _extra_per_slot + _prox_delta + _self_slot_extra
        else:
            gsp_input_size = gsp_input_size + _extra_per_slot + _prox_delta + _self_slot_extra
        # Temporal stacking multiplies the total (base + all additive enrichments).
        gsp_input_size = gsp_input_size * _temporal_stack_k

        output_size = n_actions
        if network in ['DQN', 'DDQN']:
            output_size = options_per_action**n_actions

        # Store enrichment flags before super().__init__ so make_gsp_states
        # can reference them. They must be set BEFORE the Actor constructor runs
        # because Actor.__init__ → NetworkAids.__init__ → Hyperparameters.__init__
        # only reads config keys, not these attributes; we set them here directly.
        # (They are also stored on self after super() returns — this pre-assignment
        # is to make them available if any super().__init__ code calls back into
        # Agent methods, which currently does not happen but guards future changes.)
        self._gsp_input_include_goal = _include_goal
        self._gsp_input_include_cyl_rel = _include_cyl_rel
        self._gsp_input_full_prox = _full_prox
        self._gsp_input_include_payload_state = _include_payload_state
        self._gsp_input_include_self_dynamics = _include_self_dynamics
        self._gsp_input_temporal_stack_k = _temporal_stack_k

        gsp_rl_args = {
            'config': config,
            'network': network,
            'id':id,
            'input_size':n_obs,
            'output_size':output_size,
            'min_max_action': min_max_action,
            'meta_param_size':meta_param_size, 
            'gsp':gsp,
            'recurrent_gsp':recurrent,
            'attention': attention,
            'gsp_input_size': gsp_input_size,
            'gsp_output_size': gsp_output_size,
            'gsp_min_max_action': gsp_min_max_action,
            'gsp_look_back':gsp_look_back,
            'gsp_sequence_length': gsp_sequence_length
        }
        super().__init__(**gsp_rl_args)

        self._n_agents = n_agents
        self._network = network
        self._n_actions = n_actions
        self._neighbors = neighbors
        self._broadcast = broadcast
        self._n_hop_neighbors = n_hop_neighbors
        self.neighbors_dict = {}
        self._options_per_action = options_per_action
        self._prox_filter_angle_deg = prox_filter_angle_deg


        # Ring buffer slot size: when temporal stacking is active (K>1), each slot
        # stores a single-step vector of size gsp_network_input // K. The K-step
        # stacked output is assembled in make_gsp_states from the last K slots.
        # When K=1 (default), slot size equals gsp_network_input — identical to
        # previous behavior so K=1 is a strict no-op.
        _k = getattr(self, '_gsp_input_temporal_stack_k', 1)
        _ring_slot_size = self.gsp_network_input // _k

        if self._neighbors or self._broadcast:
            # Per-agent observation ring buffers: GSP-N and GSP-B both produce
            # per-agent self-centric views, so each agent has its own history.
            self.gsp_observation = []
            for _ in range(self._n_agents):
                self.gsp_observation.append([[0 for _ in range(_ring_slot_size)] for _ in range(self.gsp_sequence_length)])
        else:
            self.gsp_observation = [[0 for _ in range(_ring_slot_size)] for _ in range(self.gsp_sequence_length)]

        # Per-agent LSTM hidden state for R-GSP-N inference
        self._agent_hidden_states = {}
        if self._neighbors and recurrent:
            for i in range(self._n_agents):
                self._agent_hidden_states[i] = None  # None = zeros on first call

        self._ROBOT_PROXIMITY_ANGLES = [7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5,
                                       112.5, 127.5, 142.5, 157.5, 172.5, -172.5, 
                                       -157.5, -142.5, -127.5, -112.5, -97.5, 
                                       -82.5, -67.5, -52.5, -37.5, -22.5, -7.5]
        if self._neighbors:
            self.build_neighbors()

        # Candidate A — future-prox delayed-label buffer. Active only when
        # GSP_PREDICTION_TARGET == 'future_prox'. Stores (state_per_robot,
        # gsp_obs_per_robot) snapshots so that K steps later we can pair the
        # snapshot with the robot's own current proximity reading as the label.
        # FIFO of length up to K+1 — once full, push followed by pop yields
        # the K-step-ago entry. K=GSP_PREDICTION_HORIZON.
        self._gsp_label_buffer: deque = deque()

    @property
    def gsp_neighbors(self):
        return self._neighbors

    @property
    def gsp_broadcast(self):
        return self._broadcast

    @property
    def n_agents(self):
        return self._n_agents

    def reset_hidden_states(self):
        """Reset all per-agent LSTM hidden states. Call at episode boundaries."""
        for i in self._agent_hidden_states:
            self._agent_hidden_states[i] = None

    def push_pending_gsp_obs(self, state_per_robot, gsp_obs_per_robot):
        """Future-prox mode: snapshot per-robot (state, gsp_obs) for label maturation
        K steps later. No-op when target != 'future_prox'."""
        if getattr(self, 'gsp_prediction_target', 'delta_theta') != 'future_prox':
            return
        self._gsp_label_buffer.append({
            'state_per_robot': [np.asarray(s).copy() for s in state_per_robot],
            'gsp_obs_per_robot': [np.asarray(g).copy() for g in gsp_obs_per_robot],
        })

    def pop_matured_gsp_label(self, current_prox_per_robot):
        """Future-prox mode: if buffer has K+1 entries, pop the oldest snapshot and
        return it paired with current per-robot prox as the label. Returns None when
        buffer is too small or target != 'future_prox'."""
        if getattr(self, 'gsp_prediction_target', 'delta_theta') != 'future_prox':
            return None
        K = getattr(self, 'gsp_prediction_horizon', 5)
        if len(self._gsp_label_buffer) < K + 1:
            return None
        oldest = self._gsp_label_buffer.popleft()
        return {
            'state_per_robot': oldest['state_per_robot'],
            'gsp_obs_per_robot': oldest['gsp_obs_per_robot'],
            'label_per_robot': np.asarray(current_prox_per_robot, dtype=np.float32).copy(),
        }

    def reset_gsp_label_buffer(self):
        """Future-prox mode: clear the buffer. Call at episode boundaries so labels
        from the previous episode never bleed into the next."""
        self._gsp_label_buffer.clear()

    def build_neighbors(self):
        agents_available = np.arange(self.n_agents)
        for agent in range(self.n_agents):
            neighbors = []
            for i in range(1, self._n_hop_neighbors+1):
                neighbors.append(agents_available[agent-i])
                neighbors.append(agents_available[(agent+1)%self.n_agents])
            self.neighbors_dict[agent] = neighbors
    
    def make_agent_state(self, env_obs, heading_gsp=None, global_knowledge=None):
        if heading_gsp is not None:
            # H-14 GSP-minus ablation: if the zero-out flag is set, the GSP slot
            # in the actor's augmented observation is forced to zeros regardless of
            # what the GSP head predicted. The head itself still runs and trains
            # normally; only the signal path from head to actor is severed.
            # This is the QMIP-minus test of "does the prediction contribute?".
            if getattr(self, 'gsp_zero_out_signal', False):
                gsp_output_size = getattr(self, 'gsp_network_output', 1)
                gsp_slot = np.zeros(gsp_output_size, dtype=np.float32)
            else:
                # Multi-dim GSP output support (Change 1 — GSP_OUTPUT_KIND):
                # heading_gsp may be a scalar (legacy, O=1) or a numpy array (O>1).
                # For the legacy scalar case, apply the historical degrees/10 scaling
                # so that network weights trained on 'delta_theta_1d' are compatible.
                # For vector cases (cyl_kinematics_3d/goal_4d/time_to_goal_1d) the
                # values are already in physical units from the label computation in
                # Main.py and are concatenated as-is (no extra scaling).
                heading_gsp_arr = np.asarray(heading_gsp, dtype=np.float32)
                if heading_gsp_arr.ndim == 0 or heading_gsp_arr.size == 1:
                    # Scalar path — preserve legacy degrees/10 normalization.
                    scalar_val = float(heading_gsp_arr.ravel()[0])
                    gsp_slot = np.array([np.degrees(scalar_val / 10)], dtype=np.float32)
                else:
                    # Vector path — physical units from label computation, no rescaling.
                    gsp_slot = heading_gsp_arr.ravel()
            if global_knowledge is not None:
                env_obs = np.concatenate((env_obs, gsp_slot, global_knowledge))
            else:
                env_obs = np.concatenate((env_obs, gsp_slot))
        elif global_knowledge is not None:
            env_obs = np.concatenate((env_obs, global_knowledge))
        return env_obs
    
    def make_gsp_states_broadcast(self, agent_prox_values, agent_prev_gsp):
        """Build per-agent GSP inputs for GSP-B (full-broadcast variant).

        Each agent's view is self-first: [self_prox, self_prev_gsp, other_0_prox,
        other_0_prev_gsp, other_1_prox, other_1_prev_gsp, ..., other_{n-1}_prox,
        other_{n-1}_prev_gsp]. "other" iterates all agents in ascending id order,
        skipping self. Total length = 2 * n_agents.

        Known limitation: the network input size is coupled to n_agents, so a
        trained GSP-B policy does not transfer to teams of different size. This
        is the tradeoff vs GSP-N, which uses fixed (self + n_hop_neighbors * 2)
        inputs and transfers across team sizes.
        """
        states = []
        for agent in range(self._n_agents):
            agent_state = np.zeros(self.gsp_network_input)
            # Self first
            agent_state[0] = agent_prox_values[agent]
            agent_state[1] = agent_prev_gsp[agent]
            i = 2
            # Then every other agent in ascending id order, skipping self
            for other in range(self._n_agents):
                if other == agent:
                    continue
                agent_state[i] = agent_prox_values[other]
                agent_state[i + 1] = agent_prev_gsp[other]
                i += 2
            # Maintain gsp_observation ring buffer the same way make_gsp_states does,
            # so recurrent/attention variants can still see sequences if added later.
            self.gsp_observation[agent].pop(0)
            self.gsp_observation[agent].append(agent_state)
            states.append(agent_state)
        return states

    def make_gsp_states(self, agent_prox_values, agent_prev_gsp, return_prox_flags=False,
                        env_observations=None, payload_state=None, self_dynamics=None):
        """Build per-agent GSP input vectors for GSP-N (neighbor) variant.

        Base layout per agent (2 dims for self + 2 per neighbor pair):
            [self_avg_prox, self_prev_gsp, n0_prox, n0_prev_gsp, ...]

        Optional enrichment (Change 2 — GSP_INPUT_INCLUDE_* flags):
            GSP_INPUT_INCLUDE_GOAL:    appends (cos(angle_to_goal), sin(angle_to_goal))
                                       to the self-slot using env_observations[i][1].
            GSP_INPUT_INCLUDE_CYL_REL: appends (dist_to_cyl, angle_to_cyl) to the
                                       self-slot using env_observations[i][4:6].
            GSP_INPUT_FULL_PROX:       replaces self_avg_prox (1 value) with the full
                                       24-dim raw proximity vector from
                                       env_observations[i][7:31], net +23 dims.

        Optional enrichment (Change 3 — new flags, GSP-N self-slot only):
            GSP_INPUT_INCLUDE_PAYLOAD_STATE: appends 5 dims to self-slot:
                (payload_vx, payload_vy, payload_omega, payload_to_goal_dx, payload_to_goal_dy)
                Requires payload_state kwarg: dict with per-robot keys
                  'vx', 'vy', 'omega', 'dx_to_goal', 'dy_to_goal' (lists/arrays, indexed by agent).
            GSP_INPUT_INCLUDE_SELF_DYNAMICS: appends 4 dims to self-slot:
                (self_vx, self_vy, force_magnitude, force_angle)
                Requires self_dynamics kwarg: dict with per-robot keys
                  'vx', 'vy', 'force_mag', 'force_ang' (lists/arrays, indexed by agent).
            GSP_INPUT_TEMPORAL_STACK_K (int, default 1): after building the per-agent
                vector, flatten the last K entries from the ring buffer (current + K-1
                previous). K=1 is a strict no-op. Effective input size becomes base×K.

        Enrichment only applies to the self-slot; neighbor slots always stay at their
        compact (prox, prev_gsp) layout — those agents' goal/cyl data is unavailable
        from the current agent's perspective in a decentralized system.

        Args:
            agent_prox_values: per-agent averaged (filtered) proximity scalars.
            agent_prev_gsp: per-agent previous GSP prediction scalars.
            return_prox_flags: if True, also return the flat list of prox values used.
            env_observations: list of raw per-robot observation vectors from ARGoS.
                Required when any GSP_INPUT_INCLUDE_* or GSP_INPUT_FULL_PROX flag is
                True; ignored otherwise. Indices used:
                  [1]    — robot's angle to goal (radians)
                  [4]    — cyl distance to robot
                  [5]    — cyl angle to robot (radians)
                  [7:31] — 24-dim raw proximity readings (when GSP_INPUT_FULL_PROX)
            payload_state: dict with keys 'vx', 'vy', 'omega', 'dx_to_goal',
                'dy_to_goal' — each a list/array indexed by agent id. Required when
                GSP_INPUT_INCLUDE_PAYLOAD_STATE is True; ignored otherwise.
            self_dynamics: dict with keys 'vx', 'vy', 'force_mag', 'force_ang' —
                each a list/array indexed by agent id. Required when
                GSP_INPUT_INCLUDE_SELF_DYNAMICS is True; ignored otherwise.
        """
        include_goal = getattr(self, '_gsp_input_include_goal', False)
        include_cyl_rel = getattr(self, '_gsp_input_include_cyl_rel', False)
        full_prox = getattr(self, '_gsp_input_full_prox', False)
        include_payload_state = getattr(self, '_gsp_input_include_payload_state', False)
        include_self_dynamics = getattr(self, '_gsp_input_include_self_dynamics', False)
        temporal_stack_k = getattr(self, '_gsp_input_temporal_stack_k', 1)
        need_env_obs = include_goal or include_cyl_rel or full_prox

        # When K>1 we need to know the unflattened single-step size so we can
        # correctly index into the ring buffer. The ring buffer stores single-step
        # vectors; gsp_network_input is already total_size * K when K>1.
        # Derive the per-step size by dividing by K.
        single_step_size = self.gsp_network_input // temporal_stack_k

        states = []
        prox_flags = []
        for agent in range(self._n_agents):
            agent_state = np.zeros(single_step_size)
            neighbors = self.neighbors_dict[agent]

            # --- Self slot ---
            idx = 0
            if full_prox and need_env_obs and env_observations is not None:
                # Replace scalar avg_prox with full 24-dim raw prox vector.
                raw_prox = np.asarray(env_observations[agent][7:31], dtype=np.float32)
                agent_state[idx:idx + 24] = raw_prox
                idx += 24
            else:
                agent_state[idx] = agent_prox_values[agent]
                idx += 1
            agent_state[idx] = agent_prev_gsp[agent]
            idx += 1
            prox_flags.append(agent_prox_values[agent])

            # Optional enrichment: goal direction (cos/sin of angle_to_goal)
            if include_goal and need_env_obs and env_observations is not None:
                angle_to_goal = float(env_observations[agent][1])
                agent_state[idx] = math.cos(angle_to_goal)
                agent_state[idx + 1] = math.sin(angle_to_goal)
                idx += 2

            # Optional enrichment: cylinder relative (dist_to_cyl, angle_to_cyl)
            if include_cyl_rel and need_env_obs and env_observations is not None:
                agent_state[idx] = float(env_observations[agent][4])
                agent_state[idx + 1] = float(env_observations[agent][5])
                idx += 2

            # Optional enrichment: payload kinematics + payload-to-goal offset.
            # 5 dims: payload_vx, payload_vy, payload_omega, payload_to_goal_dx,
            # payload_to_goal_dy. All values shared across agents (same payload),
            # but indexed per agent for API consistency with self_dynamics.
            if include_payload_state and payload_state is not None:
                agent_state[idx] = float(payload_state['vx'][agent])
                agent_state[idx + 1] = float(payload_state['vy'][agent])
                agent_state[idx + 2] = float(payload_state['omega'][agent])
                agent_state[idx + 3] = float(payload_state['dx_to_goal'][agent])
                agent_state[idx + 4] = float(payload_state['dy_to_goal'][agent])
                idx += 5

            # Optional enrichment: per-robot kinematics + applied force.
            # 4 dims: self_vx, self_vy, force_magnitude, force_angle.
            if include_self_dynamics and self_dynamics is not None:
                agent_state[idx] = float(self_dynamics['vx'][agent])
                agent_state[idx + 1] = float(self_dynamics['vy'][agent])
                agent_state[idx + 2] = float(self_dynamics['force_mag'][agent])
                agent_state[idx + 3] = float(self_dynamics['force_ang'][agent])
                idx += 4

            # --- Neighbor slots (compact layout — no enrichment) ---
            for neighbor in neighbors:
                agent_state[idx] = agent_prox_values[neighbor]
                agent_state[idx + 1] = agent_prev_gsp[neighbor]
                prox_flags.append(agent_prox_values[neighbor])
                idx += 2

            # Update ring buffer with the new single-step vector.
            # For K=1 the ring buffer stores full-size vectors (same as before).
            # For K>1 the ring buffer stores single-step vectors; the stacked output
            # is assembled below from the last K entries.
            self.gsp_observation[agent].pop(0)
            self.gsp_observation[agent].append(agent_state)

            # Temporal stacking: flatten last K entries from ring buffer.
            # gsp_observation[agent] is a list of single-step vectors, newest last.
            # K=1 returns the single-step vector unchanged — strict no-op.
            if temporal_stack_k == 1:
                stacked = agent_state
            else:
                # Take the last K entries (newest at end); flatten in temporal order
                # oldest-first so the model sees a causal sequence.
                history = self.gsp_observation[agent]
                k_entries = history[-temporal_stack_k:]
                stacked = np.concatenate(k_entries).astype(np.float32)

            states.append(stacked)
        if return_prox_flags:
            return states, prox_flags
        return states
    
    def filter_prox_values(self, prox_values, angle_to_cyl):
        if angle_to_cyl > 0:
            if angle_to_cyl > 180-self._prox_filter_angle_deg:
                cw_lim = angle_to_cyl + self._prox_filter_angle_deg - 360
            else:
                cw_lim = angle_to_cyl+self._prox_filter_angle_deg
            ccw_lim = angle_to_cyl - self._prox_filter_angle_deg
        elif angle_to_cyl < 0:
            if angle_to_cyl < -180 +self._prox_filter_angle_deg:
                ccw_lim = angle_to_cyl-self._prox_filter_angle_deg+360
            else:
                ccw_lim = angle_to_cyl - self._prox_filter_angle_deg
            cw_lim = angle_to_cyl + self._prox_filter_angle_deg
        else:
            cw_lim = self._prox_filter_angle_deg
            ccw_lim = -self._prox_filter_angle_deg

        index = []
        filtered_prox_values = []
        if angle_to_cyl > 180 - self._prox_filter_angle_deg:
            for i in range(len(self._ROBOT_PROXIMITY_ANGLES)):
                if self._ROBOT_PROXIMITY_ANGLES[i] > ccw_lim:
                    index.append(i)
                elif self._ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                    index.append(i)
                else:
                    filtered_prox_values.append(prox_values[i])
        elif angle_to_cyl < -180+self._prox_filter_angle_deg:
            for i in range(len(self._ROBOT_PROXIMITY_ANGLES)):
                if self._ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                    index.append(i)
                elif self._ROBOT_PROXIMITY_ANGLES[i] > ccw_lim:
                    index.append(i)
                else:
                    filtered_prox_values.append(prox_values[i]) 
        else:
            for i in range(len(self._ROBOT_PROXIMITY_ANGLES)):
                if self._ROBOT_PROXIMITY_ANGLES[i] > ccw_lim and self._ROBOT_PROXIMITY_ANGLES[i] < cw_lim:
                    index.append(i)
                else:
                    filtered_prox_values.append(prox_values[i])
        return filtered_prox_values, index
    
    def choose_agent_action(self, observation, failures, test=False):
        if self._network == 'None':
            # Not sure what to do here for no learning
            return [0, 0, 0], 0

        if failures:
            self.failed = True
            return self.failure_action, self.failure_action_code

        self.failed = False
        if self.networks['learning_scheme'] in ['DQN', 'DDQN']:
            action_num = self.choose_action(observation, self.networks, test)
            actions = self.parse_action(action_num)

        if self.networks['learning_scheme'] in ['DDPG', 'TD3']:
            actions = self.choose_action(observation, self.networks, test)
            actions = np.pad(actions, (0, 1))
            action_num = None

        return actions, action_num
    
    def choose_agent_gsp(self, agent_gsp_states, test = False):
        if self._neighbors or self._broadcast:
            # Per-agent predictions with self-centric inputs. GSP-N (neighbors)
            # and GSP-B (broadcast) share the same per-agent forward-pass shape;
            # only the input vector differs. Non-recurrent broadcast uses the
            # same stateless path as non-recurrent neighbors.
            actions = []
            for i in range(self._n_agents):
                if self.recurrent_gsp:
                    hidden = self._agent_hidden_states.get(i)
                    obs = T.tensor(np.array(self.gsp_observation[i]), dtype=T.float).to(
                        self.gsp_networks['actor'].device)
                    # RDDPG actor forward returns (action, (h_n, c_n))
                    with T.no_grad():
                        action_tensor, new_hidden = self.gsp_networks['actor'](obs, hidden=hidden)
                    self._agent_hidden_states[i] = (
                        new_hidden[0].detach(), new_hidden[1].detach()
                    )
                    # Take the last timestep's action
                    actions.append(action_tensor[-1].cpu().detach().numpy())
                else:
                    actions.append(self.choose_action(agent_gsp_states[i], self.gsp_networks, test))
            return actions
        else:
            if self.recurrent_gsp:
                self.gsp_observation.append(agent_gsp_states)
                self.gsp_observation.pop(0)
                action = self.choose_action(self.gsp_observation, self.gsp_networks, test)
                return action
            
            observation = np.array(agent_gsp_states)
            return self.choose_action(observation, self.gsp_networks, test)

    def parse_action(self, action_num):
        '''
        This function will parse the number action to
        a set of wheel actions:

        0 - (- 1,-1)
        1 - (-1, 0)
        2 - (-1, 1)
        3 - (0, -1)
        4 - (0, 0)
        5 - (0, 1)
        6 - (1, -1)
        7 - (1, 0)
        8 - (1, 1)
        '''
        if action_num < 0 or action_num >=self._options_per_action**self._n_actions:
            raise Exception('Action Number Out of Range:'+str(action_num))
        l_wheel = round((math.floor(action_num/self._options_per_action) - 1)/10.0, 1)
        r_wheel = round((action_num%self._options_per_action - 1)/10.0, 1)
        # Trailing zero is hardcoded control for gripper
        return np.array([l_wheel, r_wheel, 0])
    
    def store_agent_transition(self, s, a, r, s_, d, gsp_obs=None, gsp_label=None):
        if self.networks['replay'].action_type == 'Discrete':
            a = a[0]
        elif self.networks['replay'].action_type == 'Continuous':
            a = np.array(a[1][0:2])
        return super().store_agent_transition(s, a, r, s_, d, gsp_obs=gsp_obs, gsp_label=gsp_label)
    
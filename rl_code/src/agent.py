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
        # Multi-dim GSP output support (Change 1 — GSP_OUTPUT_KIND).
        # Source of truth for the size dict is GSP-RL learning_aids.py:287.
        # This local copy is kept in sync via test_multi_dim_gsp_input_size (sync
        # test in tests/test_agent/test_multi_dim_gsp.py). Update both together.
        # A value of None marks a HORIZON-COUPLED kind whose output dim is not a
        # fixed constant but equals GSP_PREDICTION_HORIZON (resolved below). Both
        # this local copy and GSP-RL's copy resolve it from the SAME config key so
        # the head output width and the actor/neighbor input width always agree.
        _GSP_OUTPUT_KIND_SIZES = {
            'delta_theta_1d': 1,
            'future_prox_1d': 1,
            'cyl_kinematics_3d': 3,
            'cyl_kinematics_goal_4d': 4,
            'time_to_goal_1d': 1,
            'neighbor_force_1d': 1,
            # delta_theta_traj: size == K == GSP_PREDICTION_HORIZON. The label is the
            # per-step payload-rotation trajectory [Δθ(t→t+1), …, Δθ(t+K-1→t+K)].
            'delta_theta_traj': None,
            # goal_progress_traj: size == K. The label is the per-step payload
            # progress-to-goal trajectory [Δd(t→t+1), …] with Δd = prev_dist2goal
            # − curr_dist2goal (positive = toward goal — the exact quantity from
            # cyl_kinematics_goal_4d's 4th component). RAW meters, no scaling.
            'goal_progress_traj': None,
            # cyl_displacement_traj: size == 2K. The label is the per-step payload
            # displacement trajectory flattened [Δx1,Δy1,…,ΔxK,ΔyK]. RAW meters.
            'cyl_displacement_traj': None,
        }
        # Horizon multiplier per horizon-coupled kind (dict value None above):
        # effective size == mult * GSP_PREDICTION_HORIZON. Kept in lockstep with
        # the GSP-RL copy (learning_aids.py) via the sync tests.
        _GSP_TRAJ_KIND_HORIZON_MULTIPLIER = {
            'delta_theta_traj': 1,
            'goal_progress_traj': 1,
            'cyl_displacement_traj': 2,
        }
        # Trajectory PREDICTION TARGETS whose names double as their required
        # GSP_OUTPUT_KIND (auto-derived below; explicit contradiction rejected).
        _GSP_TRAJ_TARGETS = (
            'delta_theta_traj', 'goal_progress_traj', 'cyl_displacement_traj'
        )
        _gsp_output_kind = str(config.get('GSP_OUTPUT_KIND', 'delta_theta_1d'))
        if _gsp_output_kind not in _GSP_OUTPUT_KIND_SIZES:
            raise ValueError(
                f"Unknown GSP_OUTPUT_KIND '{_gsp_output_kind}'. "
                f"Valid values: {list(_GSP_OUTPUT_KIND_SIZES)}"
            )
        # Keep GSP_OUTPUT_KIND consistent with GSP_PREDICTION_TARGET — mirror of
        # GSP-RL learning_aids.py (#29). The GSP head OUTPUT width is fixed by the
        # TARGET there; here on the host side the same K sizes the head INPUT and
        # the neighbor-shared prev_gsp slots. When the target is the size-K
        # trajectory but GSP_OUTPUT_KIND was left at the scalar default, the two
        # submodules disagreed on K: GSP-RL built a width-K head while RL-CT sized
        # the input/neighbor slots for K=1. The runtime symptom was the actor
        # forward pass `mat1 and mat2 shapes cannot be multiplied (64x40 and
        # 36x64)` — the augmented actor-state width diverged from the actor net's
        # input width. Auto-derive the kind (and therefore K) from the target when
        # left at the scalar default, and reject an explicit contradiction loudly.
        # (Confirmed 2026-07-08; this must stay in lockstep with the GSP-RL copy.)
        _prediction_target = str(config.get('GSP_PREDICTION_TARGET', 'delta_theta'))
        if _prediction_target in _GSP_TRAJ_TARGETS:
            if _gsp_output_kind == 'delta_theta_1d':
                _gsp_output_kind = _prediction_target
            elif _gsp_output_kind != _prediction_target:
                raise ValueError(
                    f"GSP_PREDICTION_TARGET='{_prediction_target}' requires "
                    f"GSP_OUTPUT_KIND='{_prediction_target}' (the size-K trajectory "
                    f"output); got GSP_OUTPUT_KIND='{_gsp_output_kind}'."
                )
        # K = effective output dims for the GSP head.  Used to compute gsp_input_size
        # so the head's recurrent prev_gsp slot accommodates the full prediction vector.
        # For horizon-coupled kinds (dict value None) the size is
        # multiplier * GSP_PREDICTION_HORIZON (mult 1 for delta_theta_traj /
        # goal_progress_traj, 2 for cyl_displacement_traj's per-step Δx,Δy pairs).
        _K = _GSP_OUTPUT_KIND_SIZES[_gsp_output_kind]
        if _K is None:
            _horizon = int(config.get('GSP_PREDICTION_HORIZON', 5))
            if _horizon < 1:
                raise ValueError(
                    f"GSP_OUTPUT_KIND='{_gsp_output_kind}' requires "
                    f"GSP_PREDICTION_HORIZON >= 1, got {_horizon}"
                )
            _K = _GSP_TRAJ_KIND_HORIZON_MULTIPLIER[_gsp_output_kind] * _horizon

        if neighbors:
            # (1 + K) inputs from ownship (avg_prox × 1, prev_gsp × K)
            # (1 + K) inputs from each neighbor (avg_prox × 1, prev_gsp × K)
            # 2*n_hop_neighbors for symmetry in both CW and CCW
            gsp_input_size = (1 + _K) + (1 + _K) * (n_hop_neighbors * 2)
        if broadcast:
            # GSP-B: each agent's view is (self_prox, self_prev_gsp) + (other_prox, other_prev_gsp)
            # for all (n_agents - 1) other agents. Total 2*n_agents. Known limitation:
            # coupled to team size, not transferable across num_robots.
            gsp_input_size = 2 * n_agents

        # Determinism flag (Phase 4). When true, the caller must have already applied
        # determinism settings via apply_determinism_settings() before constructing
        # the Agent. This attribute is stored on the Agent so it can be queried by
        # tests and by Main.py. Default false so all existing batches are unaffected.
        self.determinism_enabled = bool(config.get('DETERMINISM_ENABLED', False))

        # Input enrichment flags (Change 2). Computed before super().__init__ so
        # the effective gsp_input_size can be passed to the parent Actor constructor.
        # Each flag adds extra dimensions to the per-agent slice in make_gsp_states.
        _include_goal = bool(config.get('GSP_INPUT_INCLUDE_GOAL', False))
        _include_cyl_rel = bool(config.get('GSP_INPUT_INCLUDE_CYL_REL', False))
        # Explicit wrap-safe one-step change in the robot's WORLD-FRAME bearing
        # around the cylinder. The GSP target (cylinder rotation) is ~0.77-0.89
        # predictable from this delta (verified on run h5), whereas the delta of the
        # BODY-FRAME angle_to_cyl (env_observations[i][5]) correlates only ~0.003
        # with the target. The world-frame bearing atan2(robot_y - cyl_y,
        # robot_x - cyl_x) is computed in Main.py from world positions and passed in
        # via the cyl_bearing_delta arg. This +1 self-slot dim feeds that pre-computed
        # wrap-safe delta (radians). See make_gsp_states.
        _include_cyl_bearing_delta = bool(config.get('GSP_INPUT_INCLUDE_CYL_BEARING_DELTA', False))
        _full_prox = bool(config.get('GSP_INPUT_FULL_PROX', False))
        # Change 3 enrichment flags (self-slot additions, GSP-N only):
        #   GSP_INPUT_INCLUDE_PAYLOAD_STATE: +5 dims (payload vx/vy/omega + payload-to-goal dx/dy)
        #   GSP_INPUT_INCLUDE_SELF_DYNAMICS: +4 dims (robot vx/vy + force magnitude/angle)
        #   GSP_INPUT_TEMPORAL_STACK_K:      int ≥1; K>1 stacks last K obs, multiplies total size
        _include_payload_state = bool(config.get('GSP_INPUT_INCLUDE_PAYLOAD_STATE', False))
        _include_self_dynamics = bool(config.get('GSP_INPUT_INCLUDE_SELF_DYNAMICS', False))
        # Eval-time neighbor ablation (correctness-critical). When True, the neighbor
        # region of every per-agent GSP input vector is zeroed in make_gsp_states
        # (before ring-buffer/temporal stacking), neutralizing the neighbor signal
        # while leaving the self-slot + enrichment dims and input size untouched.
        # Default False is a strict bit-exact no-op vs current behavior.
        self._gsp_eval_ablate_neighbors = bool(config.get('GSP_EVAL_ABLATE_NEIGHBORS', False))
        # M2 — eval-time GSP PREDICTION ablation. The flag is host-side: parsed on
        # the Agent, consumed in Main.py at the single next_heading_gsp injection
        # site via src.pred_ablation.apply_pred_ablation. Modes:
        #   none        -> identity no-op (default; keeps training bit-exact)
        #   zero        -> replace the prediction with zeros
        #   shuffle     -> permute the prediction dims (same multiset)
        #   frozen_mean -> replace with the per-episode running mean of predictions
        # See docs/research/2026-07-04-gsp-actor-usage-instrumentation-prereg.md (M2).
        # Mirrors the GSP_ZERO_OUT_SIGNAL host-side flag pattern; default 'none' is
        # a strict bit-exact no-op vs current behavior.
        self.gsp_eval_ablate_pred = str(config.get('GSP_EVAL_ABLATE_PRED', 'none'))
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
        #   GSP_INPUT_INCLUDE_CYL_BEARING_DELTA: +1 self-slot (wrap-safe Δ angle_to_cyl, rad)
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
        _extra_per_slot = (2 if _include_goal else 0) + (2 if _include_cyl_rel else 0) + (1 if _include_cyl_bearing_delta else 0)
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
        self._gsp_input_include_cyl_bearing_delta = _include_cyl_bearing_delta
        self._gsp_input_full_prox = _full_prox
        self._gsp_input_include_payload_state = _include_payload_state
        self._gsp_input_include_self_dynamics = _include_self_dynamics
        self._gsp_input_temporal_stack_k = _temporal_stack_k
        # Deterministic (greedy) GSP prediction. The GSP head is a SUPERVISED
        # regressor (MSE against a true environment label), so adding DDPG
        # exploration noise to its rollout prediction is a category error:
        # the noise dominates the logged prediction, feeds back as the head's
        # own prev_gsp input, and hands the policy noise instead of signal.
        # When True, the prediction is greedy even during training (test=False).
        # Default False keeps all existing experiments bit-identical.
        self._gsp_prediction_deterministic = bool(config.get('GSP_PREDICTION_DETERMINISTIC', False))

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

        self._ROBOT_PROXIMITY_ANGLES = np.array([
            7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 97.5,
            112.5, 127.5, 142.5, 157.5, 172.5, -172.5,
            -157.5, -142.5, -127.5, -112.5, -97.5,
            -82.5, -67.5, -52.5, -37.5, -22.5, -7.5,
        ], dtype=np.float64)
        if self._neighbors:
            self.build_neighbors()

        # Candidate A — delayed-label FIFO. Active for any DELAYED-LABEL target:
        # GSP_PREDICTION_TARGET in {'future_prox', 'neighbor_force',
        # 'delta_theta_horizon'}. Stores (state_per_robot, gsp_obs_per_robot,
        # payload_angle_deg) snapshots so that K steps later we can pair the
        # t-snapshot with a label observed at t+K.
        #   - future_prox:        label_i = robot i's own current proximity at t+K.
        #   - neighbor_force:      label_i = mean applied force-magnitude of the OTHER
        #                          robots at t+K (mean_{j != i} force_magnitude[t+K, j]).
        #   - delta_theta_horizon: label   = wrap-safe payload rotation over the window,
        #                          cyl_angle(t+K) - cyl_angle(t) in degrees, SHARED by
        #                          every robot. This one needs the t (pushed-step)
        #                          payload angle to form the delta, so the FIFO entry
        #                          also carries payload_angle_deg; the driver reads it
        #                          back from the matured pop and subtracts it from the
        #                          current angle (see Main.py). For future_prox /
        #                          neighbor_force payload_angle_deg is simply None.
        # All reuse the SAME FIFO; only the VALUE(s) the caller passes to
        # pop_matured_gsp_label differs. FIFO of length up to K+1 — once full,
        # push followed by pop yields the K-step-ago entry. K=GSP_PREDICTION_HORIZON.
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

    # Targets whose GSP label is only observable K steps after the state is seen.
    # They all share the same push/pop FIFO; only the label VALUE the caller
    # supplies to pop_matured_gsp_label differs (see the buffer comment above).
    # 'delta_theta_traj' additionally uses the pushed-step payload angle carried
    # in EVERY FIFO entry (payload_angle_deg) to reconstruct the K-step angle
    # window and form the size-K per-step rotation trajectory.
    # 'goal_progress_traj' / 'cyl_displacement_traj' (the GLOBAL trajectory
    # targets, 2026-07-09) likewise use the per-step payload_track dict
    # ({'dist2goal','cyl_x','cyl_y'}) carried in every entry to reconstruct the
    # K-step payload track window and form the size-K progress / size-2K
    # displacement trajectory (raw meters, no scaling).
    _DELAYED_LABEL_TARGETS = (
        'future_prox', 'neighbor_force', 'delta_theta_traj',
        'goal_progress_traj', 'cyl_displacement_traj',
    )

    def _is_delayed_label_target(self):
        return getattr(self, 'gsp_prediction_target', 'delta_theta') in self._DELAYED_LABEL_TARGETS

    def push_pending_gsp_obs(self, state_per_robot, gsp_obs_per_robot,
                             payload_angle_deg=None, e2e_transition=None,
                             payload_track=None):
        """Delayed-label mode: snapshot per-robot (state, gsp_obs) for label
        maturation K steps later. No-op when the target is not a delayed-label
        target (see _DELAYED_LABEL_TARGETS).

        payload_angle_deg (optional): the payload's absolute rotation angle
        (degrees) at the pushed step. Needed by 'delta_theta_traj', whose label is
        the SEQUENCE of per-step rotations over the K-step window
        [Δθ(t→t+1), …, Δθ(t+K-1→t+K)]; every step's angle is carried here so the
        matured pop can hand back the full ordered angle window and the driver can
        difference consecutive entries. Left None for future_prox / neighbor_force
        (whose labels are fully computed at maturity from the current step).

        payload_track (optional): dict of per-step GLOBAL payload scalars
        ({'dist2goal': cyl distance-to-goal (m), 'cyl_x': payload x (m),
        'cyl_y': payload y (m)}) at the pushed step. Needed by the GLOBAL
        trajectory targets ('goal_progress_traj', 'cyl_displacement_traj'),
        whose labels are per-step differences over the K-step window; every
        step's track is carried here (mirroring payload_angle_deg) so the
        matured pop can hand back the full ordered payload_track_window and the
        driver can difference consecutive entries. Left None for all other
        targets — a strict no-op (the window is all-None).

        e2e_transition (optional): an opaque per-step payload (any object) carried
        verbatim through the FIFO and returned by pop_matured_gsp_label. Used by the
        E2E path (GSP_E2E_ENABLED) so the main-replay RL transition stored at step t
        matures co-indexed with the SAME K-step trajectory label the head regresses
        against — in E2E mode the head trains ONLY from learn_DDQN_e2e, so its label
        MUST be the future trajectory, not the immediate scalar. Default None is a
        strict no-op (matured['e2e_transition'] is None), byte-identical to the
        prior delayed-label behavior."""
        if not self._is_delayed_label_target():
            return
        self._gsp_label_buffer.append({
            'state_per_robot': [np.asarray(s).copy() for s in state_per_robot],
            'gsp_obs_per_robot': [np.asarray(g).copy() for g in gsp_obs_per_robot],
            'payload_angle_deg': (
                float(payload_angle_deg) if payload_angle_deg is not None else None
            ),
            'payload_track': (
                {k: float(v) for k, v in payload_track.items()}
                if payload_track is not None else None
            ),
            'e2e_transition': e2e_transition,
        })

    def pop_matured_gsp_label(self, current_label_per_robot):
        """Delayed-label mode: if the buffer holds K+1 entries, pop the oldest
        (t-K) snapshot and pair it with the per-robot label observed at the CURRENT
        step. The caller owns the label VALUE:
          - future_prox        → current per-robot proximity;
          - neighbor_force      → current per-robot neighbor-mean force-magnitude;
          - delta_theta_traj    → the driver instead reads back the ordered
                                   'payload_angle_window' (the K+1 payload angles
                                   spanning t-K … t) and differences consecutive
                                   entries into the size-K wrapped rotation
                                   trajectory itself; it may pass None here.
          - goal_progress_traj / cyl_displacement_traj
                                → the driver instead reads back the ordered
                                   'payload_track_window' (the K+1 payload-track
                                   dicts spanning t-K … t) and differences
                                   consecutive entries into the size-K progress /
                                   size-2K displacement trajectory itself; it may
                                   pass None here.
        The returned dict always includes:
          - 'payload_angle_deg'    : the pushed-step (t-K) payload angle, or None;
          - 'payload_angle_window' : an ordered list of the K+1 payload angles held
                                     in the buffer at pop time (oldest→newest, i.e.
                                     [angle(t-K), …, angle(t)]). Entries are None
                                     where no angle was supplied at push time.
          - 'payload_track_window' : an ordered list of the K+1 payload-track dicts
                                     held in the buffer at pop time (oldest→newest).
                                     Entries are None where no track was supplied at
                                     push time (all existing targets).
        When current_label_per_robot is None, 'label_per_robot' is None (the driver
        fills the label from payload_angle_window / payload_track_window).
        Returns None when the buffer is too small or the target is not a
        delayed-label target."""
        if not self._is_delayed_label_target():
            return None
        K = getattr(self, 'gsp_prediction_horizon', 5)
        if len(self._gsp_label_buffer) < K + 1:
            return None
        # Snapshot the full ordered angle/track windows (oldest→newest) BEFORE
        # popping so the trajectory driver can difference consecutive entries.
        angle_window = [e.get('payload_angle_deg') for e in self._gsp_label_buffer]
        track_window = [e.get('payload_track') for e in self._gsp_label_buffer]
        oldest = self._gsp_label_buffer.popleft()
        label = (
            None if current_label_per_robot is None
            else np.asarray(current_label_per_robot, dtype=np.float32).copy()
        )
        return {
            'state_per_robot': oldest['state_per_robot'],
            'gsp_obs_per_robot': oldest['gsp_obs_per_robot'],
            'payload_angle_deg': oldest.get('payload_angle_deg'),
            'payload_angle_window': angle_window,
            'payload_track_window': track_window,
            'label_per_robot': label,
            'e2e_transition': oldest.get('e2e_transition'),
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
                if getattr(self, 'gsp_jepa_enabled', False):
                    gsp_output_size = getattr(self, 'gsp_encoder_dim', 32)
                gsp_slot = np.zeros(gsp_output_size, dtype=np.float32)
            else:
                # Multi-dim GSP output support (Change 1 — GSP_OUTPUT_KIND):
                # heading_gsp may be a scalar (legacy, O=1) or a numpy array (O>1).
                # For the legacy scalar case, apply the historical degrees/10 scaling
                # so that network weights trained on 'delta_theta_1d' are compatible.
                # For vector cases (cyl_kinematics_3d/goal_4d/*_traj) the values are
                # already in physical units from the label computation in Main.py
                # and are concatenated as-is (no extra scaling).
                # JEPA path: heading_gsp is the encoder latent — concatenate raw,
                # no scaling. Detected by the gsp_jepa_enabled FLAG (when JEPA is
                # on, choose_agent_gsp always emits the latent). The former
                # width heuristic (size>5 == latent) misfires for wide non-JEPA slots
                # (cyl_displacement_traj: 2K=10 at the default horizon); the
                # raveled result was coincidentally identical, but the
                # discrimination is now explicit.
                heading_gsp_arr = np.asarray(heading_gsp, dtype=np.float32)
                if getattr(self, 'gsp_jepa_enabled', False):
                    # JEPA latent vector — concatenate raw, skip degrees/10.
                    gsp_slot = heading_gsp_arr.ravel()
                elif heading_gsp_arr.ndim == 0 or heading_gsp_arr.size == 1:
                    # Scalar path — preserve legacy degrees/10 normalization.
                    scalar_val = float(heading_gsp_arr.ravel()[0])
                    gsp_slot = np.array([np.degrees(scalar_val / 10)], dtype=np.float32)
                else:
                    # Vector path — spliced as-is (no per-slot rescaling here).
                    # Units are whatever the label pipeline trained the head
                    # on: raw physical units by default, or meters ×
                    # GSP_TRAJ_LABEL_SCALE for the metric trajectory kinds
                    # (Main.py applies that scale inside the label builder).
                    gsp_slot = heading_gsp_arr.ravel()
            # GSP_E2E_NORMALIZE_FEATURE (opt-in): standardize the spliced GSP
            # prediction to ~unit variance so it lands on the same scale as the
            # O(1) egocentric obs (the raw slot std is ~0.024, so ACTOR_USE_LAYER_NORM
            # — which normalizes the whole vector, not per-feature — cannot let the
            # actor weight it). Uses self.gsp_feature_stats, the SAME shared
            # RunningStandardizer the E2E learn splices (learn_DDQN_e2e /
            # learn_TD3_e2e) update; acting READS frozen stats and never updates
            # them, so train and eval standardize identically. Guards:
            #   * flag off -> self.gsp_feature_stats is None -> byte-identical no-op.
            #   * zero-out (H-14) severs the signal to a constant zero; leave it
            #     zeroed (do NOT standardize a deliberately-severed slot).
            #   * only standardize when the slot width matches the standardizer dim
            #     (K). The JEPA-latent path (gsp_jepa_enabled) and latent-primary are
            #     NOT the scalar/K prediction this lever targets, so their width
            #     won't match and they are skipped.
            # Composition with GSP_EVAL_ABLATE_PRED: that ablation transforms
            # heading_gsp UPSTREAM (in Main.py) before this method runs, so frozen_mean
            # arrives here as the per-episode running MEAN of predictions; standardizing
            # it with stats whose mean ~matches yields ~0 — the ablation still severs
            # the signal, now on the normalized scale.
            _stats = getattr(self, 'gsp_feature_stats', None)
            if (
                _stats is not None
                and not getattr(self, 'gsp_zero_out_signal', False)
                and gsp_slot.shape[0] == _stats.dim
            ):
                # Eval feature-stats warm-up (GSP_EVAL_FEATURE_STATS_WARMUP_
                # EPISODES): checkpoints saved before GSP-RL#37 carry no
                # standardizer state, so a fresh eval process would standardize
                # with the identity (count==0) — the 2026-07-10 eval-restore
                # incident. During the burn-in episodes Main.py sets this flag
                # and the acting splice — the one place that sees the slot at
                # the exact per-kind scale the learn splice standardized —
                # folds the live prediction into the stats. Guards (review
                # findings, 2026-07-10):
                #   * standardize FIRST with the frozen stats, update AFTER —
                #     the learn splice's BatchNorm-style order; also keeps the
                #     first burn-in steps on the count==0 identity instead of
                #     (x-mean)/sqrt(eps) garbage.
                #   * skip the exact all-zeros slot: Main.py's episode-init
                #     make_agent_state call runs before any head inference with
                #     next_heading_gsp zero-initialized — a placeholder, not a
                #     prediction (a continuous head emitting exact 0.0 across
                #     all K dims has measure zero).
                #   * Welford-mode only: the EMA half-life clock ticks per
                #     UPDATE call, and warm-up calls it per robot-step on
                #     single samples — a different estimator than the per-
                #     learn-step batch updates it was trained with. EMA-mode
                #     stats must come from the checkpoint npz instead.
                # Off (default) preserves the acting-reads-frozen-stats
                # contract byte-identically.
                _warm = (
                    getattr(self, 'gsp_eval_stats_warmup_active', False)
                    and _stats.ema_halflife == 0
                    and np.any(gsp_slot)
                )
                _raw_slot = gsp_slot if _warm else None
                gsp_slot = _stats.standardize(gsp_slot)
                if _warm:
                    _stats.update(_raw_slot.reshape(1, -1))
            # GSP_E2E_SPLICE_GAIN (GSP-RL#39): fixed constant salience gain,
            # the LAST transform before concatenation — must mirror the learn
            # splices exactly (learning_aids.py applies the same attr after the
            # optional standardizer). 1.0 (default) = exact no-op. Not applied
            # to a zero-out-severed slot (0 × gain = 0 anyway, but skipping
            # keeps the guard structure symmetric with the standardizer block).
            _gain = float(getattr(self, 'gsp_e2e_splice_gain', 1.0))
            if _gain != 1.0 and not getattr(self, 'gsp_zero_out_signal', False):
                gsp_slot = gsp_slot * _gain
            # Latent-primary (GSP_ACTOR_LATENT_PRIMARY): drop the raw env_obs block
            # so the actor's input is [latent | global] (or [latent]) — the actor is
            # forced to route through the encoder latent. Must stay in lockstep with
            # the GSP-RL fork's network_input_size (=enc_dim, not input_size+enc_dim)
            # and the coupled-splice slot (gsp_idx=0); otherwise the runtime obs width
            # mismatches the Q-net and loading crashes with a state_dict size error.
            _lp_base = () if getattr(self, 'gsp_actor_latent_primary', False) else (env_obs,)
            if global_knowledge is not None:
                env_obs = np.concatenate((*_lp_base, gsp_slot, global_knowledge))
            else:
                env_obs = np.concatenate((*_lp_base, gsp_slot))
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
                        env_observations=None, payload_state=None, self_dynamics=None,
                        cyl_bearing_delta=None):
        """Build per-agent GSP input vectors for GSP-N (neighbor) variant.

        Base layout per agent (2 dims for self + 2 per neighbor pair):
            [self_avg_prox, self_prev_gsp, n0_prox, n0_prev_gsp, ...]

        Optional enrichment (Change 2 — GSP_INPUT_INCLUDE_* flags):
            GSP_INPUT_INCLUDE_GOAL:    appends (cos(angle_to_goal), sin(angle_to_goal))
                                       to the self-slot using env_observations[i][1].
            GSP_INPUT_INCLUDE_CYL_REL: appends (dist_to_cyl, angle_to_cyl) to the
                                       self-slot using env_observations[i][4:6].
            GSP_INPUT_INCLUDE_CYL_BEARING_DELTA: appends the wrap-safe signed one-step
                                       change in the robot's WORLD-FRAME bearing around
                                       the cylinder as a single self-slot dim. The value
                                       is computed in Main.py from world positions
                                       (atan2(robot_y − cyl_y, robot_x − cyl_x)) and passed
                                       in via the cyl_bearing_delta kwarg. Placed
                                       immediately after cyl_rel. 0.0 when the arg is None.
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
            cyl_bearing_delta: dict with key 'delta' — a list/array of the wrap-safe
                one-step world-frame bearing delta (radians) indexed by agent id.
                Computed in Main.py from world positions. Required when
                GSP_INPUT_INCLUDE_CYL_BEARING_DELTA is True; when None the delta dim
                is written as 0.0.
        """
        include_goal = getattr(self, '_gsp_input_include_goal', False)
        include_cyl_rel = getattr(self, '_gsp_input_include_cyl_rel', False)
        include_cyl_bearing_delta = getattr(self, '_gsp_input_include_cyl_bearing_delta', False)
        full_prox = getattr(self, '_gsp_input_full_prox', False)
        include_payload_state = getattr(self, '_gsp_input_include_payload_state', False)
        include_self_dynamics = getattr(self, '_gsp_input_include_self_dynamics', False)
        temporal_stack_k = getattr(self, '_gsp_input_temporal_stack_k', 1)
        # cyl_bearing_delta no longer needs env_observations — the value is
        # pre-computed in Main.py from world positions and passed via the arg.
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
            # Multi-dim GSP output: write K dims for the self prev_gsp slot.
            # When K=1 (legacy), this is identical to the previous scalar write.
            # When K>1 (cyl_kinematics_3d/goal_4d), the full prediction vector from
            # the previous step is stored so the head sees its own prior output.
            _slot_k = self.gsp_network_output  # set by super().__init__ from gsp_output_size_effective
            _prev = np.asarray(agent_prev_gsp[agent], dtype=np.float32).ravel()
            if _prev.size != _slot_k:
                # Defensive: pad/truncate if sizes mismatch (should not happen at
                # steady state, but protects the first step when next_heading_gsp
                # is initialised to zeros of shape (num_robots, K)).
                _prev = np.resize(_prev, _slot_k)
            agent_state[idx:idx + _slot_k] = _prev
            idx += _slot_k
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

            # Optional enrichment: wrap-safe one-step change in the robot's WORLD-FRAME
            # bearing around the cylinder. The GSP target (cylinder rotation) is
            # ~0.77-0.89 predictable from this delta (verified on run h5), whereas the
            # delta of the body-frame angle_to_cyl correlates only ~0.003. The value is
            # computed in Main.py from world positions (atan2(robot_y − cyl_y,
            # robot_x − cyl_x), wrap-safe delta vs the previous step) and passed in via
            # the cyl_bearing_delta arg. This keeps the self-slot layout position (right
            # after cyl_rel) and the +1 size (_extra_per_slot) unchanged. When the arg
            # is None (default) the dim is written as 0.0.
            if include_cyl_bearing_delta:
                agent_state[idx] = (
                    float(cyl_bearing_delta['delta'][agent])
                    if cyl_bearing_delta is not None else 0.0
                )
                idx += 1

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
            # Each neighbor slot is (1 + K) dims: avg_prox × 1, prev_gsp × K.
            # When K=1 (legacy) this is identical to the previous 2-dim write.
            _neighbor_region_start = idx  # capture before the loop (after self+enrichment)
            for neighbor in neighbors:
                agent_state[idx] = agent_prox_values[neighbor]
                idx += 1
                _nbr_prev = np.asarray(agent_prev_gsp[neighbor], dtype=np.float32).ravel()
                if _nbr_prev.size != _slot_k:
                    _nbr_prev = np.resize(_nbr_prev, _slot_k)
                agent_state[idx:idx + _slot_k] = _nbr_prev
                prox_flags.append(agent_prox_values[neighbor])
                idx += _slot_k
            _neighbor_region_end = idx  # capture after the loop

            # Eval-time neighbor ablation: neutralize the neighbor region in place,
            # BEFORE the vector is pushed to the ring buffer / temporally stacked, so
            # the zeros propagate through stacking. The self-slot and all enrichment
            # dims (everything before _neighbor_region_start) are untouched. Works for
            # any _slot_k (K=1 and K>1) and for the temporal-stack-K path.
            if getattr(self, '_gsp_eval_ablate_neighbors', False):
                agent_state[_neighbor_region_start:_neighbor_region_end] = 0.0

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

        angles = self._ROBOT_PROXIMITY_ANGLES  # precomputed np.array
        if angle_to_cyl > 180 - self._prox_filter_angle_deg:
            # Wrap-around positive: indexed (filtered out) when angle > ccw_lim OR < cw_lim
            mask_indexed = (angles > ccw_lim) | (angles < cw_lim)
        elif angle_to_cyl < -180 + self._prox_filter_angle_deg:
            # Wrap-around negative: indexed when angle < cw_lim OR > ccw_lim
            mask_indexed = (angles < cw_lim) | (angles > ccw_lim)
        else:
            # Normal: indexed when ccw_lim < angle < cw_lim
            mask_indexed = (angles > ccw_lim) & (angles < cw_lim)
        index = list(np.where(mask_indexed)[0])
        filtered_prox_values = list(np.asarray(prox_values)[~mask_indexed])
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
        # The GSP head is a supervised regressor; when configured deterministic,
        # its prediction is greedy (no exploration noise) even during training.
        # Only the choose_action calls below route through the noise-adding path;
        # the JEPA encoder and the recurrent direct-forward path already run
        # noise-free, so gsp_test does not gate them.
        gsp_test = test or getattr(self, '_gsp_prediction_deterministic', False)
        # JEPA path: run the online encoder on each agent's GSP input state.
        # Returns a list of 32-d latent vectors (one per agent), or a single
        # 32-d array for the non-neighbor/non-broadcast flat case.
        if getattr(self, 'gsp_jepa_enabled', False):
            enc = self.gsp_encoder_online
            enc.eval()
            with T.no_grad():
                if self._neighbors or self._broadcast:
                    latents = []
                    for i in range(self._n_agents):
                        state_np = np.asarray(agent_gsp_states[i], dtype=np.float32)
                        state_t = T.tensor(state_np, dtype=T.float32).unsqueeze(0).to(enc.device)
                        latent = enc(state_t).squeeze(0).cpu().numpy()
                        latents.append(latent)
                    return latents
                else:
                    state_np = np.asarray(agent_gsp_states, dtype=np.float32)
                    state_t = T.tensor(state_np, dtype=T.float32).unsqueeze(0).to(enc.device)
                    latent = enc(state_t).squeeze(0).cpu().numpy()
                    return latent

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
                    actions.append(self.choose_action(agent_gsp_states[i], self.gsp_networks, gsp_test))
            return actions
        else:
            if self.recurrent_gsp:
                self.gsp_observation.append(agent_gsp_states)
                self.gsp_observation.pop(0)
                action = self.choose_action(self.gsp_observation, self.gsp_networks, gsp_test)
                return action

            observation = np.array(agent_gsp_states)
            return self.choose_action(observation, self.gsp_networks, gsp_test)

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
    
    def store_agent_transition(self, s, a, r, s_, d, gsp_obs=None, gsp_label=None, phi=None):
        if self.networks['replay'].action_type == 'Discrete':
            a = a[0]
        elif self.networks['replay'].action_type == 'Continuous':
            a = np.array(a[1][0:2])
        return super().store_agent_transition(s, a, r, s_, d, gsp_obs=gsp_obs, gsp_label=gsp_label, phi=phi)
    
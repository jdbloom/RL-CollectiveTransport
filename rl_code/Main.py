from urllib.parse import uses_relative
#import python_code.Agent as Agent
import src.agent as Agent
from src.env import calculate_gsp_reward, ZMQ_Utility, angle_normalize_signed_deg
from src.knowledge import build_global_knowledge, build_g_knowledge_all
from src.hdf5_logger import HDF5Logger
from src.pred_ablation import apply_pred_ablation, RunningMeanState
from src.zmq_diagnostics import DiagnosticSocket
from src.diagnostics import ExperimentLogger

#from python_code.comms_viz import viz

import argparse
from collections import namedtuple
from struct import pack, unpack, Struct
import numpy as np
import math

import copy
import zmq
import csv
import os
import time
import torch as T
import matplotlib.pyplot as plt
import yaml
import logging
import traceback

Utility = ZMQ_Utility()

# get path to containing folder so this works where ever it is used
containing_folder = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("recording_path")
parser.add_argument("--test", default = False, action = "store_true")
parser.add_argument("--model_path")
parser.add_argument("--best_gsp_ckpt", default=None,
                    help="Path to a saved GSP-head snapshot (from Task 1). "
                         "If set in test mode, the GSP head is loaded from this "
                         "checkpoint AFTER load_model, overriding the bundled final weights.")
parser.add_argument("--trained_num_robots")                                          # if we are testing a model trained on a different number of robots. This should be set to the training number of robots so that the network is built properly.
parser.add_argument("--no_print", default = False, action = "store_true")
parser.add_argument("--independent_learning", default = False, action = "store_true")
parser.add_argument("--global_knowledge", default = False, action = "store_true")   # append knowledge of other agents to the observation space
parser.add_argument("--share_prox_values", default=False, action = 'store_true')    # Robots will share their averaged prox values with eachother

args = parser.parse_args()

recording_path = os.path.join(containing_folder, args.recording_path)
exp_logger = ExperimentLogger(os.path.basename(recording_path))
log = exp_logger.get_logger("main")
log.info("Starting experiment: %s", recording_path)
config_path = os.path.join(recording_path, 'agent_config.yml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

if args.model_path is not None:
    model_file_path = os.path.join(containing_folder, args.model_path)
learning_scheme = config['LEARNING_SCHEME']
learn_every = int(config.get('LEARN_EVERY', 1))
port = str(config['PORT'])
test_mode = args.test
train_mode = not test_mode
#
# Initialize zmq
#
# Create context
context = zmq.Context()
# create socket
socket = context.socket(zmq.REP)
# wait for connections on specified port, defaults to 55555
socket.bind("tcp://*:" + port)
socket = DiagnosticSocket(
    socket, os.path.basename(recording_path),
    logger=exp_logger.get_logger("zmq"),
)
socket.setsockopt(zmq.RCVTIMEO, 600000)  # 10 minute recv timeout — detect ARGoS crashes
print("Server Started")
# Get Parameters
Utility.get_params(socket.recv())
if not args.no_print:
    print("PARAMETERS:")
    print("  num_robots ----", Utility.params['num_robots'])
    print("  num_obstacles -", Utility.params['num_obstacles'])
    print("  num_obs -------", Utility.params['num_obs'])
    print("  alphabet_size -", Utility.params['alphabet_size'])
    print("  num_actions ---", Utility.params['num_actions'])
    print("  num_stats -----", Utility.params['num_stats'])

Utility.set_obstacles_fields()
# Path to save data
data_file_path = recording_path + '/Data/'

# Initialize HDF5 logger (one per experiment). Commit shas + branches come from
# the dispatcher's pre-launch code_verification step; they are written as h5
# root attrs so cross-machine comparisons can filter by code version.
hdf5_path = os.path.join(recording_path, os.path.basename(recording_path) + ".h5")
hdf5_writer = HDF5Logger(
    hdf5_path,
    stelaris_sha=config.get("STELARIS_SHA"),
    rl_ct_sha=config.get("RL_CT_SHA"),
    gsp_rl_sha=config.get("GSP_RL_SHA"),
    stelaris_branch=config.get("STELARIS_BRANCH"),
    rl_ct_branch=config.get("RL_CT_BRANCH"),
    gsp_rl_branch=config.get("GSP_RL_BRANCH"),
    count_episodes=train_mode,  # public counter reflects TRAINING episodes only
)

# Per-episode diagnostics (FAU / weight norms / effective rank / Q-gap / pred
# diversity). Opt-in via config['DIAGNOSTICS_ENABLED']. The rolling gsp_obs_pool
# buffers recent head-input vectors so the eval batch for GSP diagnostics can
# be drawn from the same distribution the head actually sees during training.
# Cap at max size so memory stays bounded on long runs.
_DIAG_POOL_MAX_SIZE = 8192
diag_gsp_obs_pool: list = []
diag_eval_batch_frozen: bool = False
diag_episode_predictions: list = []  # per-step GSP predictions this episode, reset each ep

if args.share_prox_values:
    num_obs = Utility.params['num_obs'] +Utility.params['num_robots']   #need to account for num_robots extra observations
elif args.global_knowledge:
    num_obs = Utility.params['num_obs']+(Utility.params['num_robots']-1)*4  #need to account for the x and y positions and the x and y velocitis for each robot
else:
    num_obs = Utility.params['num_obs']

agent_nn_args = {
    'config': config,
    'network': config['LEARNING_SCHEME'],
    'n_agents': Utility.params['num_robots'],
    'n_obs': num_obs, # + 6,  # to account for the sin, cos, and tan of the two angles
    'n_actions': Utility.params['num_actions']-1,  #remove control of the gripper
    'options_per_action':config['OPTIONS_PER_ACTION'],
    'min_max_action':config['MIN_MAX_ACTION'],
    'meta_param_size':config['META_PARAM_SIZE'],
    'gsp': config['GSP'],
    'recurrent': config['RECURRENT'],
    'attention': config['ATTENTION'],
    'neighbors': config['NEIGHBORS'],
    'broadcast': config.get('BROADCAST', False),
    'gsp_input_size':config['GSP_INPUT_SIZE'],
    'gsp_output_size':config['GSP_OUTPUT_SIZE'],
    'gsp_look_back':config['GSP_LOOK_BACK'],
    'gsp_min_max_action':config['GSP_MIN_MAX_ACTION'],
    'gsp_sequence_length':config['GSP_SEQUENCE_LENGTH'],
    'prox_filter_angle_deg':config['PROX_FILTER_ANGLE_DEG'],
}


if args.independent_learning:
    models = [Agent.Agent(id=i, **agent_nn_args) for i in range(Utility.params['num_robots'])]
    if test_mode:
        [models[i].load_model(model_file_path) for i in range(Utility.params['num_robots'])]
        if args.best_gsp_ckpt:
            log.info(f'Loading best GSP-head checkpoint from {args.best_gsp_ckpt}')
            for m in models:
                m.load_gsp_head_snapshot(args.best_gsp_ckpt)
else:
    if args.trained_num_robots is not None:
        agent_nn_args['n_agents'] = int(args.trained_num_robots)
        model = Agent.Agent(id = 0, **agent_nn_args)
    else:
        model = Agent.Agent(id = 0, **agent_nn_args)
    if test_mode:
        model.load_model(model_file_path)
        if args.best_gsp_ckpt:
            log.info(f'Loading best GSP-head checkpoint from {args.best_gsp_ckpt}')
            model.load_gsp_head_snapshot(args.best_gsp_ckpt)


# Send acknowledgment
socket.send(b"ok")

# Prism handshake (non-uniform objects)
if Utility.params['num_prisms'] > 0:
    Utility.set_prism_sizes()
    prism_sizes = Utility.parse_prism_sizes(socket.recv())
    socket.send(b"ok")
    Utility.set_prism_points(prism_sizes)
    prism_points = Utility.parse_prism_points(socket.recv())
    socket.send(b"ok")

#######################################################################
#                           MAIN LOOP
#######################################################################
exp_done = False
ep_counter = 0
exp_rewards = []
exp_mean_rewards = []
high_score = -np.inf
mean_axis = []
experiment_start_time = time.time()
Testing_Failures = 0
Testing_Successes = 0
var_grad = 0
gate = 0
gate_stats = 0
obstacles = 0
obstacle_stats = 0
ep_ticks = 0

# GSP_OUTPUT_KIND — multi-target label computation support.
# Read once at startup; default preserves legacy behavior.
_gsp_output_kind = str(config.get('GSP_OUTPUT_KIND', 'delta_theta_1d'))

# delta_theta_traj E2E label scaling. The per-step trajectory _traj is built in
# DEGREES (differences of obj_stats[5]). The single-step delta_theta label
# (env.py calculate_gsp_reward) does `radians(diff) * 100` clipped to [-1,1] —
# env.py itself flags that ×100 SATURATES (max rotation ~0.09 rad/step → 9.0 →
# clipped to 1.0), degenerating the regression into near-binary classification.
# For the trajectory label we use the NON-saturating scale env.py's own comment
# says was intended — radians × 10 (0.09 rad × 10 = 0.9, comfortably inside
# [-1,1], no clip) — so the per-step targets keep their magnitude and the head
# learns a real regression, not a sign bit. Combined deg→scaled factor:
#   label = radians(traj_deg) * 10  ==  traj_deg * (pi/180) * 10.
# Overridable via GSP_DELTA_THETA_TRAJ_LABEL_SCALE (applied to the RADIAN value).
_delta_theta_traj_label_rad_scale = float(
    config.get('GSP_DELTA_THETA_TRAJ_LABEL_SCALE', 10.0)
)
_delta_theta_traj_label_scale = (
    (math.pi / 180.0) * _delta_theta_traj_label_rad_scale
)

# GSP_TRAJ_LABEL_SCALE — fixed multiplier on the RAW-METER trajectory labels
# (goal_progress_traj / cyl_displacement_traj), applied inside
# _build_traj_label_from_windows so the head-store path, the E2E path, and the
# h5 gsp_target logging all see ONE consistent target definition. Default 1.0
# = byte-identical raw meters.
#
# Why (2026-07-10): raw meter labels (std ~2.5e-3/step) forced GSP_E2E_LAMBDA
# to 40000 (F15 loss balance) and a post-hoc feature standardizer at the actor
# splice — a route with three measured failure classes (Welford inflation, EMA
# drift dominance, eval-restore gap; closed GSP-RL#36 + campaign correction).
# Scaling the LABEL at the source (~400 → std ~1.0) makes the head emit O(1)
# outputs directly: λ recalibrates to O(0.25-1), the splice needs no
# standardizer (GSP_E2E_NORMALIZE_FEATURE off), the scale is baked into the
# checkpointed head weights (nothing to persist or warm up at eval), and
# train/eval standardize identically by construction. Mirrors the pre-existing
# GSP_DELTA_THETA_TRAJ_LABEL_SCALE pattern: a fixed, STATELESS config scale —
# deliberately not running label statistics. Pair scales that push |label|
# past tanh range with GSP_E2E_LINEAR_OUTPUT=true.
_gsp_traj_label_scale = float(config.get('GSP_TRAJ_LABEL_SCALE', 1.0))
log.info("GSP_TRAJ_LABEL_SCALE = %s", _gsp_traj_label_scale)
if _gsp_traj_label_scale != 1.0:
    # The GSP head output is BOUNDED either way: tanh by default, and
    # GSP_E2E_LINEAR_OUTPUT is a hard clamp at ±MIN_MAX_ACTION (ddpg.py), so
    # labels pushed past that bound are unfittable (zero gradient at the
    # clamp). Choose the scale so |label| stays inside the bound — measured
    # cdt tails: per-step displacement absmax ~9.4e-3 m ⇒ scale 80 → 0.75.
    log.warning(
        "GSP_TRAJ_LABEL_SCALE=%s: head output is bounded (tanh or clamp at "
        "±MIN_MAX_ACTION) — verify scaled |label| max stays inside the bound "
        "(and note gsp_label_std/gsp_mse h5 metrics change units vs scale=1 runs)",
        _gsp_traj_label_scale,
    )

# K-step trajectory targets — all share the delayed FIFO (push per step, pop at
# maturity K steps later) and the auto-derived horizon-coupled GSP_OUTPUT_KIND.
# Kept in lockstep with agent.py (_GSP_TRAJ_TARGETS) and GSP-RL learning_aids.py.
#   delta_theta_traj      (size K)  per-step payload rotation, scaled (see above)
#   goal_progress_traj    (size K)  per-step payload progress-to-goal delta
#                                   (prev_cyl_dist2goal − curr, positive = toward
#                                   goal — the exact quantity from the
#                                   cyl_kinematics_goal_4d kind's 4th component).
#                                   meters × GSP_TRAJ_LABEL_SCALE (default 1.0
#                                   = raw; λ from measured label std, F15).
#   cyl_displacement_traj (size 2K) per-step payload (Δx, Δy), flattened
#                                   [Δx1,Δy1,…,ΔxK,ΔyK]. meters ×
#                                   GSP_TRAJ_LABEL_SCALE (default 1.0 = raw).
_GSP_TRAJ_TARGETS = (
    'delta_theta_traj', 'goal_progress_traj', 'cyl_displacement_traj'
)


def _build_traj_label_from_windows(pred_target, ang_win, trk_win):
    """Build the matured K-step trajectory label for `pred_target` from the
    FIFO's ordered windows (oldest→newest, length K+1).

    delta_theta_traj      → size-K wrapped per-step rotation (degrees; the
                            caller applies _delta_theta_traj_label_scale on the
                            E2E path, matching the pre-existing behavior).
    goal_progress_traj    → size-K per-step progress-to-goal delta, meters
                            × _gsp_traj_label_scale.
    cyl_displacement_traj → size-2K flattened per-step (Δx, Δy), meters
                            × _gsp_traj_label_scale.
    """
    if pred_target == 'delta_theta_traj':
        return np.array([
            angle_normalize_signed_deg(
                float(ang_win[k + 1]) - float(ang_win[k])
            )
            for k in range(len(ang_win) - 1)
        ], dtype=np.float32)
    if pred_target == 'goal_progress_traj':
        return np.array([
            (float(trk_win[k]['dist2goal']) - float(trk_win[k + 1]['dist2goal']))
            * _gsp_traj_label_scale
            for k in range(len(trk_win) - 1)
        ], dtype=np.float32)
    if pred_target == 'cyl_displacement_traj':
        _disp = []
        for k in range(len(trk_win) - 1):
            _disp.append(float(trk_win[k + 1]['cyl_x']) - float(trk_win[k]['cyl_x']))
            _disp.append(float(trk_win[k + 1]['cyl_y']) - float(trk_win[k]['cyl_y']))
        # Multiply in float64 BEFORE the float32 cast (single rounding, same
        # convention as the goal_progress branch) so offline recomputation
        # from raw h5 cyl positions reproduces the stored labels bit-exactly.
        return (
            np.asarray(_disp, dtype=np.float64) * _gsp_traj_label_scale
        ).astype(np.float32)
    raise ValueError(f"not a trajectory target: {pred_target}")

# Input enrichment flags — need to pass env_observations to make_gsp_states
# when any flag is active. Read once so the hot loop avoids repeated dict lookup.
_gsp_input_include_goal = bool(config.get('GSP_INPUT_INCLUDE_GOAL', False))
_gsp_input_include_cyl_rel = bool(config.get('GSP_INPUT_INCLUDE_CYL_REL', False))
_gsp_input_full_prox = bool(config.get('GSP_INPUT_FULL_PROX', False))
_gsp_input_needs_env_obs = _gsp_input_include_goal or _gsp_input_include_cyl_rel or _gsp_input_full_prox

# Wrap-safe one-step delta of the robot's WORLD-FRAME bearing around the cylinder.
# Computed here from world positions (atan2(robot_y - cyl_y, robot_x - cyl_x)) and
# passed to make_gsp_states via the cyl_bearing_delta arg. The body-frame angle_to_cyl
# delta correlates only ~0.003 with the GSP target; the world-frame bearing delta
# correlates ~0.77-0.89 (verified on run h5).
_gsp_input_include_cyl_bearing_delta = bool(config.get('GSP_INPUT_INCLUDE_CYL_BEARING_DELTA', False))

# Change 3 enrichment flags (GSP-N self-slot additions).
_gsp_input_include_payload_state = bool(config.get('GSP_INPUT_INCLUDE_PAYLOAD_STATE', False))
_gsp_input_include_self_dynamics = bool(config.get('GSP_INPUT_INCLUDE_SELF_DYNAMICS', False))
_gsp_input_temporal_stack_k = int(config.get('GSP_INPUT_TEMPORAL_STACK_K', 1))

# H-phase5-2 reward shaping. When > 0, gsp_reward (negative penalty for prediction
# error from env.calculate_gsp_reward, range [-2, 0]) is added to the actor's
# training reward stream at coef * gsp_reward[i] per robot per timestep. Default
# 0.0 preserves bit-identical reward stream for all pre-Phase-5.2 batches.
# See docs/predictions/2026-04-30-h-phase5-2-prereg.md.
_gsp_reward_coef = float(config.get('GSP_REWARD_COEF', 0.0))
log.info("GSP_REWARD_COEF = %s", _gsp_reward_coef)
# H-phase5-4 random-noise control. When True, the per-step augmentation at
# Main.py:772 uses a random Gaussian-squared penalty in [-2, 0] of matched
# magnitude instead of gsp_reward. Disambiguates "prediction signal helps"
# from "any aux reward of right scale helps." Default False is unchanged.
_gsp_reward_random_noise = bool(config.get('GSP_REWARD_RANDOM_NOISE', False))
log.info("GSP_REWARD_RANDOM_NOISE = %s", _gsp_reward_random_noise)

# M2 — eval-time GSP prediction ablation (GSP_EVAL_ABLATE_PRED).
# Read once at startup. The prediction transform is applied at the single
# next_heading_gsp injection site via src.pred_ablation.apply_pred_ablation.
# 'none' (default) is a literal identity no-op → training path stays bit-exact.
# The 'shuffle' mode needs a seeded rng (deterministic per SEED); the
# 'frozen_mean' mode needs a per-episode running-mean accumulator (reset at
# episode start). Both are inert on the default 'none' path.
# See docs/research/2026-07-04-gsp-actor-usage-instrumentation-prereg.md (M2).
_gsp_eval_ablate_pred = str(config.get('GSP_EVAL_ABLATE_PRED', 'none'))
log.info("GSP_EVAL_ABLATE_PRED = %s", _gsp_eval_ablate_pred)
_pred_ablation_rng = np.random.default_rng(int(config.get('SEED', 0)))
# Per-episode running-mean accumulator for the 'frozen_mean' mode. Reconstructed
# at each episode boundary (see episode-init block) so the mean never bleeds
# across episodes. Initialized here for module scope; reset per episode below.
_pred_frozen_mean_state = RunningMeanState()

# Eval-time feature-stats warm-up (GSP_EVAL_FEATURE_STATS_WARMUP_EPISODES).
# The GSP_E2E_NORMALIZE_FEATURE standardizer's running stats are part of the
# policy, but checkpoints saved before GSP-RL#37 never persisted them — a
# fresh eval process reconstructs the standardizer cold, standardize() is the
# identity, and the actor receives the raw tiny-scale feature it was NOT
# trained on (the 2026-07-10 incident that voided the abl500r2 verdict).
# When W > 0 in test mode, the first W episodes are BURN-IN: the acting splice
# (Agent.make_agent_state) updates the stats from the live post-scale
# predictions, and the GSP_EVAL_ABLATE_PRED transform is deferred (burn-in
# runs as 'none' so the stats reflect the TRUE prediction distribution, not an
# ablated one). From episode W on, the stats freeze and the configured
# ablation applies — measured episodes see warm stats. Analyses MUST drop
# episodes < W (read this key from agent_config.yml). Default 0 → inert.
# Caveat: warm-up estimates the FINAL head's output stats, while training
# standardized with all-history stats; valid when the training run's
# e2e_gsp_feature_std_postnorm held ~1.0 (stationary feature scale), as in the
# lambda=100 dtraj campaign cells this exists to re-verdict.
_gsp_eval_stats_warmup_eps = int(config.get('GSP_EVAL_FEATURE_STATS_WARMUP_EPISODES', 0))
log.info("GSP_EVAL_FEATURE_STATS_WARMUP_EPISODES = %s", _gsp_eval_stats_warmup_eps)
_in_stats_warmup = False
if _gsp_eval_stats_warmup_eps > 0 and test_mode:
    # Post-GSP-RL#37 checkpoints restore the training stats via load_model
    # (which already ran above). Warming on top of restored stats would
    # overwrite the exact calibration the persistence fix preserves — auto-
    # disable and say so. Mixed batches (old + new checkpoints) can then share
    # one YAML: new ones restore, old ones warm up.
    _stats_models = models if args.independent_learning else [model]
    if any(
        getattr(_m, 'gsp_feature_stats', None) is not None
        and _m.gsp_feature_stats.count > 0
        for _m in _stats_models
    ):
        log.info(
            "feature stats restored from checkpoint (count>0) — warm-up "
            "disabled; restored training stats are authoritative"
        )
        _gsp_eval_stats_warmup_eps = 0
    elif args.best_gsp_ckpt:
        # The GSP head was swapped to a different episode's snapshot AFTER
        # load_model, so the warmed stats will describe the SNAPSHOT head's
        # output stream. That is self-consistent for the warm-up path, but
        # flag the combination loudly — for restored-stats runs the same
        # combination is episode-mismatched with no warning possible.
        log.warning(
            "warm-up active with --best_gsp_ckpt: stats will be warmed on "
            "the snapshot head's predictions"
        )

# M4 — candidate-target logging (GSP_LOG_CANDIDATE_TARGETS).
# When 1, all four candidate GSP targets (delta_theta, future_prox, cyl_kin Δx/Δy/Δθ,
# centroid-to-goal) are computed EVERY step regardless of the active GSP_OUTPUT_KIND
# and buffered for per-step h5 datasets. Default 0 → zero behavior change (the
# candidate block is skipped and no datasets are written).
# See docs/research/2026-07-04-gsp-actor-usage-instrumentation-prereg.md (M4).
_gsp_log_candidate_targets = bool(int(config.get('GSP_LOG_CANDIDATE_TARGETS', 0)))
log.info("GSP_LOG_CANDIDATE_TARGETS = %s", _gsp_log_candidate_targets)

# Ring buffer for previous-step payload state (needed for velocity computation).
# comX_prev, comY_prev, cyl_angle_prev are the payload position at t-1.
# Initialized to None; on the first step the velocity terms default to zero.
_prev_payload_comX: float = None
_prev_payload_comY: float = None
_prev_payload_cyl_angle: float = None

# Ring buffer for previous-step per-robot positions (needed for self_vx/vy).
# Initialized to None; on the first step velocity defaults to zero.
_prev_robot_x: list = None
_prev_robot_y: list = None

# Ring buffer for previous-step per-robot WORLD-FRAME bearing around the cylinder
# (radians), needed for the wrap-safe cyl-bearing delta. None until the first step;
# first step yields delta = 0.0.
_prev_cyl_bearing: list = None

try:
    while not exp_done:
        #receive initial observations
        msgs = socket.recv_multipart()
        exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
        socket.set_episode(ep_counter)
        log.info("Episode %d starting", ep_counter)

        if not exp_done:
            time_steps = 0

            agent_prox_flags = []
            last_object_heading = None

            # Multi-dim GSP output (Change 1 — GSP_OUTPUT_KIND):
            # Always 2D (num_robots, K) so all downstream code gets a consistent array.
            # For K=1 (legacy delta_theta_1d) next_heading_gsp[i] is a 1-element array
            # instead of a scalar — make_agent_state handles both via the ndim/size check.
            _model_for_gsp_k = model if not args.independent_learning else models[0]
            if config.get('GSP_JEPA_ENABLED', False):
                # JEPA path: actor input slot for GSP signal is the encoder latent (default 32).
                _gsp_K = int(config.get('GSP_ENCODER_DIM', 32)) if config.get('GSP') else 1
            else:
                _gsp_K = getattr(_model_for_gsp_k, 'gsp_network_output', 1) if config.get('GSP') else 1
            next_heading_gsp = np.zeros((Utility.params['num_robots'], _gsp_K))
            old_heading_gsp = np.zeros((Utility.params['num_robots'], _gsp_K))
            episode_gsp_rewards = np.zeros(Utility.params['num_robots'])

            # Reset Change-3 prev-step ring buffers at episode boundaries so
            # velocity terms never bleed across episodes.
            _prev_payload_comX = None
            _prev_payload_comY = None
            _prev_payload_cyl_angle = None
            _prev_robot_x = None
            _prev_robot_y = None
            _prev_cyl_bearing = None
            # M2: reset the frozen_mean running-mean accumulator at every episode
            # boundary so the prediction mean never bleeds across episodes.
            _pred_frozen_mean_state = RunningMeanState()

            # Eval feature-stats warm-up phase for this episode (see the
            # GSP_EVAL_FEATURE_STATS_WARMUP_EPISODES block above). The flag on
            # each Agent gates the stats update inside make_agent_state; while
            # warm-up is active the M2 ablation transform is deferred (guarded
            # at both injection sites below).
            _in_stats_warmup = bool(
                test_mode and ep_counter < _gsp_eval_stats_warmup_eps
            )
            if _gsp_eval_stats_warmup_eps > 0:
                for _m in (models if args.independent_learning else [model]):
                    _m.gsp_eval_stats_warmup_active = _in_stats_warmup

            # Receive initial observations from the environment
            env_observations, failures, rewards, stats, robot_stats, obj_stats = Utility.parse_msgs(msgs)
            old_cyl_ang = obj_stats[5]

            # Multi-target label state tracking (GSP_OUTPUT_KIND != delta_theta_1d).
            # prev_obj_stats: cylinder position/heading snapshot from previous step,
            #   used to compute (cyl_Δx, cyl_Δy, cyl_Δθ) for cyl_kinematics_* kinds.
            # prev_cyl_dist2goal: cyl_dist2goal at previous step, used to compute
            #   group_centroid_Δ_to_goal via -Δ(cyl_dist2goal).
            # ep_step_counter: step count within current episode for time_to_goal_1d.
            prev_obj_stats = obj_stats.copy()
            prev_cyl_dist2goal = float(env_observations[0][6]) if len(env_observations) > 0 else 0.0
            ep_step_counter = 0

            # Raw (pre-scale/clip) diff_rad accumulator for the 2026-04-20 signal-
            # distribution diagnostic. Populated in the env.calculate_gsp_reward
            # call inside the step loop; flushed to the episode HDF5 at episode_done.
            _gsp_raw_diff_episode = []

            if Utility.params['num_obstacles'] > 0:
                obstacle_stats = Utility.parse_obstacle_stats(msgs[7])
            elif Utility.params['use_gate'] == 1:
                gate_stats = Utility.parse_gate_stats(msgs[7])

            agent_states = []
            force_mags = []
            force_angs = []
            if args.independent_learning:
                running_reward = []
            else:
                running_reward = 0
        
            for i in range(Utility.params['num_robots']):
                if failures[i][0]:
                    agent_prox_flags.append(0)
                else:
                    prox_values = env_observations[i][7:]
                    # Add logic to filter prox values that are observing the object
                    prox_values, filtered_indeces = model.filter_prox_values(prox_values, env_observations[i][5])
                    for j in range(len(filtered_indeces)):
                        env_observations[i][7+filtered_indeces[j]] = 0.0
                    prox_value = np.sum(prox_values)
                    agent_prox_flags.append(prox_value/float(len(filtered_indeces)))
        
            #Define Global Knowledge: [positions, velocities]
            # T7: vectorized via src.knowledge (inert, golden-verified).
            global_knowledge = build_global_knowledge(robot_stats, stats)
            g_knowledge_all = build_g_knowledge_all(global_knowledge)

            for i in range(Utility.params['num_robots']):
                g_knowledge = g_knowledge_all[i]
                if args.independent_learning:
                    running_reward.append(0)
                    if config['GSP']:
                        if args.global_knowledge:
                            agent_state = models[i].make_agent_state(env_observations[i], heading_gsp = next_heading_gsp[i], global_knowledge=g_knowledge) 
                        else:
                            agent_state = models[i].make_agent_state(env_observations[i], heading_gsp = next_heading_gsp[i])
                    else:
                        if args.global_knowledge:
                            agent_state = models[i].make_agent_state(env_observations[i], global_knowledge = g_knowledge)
                        else:
                            agent_state = env_observations[i]
                    
                else:
                    if config['GSP']:
                        if args.global_knowledge:
                            agent_state = model.make_agent_state(env_observations[i], heading_gsp=next_heading_gsp[i], global_knowledge=g_knowledge)
                        else:
                            agent_state = model.make_agent_state(env_observations[i], heading_gsp=next_heading_gsp[i])
                    else: 
                        if args.share_prox_values:
                            agent_state = np.concatenate((env_observations[i], agent_prox_flags))
                        else:
                            if args.global_knowledge:
                                agent_state = model.make_agent_state(env_observations[i], global_knowledge=g_knowledge)
                            else:
                                agent_state = env_observations[i]
                agent_states.append(agent_state)
                force_mags.append(stats[i][0])
                force_angs.append(stats[i][1])

            # reward is the same across all agents. If it were per agent then this would need to move into the loop above
            if args.independent_learning:
                for i in range(Utility.params['num_robots']):
                    running_reward[i]+= rewards[i]
            else:
                running_reward += rewards[0]
            # failures should all be false because we havent started the episode yet
            failure = failures[0]

            #
            # Start the Episode Loop
            #

            # Churn diagnostic: snapshot network weights at episode start so that
            # after the episode's learn steps complete we can measure activation
            # churn (L2 distance of outputs before vs after). Only snapshot when
            # diagnostics are enabled — zero overhead on legacy runs.
            # Strategy: start-of-episode vs end-of-episode.  This captures the
            # cumulative weight change across all learn steps within the episode
            # (every learn_every timesteps), which is the most representative
            # "update" the network actually received.  No extra learn call is
            # triggered — we reuse the natural training boundary.
            _churn_actor_before = None
            _churn_gsp_before = None
            if (not args.independent_learning
                    and getattr(model, 'diagnostics_enabled', False)
                    and getattr(model, 'diagnose_churn', True)):
                _actor_net = model._main_network(model.networks)
                if _actor_net is not None:
                    _churn_actor_before = copy.deepcopy(_actor_net.state_dict())
                if model.gsp_networks is not None:
                    _gsp_net = model._main_network(model.gsp_networks)
                    if _gsp_net is not None:
                        _churn_gsp_before = copy.deepcopy(_gsp_net.state_dict())

            episode_start_time = time.time()
            while not episode_done:
                if not exp_done:
                    reward = []
                    actions = []
                    actions_to_take = []
                    time_steps += 1
                    robot_failures = []

                    for i in range(Utility.params['num_robots']):
                        # Choose an action
                        if args.independent_learning:
                            action, action_num = models[i].choose_agent_action(agent_states[i], failures[i], test_mode)
                        else:
                            action, action_num = model.choose_agent_action(agent_states[i], failures[i], test_mode)
                        actions_to_take.append(action)
                        actions.append(action_num)

                    old_failures = failures[:]
                    # Take Step
                    socket.send(Utility.serialize_actions(actions_to_take))
                    msgs = socket.recv_multipart()

                    exp_done, episode_done, reached_goal = Utility.parse_status(msgs[0])
                    env_observations, failures, rewards, stats, robot_stats, obj_stats = Utility.parse_msgs(msgs)
                    com_X_poses = obj_stats[7]
                    com_Y_poses = obj_stats[8]
                    robot_x_pos = []
                    robot_y_pos = []
                    robot_angle = []
                    for i in range(Utility.params['num_robots']):
                        robot_x_pos.append(robot_stats[i][0])
                        robot_y_pos.append(robot_stats[i][1])
                        robot_angle.append(robot_stats[i][5])
                    if Utility.params['num_obstacles'] > 0:
                        obstacle_stats = Utility.parse_obstacle_stats(msgs[7])
                    elif Utility.params['use_gate'] == 1:
                        gate_stats = Utility.parse_gate_stats(msgs[7])

                    ############################## gsp REWARD ##############################################
                    gsp_reward, label, gsp_squared_error, raw_diff_rad = calculate_gsp_reward(
                        config['GSP'],
                        old_cyl_ang,
                        obj_stats[5],
                        next_heading_gsp,
                        Utility.params['num_robots']
                    )
                    # Diagnostic (2026-04-20 audit): accumulate raw per-step rotation
                    # BEFORE the ×100 / clip[-1,1] step in env.calculate_gsp_reward.
                    # Lets us measure the true signal distribution the supervised MSE
                    # head is trying to predict and decide whether the current scaling
                    # destroys the regression target.
                    if _gsp_raw_diff_episode is not None:
                        _gsp_raw_diff_episode.append(raw_diff_rad)
                    # print('[MAIN] GSP Reward', gsp_reward)
                    # print('[MAIN] GSP Label ', label)

                    # Multi-target label computation (GSP_OUTPUT_KIND).
                    # Computed once per timestep; stored per robot in the GSP
                    # transition store block below. The scalar `label` from
                    # calculate_gsp_reward is still used for logging/reward.
                    ep_step_counter += 1
                    if _gsp_output_kind == 'cyl_kinematics_3d':
                        # (cyl_Δx, cyl_Δy, cyl_Δθ): delta cylinder position + heading
                        # obj_stats: [0]=x_pos, [1]=y_pos, [5]=z_deg (heading)
                        _multi_label = np.array([
                            float(obj_stats[0]) - float(prev_obj_stats[0]),  # Δx
                            float(obj_stats[1]) - float(prev_obj_stats[1]),  # Δy
                            float(obj_stats[5]) - float(prev_obj_stats[5]),  # Δθ
                        ], dtype=np.float32)
                    elif _gsp_output_kind == 'cyl_kinematics_goal_4d':
                        # (cyl_Δx, cyl_Δy, cyl_Δθ, group_centroid_Δ_to_goal)
                        # group_centroid_Δ_to_goal: negative change in cyl_dist2goal
                        # (positive = centroid moved toward goal).
                        curr_cyl_dist2goal = float(env_observations[0][6]) if len(env_observations) > 0 else 0.0
                        _centroid_delta_to_goal = prev_cyl_dist2goal - curr_cyl_dist2goal
                        _multi_label = np.array([
                            float(obj_stats[0]) - float(prev_obj_stats[0]),  # Δx
                            float(obj_stats[1]) - float(prev_obj_stats[1]),  # Δy
                            float(obj_stats[5]) - float(prev_obj_stats[5]),  # Δθ
                            _centroid_delta_to_goal,
                        ], dtype=np.float32)
                        prev_cyl_dist2goal = curr_cyl_dist2goal
                    elif _gsp_output_kind == 'time_to_goal_1d':
                        # Regression on remaining episode steps until success.
                        # At success (reached_goal=True) this is 0; otherwise we
                        # don't know the future horizon, so we use 0 for non-terminal
                        # steps and record 0 at success. This is a sparse target but
                        # computable without lookahead.
                        _multi_label = np.array([0.0 if reached_goal else 0.0], dtype=np.float32)
                    else:
                        # delta_theta_1d or future_prox_1d: scalar label unchanged
                        _multi_label = np.array([label], dtype=np.float32)

                    # M4 candidate-target logging (GSP_LOG_CANDIDATE_TARGETS).
                    # Compute ALL FOUR candidate targets EVERY step, independently of
                    # the active GSP_OUTPUT_KIND above (so the offline M4 analysis can
                    # rank task-relevance without re-running the sim per target). This
                    # block is behind the flag → default off → strict no-op. It reads
                    # prev_obj_stats (previous step) BEFORE it is overwritten below, and
                    # computes centroid-to-goal from the live cyl_dist2goal without
                    # touching the prev_cyl_dist2goal advance owned by the active-kind
                    # branch above. future_prox candidate = mean current per-robot
                    # proximity (raw summed prox over the prox window); the offline
                    # analysis shifts it by the horizon K to form the future-prox target.
                    if _gsp_log_candidate_targets:
                        _cand_cyl_kin = [
                            float(obj_stats[0]) - float(prev_obj_stats[0]),  # Δx
                            float(obj_stats[1]) - float(prev_obj_stats[1]),  # Δy
                            float(obj_stats[5]) - float(prev_obj_stats[5]),  # Δθ
                        ]
                        _cand_curr_cyl_dist2goal = (
                            float(env_observations[0][6]) if len(env_observations) > 0 else 0.0
                        )
                        _cand_centroid_goal = prev_cyl_dist2goal - _cand_curr_cyl_dist2goal
                        _cand_prox_sums = [
                            float(np.sum(env_observations[i][7:]))
                            for i in range(Utility.params['num_robots'])
                        ]
                        _cand_future_prox = (
                            float(np.mean(_cand_prox_sums)) if _cand_prox_sums else 0.0
                        )
                        hdf5_writer.record_candidate_targets(
                            delta_theta=float(label),
                            future_prox=_cand_future_prox,
                            cyl_kin=_cand_cyl_kin,
                            centroid_goal=_cand_centroid_goal,
                        )

                    # Update previous cylinder stats for next step's delta computation.
                    prev_obj_stats = obj_stats.copy()

                    e2e_gsp_label = None
                    if config.get('GSP_E2E_ENABLED'):
                        e2e_gsp_label = _multi_label
                    for i in range(len(gsp_reward)):
                        episode_gsp_rewards[i] += gsp_reward[i]

                    old_cyl_ang = obj_stats[5]

                    old_agent_prox_flags = list(agent_prox_flags)
                    neighbors_old_heading_gsp = old_heading_gsp.copy()
                    old_heading_gsp = next_heading_gsp.copy()

                    new_agent_states = []
                    force_mags = []
                    force_angs = []
                    r = []
                    agent_prox_flags = []
                    next_object_heading = np.zeros(Utility.params['num_robots'])
                
                    # Build proximity observation
                    for i in range(Utility.params['num_robots']):
                        robot_failures.append(failures[i][0])
                        if failures[i][0]:
                            agent_prox_flags.append(0)
                        else:
                            prox_values = env_observations[i][7:]
                            prox_values, filtered_indeces = model.filter_prox_values(prox_values, env_observations[i][5])
                            for j in range(len(filtered_indeces)):
                                env_observations[i][7+filtered_indeces[j]] = 0.0
                            prox_value = np.sum(prox_values)              
                            agent_prox_flags.append(prox_value/float(len(filtered_indeces)))

                    e2e_gsp_obs = [None] * Utility.params['num_robots']
                    # The main replay must carry the raw GSP-input vector (gsp_obs)
                    # whenever the actor learn step re-encodes it WITH gradient. Two
                    # paths need it: the legacy scalar e2e coupling (GSP_E2E_ENABLED)
                    # and the coupled-JEPA fix (GSP_JEPA_COUPLE_VALUE), which flows the
                    # DDQN value gradient into the JEPA online encoder. Unified here so
                    # both the population site and the store call sites share one gate.
                    _needs_gsp_obs = bool(config.get('GSP_E2E_ENABLED')) or bool(
                        config.get('GSP_JEPA_COUPLE_VALUE')
                    )
                    # H-14 / first-principles diagnostic: capture the GSP head's per-robot
                    # input vector at this timestep so it can be logged alongside gsp_target.
                    # Set in the GSP branch below; remains None for non-GSP runs.
                    gsp_obs_per_robot = None
                    # The actual GSP head input vector(s) this timestep. Used only to
                    # populate the rolling diagnostics pool so freeze_diagnostic_batch
                    # gets shape-correct samples. Distinct from gsp_obs_per_robot (which
                    # is h5-logged per-robot, shape (R, 1) for plain-GSP) because the
                    # plain-GSP head takes one shared (GSP_INPUT_SIZE,) vector, not R
                    # scalar inputs. Conflating the two (the original B-004 regression)
                    # caused all plain-GSP cells to crash in freeze_diagnostic_batch at
                    # DIAGNOSTICS_FREEZE_EPISODE with a shape-mismatch in the head's fc1.
                    diag_gsp_head_input = None

                    # Change 3: build payload_state and self_dynamics dicts for the
                    # enrichment flags. These are computed once per timestep and passed
                    # to both the predict and store-transition make_gsp_states calls.
                    # When the flags are False the dicts are None and ignored by agent.py.
                    _payload_state_arg = None
                    _self_dynamics_arg = None
                    if _gsp_input_include_payload_state:
                        _comX_now = float(obj_stats[7])
                        _comY_now = float(obj_stats[8])
                        _cyl_ang_now = float(obj_stats[5])
                        # Velocity: zero on first step (prev buffer not yet populated).
                        _pl_vx = _comX_now - float(_prev_payload_comX) if _prev_payload_comX is not None else 0.0
                        _pl_vy = _comY_now - float(_prev_payload_comY) if _prev_payload_comY is not None else 0.0
                        _pl_omega = _cyl_ang_now - float(_prev_payload_cyl_angle) if _prev_payload_cyl_angle is not None else 0.0
                        # Payload-to-goal offset (normalized by distance_to_goal_normalization_factor).
                        # cyl_dist2goal and cyl_angle2goal are available from env_observations[0][6]
                        # and obj_stats[6] respectively. Reconstruct goal absolute position from
                        # cylinder CoM + distance*cos/sin(angle_to_goal).
                        _cyl_dist2goal = float(env_observations[0][6]) if len(env_observations) > 0 else 0.0
                        _cyl_ang2goal = float(obj_stats[6])
                        _norm = float(Utility.params.get('distance_to_goal_normalization_factor', 1.0))
                        if _norm == 0.0:
                            _norm = 1.0
                        _goal_x = _comX_now + _cyl_dist2goal * math.cos(math.radians(_cyl_ang2goal))
                        _goal_y = _comY_now + _cyl_dist2goal * math.sin(math.radians(_cyl_ang2goal))
                        _dx_to_goal = (_goal_x - _comX_now) / _norm
                        _dy_to_goal = (_goal_y - _comY_now) / _norm
                        # Payload state is the same for all agents (shared payload).
                        _n_r = Utility.params['num_robots']
                        _payload_state_arg = {
                            'vx': [_pl_vx] * _n_r,
                            'vy': [_pl_vy] * _n_r,
                            'omega': [_pl_omega] * _n_r,
                            'dx_to_goal': [_dx_to_goal] * _n_r,
                            'dy_to_goal': [_dy_to_goal] * _n_r,
                        }
                        # Update prev-step payload buffer for next timestep.
                        _prev_payload_comX = _comX_now
                        _prev_payload_comY = _comY_now
                        _prev_payload_cyl_angle = _cyl_ang_now

                    if _gsp_input_include_self_dynamics:
                        _n_r = Utility.params['num_robots']
                        _self_vx = []
                        _self_vy = []
                        for _ri in range(_n_r):
                            _rx_now = float(robot_stats[_ri][0])
                            _ry_now = float(robot_stats[_ri][1])
                            _prev_rx = float(_prev_robot_x[_ri]) if _prev_robot_x is not None else _rx_now
                            _prev_ry = float(_prev_robot_y[_ri]) if _prev_robot_y is not None else _ry_now
                            _self_vx.append(_rx_now - _prev_rx)
                            _self_vy.append(_ry_now - _prev_ry)
                        _self_dynamics_arg = {
                            'vx': _self_vx,
                            'vy': _self_vy,
                            'force_mag': [float(stats[_ri][0]) for _ri in range(_n_r)],
                            'force_ang': [float(stats[_ri][1]) for _ri in range(_n_r)],
                        }
                        # Update prev-step robot position buffer for next timestep.
                        _prev_robot_x = [float(robot_stats[_ri][0]) for _ri in range(_n_r)]
                        _prev_robot_y = [float(robot_stats[_ri][1]) for _ri in range(_n_r)]

                    # Wrap-safe world-frame cyl-bearing delta. Computed once per
                    # timestep from world positions and passed to BOTH make_gsp_states
                    # calls. The prev-bearing buffer advances exactly once per step here.
                    _cyl_bearing_delta_arg = None
                    if _gsp_input_include_cyl_bearing_delta:
                        _n_r = Utility.params['num_robots']
                        _bearings_now = []
                        _deltas = []
                        # Reference point is the CYLINDER position (obj_stats[0]/[1],
                        # logged as cyl_x_pos/cyl_y_pos), NOT the payload COM
                        # (obj_stats[7]/[8]). The bearing-around-cylinder delta vs the
                        # cylinder centre correlates ~0.77 with the target; vs the COM it
                        # correlates ~0 (verified on run h5).
                        _cyl_x = float(obj_stats[0])
                        _cyl_y = float(obj_stats[1])
                        for _ri in range(_n_r):
                            _bearing = math.atan2(
                                float(robot_stats[_ri][1]) - _cyl_y,
                                float(robot_stats[_ri][0]) - _cyl_x,
                            )
                            if _prev_cyl_bearing is None:
                                _d = 0.0
                            else:
                                _d = _bearing - float(_prev_cyl_bearing[_ri])
                                _d = (_d + math.pi) % (2 * math.pi) - math.pi
                            _bearings_now.append(_bearing)
                            _deltas.append(_d)
                        _cyl_bearing_delta_arg = {'delta': _deltas}
                        # Advance prev-bearing buffer exactly once per timestep.
                        _prev_cyl_bearing = _bearings_now

                    if config['GSP']:
                        # GSP Predict
                        if args.independent_learning:
                            for i in range(Utility.params['num_robots']):
                                next_object_heading[i] = models[i].choose_agent_gsp(agent_prox_flags, test_mode)
                                next_heading_gsp[i] = next_object_heading[i]
                                # M2 eval-time prediction ablation (injection site).
                                # 'none' (default) is a literal identity no-op.
                                # Deferred during the feature-stats warm-up burn-in
                                # (stats must reflect the true prediction stream).
                                if _gsp_eval_ablate_pred != 'none' and not _in_stats_warmup:
                                    next_heading_gsp[i] = apply_pred_ablation(
                                        next_heading_gsp[i], _gsp_eval_ablate_pred,
                                        _pred_ablation_rng, _pred_frozen_mean_state,
                                    )
                        else:
                            if model.gsp_neighbors:
                                # Pass env_observations when input enrichment flags are active.
                                _env_obs_arg = env_observations if _gsp_input_needs_env_obs else None
                                agent_gsp_states = model.make_gsp_states(
                                    agent_prox_flags, old_heading_gsp,
                                    env_observations=_env_obs_arg,
                                    payload_state=_payload_state_arg,
                                    self_dynamics=_self_dynamics_arg,
                                    cyl_bearing_delta=_cyl_bearing_delta_arg,
                                )
                                ctde_gsp = model.choose_agent_gsp(agent_gsp_states, test_mode)
                                gsp_obs_per_robot = agent_gsp_states
                                # GSP-N head takes (GSP_INPUT_SIZE,) per robot; agent_gsp_states
                                # is already shape (R, GSP_INPUT_SIZE) — use directly.
                                diag_gsp_head_input = agent_gsp_states
                            elif model.gsp_broadcast:
                                # GSP-B: per-agent self-centric view with full-broadcast
                                # [self_prox, self_prev_gsp, other_i_prox, other_i_prev_gsp, ...]
                                agent_gsp_states = model.make_gsp_states_broadcast(agent_prox_flags, old_heading_gsp)
                                ctde_gsp = model.choose_agent_gsp(agent_gsp_states, test_mode)
                                gsp_obs_per_robot = agent_gsp_states
                                diag_gsp_head_input = agent_gsp_states
                            else:
                                # GSP single-shot: head sees each robot's own scalar prox only.
                                # Stored as (R, 1) so the h5 gsp_obs dataset has canonical (T, R, D)
                                # shape — needed by scripts/future_prox_recorrelation.py to
                                # reconstruct per-robot labels at t+K horizon. Previously this
                                # branch left gsp_obs_per_robot=None, blocking the recomputed
                                # per-robot corr metric for plain GSP cells (BLOCKED B-004).
                                ctde_gsp = model.choose_agent_gsp(agent_prox_flags, test_mode)
                                gsp_obs_per_robot = np.asarray(agent_prox_flags, dtype=np.float32).reshape(-1, 1)
                                # The actual head input is one shared (GSP_INPUT_SIZE,) vector —
                                # the full agent_prox_flags list. Wrap in a length-1 batch dim so
                                # the pool-populating loop yields one sample per step.
                                diag_gsp_head_input = np.asarray(agent_prox_flags, dtype=np.float32).reshape(1, -1)
                            for i in range(Utility.params['num_robots']):
                                # Multi-dim GSP output: store the full K-dim prediction vector
                                # for each robot. Per-agent predictions come back as either a
                                # numpy array (DDPG / continuous head) or a torch tensor
                                # (depending on choose_action's return path). Handle both.
                                # NOTE: do NOT use `[-1]` here — that would truncate the K-dim
                                # vector to its last element. Take the whole per-agent prediction.
                                _pred_raw = ctde_gsp[i] if len(ctde_gsp) > 1 else ctde_gsp
                                if hasattr(_pred_raw, 'detach'):
                                    _pred_raw = _pred_raw.detach().cpu().numpy()
                                _pred_vec = np.asarray(_pred_raw, dtype=np.float32).ravel()
                                if _pred_vec.size != _gsp_K:
                                    _pred_vec = np.resize(_pred_vec, _gsp_K)
                                next_heading_gsp[i] = _pred_vec
                                # M2 eval-time prediction ablation (injection site).
                                # Applied immediately after next_heading_gsp[i] is set
                                # and BEFORE make_agent_state consumes it. 'none'
                                # (default) is a literal identity no-op → bit-exact.
                                # Deferred during the feature-stats warm-up burn-in
                                # (stats must reflect the true prediction stream).
                                if _gsp_eval_ablate_pred != 'none' and not _in_stats_warmup:
                                    next_heading_gsp[i] = apply_pred_ablation(
                                        next_heading_gsp[i], _gsp_eval_ablate_pred,
                                        _pred_ablation_rng, _pred_frozen_mean_state,
                                    )
                        # print("-------------------------------------------------")
                        # print('[GSP]', next_heading_gsp)

                        # Store GSP Transition — guard by per-robot force magnitude.
                        # GSP_STORE_FORCE_THRESHOLD concentrates training on samples where
                        # the robot is actively applying force (top ~25% of samples at
                        # threshold ~4.0), which multiplies the linear-R² ceiling of the
                        # prediction problem 3–4× (see
                        # docs/research/2026-04-13-gsp-ddpg-vs-attention-collapse.md).
                        # 0.0 = filter disabled (legacy behavior).
                        force_thr = float(config.get('GSP_STORE_FORCE_THRESHOLD', 0.0))
                        if model.gsp_neighbors:
                            _env_obs_arg = env_observations if _gsp_input_needs_env_obs else None
                            states, state_prox_flags = model.make_gsp_states(
                                old_agent_prox_flags, neighbors_old_heading_gsp, True,
                                env_observations=_env_obs_arg,
                                payload_state=_payload_state_arg,
                                self_dynamics=_self_dynamics_arg,
                                cyl_bearing_delta=_cyl_bearing_delta_arg,
                            )
                            new_states = model.make_gsp_states(
                                agent_prox_flags, old_heading_gsp,
                                env_observations=_env_obs_arg,
                                payload_state=_payload_state_arg,
                                self_dynamics=_self_dynamics_arg,
                                cyl_bearing_delta=_cyl_bearing_delta_arg,
                            )
                            if _needs_gsp_obs:
                                for i in range(Utility.params['num_robots']):
                                    e2e_gsp_obs[i] = np.array(states[i], dtype=np.float32)

                            # Candidate A: delayed-label targets — store transitions whose label
                            # is only observable K steps after the state is seen. Buffer accumulates
                            # (state_t) snapshots; only when matured at t+K do we have
                            # (state_{t-K}, label_t) pairs to store. The delayed-label targets share
                            # this FIFO — only the label VALUE differs:
                            #   future_prox   → label_i = robot i's own current proximity at t+K.
                            #   neighbor_force→ label_i = mean applied force-magnitude of the OTHER
                            #                   robots at t+K (mean_{j != i} force_magnitude[t+K, j]).
                            #                   force_magnitude[j] is stats[j][0] at the current step.
                            #   delta_theta_traj → label = size-K payload-rotation TRAJECTORY
                            #                   over the next K steps, the vector
                            #                   [Δθ(t→t+1), …, Δθ(t+K-1→t+K)], the SAME K-vector
                            #                   for every robot. Every step's payload angle
                            #                   (obj_stats[5], degrees) is pushed into the FIFO; at
                            #                   maturity the pop returns the ordered K+1-angle window
                            #                   [angle(t), …, angle(t+K)] and we difference consecutive
                            #                   entries. angle_normalize_signed_deg wraps each per-step
                            #                   difference into [-180,180) so a boundary crossing
                            #                   (e.g. 179 → -179) is a small per-step delta, not ~358.
                            #   goal_progress_traj → label = size-K GLOBAL payload progress-to-goal
                            #                   TRAJECTORY over the next K steps: per-step
                            #                   prev_cyl_dist2goal − curr_cyl_dist2goal (positive =
                            #                   toward goal), the SAME K-vector for every robot.
                            #                   Every step's payload track ({dist2goal, cyl_x, cyl_y})
                            #                   is pushed into the FIFO; at maturity the pop returns
                            #                   the ordered K+1-entry track window and we difference
                            #                   consecutive dist2goal entries. RAW meters, no scaling.
                            #   cyl_displacement_traj → label = size-2K GLOBAL payload displacement
                            #                   TRAJECTORY over the next K steps, flattened
                            #                   [Δx1,Δy1,…,ΔxK,ΔyK] from the same track window
                            #                   (cyl_x/cyl_y = obj_stats[0]/[1]). RAW meters.
                            _gsp_pred_target = getattr(model, 'gsp_prediction_target', 'delta_theta')
                            if (_gsp_pred_target in ('future_prox', 'neighbor_force')
                                    or _gsp_pred_target in _GSP_TRAJ_TARGETS):
                                if _gsp_pred_target in _GSP_TRAJ_TARGETS:
                                    # Carry the CURRENT payload angle (degrees) / payload track
                                    # so the full window is available to build the per-step
                                    # trajectory.
                                    # In E2E mode the head trains ONLY from learn_DDQN_e2e (the
                                    # head's own replay is unused), so this head-store FIFO push
                                    # is skipped there — the E2E delayed store runs its OWN single
                                    # push+pop at the RL-transition store site (below), where the
                                    # next-state + guards are available. Non-E2E keeps this push.
                                    if not config.get('GSP_E2E_ENABLED'):
                                        if _gsp_pred_target == 'delta_theta_traj':
                                            model.push_pending_gsp_obs(
                                                states, states, payload_angle_deg=float(obj_stats[5])
                                            )
                                        else:
                                            # Global targets: carry the payload track (raw meters).
                                            model.push_pending_gsp_obs(
                                                states, states,
                                                payload_track={
                                                    'dist2goal': (
                                                        float(env_observations[0][6])
                                                        if len(env_observations) > 0 else 0.0
                                                    ),
                                                    'cyl_x': float(obj_stats[0]),
                                                    'cyl_y': float(obj_stats[1]),
                                                },
                                            )
                                    # Label is built from the returned window → pass None.
                                    _current_label = None
                                elif _gsp_pred_target == 'neighbor_force':
                                    model.push_pending_gsp_obs(states, states)
                                    _n_r = Utility.params['num_robots']
                                    _force_mags_now = np.asarray(
                                        [float(stats[j][0]) for j in range(_n_r)],
                                        dtype=np.float32,
                                    )
                                    # Per-robot mean force of the OTHER robots at the current step.
                                    _force_sum = float(_force_mags_now.sum())
                                    if _n_r > 1:
                                        _current_label = (
                                            (_force_sum - _force_mags_now) / (_n_r - 1)
                                        ).astype(np.float32)
                                    else:
                                        _current_label = np.zeros(_n_r, dtype=np.float32)
                                else:  # future_prox
                                    model.push_pending_gsp_obs(states, states)
                                    _current_label = np.asarray(agent_prox_flags, dtype=np.float32)
                                # E2E trajectory targets own the FIFO at the RL-store site
                                # (single push+pop there); skip the head-store pop here so
                                # the two paths never double-consume the same buffer.
                                _skip_head_store_pop = bool(
                                    config.get('GSP_E2E_ENABLED')
                                    and _gsp_pred_target in _GSP_TRAJ_TARGETS
                                )
                                matured = (
                                    None if _skip_head_store_pop
                                    else model.pop_matured_gsp_label(_current_label)
                                )
                                if matured is not None:
                                    if _gsp_pred_target in _GSP_TRAJ_TARGETS:
                                        # Build the size-K (or 2K) per-step trajectory from the
                                        # ordered FIFO window (oldest→newest), shared by all robots.
                                        # delta_theta_traj: wrapped rotation (degrees); global
                                        # targets: raw-meter progress / displacement deltas.
                                        _traj = _build_traj_label_from_windows(
                                            _gsp_pred_target,
                                            matured['payload_angle_window'],
                                            matured['payload_track_window'],
                                        )
                                        _matured_labels = [
                                            _traj for _ in range(Utility.params['num_robots'])
                                        ]
                                    else:
                                        _matured_labels = [
                                            float(matured['label_per_robot'][i])
                                            for i in range(Utility.params['num_robots'])
                                        ]
                                    for i in range(Utility.params['num_robots']):
                                        s_to_store = matured['state_per_robot'][i]
                                        label_to_store = _matured_labels[i]
                                        model.store_gsp_transition(s_to_store, label_to_store, 0, s_to_store, 0)
                                        hdf5_writer.record_stored_transition(label_to_store, s_to_store)
                            else:
                                # Multi-target label: use _multi_label for all non-future_prox kinds.
                                # For scalar kinds (_multi_label.size==1) the store_gsp_transition
                                # call is identical to the legacy path. For vector kinds, the numpy
                                # array is stored as the action field in the replay buffer.
                                _label_to_store = _multi_label if _multi_label.size > 1 else float(_multi_label[0])
                                for i in range(Utility.params['num_robots']):
                                    if np.sum(state_prox_flags[i]) > 0 and stats[i][0] > force_thr:
                                        if model.gsp_networks['learning_scheme'] == 'attention':
                                            model.store_gsp_transition(states[i], label, 0, 0, 0)
                                            hdf5_writer.record_stored_transition(label, states[i])
                                        else:
                                            state = states[i]
                                            new_state = new_states[i]
                                            model.store_gsp_transition(state, _label_to_store, 0, new_state, 0)
                                            hdf5_writer.record_stored_transition(_label_to_store, state)
                        elif model.gsp_broadcast:
                            states = model.make_gsp_states_broadcast(old_agent_prox_flags, neighbors_old_heading_gsp)
                            new_states = model.make_gsp_states_broadcast(agent_prox_flags, old_heading_gsp)
                            _label_to_store = _multi_label if _multi_label.size > 1 else float(_multi_label[0])
                            for i in range(Utility.params['num_robots']):
                                if states[i][0] != 0 and stats[i][0] > force_thr:
                                    model.store_gsp_transition(states[i], _label_to_store, 0, new_states[i], 0)
                                    hdf5_writer.record_stored_transition(_label_to_store, states[i])
                        else:
                            _label_to_store = _multi_label if _multi_label.size > 1 else float(_multi_label[0])
                            for i in range(Utility.params['num_robots']):
                                state = np.array(old_agent_prox_flags)
                                if np.sum(state) > 0 and stats[i][0] > force_thr:
                                    if model.gsp_networks['learning_scheme'] == 'attention':
                                        model.store_gsp_transition(state, label, 0, 0, 0)
                                        hdf5_writer.record_stored_transition(label, state)
                                    elif args.independent_learning:
                                        new_state = np.array(agent_prox_flags)
                                        models[i].store_gsp_transition(state, _label_to_store, 0, new_state, 0)
                                        hdf5_writer.record_stored_transition(_label_to_store, state)
                                    else:
                                        new_state = np.array(agent_prox_flags)
                                        model.store_gsp_transition(state, _label_to_store, 0, new_state, 0)
                                        hdf5_writer.record_stored_transition(_label_to_store, state)


                    #Define Global Knowledge: [positions, velocities]
                    # T7: vectorized via src.knowledge (inert, golden-verified).
                    global_knowledge = build_global_knowledge(robot_stats, stats)
                    g_knowledge_all = build_g_knowledge_all(global_knowledge)

                    for i in range(Utility.params['num_robots']):
                        g_knowledge = g_knowledge_all[i]
                        prox_values = env_observations[i][7:]
                        prox_value = np.sum(prox_values)
                        rewards[i] += (-1)*prox_value
                        # H-phase5-2 reward shaping: when GSP_REWARD_COEF > 0, add
                        # gsp_reward[i] (signed prediction-error penalty in [-2, 0])
                        # scaled by coef to the actor's training reward. Default 0.0
                        # is bit-identical (the if branch is skipped). gsp_reward was
                        # computed at line 429 by env.calculate_gsp_reward and is in
                        # scope here as a per-robot list.
                        if _gsp_reward_coef > 0.0:
                            if _gsp_reward_random_noise:
                                # Replace gsp_reward with a Gaussian-squared
                                # penalty of matched magnitude. Same clip range
                                # [-2, 0] so the additive perturbation is
                                # statistically comparable to gsp_reward.
                                _noise = float(np.clip(-(np.random.normal(0.0, 1.0))**2, -2.0, 0.0))
                                rewards[i] += _gsp_reward_coef * _noise
                            else:
                                rewards[i] += _gsp_reward_coef * float(gsp_reward[i])
                        force_mags.append(stats[i][0])
                        force_angs.append(stats[i][1])

                        if args.independent_learning:
                            if config['GSP']:
                                if args.global_knowledge:
                                    new_agent_state = models[i].make_agent_state(env_observations[i], heading_gsp = next_heading_gsp[i], global_knowledge=g_knowledge) 
                                else:
                                    new_agent_state = models[i].make_agent_state(env_observations[i], heading_gsp = next_heading_gsp[i])
                            else:
                                if args.global_knowledge:
                                    new_agent_state = models[i].make_agent_state(env_observations[i], global_knowledge = g_knowledge)
                                else:
                                    new_agent_state = env_observations[i]
                            
                        else:
                            if config['GSP']:
                                if args.global_knowledge:
                                    new_agent_state = model.make_agent_state(env_observations[i], heading_gsp=next_heading_gsp[i], global_knowledge=g_knowledge)
                                else:
                                    new_agent_state = model.make_agent_state(env_observations[i], heading_gsp=next_heading_gsp[i])
                            else: 
                                if args.share_prox_values:
                                    new_agent_state = np.concatenate((env_observations[i], agent_prox_flags))
                                else:
                                    if args.global_knowledge:
                                        new_agent_state = model.make_agent_state(env_observations[i], global_knowledge=g_knowledge)
                                    else:
                                        new_agent_state = env_observations[i]

                        new_agent_states.append(new_agent_state)
                        # Successor-Features cumulant phi (GSP_SF_ENABLED). The GSP-RL
                        # SF head trains psi to predict the discounted sum of this
                        # per-step, reward-relevant, low-dim vector; Q = psi . w. The
                        # reward comes from the C++ sim as a single scalar (it is NOT
                        # decomposed on the wire), so we source phi from the
                        # already-available post-step observation + force stats — the
                        # same quantities the shaped reward is built from
                        # (direction-to-goal, gate/payload progress, force). d_phi=5.
                        # None when SF is off -> byte-identical store (no phi column).
                        sf_phi = None
                        if config.get('GSP_SF_ENABLED'):
                            _obs = env_observations[i]
                            sf_phi = np.array([
                                float(_obs[0]),                 # robot_dist2goal
                                float(_obs[6]),                 # cyl_dist2goal (payload/gate progress)
                                float(np.cos(_obs[5])),         # cyl bearing alignment (direction-to-goal)
                                float(stats[i][0]),             # applied force magnitude
                                float(rewards[i][0]),           # scalar reward (anchors w.phi ~= r)
                            ], dtype=np.float32)
                        # Trajectory-target E2E (delta_theta_traj / goal_progress_traj /
                        # cyl_displacement_traj): the RL transition AND its size-K (2K)
                        # trajectory label are stored by the DELAYED FIFO path (they
                        # only mature K steps later), so the immediate store here is
                        # skipped for those configs to avoid double-storing the
                        # transition. All other configs store immediately as before.
                        _defer_immediate_e2e_store = bool(
                            config.get('GSP_E2E_ENABLED')
                            and getattr(model, 'gsp_prediction_target', 'delta_theta')
                            in _GSP_TRAJ_TARGETS
                        )
                        if time_steps > 2 and not _defer_immediate_e2e_store:
                            if train_mode:
                                if learning_scheme != 'None':
                                    if not old_failures[i] and not failures[i]:
                                        if not episode_done:
                                            if args.independent_learning:
                                                models[i].store_agent_transition(agent_states[i],
                                                                    (actions[i], actions_to_take[i]),
                                                                    rewards[i],
                                                                    new_agent_states[i],
                                                                    episode_done,
                                                                    gsp_obs=e2e_gsp_obs[i] if _needs_gsp_obs else None,
                                                                    gsp_label=e2e_gsp_label if config.get('GSP_E2E_ENABLED') else None,
                                                                    phi=sf_phi)
                                            else:
                                                model.store_agent_transition(agent_states[i],
                                                                    (actions[i], actions_to_take[i]),
                                                                    rewards[i],
                                                                    new_agent_states[i],
                                                                    episode_done,
                                                                    gsp_obs=e2e_gsp_obs[i] if _needs_gsp_obs else None,
                                                                    gsp_label=e2e_gsp_label if config.get('GSP_E2E_ENABLED') else None,
                                                                    phi=sf_phi)

                        r.append(rewards[i][0])

                    # --- trajectory-target E2E delayed main-replay store (Bug C Option 1) ---
                    # In E2E mode the head trains ONLY from learn_DDQN_e2e's MSE, so the
                    # main-replay gsp_label must be the FUTURE K-step (or 2K for
                    # cyl_displacement_traj) trajectory, which only matures K steps after
                    # the state is seen. Push the full per-step RL transition + store
                    # guards through the delayed FIFO (one entry per step, here where
                    # next-state + guards are complete), and at maturity store
                    # (state_{t-K}, traj_label) co-indexed. The immediate store above is
                    # skipped for these configs (_defer_immediate_e2e_store), so the
                    # transition is stored exactly once, with the correct label.
                    # Labels: delta_theta_traj is scaled by _delta_theta_traj_label_scale
                    # (pre-existing); goal_progress_traj / cyl_displacement_traj are
                    # meters × GSP_TRAJ_LABEL_SCALE (applied inside the builder;
                    # default 1.0 = raw, λ from measured label std, F15 lesson).
                    #
                    # Shared-model only: `model` is undefined under --independent_learning
                    # (each robot has its own models[i] with its own delayed-label FIFO,
                    # reset per-model at episode end), so a single shared push/pop cannot
                    # co-index correctly. This mirrors the head-store block above, which is
                    # likewise gated under the shared-model branch. E2E trajectory targets
                    # are single-model coordination studies; if independent_learning support
                    # is ever needed, route the push/pop through each models[i] FIFO.
                    _gsp_pred_target_e2e = getattr(model, 'gsp_prediction_target', 'delta_theta') \
                        if not args.independent_learning else 'delta_theta'
                    if (not args.independent_learning
                            and config.get('GSP_E2E_ENABLED')
                            and _gsp_pred_target_e2e in _GSP_TRAJ_TARGETS):
                        # gsp_obs is only captured for the neighbors (GSP-N) head
                        # (populated inside `if model.gsp_neighbors`); a broadcast /
                        # plain-GSP trajectory-E2E run would silently store zeroed gsp_obs
                        # and train the head on all-zero inputs. Fail loudly instead.
                        if not getattr(model, 'gsp_neighbors', False):
                            raise ValueError(
                                "GSP_E2E_ENABLED + GSP_PREDICTION_TARGET="
                                f"{_gsp_pred_target_e2e} "
                                "is currently supported only for the GSP-N (neighbors) head "
                                "— the E2E gsp_obs is captured only on that path."
                            )
                        _n_r = Utility.params['num_robots']
                        _e2e_tx = {
                            'agent_state': [np.asarray(agent_states[i], dtype=np.float32).copy() for i in range(_n_r)],
                            'action': [(actions[i], actions_to_take[i]) for i in range(_n_r)],
                            'reward': [rewards[i] for i in range(_n_r)],
                            'new_agent_state': [np.asarray(new_agent_states[i], dtype=np.float32).copy() for i in range(_n_r)],
                            'done': bool(episode_done),
                            'gsp_obs': [np.asarray(e2e_gsp_obs[i], dtype=np.float32).copy() if e2e_gsp_obs[i] is not None else None for i in range(_n_r)],
                            'phi': [None] * _n_r,
                            # Store guards captured at push time (t-K); replayed at maturity.
                            'guard_time_steps': int(time_steps),
                            'guard_train_mode': bool(train_mode),
                            'guard_learning_scheme': learning_scheme,
                            'guard_old_failures': [bool(old_failures[i]) for i in range(_n_r)],
                            'guard_failures': [bool(failures[i][0]) for i in range(_n_r)],
                            'guard_episode_done': bool(episode_done),
                        }
                        model.push_pending_gsp_obs(
                            agent_states, [e2e_gsp_obs[i] if e2e_gsp_obs[i] is not None
                                           else np.zeros(1, dtype=np.float32)
                                           for i in range(_n_r)],
                            payload_angle_deg=float(obj_stats[5]),
                            payload_track={
                                'dist2goal': (
                                    float(env_observations[0][6])
                                    if len(env_observations) > 0 else 0.0
                                ),
                                'cyl_x': float(obj_stats[0]),
                                'cyl_y': float(obj_stats[1]),
                            },
                            e2e_transition=_e2e_tx,
                        )
                        _matured_e2e = model.pop_matured_gsp_label(None)
                        if _matured_e2e is not None and _matured_e2e.get('e2e_transition') is not None:
                            _traj_e2e = _build_traj_label_from_windows(
                                _gsp_pred_target_e2e,
                                _matured_e2e['payload_angle_window'],
                                _matured_e2e['payload_track_window'],
                            )
                            if _gsp_pred_target_e2e == 'delta_theta_traj':
                                # Pre-existing non-saturating rotation scale (deg → rad×10).
                                _traj_label = (_traj_e2e * _delta_theta_traj_label_scale).astype(np.float32)
                            else:
                                # Global targets (goal_progress_traj / cyl_displacement_traj):
                                # meters × GSP_TRAJ_LABEL_SCALE, applied inside the builder
                                # (default 1.0 = raw; λ from measured label std, F15).
                                _traj_label = _traj_e2e.astype(np.float32)
                            _tx = _matured_e2e['e2e_transition']
                            if (_tx['guard_time_steps'] > 2
                                    and _tx['guard_train_mode']
                                    and _tx['guard_learning_scheme'] != 'None'
                                    and not _tx['guard_episode_done']):
                                for i in range(_n_r):
                                    if (not _tx['guard_old_failures'][i]
                                            and not _tx['guard_failures'][i]):
                                        # Shared-model only (guarded above): use `model`.
                                        model.store_agent_transition(
                                            _tx['agent_state'][i],
                                            _tx['action'][i],
                                            _tx['reward'][i],
                                            _tx['new_agent_state'][i],
                                            _tx['done'],
                                            gsp_obs=_tx['gsp_obs'][i],
                                            gsp_label=_traj_label,
                                            phi=_tx['phi'][i],
                                        )
                            # Quick label-distribution telemetry (min/max/std over the
                            # scaled per-step targets) so a crushed/near-binary label is
                            # visible in the log without waiting for h5 analysis.
                            hdf5_writer.record_stored_transition(_traj_label, _tx['gsp_obs'][0])

                    if train_mode and config['LEARNING_SCHEME'] != 'None':
                        if time_steps % learn_every == 0:
                            if args.independent_learning:
                                # Aggregate GSP losses across per-robot models to a single
                                # scalar per learn tick. Otherwise the 1D gsp_loss dataset
                                # would have (num_learn_steps × num_robots) entries in
                                # independent mode vs. num_learn_steps in shared mode,
                                # breaking cross-mode comparability of the
                                # information-collapse diagnostic.
                                for i in range(Utility.params['num_robots']):
                                    loss = models[i].learn()
                                    # TD3's learn_TD3 returns (0, 0) on non-actor-update steps;
                                    # unwrap so the hdf5 logger's 1D loss array stays homogeneous.
                                    if isinstance(loss, tuple):
                                        loss = loss[0]
                                gsp_losses = [
                                    m.last_gsp_loss for m in models
                                    if getattr(m, "last_gsp_loss", None) is not None
                                ]
                                if gsp_losses:
                                    hdf5_writer.record_gsp_loss(float(np.mean(gsp_losses)))
                                if config.get('GSP_JEPA_ENABLED'):
                                    for _m in models:
                                        _js = getattr(_m, 'last_gsp_jepa_stats', None)
                                        if _js is not None:
                                            hdf5_writer.record_jepa_pred_mse(_js.get('pred_mse', 0.0))
                                            hdf5_writer.record_jepa_latent_var(_js.get('var', 0.0))
                                            hdf5_writer.record_jepa_latent_rank(_js.get('rank', 0.0))
                                            break  # one learn step per tick; first non-None wins
                                if config.get('GSP_E2E_ENABLED'):
                                    e2e_diag = getattr(models[0], 'last_e2e_diagnostics', None)
                                    if e2e_diag is not None:
                                        hdf5_writer.record_e2e_diagnostics(e2e_diag)
                            else:
                                loss = model.learn()
                                if isinstance(loss, tuple):
                                    loss = loss[0]
                                gsp_step_loss = getattr(model, "last_gsp_loss", None)
                                if gsp_step_loss is not None:
                                    hdf5_writer.record_gsp_loss(gsp_step_loss)
                                if config.get('GSP_JEPA_ENABLED'):
                                    _js = getattr(model, 'last_gsp_jepa_stats', None)
                                    if _js is not None:
                                        hdf5_writer.record_jepa_pred_mse(_js.get('pred_mse', 0.0))
                                        hdf5_writer.record_jepa_latent_var(_js.get('var', 0.0))
                                        hdf5_writer.record_jepa_latent_rank(_js.get('rank', 0.0))
                                if config.get('GSP_E2E_ENABLED'):
                                    e2e_diag = getattr(model, 'last_e2e_diagnostics', None)
                                    if e2e_diag is not None:
                                        hdf5_writer.record_e2e_diagnostics(e2e_diag)
                        else:
                            loss = 0
                    else:
                        loss = 0

                    if args.independent_learning:
                        for i in range(Utility.params['num_robots']):
                            running_reward[i] += r[i]
                    else:
                        running_reward += np.average(r)
                    # Store New Observations
                    agent_states = new_agent_states
                    actions = []

                    # Calculate average force vector
                    average_force_mag = None
                    average_force_ang = None
                    for i in range(Utility.params['num_robots']):
                        if average_force_mag is None:
                            average_force_mag = force_mags[i]
                            average_force_ang = force_angs[i]
                        else:
                            angle = abs(average_force_ang - force_angs[i])
                            #average_force_mag = math.sqrt(average_force_mag**2 + force_mags[i]**2 + 2*(average_force_mag)*(force_mags[i])*math.cos(math.radians(angle)))
                            #average_force_ang = math.asin(force_mags[i]*math.sin(math.radians(180 - angle)) / average_force_mag)
                            average_force_mag = 0
                            average_force_ang = 0

                    if type(gate_stats) != int:
                        gate = []
                        for i in range(len(gate_stats)):
                            gate.append(gate_stats[i])
                    if type(obstacle_stats) != int:
                        obstacles = []
                        for i in range(len(obstacle_stats)):
                            obstacles.append(obstacle_stats[i])
                    if args.independent_learning:
                        tmp_epsilon = models[0].epsilon
                    else:
                        tmp_epsilon = model.epsilon

                    # gsp_target: broadcast the scalar payload delta-theta label to per-robot list
                    # so it aligns with the (timesteps × robots) HDF5 schema. Needed for the
                    # information-collapse diagnostic (gsp_output_std, gsp_pred_target_corr).
                    gsp_target_per_robot = [float(label)] * Utility.params['num_robots']
                    hdf5_writer.writerow(r, tmp_epsilon, reached_goal, loss, force_mags, force_angs,
                                    [average_force_mag, math.degrees(average_force_ang)], obj_stats[0], obj_stats[1],
                                    obj_stats[5], gate, obstacles, gsp_reward, next_heading_gsp,
                                    time.time() - episode_start_time, robot_x_pos, robot_y_pos, robot_angle,
                                    robot_failures, com_X_poses=com_X_poses, com_Y_poses=com_Y_poses,
                                    gsp_target=gsp_target_per_robot, gsp_squared_error=gsp_squared_error,
                                    gsp_obs=gsp_obs_per_robot)

                    # Populate diagnostics pools from the live training loop.
                    # See docs/specs/2026-04-17-diagnostics-instrumentation.md.
                    if getattr(model, 'diagnostics_enabled', False):
                        # Rolling pool of recent GSP head inputs. Uses diag_gsp_head_input
                        # (the actual head-input shape, (N, GSP_INPUT_SIZE)) rather than
                        # gsp_obs_per_robot, because the latter is shape (R, 1) for plain-GSP
                        # (h5 per-robot dataset) and would crash freeze_diagnostic_batch →
                        # head.fc1 with a shape mismatch. See B-008 postmortem.
                        # Capped to keep memory bounded on long runs.
                        if diag_gsp_head_input is not None:
                            for obs in diag_gsp_head_input:
                                diag_gsp_obs_pool.append(np.asarray(obs, dtype=np.float32))
                            while len(diag_gsp_obs_pool) > _DIAG_POOL_MAX_SIZE:
                                diag_gsp_obs_pool.pop(0)
                        # Accumulate this-episode GSP predictions for the diversity entropy metric.
                        if next_heading_gsp is not None:
                            diag_episode_predictions.extend(
                                float(v) for v in np.asarray(next_heading_gsp, dtype=np.float32).ravel()
                            )

                    if episode_done:
                        if args.independent_learning:
                            for m in models:
                                if hasattr(m, 'reset_hidden_states'):
                                    m.reset_hidden_states()
                                if hasattr(m, 'reset_gsp_label_buffer'):
                                    m.reset_gsp_label_buffer()
                        else:
                            if hasattr(model, 'reset_hidden_states'):
                                model.reset_hidden_states()
                            if hasattr(model, 'reset_gsp_label_buffer'):
                                model.reset_gsp_label_buffer()

                        # Phase 4 — cross-target plasticity-recovery hook.
                        # GSP_TARGET_SWITCH_AT_EP: episode at which the GSP prediction
                        # target is swapped. GSP_TARGET_SWITCH_TO: the new target string.
                        # Default values (0, '') keep the conditional permanently False,
                        # making this a strict no-op for all historical runs.
                        # Only applies to shared-model mode (not independent_learning)
                        # because independent models would each need their own switch
                        # logic and the OCP experiment is a single-model investigation.
                        _switch_at = int(config.get('GSP_TARGET_SWITCH_AT_EP', 0))
                        _switch_to = str(config.get('GSP_TARGET_SWITCH_TO', ''))
                        if (not args.independent_learning
                                and _switch_at > 0
                                and ep_counter == _switch_at
                                and _switch_to
                                and hasattr(model, 'gsp_prediction_target')):
                            old_target = model.gsp_prediction_target
                            model.gsp_prediction_target = _switch_to
                            model.reset_gsp_label_buffer()
                            log.info(
                                "GSP target switched at ep %d: %s -> %s",
                                ep_counter, old_target, _switch_to,
                            )

                        run_time = time.time() - episode_start_time

                        # Per-episode diagnostics hook. Runs before write_episode so the
                        # diag_* attrs and the (optional) diag_eval_batch_states dataset
                        # land on the same episode group. Gated on DIAGNOSTICS_ENABLED.
                        if getattr(model, 'diagnostics_enabled', False):
                            freeze_ep = getattr(model, 'diagnostics_freeze_episode', 50)
                            cadence = getattr(model, 'diagnostics_cadence', 10)
                            # Freeze the eval batch once, on/after freeze_ep, when the
                            # replay buffer is big enough and the gsp_obs pool has ≥
                            # batch_size samples.
                            if (
                                not diag_eval_batch_frozen
                                and ep_counter >= freeze_ep
                            ):
                                pool_np = (
                                    np.stack(diag_gsp_obs_pool)
                                    if len(diag_gsp_obs_pool) >= model.diagnostics_batch_size
                                    else None
                                )
                                model.freeze_diagnostic_batch(gsp_obs_pool=pool_np)
                                if getattr(model, 'diag_actor_eval_batch', None) is not None:
                                    diag_eval_batch_frozen = True
                                    hdf5_writer.record_eval_batch_states(
                                        model.diag_actor_eval_batch
                                    )
                            # Compute diagnostics on the cadence schedule once frozen.
                            if (
                                diag_eval_batch_frozen
                                and (ep_counter - freeze_ep) % cadence == 0
                            ):
                                preds = (
                                    np.asarray(diag_episode_predictions, dtype=np.float32)
                                    if diag_episode_predictions
                                    else None
                                )
                                # Capture end-of-episode snapshots for churn computation.
                                # The "after" snapshot is taken here — after all in-episode
                                # learn steps have completed — paired with the "before"
                                # snapshot taken at episode start above. This gives churn
                                # over the episode's cumulative weight update with zero
                                # extra learn calls.
                                _churn_actor_after = None
                                _churn_gsp_after = None
                                if getattr(model, 'diagnose_churn', True):
                                    _actor_net = model._main_network(model.networks)
                                    if _actor_net is not None:
                                        _churn_actor_after = copy.deepcopy(
                                            _actor_net.state_dict()
                                        )
                                    if model.gsp_networks is not None:
                                        _gsp_net = model._main_network(model.gsp_networks)
                                        if _gsp_net is not None:
                                            _churn_gsp_after = copy.deepcopy(
                                                _gsp_net.state_dict()
                                            )
                                diag_result = model.compute_diagnostics(
                                    gsp_predictions_this_episode=preds,
                                    actor_before_state_dict=_churn_actor_before,
                                    actor_after_state_dict=_churn_actor_after,
                                    gsp_before_state_dict=_churn_gsp_before,
                                    gsp_after_state_dict=_churn_gsp_after,
                                )
                                if diag_result:
                                    hdf5_writer.record_episode_diagnostics(diag_result)
                            # Reset per-episode prediction accumulator regardless of cadence.
                            diag_episode_predictions = []

                        # 2026-04-20 signal-distribution diagnostic: compute per-episode
                        # stats of raw_diff_rad (pre-scale, pre-clip) to measure what the
                        # supervised MSE target actually looks like before env.py applies
                        # ×100 / clip[-1,1]. Answers: is the label-clipping destroying a
                        # fine-grained signal, or capturing an already-saturated one?
                        if _gsp_raw_diff_episode:
                            _arr = np.asarray(_gsp_raw_diff_episode, dtype=np.float32)
                            _abs = np.abs(_arr)
                            _diag = {
                                'diag_raw_diff_rad_mean': float(np.mean(_arr)),
                                'diag_raw_diff_rad_std': float(np.std(_arr)),
                                'diag_raw_diff_rad_abs_mean': float(np.mean(_abs)),
                                'diag_raw_diff_rad_abs_p50': float(np.percentile(_abs, 50)),
                                'diag_raw_diff_rad_abs_p95': float(np.percentile(_abs, 95)),
                                'diag_raw_diff_rad_abs_max': float(np.max(_abs)),
                                # Clip frac: fraction of steps where |diff*100| >= 1
                                # (i.e., the label was saturated to ±1 after scaling).
                                'diag_raw_diff_rad_clip_frac': float(np.mean(_abs >= 0.01)),
                                'diag_raw_diff_rad_n_steps': float(len(_arr)),
                            }
                            hdf5_writer.record_episode_diagnostics(_diag)

                        # Phase 4 loss-step correlation — flush per-episode batch corr samples.
                        # The actor accumulates one float per GSP learn step in
                        # last_gsp_loss_step_corr_samples; we consume them here at episode
                        # boundary and clear the list so they don't leak into the next episode.
                        # Independent-learning mode: aggregate across per-robot models.
                        if args.independent_learning:
                            _all_corr_samples = []
                            for _m in models:
                                _samples = getattr(_m, 'last_gsp_loss_step_corr_samples', [])
                                _all_corr_samples.extend(_samples)
                                _m.last_gsp_loss_step_corr_samples = []
                            for _c in _all_corr_samples:
                                hdf5_writer.record_gsp_loss_step_corr(_c)
                        else:
                            _samples = getattr(model, 'last_gsp_loss_step_corr_samples', [])
                            for _c in _samples:
                                hdf5_writer.record_gsp_loss_step_corr(_c)
                            model.last_gsp_loss_step_corr_samples = []

                        # h5py is a hard dep of src.hdf5_logger, so the previous HAS_HDF5
                        # gate was always-true dead code. Removed during the same cleanup
                        # that dropped the data_logger references.
                        hdf5_writer.write_episode(ep_counter)
                        log.info(
                            "Episode %d done: success=%s duration=%.1fs timesteps=%d",
                            ep_counter, reached_goal, run_time, time_steps,
                        )
                        if not args.no_print:
                            print('[RUN TIME] %.2f' % run_time)
                        if args.independent_learning:
                            exp_rewards.append(np.average(running_reward))
                        else:
                            exp_rewards.append(running_reward)
                        if not reached_goal:
                            if not args.no_print:
                                print("Episode", ep_counter ,"timed out")
                            # Feature-stats warm-up burn-in episodes run with
                            # the ablation deferred and evolving stats — they
                            # are not measurement episodes and must not dilute
                            # the headline success percentage.
                            if test_mode and not _in_stats_warmup:
                                Testing_Failures += 1
                        else:
                            if not args.no_print:
                                print("Episode", ep_counter ,"reached goal")
                            if test_mode and not _in_stats_warmup:
                                Testing_Successes += 1
                        if not args.no_print:
                            for i in range(Utility.params['num_robots']):
                                if args.independent_learning:
                                    print('Agent', i, 'reward %.1f' % running_reward[i],
                                            'epsilon:%.2f' % models[i].epsilon,
                                            'steps:', models[i].learn_step_counter)
                                else:
                                    print('Agent', i, 'reward %.1f' % running_reward[0],
                                            'epsilon:%.2f' % model.epsilon,
                                            'steps:', model.networks['learn_step_counter'])
                                    print('gsp rewards %.2f' % episode_gsp_rewards[i])

                        if ep_counter % 10 == 0:
                            exp_mean_rewards.append(np.mean(exp_rewards))
                            exp_rewards = []
                            file_name = 'Episode_'+str(ep_counter)
                            path = recording_path + "/Models/" +file_name
                            if train_mode:
                                if args.independent_learning:
                                    for i in range(Utility.params['num_robots']):
                                        models[i].save_model(path)
                                else:
                                    model.save_model(path)
                            if not args.no_print:
                                print('reward last 10 eps:%.2f'%exp_mean_rewards[-1],'\n')

                        # GSP head snapshot checkpointing (Task 1 of the stability plan).
                        # Captures the GSP prediction-network weights every N episodes so
                        # post-hoc best-checkpoint selection can recover the top-correlation
                        # predictor even if it regresses during later training. 0 disables.
                        gsp_ckpt_every = int(config.get('GSP_CHECKPOINT_EVERY', 0))
                        if (train_mode and gsp_ckpt_every > 0
                                and ep_counter > 0 and ep_counter % gsp_ckpt_every == 0
                                and not args.independent_learning):
                            try:
                                ckpt_dir = os.path.join(recording_path, 'Models', 'gsp_snapshots')
                                os.makedirs(ckpt_dir, exist_ok=True)
                                ckpt_path = os.path.join(ckpt_dir, f'gsp_ep{ep_counter:04d}.pt')
                                model.save_gsp_head_snapshot(ckpt_path)
                                idx_path = os.path.join(ckpt_dir, 'index.json')
                                import json as _json_ckpt
                                idx = []
                                if os.path.exists(idx_path):
                                    try:
                                        idx = _json_ckpt.load(open(idx_path))
                                    except (ValueError, OSError):
                                        idx = []
                                idx.append({'episode': ep_counter, 'path': ckpt_path})
                                _json_ckpt.dump(idx, open(idx_path, 'w'))
                            except Exception as _e:
                                log.warning(f'GSP checkpoint save failed at ep {ep_counter}: {_e}')
                        ep_counter += 1

                        # Send acknowledgment
                        socket.send(b"ok")
    print("[RUN TIME] Experiment: %.2f" % (time.time() - experiment_start_time))
    if test_mode:
        print('Experiment:', args.recording_path)
        print("[Statistics] Success Percentage", (Testing_Successes/(Testing_Successes+Testing_Failures)))
        print("[Statistics] Failure Percentage", (Testing_Failures/(Testing_Successes+Testing_Failures)))
    print("Closing Server")
    #socket.unbind("tcp://:" + port)
    #socket.close()
    print("Experiment Done\n")
    exp_logger.finish(success=True)
except zmq.error.Again:
    error_msg = f"ZMQ timeout at episode {ep_counter} — ARGoS likely crashed"
    log.critical(error_msg)
    exp_logger.write_crash_dump(
        last_state={"episode": ep_counter, "timestep": time_steps if 'time_steps' in dir() else 0,
                    "msg_count": socket._msg_count},
        error_message=error_msg,
    )
    exp_logger.finish(success=False, error_message=error_msg)
    raise
except Exception as e:
    error_msg = f"Unexpected error at episode {ep_counter if 'ep_counter' in dir() else '?'}: {e}"
    log.critical(error_msg)
    log.critical(traceback.format_exc())
    exp_logger.write_crash_dump(
        last_state={"episode": ep_counter if 'ep_counter' in dir() else -1},
        error_message=f"{error_msg}\n{traceback.format_exc()}",
    )
    exp_logger.finish(success=False, error_message=error_msg)
    raise

"""OBSTACLE-CONTACT rule (operator-directed, 2026-07-10) — flag-gated, default OFF.

Mechanism (the reason this exists): the legacy obstacle signal is a soft,
per-robot LOCAL proximity penalty (Main.py reward loop: ``rewards[i] +=
-sum(own prox_values)``) — a far-side robot pushing the payload into an
obstacle near a teammate feels nothing (the 24-dim proximity ring is local,
and the penalty is own-sensor only). Terminating the episode on obstacle
contact with a large SHARED negative reward creates a natural occlusion:
only the near robot senses the obstacle; the far-side robots' only
anticipatory signal is the GSP payload-motion forecast — making the forecast
decision-relevant through consequence. Complements the advantage-splice
(GSP_SPLICE_ADVANTAGE_ONLY, RL-CT#40), which makes the forecast usable as
action discrimination; endgame is the 2x2 (contact-rule x splice).

Flags (agent_config.yml, launcher passthrough on the stelaris side):

  OBSTACLE_CONTACT_TERMINATE  bool, default False. On a contact event the
      episode is LOGICALLY terminated: the contact-step transition is stored
      with done=True (cuts the DDQN bootstrap: q_next[dones]=0 in GSP-RL
      learning_aids), and no further transitions (agent or GSP-head) are
      stored for the rest of the episode. NOTE — the ARGoS episode state
      machine is C++-owned (collectiveRlTransport.cpp IsEpisodeFinished =
      goal|timeout; the ZMQ actions message has no abort channel), so the
      PHYSICAL episode runs on to its natural end while Python is in a
      learner-invisible "zombie" phase. Learner-visible semantics are exactly
      those of a terminated episode; wall-clock is unchanged. A C++ abort
      channel is the pre-named follow-up if the screen promotes and episode
      turnover becomes the bottleneck.

  OBSTACLE_CONTACT_PENALTY    float, default 0.0. Added on the contact step
      to EVERY robot's individual reward (shared consequence). Expected
      negative (e.g. -10.0). Per-robot reward convention: the SAME value is
      added to each robot's own reward stream — do NOT divide by num_robots
      (rewards in this codebase are per-robot; the summed/num_robots
      normalization lives in analysis, never here).
      UNITS: the value is in RAW env-reward units — the same stream and units
      as the C++ dense reward (-2 + direction per step,
      collectiveRlTransport.cpp GetObservations) and the -sum(prox) shaping.
      It is applied BEFORE (upstream of) REWARD_SCALE: like every other
      reward component it is stored raw and multiplied by REWARD_SCALE inside
      the TD target (GSP-RL gsp_rl/src/actors/learning_aids.py:762,
      `target = reward_scale * rewards + gamma * bootstrap`). At the current
      recipe's REWARD_SCALE=0.1, a YAML value of -10.0 reaches the learner as
      -1.0. Chosen because it is the least surprising: every reward number in
      this codebase (dense reward, prox shaping, GSP_REWARD_COEF) is quoted
      in raw units and scaled at the same single site.
      MAGNITUDE WARNING (terminate mode): this env has NO terminal success
      bonus (goal_reward="0" in the argos template) and an all-negative dense
      reward, so at gamma near 1 the continuation value is a large NEGATIVE
      number (~ -2/step over the remaining horizon) — terminating EARLY is
      implicitly REWARDING, and a small penalty makes obstacle contact an
      attractive escape hatch (suicide-by-obstacle attractor). To make
      contact strictly worse than continuing, |penalty| must exceed the
      magnitude of the escaped remaining cost (~2 x effective remaining
      horizon in raw units). See the 2026-07-10 contact-rule pre-registration
      for the quantitative screen design around this.

  OBSTACLE_CONTACT_DIST       float, default derived from the ARGoS geometry:
      obstacle_radius + footbot_radius + skin epsilon. Sources:
        footbot radius   0.085036758 m  (argos/collectiveRlTransport.cpp:20,
                                         FOOTBOT_RADIUS)
        obstacle radius  0.5 m          (argos/generate_argos.py
                                         --obstacle_radius default; per-cell
                                         override OBSTACLE_RADIUS is read here
                                         when present in agent_config.yml)
        skin epsilon     0.01 m         (one-control-step penetration slack)
      Default at stock geometry: 0.5 + 0.085036758 + 0.01 = 0.595036758 m.

The two flags COMPOSE: penalty-only (TERMINATE=false, PENALTY<0) is a
legitimate dense-task control arm (penalty re-applies on every contact step,
episode continues); terminate-only is legal too.

CONTACT definition: min robot-obstacle CENTER distance over all
(robot, obstacle) pairs < OBSTACLE_CONTACT_DIST, computed from the global sim
state Main.py already parses every step (robot_stats x/y, obstacle_stats
x/y pairs). CTDE: the reward side may read global state; OBSERVATIONS ARE
UNTOUCHED — no robot gains any new sensory channel from this rule.

Fail-loud contract: Main.py logs exactly one line at startup —
"OBSTACLE_CONTACT rule: ENGAGED (terminate=X penalty=Y dist=Z)" or
"OBSTACLE_CONTACT rule: off" — keyed on the effective gate; every contact
event is logged at INFO (episode, step, robot, obstacle, distance); per
episode the h5 gets contact_terminated/contact_step/contact_count attrs
(only when the rule is engaged — the off path stays byte-identical).
"""

import numpy as np

# argos/collectiveRlTransport.cpp:20 — static const Real FOOTBOT_RADIUS.
FOOTBOT_RADIUS_M = 0.085036758

# argos/generate_argos.py --obstacle_radius argparse default (a string "0.5"
# substituted into $$obstacle_radius$$; collectiveRlTransport.cpp reads it
# into m_fObstacleRadius). Per-cell overrides arrive via the OBSTACLE_RADIUS
# key in agent_config.yml (stelaris flag_manifest.yaml scale-geom flags).
DEFAULT_OBSTACLE_RADIUS_M = 0.5

# Skin epsilon over the exact center-to-center touching distance. Covers the
# per-control-step penetration slack (the check runs on post-step positions,
# so a robot can be up to one step's travel inside the touching distance
# before the first check sees it).
CONTACT_EPS_M = 0.01


class ContactRule:
    """Parsed OBSTACLE_CONTACT_* config + the per-step contact detector.

    Pure and stateless across steps (Main.py owns the per-episode state), so
    the geometry and gating are unit-testable without ZMQ/ARGoS.
    """

    def __init__(self, config, num_obstacles):
        self.terminate = bool(config.get('OBSTACLE_CONTACT_TERMINATE', False))
        # float() the penalty defensively: YAML may deliver int -10.
        _pen = config.get('OBSTACLE_CONTACT_PENALTY')
        self.penalty = 0.0 if _pen is None else float(_pen)

        # Threshold: explicit OBSTACLE_CONTACT_DIST wins; otherwise derive
        # from the geometry the run actually uses (OBSTACLE_RADIUS override
        # respected). Only None degrades to the derived default — a valid
        # falsy 0.0 would be an explicit (if nonsensical) operator choice and
        # is rejected below, never silently remapped.
        _obstacle_radius = config.get('OBSTACLE_RADIUS')
        _obstacle_radius = (
            DEFAULT_OBSTACLE_RADIUS_M if _obstacle_radius is None
            else float(_obstacle_radius)
        )
        _dist = config.get('OBSTACLE_CONTACT_DIST')
        self.contact_dist = (
            _obstacle_radius + FOOTBOT_RADIUS_M + CONTACT_EPS_M
            if _dist is None else float(_dist)
        )

        self.enabled = self.terminate or self.penalty != 0.0

        # Fail loud on misconfiguration instead of silently never firing.
        if self.enabled and int(num_obstacles) <= 0:
            raise ValueError(
                "OBSTACLE_CONTACT rule engaged (terminate="
                f"{self.terminate} penalty={self.penalty}) but the "
                f"environment has num_obstacles={num_obstacles} — the rule "
                "can never fire. Fix the cell config (NUM_OBSTACLES > 0) or "
                "drop the OBSTACLE_CONTACT_* keys."
            )
        if self.enabled and self.contact_dist <= 0.0:
            raise ValueError(
                "OBSTACLE_CONTACT rule engaged with non-positive "
                f"OBSTACLE_CONTACT_DIST={self.contact_dist} — the contact "
                "predicate can never fire."
            )

    def startup_line(self):
        """The exactly-one startup log line, keyed on the effective gate."""
        if self.enabled:
            return (
                "OBSTACLE_CONTACT rule: ENGAGED (terminate=%s penalty=%s "
                "dist=%s)" % (self.terminate, self.penalty, self.contact_dist)
            )
        return "OBSTACLE_CONTACT rule: off"

    def detect(self, robot_stats, obstacle_stats):
        """Min center-distance contact check on post-step global sim state.

        robot_stats: per-robot arrays [x, y, z, x_deg, y_deg, z_deg]
            (ZMQ_Utility.parse_robot_stats output).
        obstacle_stats: flat (num_obstacles*2,) array [x0, y0, x1, y1, ...]
            (ZMQ_Utility.parse_obstacle_stats output).

        Returns (contact: bool, robot_idx: int, obstacle_idx: int,
        min_dist: float) — indices/distance of the closest pair whether or
        not it breaches the threshold (the event log records the distance).
        """
        obs_xy = np.asarray(obstacle_stats, dtype=np.float64).reshape(-1, 2)
        rob_xy = np.asarray(
            [[r[0], r[1]] for r in robot_stats], dtype=np.float64
        )
        d = np.linalg.norm(rob_xy[:, None, :] - obs_xy[None, :, :], axis=2)
        ri, oi = np.unravel_index(int(np.argmin(d)), d.shape)
        min_d = float(d[ri, oi])
        return (min_d < self.contact_dist), int(ri), int(oi), min_d

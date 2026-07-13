"""GOAL-ENTRY bonus rule (Option A, operator-approved 2026-07-13) — flag-gated, default OFF.

Mechanism (the reason this exists): the env reward is all-negative
(-2 + direction per step, goal_reward="0"), so at gamma near 1 the
continuation value is a large NEGATIVE number and EARLY termination is
implicitly rewarding — the contact-terminate rule (src/contact_rule.py)
created a cheap exit (suicide-by-obstacle attractor) that the punishment
lever cannot re-order (Q_TARGET_CLIP floors the whole target at -1000; the
-9000 arm measured null). Option A re-orders the exits from the other side:
a large SHARED bonus on the payload's FIRST entry into the goal region makes
clean success the highest-value exit
(docs/research/2026-07-13-reward-structure-options.md in the stelaris
superrepo, section 2.3 / Option A).

The learner-invisibility trap this module's design dodges (research doc
section 2.5): the ARGoS success terminal itself is NEVER stored — Main.py's
legacy `if not episode_done:` store guard drops the ARGoS-terminal
transition, so a bonus paid by the C++ goal_reward (or detected at exactly
the success radius) would never reach any replay buffer. The rule therefore
fires strictly BEFORE physical success: it detects entry into an OUTER
radius (success radius + margin), stores THAT step's transition with
done=True (cutting the DDQN bootstrap: q_next[dones]=0 in GSP-RL
learning_aids, so the Q-target is exactly the grounded bonus-bearing
reward), and suppresses all later stores while the physical episode runs on
to its natural ARGoS success a few steps later (the same learner-invisible
"zombie" phase as the contact rule — those zombie steps are what mature the
E2E FIFO's bonus-bearing entry).

Flags (agent_config.yml, launcher passthrough on the stelaris side):

  GOAL_BONUS          float, default 0.0 = OFF (byte-identical legacy path).
      Added on the goal-entry step to EVERY robot's individual reward
      (shared consequence — same convention as OBSTACLE_CONTACT_PENALTY).
      Per-robot reward convention: the SAME value is added to each robot's
      own reward stream — do NOT divide by num_robots.
      UNITS: raw env-reward units, UPSTREAM of REWARD_SCALE — the same
      stream as the C++ dense reward and the contact penalty; scaled once
      at the TD target (GSP-RL gsp_rl/src/actors/learning_aids.py,
      `target = reward_scale * rewards + gamma * bootstrap`). At the
      current recipe's REWARD_SCALE=0.1 and Q_TARGET_CLIP=1000, a YAML
      value of 10000.0 reaches the learner as a grounded target of ~+999.8
      — the clip edge without crossing it (research doc section 2.3).
      When engaged, goal entry ALWAYS logically terminates (no separate
      terminate flag: a bonus the learner keeps bootstrapping past would
      be farmable and would not be a grounded terminal).

  GOAL_TERMINAL_DIST  float, default derived from the env's success radius:
      GOAL_RADIUS (agent_config key when set, else 2.0 — the
      argos/generate_argos.py --goal_radius default substituted into the
      template's `threshold`/`min_threshold`, which
      collectiveRlTransport.cpp reads into m_fThreshold; threshold ==
      min_threshold so the success radius is CONSTANT over training)
      + GOAL_ENTRY_MARGIN_M (0.25 m). The margin MUST keep the threshold
      strictly above the success radius: ObjectAtTarget() flags success on
      the same post-step message a radius-equal detector would first fire
      on, episode_done arrives True, and the legacy store guard drops the
      bonus-bearing transition — the exact section-2.5 trap. The margin
      also supplies the E2E FIFO's K-step maturation window: the payload's
      margin->radius travel is the zombie phase in which the bonus-bearing
      FIFO entry receives its K maturing pushes (at typical payload speeds
      ~0.005-0.02 m/step, 0.25 m gives ~12-50 zombie steps vs the default
      K=5 horizon). Too-small margins surface LOUDLY as goal_store_dropped
      episodes, never silently.

DETECTION: cyl_dist2goal < GOAL_TERMINAL_DIST, strict `<` (mirrors both
ObjectAtTarget()'s `< m_fThreshold` and the contact rule's predicate).
cyl_dist2goal is observation slot 6 (raw meters, un-normalized:
collectiveRlTransport.cpp GetObservations stores cVecObject2Goal.Length()
directly), already parsed by Main.py every step — the SAME quantity the C++
success check compares against m_fThreshold. CTDE: the reward side may read
it; OBSERVATIONS ARE UNTOUCHED — no robot gains any new sensory channel.

INTERACTION with the contact rule (first-event-wins): whichever logical
termination fires first owns the episode — a contact before goal entry
keeps contact semantics (the goal detector is suppressed in the contact
zombie phase), a goal entry before contact books the bonus terminal (the
contact detector is suppressed in the goal zombie phase). On the degenerate
SAME-step tie the contact wins (the catastrophic event is the conservative
call; the goal detector is additionally gated on `not _contact_now`).

Fail-loud contract: Main.py logs exactly one line at startup —
"GOAL_BONUS rule: ENGAGED (bonus=X dist=Y)" or "GOAL_BONUS rule: off" —
keyed on the effective gate; every goal-entry event is logged at INFO
(episode, step, distance, bonus); per episode the h5 gets
goal_terminal/goal_step/goal_store_dropped attrs (only when the rule is
engaged — the off path stays byte-identical).
"""

# argos/generate_argos.py --goal_radius argparse default (a string "2"
# substituted into $$goal_radius$$; collectiveRlTransport.cpp reads it into
# m_fThreshold via the template's `threshold` attribute, with min_threshold
# equal so the radius never decays below it). Per-cell overrides arrive via
# the GOAL_RADIUS key in agent_config.yml (run_baseline_experiments.py maps
# it to --goal_radius).
DEFAULT_GOAL_RADIUS_M = 2.0

# Entry margin (m) over the ARGoS success radius — see the module docstring
# (section-2.5 trap + E2E FIFO maturation window) for the derivation.
GOAL_ENTRY_MARGIN_M = 0.25


class GoalBonusRule:
    """Parsed GOAL_BONUS/GOAL_TERMINAL_DIST config + the per-step detector.

    Pure and stateless across steps (Main.py owns the per-episode state), so
    the gating and threshold arithmetic are unit-testable without ZMQ/ARGoS.
    """

    def __init__(self, config):
        # float() the bonus defensively: YAML may deliver int 10000.
        _bonus = config.get('GOAL_BONUS')
        self.bonus = 0.0 if _bonus is None else float(_bonus)

        # Effective success radius: explicit GOAL_RADIUS key wins; only None
        # degrades to the generate_argos default — a valid falsy 0.0 would
        # be an explicit operator value and is rejected by the check below,
        # never silently remapped.
        _goal_radius = config.get('GOAL_RADIUS')
        self.goal_radius = (
            DEFAULT_GOAL_RADIUS_M if _goal_radius is None
            else float(_goal_radius)
        )
        _dist = config.get('GOAL_TERMINAL_DIST')
        self.terminal_dist = (
            self.goal_radius + GOAL_ENTRY_MARGIN_M
            if _dist is None else float(_dist)
        )

        self.enabled = self.bonus != 0.0

        # Fail loud on misconfiguration instead of silently paying a bonus
        # the learner can never see (the section-2.5 learner-invisibility
        # trap): a threshold at or inside the success radius first fires on
        # the ARGoS terminal step itself, whose transition the legacy
        # `if not episode_done` store guard drops.
        if self.enabled and self.terminal_dist <= self.goal_radius:
            raise ValueError(
                "GOAL_BONUS rule engaged (bonus="
                f"{self.bonus}) with GOAL_TERMINAL_DIST="
                f"{self.terminal_dist} <= the env success radius "
                f"{self.goal_radius} — the goal-entry terminal would fire "
                "on the ARGoS success step, whose transition the legacy "
                "episode_done store guard drops: the bonus would be "
                "learner-invisible. Set GOAL_TERMINAL_DIST strictly "
                "greater than the success radius (default: radius + "
                f"{GOAL_ENTRY_MARGIN_M})."
            )

    def startup_line(self):
        """The exactly-one startup log line, keyed on the effective gate."""
        if self.enabled:
            return (
                "GOAL_BONUS rule: ENGAGED (bonus=%s dist=%s)"
                % (self.bonus, self.terminal_dist)
            )
        return "GOAL_BONUS rule: off"

    def detect(self, cyl_dist2goal):
        """Strict `<` on the post-step payload->goal distance (raw meters)."""
        return float(cyl_dist2goal) < self.terminal_dist

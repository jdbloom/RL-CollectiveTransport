"""Tests for the eval-time neighbor-ablation flag GSP_EVAL_ABLATE_NEIGHBORS.

Scientific contract: when GSP_EVAL_ABLATE_NEIGHBORS=True, the neighbor region of
every per-agent GSP input vector produced by make_gsp_states must be zeroed, while
the self-slot + all enrichment dims (everything before the neighbor region) and the
ring-buffer/temporal-stacking machinery are left byte-identical to the unablated run.
When the flag is False the output must be a strict no-op vs current behavior.

The neighbor-region boundary is computed INDEPENDENTLY in this test from the slot
sizes (self prox + K prev_gsp + enrichment), never read from the implementation's own
index counter — so a buggy implementation index cannot make a wrong test pass.
"""

import numpy as np
import pytest

from src.agent import Agent


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_config(**overrides):
    cfg = {
        "GAMMA": 0.99, "TAU": 0.005, "ALPHA": 0.001, "BETA": 0.001,
        "LR": 0.001, "EPSILON": 0.0, "EPS_MIN": 0.0, "EPS_DEC": 0.0,
        "BATCH_SIZE": 8, "MEM_SIZE": 100, "REPLACE_TARGET_COUNTER": 10,
        "NOISE": 0.0, "UPDATE_ACTOR_ITER": 1, "WARMUP": 0,
        "GSP_LEARNING_FREQUENCY": 100, "GSP_BATCH_SIZE": 8,
    }
    cfg.update(overrides)
    return cfg


def _make_agent(ablate, *, gsp_output_kind='delta_theta_1d', n_hop_neighbors=1,
                include_goal=False, include_cyl_rel=False, temporal_stack_k=1,
                n_agents=4):
    """Construct a GSP-N agent (gsp+neighbors+attention) with the ablation flag set."""
    cfg = _base_config(
        GSP_OUTPUT_KIND=gsp_output_kind,
        GSP_EVAL_ABLATE_NEIGHBORS=ablate,
        GSP_INPUT_INCLUDE_GOAL=include_goal,
        GSP_INPUT_INCLUDE_CYL_REL=include_cyl_rel,
        GSP_INPUT_TEMPORAL_STACK_K=temporal_stack_k,
    )
    return Agent(
        config=cfg,
        network='DDQN',
        n_agents=n_agents,
        n_obs=31,
        n_actions=2,
        options_per_action=3,
        id=0,
        min_max_action=0.1,
        meta_param_size=1,
        gsp=True,
        recurrent=False,
        attention=True,
        neighbors=True,
        gsp_input_size=6,   # overridden in __init__ when neighbors=True
        gsp_output_size=1,  # overridden by GSP_OUTPUT_KIND
        gsp_min_max_action=1.0,
        gsp_look_back=2,
        gsp_sequence_length=5,
        n_hop_neighbors=n_hop_neighbors,
    )


def _make_inputs(n_agents=4, K=1):
    """Distinct nonzero prox + prev_gsp so any accidental zeroing is detectable."""
    prox = [0.11 * (i + 1) for i in range(n_agents)]
    prev = np.zeros((n_agents, K), dtype=np.float32)
    for i in range(n_agents):
        for k in range(K):
            prev[i, k] = (i + 1) * 0.13 + k * 0.017 + 0.001  # never exactly 0
    return prox, prev


def _env_obs(n_agents=4):
    """Nonzero per-robot obs so goal/cyl enrichment dims are nonzero (detectable)."""
    obs = []
    for i in range(n_agents):
        v = np.arange(31, dtype=np.float32) * 0.01 + (i + 1) * 0.1 + 0.05
        obs.append(v)
    return obs


def _neighbor_region(agent, K, *, include_goal=False, include_cyl_rel=False):
    """Independently compute (start, end) of the neighbor region in the SINGLE-STEP
    per-agent vector, from slot sizes only — NOT from the implementation.

    Single-step layout (full_prox/payload/self_dynamics are all off in these tests):
        self_prox       : 1
        self_prev_gsp   : K
        goal enrichment : 2 if include_goal
        cyl_rel         : 2 if include_cyl_rel
        --- neighbor region starts here ---
        per neighbor    : (1 + K)   × n_neighbors
    """
    self_len = 1 + K
    self_len += 2 if include_goal else 0
    self_len += 2 if include_cyl_rel else 0
    n_neighbors = len(agent.neighbors_dict[0])  # ring topology: same count for all
    neighbor_len = n_neighbors * (1 + K)
    start = self_len
    end = self_len + neighbor_len
    return start, end


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNoOpWhenFlagUnset:
    def test_flag_defaults_false(self):
        agent = _make_agent(ablate=False)
        assert agent._gsp_eval_ablate_neighbors is False

    def test_flag_absent_from_config_is_false(self):
        """No GSP_EVAL_ABLATE_NEIGHBORS key at all → False (back-compat)."""
        cfg = _base_config()  # no ablation key
        agent = Agent(
            config=cfg, network='DDQN', n_agents=4, n_obs=31, n_actions=2,
            options_per_action=3, id=0, min_max_action=0.1, meta_param_size=1,
            gsp=True, recurrent=False, attention=True, neighbors=True,
            gsp_input_size=6, gsp_output_size=1, gsp_min_max_action=1.0,
            gsp_look_back=2, gsp_sequence_length=5, n_hop_neighbors=1,
        )
        assert agent._gsp_eval_ablate_neighbors is False

    @pytest.mark.parametrize("kind,K", [('delta_theta_1d', 1),
                                        ('cyl_kinematics_3d', 3),
                                        ('cyl_kinematics_goal_4d', 4)])
    def test_flag_false_bit_identical_to_baseline(self, kind, K):
        """(c) Flag False output must equal a freshly-built unablated agent's output."""
        prox, prev = _make_inputs(4, K=K)
        a_baseline = _make_agent(ablate=False, gsp_output_kind=kind)
        a_false = _make_agent(ablate=False, gsp_output_kind=kind)
        base_states = a_baseline.make_gsp_states(list(prox), prev.copy())
        false_states = a_false.make_gsp_states(list(prox), prev.copy())
        for i in range(4):
            np.testing.assert_array_equal(
                np.asarray(false_states[i]), np.asarray(base_states[i]),
                err_msg=f"kind={kind} agent={i}: flag-False is not a no-op",
            )


class TestNeighborRegionZeroed:
    @pytest.mark.parametrize("kind,K", [('delta_theta_1d', 1),
                                        ('cyl_kinematics_3d', 3),
                                        ('cyl_kinematics_goal_4d', 4)])
    def test_neighbor_region_all_zero(self, kind, K):
        """(a) With ablation ON, the neighbor region of every agent vector is zero."""
        prox, prev = _make_inputs(4, K=K)
        agent = _make_agent(ablate=True, gsp_output_kind=kind)
        start, end = _neighbor_region(agent, K)
        states = agent.make_gsp_states(list(prox), prev.copy())
        for i in range(4):
            region = np.asarray(states[i])[start:end]
            assert region.size > 0, "test bug: empty neighbor region"
            np.testing.assert_array_equal(
                region, np.zeros_like(region),
                err_msg=f"kind={kind} agent={i}: neighbor region not zeroed",
            )

    @pytest.mark.parametrize("kind,K", [('delta_theta_1d', 1),
                                        ('cyl_kinematics_3d', 3),
                                        ('cyl_kinematics_goal_4d', 4)])
    def test_self_and_enrichment_untouched(self, kind, K):
        """(b) Self-slot + enrichment region is byte-identical with flag on vs off."""
        prox, prev = _make_inputs(4, K=K)
        a_off = _make_agent(ablate=False, gsp_output_kind=kind)
        a_on = _make_agent(ablate=True, gsp_output_kind=kind)
        start, _ = _neighbor_region(a_on, K)
        off_states = a_off.make_gsp_states(list(prox), prev.copy())
        on_states = a_on.make_gsp_states(list(prox), prev.copy())
        for i in range(4):
            off_pre = np.asarray(off_states[i])[:start]
            on_pre = np.asarray(on_states[i])[:start]
            assert off_pre.size > 0, "test bug: empty self region"
            np.testing.assert_array_equal(
                on_pre, off_pre,
                err_msg=f"kind={kind} agent={i}: self/enrichment region changed by ablation",
            )

    def test_baseline_neighbor_region_is_nonzero(self):
        """Guard: without ablation the neighbor region IS populated (nonzero), so the
        zeroing test in test_neighbor_region_all_zero is actually proving something."""
        prox, prev = _make_inputs(4, K=1)
        agent = _make_agent(ablate=False)
        start, end = _neighbor_region(agent, 1)
        states = agent.make_gsp_states(list(prox), prev.copy())
        for i in range(4):
            region = np.asarray(states[i])[start:end]
            assert np.any(region != 0.0), (
                f"agent={i}: neighbor region is already zero without ablation — "
                "inputs are not exercising the neighbor slots"
            )


class TestEnrichmentSelfSlotPreserved:
    """With goal + cyl enrichment on the self-slot, ablation must zero ONLY the
    neighbor region and leave the (larger) enriched self-slot untouched."""

    def test_enriched_self_slot_untouched_and_neighbors_zeroed(self):
        prox, prev = _make_inputs(4, K=1)
        env = _env_obs(4)
        common = dict(include_goal=True, include_cyl_rel=True)
        a_off = _make_agent(ablate=False, **common)
        a_on = _make_agent(ablate=True, **common)
        start, end = _neighbor_region(a_on, 1, include_goal=True, include_cyl_rel=True)
        off_states = a_off.make_gsp_states(list(prox), prev.copy(), env_observations=env)
        on_states = a_on.make_gsp_states(list(prox), prev.copy(), env_observations=env)
        for i in range(4):
            # self+enrichment identical
            np.testing.assert_array_equal(
                np.asarray(on_states[i])[:start], np.asarray(off_states[i])[:start],
                err_msg=f"agent={i}: enriched self-slot altered by ablation",
            )
            # neighbor region zeroed
            np.testing.assert_array_equal(
                np.asarray(on_states[i])[start:end],
                np.zeros(end - start, dtype=np.asarray(on_states[i]).dtype),
                err_msg=f"agent={i}: neighbor region not zeroed under enrichment",
            )
            # enriched self-slot actually contains nonzero enrichment dims (guard)
            assert np.any(np.asarray(off_states[i])[:start] != 0.0)


class TestTemporalStackingAblation:
    """K_stack > 1: the neighbor region of EVERY stacked single-step block must be
    zero, because the zeros must propagate through the ring buffer into the stack."""

    def test_stacked_neighbor_regions_all_zero(self):
        K = 1  # GSP output dim
        kstack = 3
        prox, prev = _make_inputs(4, K=K)
        agent = _make_agent(ablate=True, temporal_stack_k=kstack)
        single_step = agent.gsp_network_input // kstack
        start, end = _neighbor_region(agent, K)
        # Step several times so the ring buffer fills with ablated single-step vectors.
        for _ in range(kstack + 2):
            states = agent.make_gsp_states(list(prox), prev.copy())
        for i in range(4):
            full = np.asarray(states[i])
            assert full.size == single_step * kstack
            for blk in range(kstack):
                base = blk * single_step
                region = full[base + start: base + end]
                np.testing.assert_array_equal(
                    region, np.zeros_like(region),
                    err_msg=f"agent={i} block={blk}: stacked neighbor region not zeroed",
                )

    def test_stacked_self_slots_untouched(self):
        K = 1
        kstack = 3
        prox, prev = _make_inputs(4, K=K)
        a_off = _make_agent(ablate=False, temporal_stack_k=kstack)
        a_on = _make_agent(ablate=True, temporal_stack_k=kstack)
        single_step = a_on.gsp_network_input // kstack
        start, _ = _neighbor_region(a_on, K)
        for _ in range(kstack + 2):
            off_states = a_off.make_gsp_states(list(prox), prev.copy())
            on_states = a_on.make_gsp_states(list(prox), prev.copy())
        for i in range(4):
            for blk in range(kstack):
                base = blk * single_step
                off_self = np.asarray(off_states[i])[base: base + start]
                on_self = np.asarray(on_states[i])[base: base + start]
                np.testing.assert_array_equal(
                    on_self, off_self,
                    err_msg=f"agent={i} block={blk}: stacked self-slot changed by ablation",
                )

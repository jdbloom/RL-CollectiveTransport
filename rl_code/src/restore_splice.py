"""Blindfold Fusion Probe — RESTORE-VIA-SPLICE (step 2, restore arms).

Pre-reg: docs/research/2026-07-22-blindfold-fusion-probe-spec.md (step 2,
"Restore via the splice path"). Council decision:
kb/wiki/studies/2026-07-22-council-gsp-direction-decision.md (stelaris repo).

The masking pilot (RL-CT#51, src/obs_mask.py) confirmed a real, restorable
deficit: with OBS_MASK_INDICES=[1] (goal-relative bearing zeroed from the
actor's native 31-dim obs) IC collapses ~0.99 -> ~0.15. This module re-injects
the TRUE (pre-mask) value of that channel through the SAME appended-feature
path the GSP prediction uses, so the probe discriminates "the task has no gap"
(Advocate) vs "the actor cannot consume appended side-information" (Skeptic) —
the two rival explanations of the five GSP nulls.

Apparatus reuse (validity-critical — the probe must test the splice GSP uses):

    * ``late_splice`` (default) rides the UNTOUCHED GSP fusion verbatim: the
      restored value is passed to ``Agent.make_agent_state`` as the
      ``heading_gsp`` argument, so it traverses the identical code — the H-14
      zero-out gate, the legacy scalar transform (``np.degrees(scalar / 10)``,
      src/agent.py make_agent_state scalar branch), the (inert-on-IC) feature
      standardizer and splice gain, and the exact concatenation slot
      ``[env_obs | gsp_slot | global_knowledge]``. Because the legacy scalar
      transform is a delta_theta-specific UNITS CONVERSION (not a
      normalization), ``splice_arg`` pre-inverts it (``radians(v) * 10``) so
      the value the actor consumes at the slot equals the TRUE channel value
      at its native obs scale (identical units to the masked native slot) to
      float32 round-trip precision. This is the ONLY delta vs a real GSP run:
      GSP feeds a head prediction in delta_theta units; the probe feeds the
      restored channel in its native units. Position, width (K=1), transform
      code, and network-input extension (+1, mirroring GSP-RL actor.py
      ``network_input_size += self.gsp_network_output``; here Main.py passes
      ``n_obs + 1``) are the same.
    * ``early_fuse`` instead injects the restored value INTO the masked slot
      of the native obs vector (un-masks it in place, on the actor's copy
      only) — no appended slot, no width change. This is the "gradient reaches
      the trunk from step 0" arm: the value re-enters through a native input
      column whose first-layer weights train from init, rather than through an
      appended slot the actor may have learned to price to zero.

Warm-noise (path-dependence probe, ``late_splice`` only): for the first M
EPISODES the appended slot carries uniform noise drawn from the channel's
plausible range (RESTORE_WARM_NOISE_RANGE, required); from episode M onward it
carries the true value. Tests whether an actor that learned to ignore the slot
can re-price it. For ``early_fuse`` the combination is rejected loudly (that
arm is defined as true-value-into-the-trunk from step 0). In test mode
warm-noise is forced inert with a loud log — an eval of a warm-noise-trained
checkpoint must consume the TRUE value, and eval configs are cloned from
training configs.

Fail-loud contract (assert-the-ENGAGED-path):

    * Default (RESTORE_SPLICE_SOURCE_INDEX absent/None) is a STRICT bit-exact
      no-op: every method returns its input unchanged / None, and setting any
      other RESTORE_* key while disabled raises (configured-but-not-engaged is
      a misconfig, never a silent skip).
    * When engaged, Main.py logs mode + index + warm-noise episodes at startup
      (describe()) and this module logs the observed value range of the
      restored channel over the first episodes (the engaged-path assert).
    * Restoring a channel NOT in OBS_MASK_INDICES makes the probe meaningless
      (the actor already sees it natively) — hard ERROR-level warning.
    * ``None`` is the only value that degrades to a default
      (``default if x is None else x``); falsy 0 / [] are never remapped.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

VALID_MODES = ("late_splice", "early_fuse")

# Engaged-path assert: accumulate + log the restored channel's observed value
# range over this many initial episodes.
RANGE_LOG_EPISODES = 3


class RestoreSplice:
    """Re-injects the TRUE value of a masked obs channel through the GSP splice.

    One instance is held by Main.py for the whole run. Main owns:
      * construction (config parse + run-shape facts),
      * the ``n_obs + extra_actor_inputs`` actor-width extension,
      * ``set_episode`` at every episode boundary,
      * ``true_values`` (post-delay, PRE-mask), ``apply_early_fuse``
        (post-mask), and ``slot_values`` / ``splice_arg`` at the two
        actor-state assembly sites (episode-init and step-loop next-state).

    Parameters (None applies the documented default; anything else validates):
      source_index: RESTORE_SPLICE_SOURCE_INDEX — obs index whose TRUE
          (pre-mask) value is restored. None (default) = feature OFF.
      mode: RESTORE_SPLICE_MODE — 'late_splice' (default) | 'early_fuse'.
      warm_noise_episodes: RESTORE_WARM_NOISE_EPISODES — int >= 0, default 0.
      warm_noise_range: RESTORE_WARM_NOISE_RANGE — [lo, hi] floats, lo < hi.
          REQUIRED when warm_noise_episodes > 0, forbidden otherwise.
      mask_indices: the engaged OBS_MASK_INDICES (ObsMask.indices).
      obs_dim: native per-robot obs width (31) for bounds-checking.
      gsp_enabled: config['GSP'] — must be False (the GSP prediction owns the
          appended slot; a run cannot splice both).
      share_prox_values: --share_prox_values — must be False (that branch
          bypasses make_agent_state entirely).
      train_mode: True for training runs; gates warm-noise (train-only).
      seed: base RNG seed (config SEED); the warm-noise stream is seeded with
          seed + 2026 so it is deterministic and independent of the
          pred-ablation stream (Main.py seeds that with the raw SEED).
    """

    def __init__(
        self,
        source_index: Optional[int],
        mode: Optional[str],
        warm_noise_episodes: Optional[int],
        warm_noise_range: Optional[Sequence[float]],
        mask_indices: Sequence[int],
        obs_dim: int,
        gsp_enabled: bool,
        share_prox_values: bool,
        train_mode: bool,
        seed: int,
    ) -> None:
        if not isinstance(obs_dim, int) or isinstance(obs_dim, bool) or obs_dim < 1:
            raise ValueError(f"obs_dim must be a positive int, got {obs_dim!r}")

        if source_index is None:
            # Feature OFF. Any other RESTORE_* key set alongside is a misconfig
            # — fail loud, never silently ignore a configured lever.
            leftover = {
                "RESTORE_SPLICE_MODE": mode,
                "RESTORE_WARM_NOISE_EPISODES": warm_noise_episodes,
                "RESTORE_WARM_NOISE_RANGE": warm_noise_range,
            }
            set_keys = [k for k, v in leftover.items() if v is not None]
            if set_keys:
                raise ValueError(
                    f"RESTORE_SPLICE_SOURCE_INDEX is None (feature off) but {set_keys} "
                    "are set — configured-but-not-engaged is a misconfig. Set the "
                    "source index or remove the other RESTORE_* keys."
                )
            self._enabled = False
            self._index: Optional[int] = None
            self._mode = "late_splice"
            self._warm_cfg = 0
            self._warm_eff = 0
            self._range: Optional[tuple] = None
            self._rng: Optional[np.random.Generator] = None
            self._episode: Optional[int] = None
            self._range_min = np.inf
            self._range_max = -np.inf
            self._range_n = 0
            return

        if isinstance(source_index, bool) or not isinstance(source_index, (int, np.integer)):
            raise TypeError(
                f"RESTORE_SPLICE_SOURCE_INDEX must be an int or None, got "
                f"{source_index!r} ({type(source_index).__name__})"
            )
        source_index = int(source_index)
        if source_index < 0 or source_index >= obs_dim:
            raise ValueError(
                f"RESTORE_SPLICE_SOURCE_INDEX {source_index} out of range for "
                f"obs_dim={obs_dim} (valid 0..{obs_dim - 1})"
            )

        # None -> documented default; never `or`-remap a valid value.
        mode = "late_splice" if mode is None else str(mode)
        if mode not in VALID_MODES:
            raise ValueError(
                f"RESTORE_SPLICE_MODE must be one of {VALID_MODES}, got {mode!r}"
            )

        warm = 0 if warm_noise_episodes is None else warm_noise_episodes
        if isinstance(warm, bool) or not isinstance(warm, (int, np.integer)):
            raise TypeError(
                f"RESTORE_WARM_NOISE_EPISODES must be an int, got {warm!r} "
                f"({type(warm).__name__})"
            )
        warm = int(warm)
        if warm < 0:
            raise ValueError(f"RESTORE_WARM_NOISE_EPISODES must be >= 0, got {warm}")
        if warm > 0 and mode == "early_fuse":
            raise ValueError(
                "RESTORE_WARM_NOISE_EPISODES > 0 is not defined for "
                "RESTORE_SPLICE_MODE='early_fuse' — the early-fuse arm is "
                "true-value-into-the-trunk from step 0 by definition. Use "
                "'late_splice' for the warm-noise (re-pricing) arm."
            )
        if warm > 0:
            if warm_noise_range is None:
                raise ValueError(
                    "RESTORE_WARM_NOISE_EPISODES > 0 requires RESTORE_WARM_NOISE_RANGE "
                    "= [lo, hi] (the channel's plausible range; e.g. [-180.0, 180.0] "
                    "for the goal-bearing channel, index 1). No silent default — the "
                    "noise must be range-matched to the channel or the warm phase is "
                    "not a matched-magnitude control."
                )
            rng_vals = list(warm_noise_range)
            if len(rng_vals) != 2:
                raise ValueError(
                    f"RESTORE_WARM_NOISE_RANGE must be [lo, hi], got {warm_noise_range!r}"
                )
            lo, hi = float(rng_vals[0]), float(rng_vals[1])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                raise ValueError(
                    f"RESTORE_WARM_NOISE_RANGE requires finite lo < hi, got [{lo}, {hi}]"
                )
            self._range = (lo, hi)
        else:
            if warm_noise_range is not None:
                raise ValueError(
                    "RESTORE_WARM_NOISE_RANGE is set but RESTORE_WARM_NOISE_EPISODES "
                    "is 0/absent — dangling key, fail loud (remove it or set the "
                    "episode count)."
                )
            self._range = None

        if gsp_enabled:
            raise ValueError(
                "RESTORE_SPLICE_SOURCE_INDEX is set but GSP is enabled — the GSP "
                "prediction owns the appended slot (make_agent_state heading_gsp); "
                "a run cannot splice both. The restore probe runs on the IC stack "
                "(GSP: false)."
            )
        if share_prox_values:
            raise ValueError(
                "RESTORE_SPLICE_SOURCE_INDEX is set with --share_prox_values — that "
                "branch bypasses make_agent_state (plain concat), so the restore "
                "cannot ride the GSP splice apparatus there. Unsupported."
            )

        if source_index not in tuple(int(i) for i in mask_indices):
            # Hard warning, not fatal: restoring an unmasked channel makes the
            # PROBE meaningless (the actor already sees the channel natively),
            # but it is a legitimate redundancy control if run deliberately.
            log.error(
                "RESTORE_SPLICE: source index %d is NOT in OBS_MASK_INDICES=%s — "
                "the actor already sees this channel natively, so the restore "
                "probe is MEANINGLESS as a fusion test. Proceeding only because "
                "a deliberate redundancy control is conceivable; if this is the "
                "blindfold probe, fix the config now.",
                source_index, list(mask_indices),
            )

        self._enabled = True
        self._index = source_index
        self._mode = mode
        self._warm_cfg = warm
        self._warm_eff = warm
        if warm > 0 and not train_mode:
            log.warning(
                "RESTORE_SPLICE: RESTORE_WARM_NOISE_EPISODES=%d is TRAIN-ONLY and "
                "this is a TEST run (eval configs are cloned from training "
                "configs) — warm-noise forced INERT; the splice carries the TRUE "
                "value from episode 0.",
                warm,
            )
            self._warm_eff = 0
        # Deterministic, independent noise stream (seed + 2026; Main.py's
        # pred-ablation stream uses the raw SEED).
        self._rng = np.random.default_rng(int(seed) + 2026)
        self._episode = None
        self._range_min = np.inf
        self._range_max = -np.inf
        self._range_n = 0

    # ── engaged-path introspection ───────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def source_index(self) -> Optional[int]:
        return self._index

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def warm_noise_episodes(self) -> int:
        """The EFFECTIVE warm-noise episode count (0 when forced inert)."""
        return self._warm_eff

    @property
    def late_splice_engaged(self) -> bool:
        return self._enabled and self._mode == "late_splice"

    @property
    def early_fuse_engaged(self) -> bool:
        return self._enabled and self._mode == "early_fuse"

    @property
    def extra_actor_inputs(self) -> int:
        """Actor-input width extension: +1 appended slot for late_splice
        (mirrors GSP-RL actor.py ``network_input_size += gsp_network_output``,
        K=1), 0 for early_fuse (in-place, no width change) and when off."""
        return 1 if self.late_splice_engaged else 0

    def describe(self) -> str:
        """One-line startup summary for Main.py's engaged-path log."""
        if not self._enabled:
            return "off (no-op)"
        parts = [
            f"ENGAGED mode={self._mode}",
            f"source_index={self._index}",
            f"warm_noise_episodes={self._warm_eff}"
            + (f" (configured {self._warm_cfg}, forced inert in test mode)"
               if self._warm_eff != self._warm_cfg else ""),
        ]
        if self._range is not None:
            parts.append(f"warm_noise_range=[{self._range[0]}, {self._range[1]}]")
        parts.append(f"extra_actor_inputs={self.extra_actor_inputs}")
        return " ".join(parts)

    # ── per-episode lifecycle ────────────────────────────────────────────────

    def set_episode(self, episode: int) -> None:
        """Advance to ``episode``. Main.py calls this at every episode start.

        Flushes the engaged-path range log for each of the first
        RANGE_LOG_EPISODES episodes and logs the warm-noise -> true-value
        phase switch when it happens.
        """
        if not self._enabled:
            return
        prev = self._episode
        self._episode = int(episode)
        if prev is None:
            return
        if prev < RANGE_LOG_EPISODES and self._range_n > 0:
            log.info(
                "RESTORE_SPLICE engaged-path check: episode %d restored channel "
                "[%d] observed range [%.4f, %.4f] over %d samples (mode=%s%s)",
                prev, self._index, self._range_min, self._range_max,
                self._range_n, self._mode,
                " — slot carried WARM NOISE this episode; range above is the "
                "TRUE channel" if prev < self._warm_eff else "",
            )
            self._range_min = np.inf
            self._range_max = -np.inf
            self._range_n = 0
        if self._warm_eff > 0 and prev < self._warm_eff <= self._episode:
            log.info(
                "RESTORE_SPLICE: warm-noise phase ended at episode %d — the "
                "appended slot now carries the TRUE channel [%d] value.",
                self._episode, self._index,
            )

    # ── value plumbing ───────────────────────────────────────────────────────

    def true_values(self, actor_env_obs: Sequence[Any]) -> Optional[np.ndarray]:
        """Per-robot TRUE value of the restored channel, read from the
        POST-DELAY, PRE-MASK actor observation (the exact per-robot observation
        the actor would have seen unmasked). None when the feature is off."""
        if not self._enabled:
            return None
        if self._episode is None:
            raise RuntimeError(
                "RESTORE_SPLICE: true_values called before set_episode — Main.py "
                "must advance the episode at every episode boundary first."
            )
        vals = np.array(
            [float(obs[self._index]) for obs in actor_env_obs], dtype=np.float64
        )
        if self._episode < RANGE_LOG_EPISODES:
            self._range_min = min(self._range_min, float(vals.min()))
            self._range_max = max(self._range_max, float(vals.max()))
            self._range_n += vals.size
        return vals

    def apply_early_fuse(
        self, masked_obs: Sequence[Any], true_vals: Optional[np.ndarray]
    ) -> Any:
        """early_fuse arm: write the TRUE value back into the masked slot of
        each per-robot actor observation (un-mask in place, on fresh copies —
        the input list and the live env_observations are never mutated). For
        late_splice / off this is a strict pass-through (same object)."""
        if not self.early_fuse_engaged:
            return masked_obs
        if true_vals is None or len(true_vals) != len(masked_obs):
            raise RuntimeError(
                "RESTORE_SPLICE early_fuse: true_vals missing or robot-count "
                "mismatch — capture them via true_values(pre-mask obs) first."
            )
        fused = []
        for i, obs in enumerate(masked_obs):
            arr = np.array(obs, dtype=np.float32, copy=True)
            arr[self._index] = true_vals[i]
            fused.append(arr)
        return fused

    def slot_values(self, true_vals: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Per-robot value the APPENDED slot should carry this step
        (late_splice only): uniform warm noise while episode < M, else the
        true value. None unless late_splice is engaged."""
        if not self.late_splice_engaged:
            return None
        if true_vals is None:
            raise RuntimeError(
                "RESTORE_SPLICE late_splice: true_vals is None — capture them via "
                "true_values(pre-mask obs) before assembling agent states."
            )
        if self._episode is None:
            raise RuntimeError(
                "RESTORE_SPLICE: slot_values called before set_episode — Main.py "
                "must advance the episode at every episode boundary first."
            )
        if self._episode < self._warm_eff:
            lo, hi = self._range
            return self._rng.uniform(lo, hi, size=len(true_vals))
        return np.asarray(true_vals, dtype=np.float64)

    @staticmethod
    def splice_arg(slot_value: float) -> float:
        """Pre-invert the legacy scalar units-conversion so the appended slot
        carries exactly ``slot_value``.

        make_agent_state's scalar branch (src/agent.py) computes
        ``np.degrees(heading_gsp / 10)`` — the historical delta_theta
        units-mapping, NOT a normalization. Passing ``radians(v) * 10`` makes
        that untouched transform the identity on v (float32 round-trip, <=1
        ulp), so the actor consumes the restored channel at its native obs
        scale through the byte-identical GSP code path.
        """
        return float(np.radians(slot_value) * 10.0)

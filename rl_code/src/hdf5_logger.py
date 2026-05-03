"""HDF5-based episode data logger — drop-in replacement for pkl data_logger.

Main.py creates one HDF5Logger per experiment. Each episode is written
as a group in the experiment's HDF5 file. No pkl files created.

Usage:
    logger = HDF5Logger("/path/to/experiment.h5")

    # Per timestep (same API as data_logger.writerow)
    logger.writerow(rewards, epsilon, termination, loss, ...)

    # End of episode
    logger.write_episode(episode_num)

    # Notify ingestion worker
    logger.notify(experiment_name)
"""

import os
import time
from typing import Optional

import h5py
import numpy as np

# Notification handled by ingestion worker (optional, external)

_SWMR_RETRY_COUNT = 3
_SWMR_RETRY_DELAY = 0.5  # seconds


def _open_h5_writer(path: str, mode: str = "a"):
    """Open an HDF5 file for writing with SWMR-compatible settings.

    Uses libver='latest' (required for SWMR) and retries up to
    _SWMR_RETRY_COUNT times on BlockingIOError (errno 35) which occurs on
    macOS APFS when an external reader briefly holds the file lock.

    Returns an open h5py.File handle; caller is responsible for closing it.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_SWMR_RETRY_COUNT):
        try:
            return h5py.File(path, mode, libver="latest")
        except BlockingIOError as exc:
            last_exc = exc
            if attempt < _SWMR_RETRY_COUNT - 1:
                time.sleep(_SWMR_RETRY_DELAY)
    raise last_exc  # type: ignore[misc]


class HDF5Logger:
    """Accumulates timestep data and writes to HDF5 at episode boundaries."""

    def __init__(
        self,
        hdf5_path: str,
        stelaris_sha: Optional[str] = None,
        rl_ct_sha: Optional[str] = None,
        gsp_rl_sha: Optional[str] = None,
        stelaris_branch: Optional[str] = None,
        rl_ct_branch: Optional[str] = None,
        gsp_rl_branch: Optional[str] = None,
    ):
        """
        Args:
            hdf5_path: File to write to.
            stelaris_sha: Outer Stelaris repo commit sha at cell launch.
                Written as the ``stelaris_sha`` root attr. Enables post-hoc
                partitioning by code version. None is allowed (legacy paths).
            rl_ct_sha: RL-CollectiveTransport submodule commit sha at launch.
                Written as ``rl_ct_sha`` root attr.
            gsp_rl_sha: GSP-RL submodule commit sha at launch. Written as
                ``gsp_rl_sha`` root attr. Third repo in the three-sha provenance
                trio — see BLOCKED B-001 in repo root.
            stelaris_branch: Branch name for outer repo — written as
                ``stelaris_branch`` root attr for human readability.
            rl_ct_branch: Branch name for submodule — written as
                ``rl_ct_branch`` root attr.
            gsp_rl_branch: Branch name for GSP-RL — written as
                ``gsp_rl_branch`` root attr.
        """
        self.hdf5_path = hdf5_path
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        # Write provenance attrs once, at file creation. Subsequent episodes
        # append without touching root attrs.
        # libver='latest' is required for SWMR; swmr_mode=True is set here so
        # external probes can open with swmr=True immediately after __init__.
        with _open_h5_writer(self.hdf5_path) as h5f:
            if stelaris_sha:
                h5f.attrs["stelaris_sha"] = str(stelaris_sha)
            if rl_ct_sha:
                h5f.attrs["rl_ct_sha"] = str(rl_ct_sha)
            if gsp_rl_sha:
                h5f.attrs["gsp_rl_sha"] = str(gsp_rl_sha)
            if stelaris_branch:
                h5f.attrs["stelaris_branch"] = str(stelaris_branch)
            if rl_ct_branch:
                h5f.attrs["rl_ct_branch"] = str(rl_ct_branch)
            if gsp_rl_branch:
                h5f.attrs["gsp_rl_branch"] = str(gsp_rl_branch)
            h5f.swmr_mode = True
        self._reset()

    def _reset(self):
        """Clear buffers for a new episode."""
        self.reward = []
        self.epsilon = []
        self.termination = []
        self.loss = []
        self.force_magnitude = []
        self.force_angle = []
        self.average_force_vector = []
        self.cyl_x_pos = []
        self.cyl_y_pos = []
        self.cyl_angle = []
        self.gate_stats = []
        self.obstacle_stats = []
        self.gsp_reward = []
        self.gsp_heading = []
        self.run_time = []
        self.robots_x_pos = []
        self.robots_y_pos = []
        self.robot_angle = []
        self.robot_failures = []
        self.com_X_pos = []
        self.com_Y_pos = []
        # GSP information-collapse diagnostics: predicted delta-theta target and squared error
        # per step, plus GSP-specific training loss per learning step.
        self.gsp_target = []
        self.gsp_squared_error = []
        self.gsp_loss = []
        # H-14 / first-principles diagnostic: the GSP head's actual input vector per
        # timestep per robot. Shape: (timesteps, num_robots, gsp_input_size).
        # When provided, enables offline supervised diagnostic training of a fresh
        # head on the same (input, label) pairs the live training saw — answering
        # "can the head learn at all when RL coupling and exploration are removed?"
        self.gsp_obs = []
        # Per-episode diagnostic scalars (FAU / weight norms / effective rank / Q-gap
        # / pred diversity). Namespaced keys prefixed with diag_*. Populated by
        # record_episode_diagnostics() once per diagnostic episode; written as HDF5
        # attrs on the episode group. See
        # docs/specs/2026-04-17-diagnostics-instrumentation.md for the full key list.
        self.episode_diagnostics: dict = {}
        # Fixed eval-batch states used for diagnostic computations. Stored as a
        # dataset on whichever episode first records them so reanalysis can
        # reconstruct exactly what was measured.
        self.eval_batch_states = None
        # E2E learning diagnostics: per-learn-step metrics from learn_DDQN_e2e.
        self.e2e_ddqn_loss = []
        self.e2e_gsp_mse_loss = []
        self.e2e_total_loss = []
        self.e2e_gsp_grad_norm = []
        self.e2e_gsp_grad_norm_pre_clip = []
        self.e2e_ddqn_grad_norm = []
        self.e2e_gsp_input_grad = []
        self.e2e_gsp_pred_mean = []
        self.e2e_gsp_pred_std = []
        self.e2e_gsp_label_mean = []
        self.e2e_gsp_label_std = []
        # Sample-quality diagnostics (schema v4+) — (label, input) pairs captured at
        # each store_gsp_transition call site. Needed because the replay buffer lives
        # in the external gsp_rl library and we have no hook to query its composition.
        # Aggregated into attrs on write_episode; not stored as full datasets.
        self.stored_gsp_labels: list = []
        self.stored_gsp_inputs: list = []
        # Phase 4 loss-step correlation diagnostic. One float per GSP learn step:
        # the Pearson correlation between the fresh forward-pass predictions used
        # in the MSE loss and the replay-buffer labels for that same batch.
        # Populated by record_gsp_loss_step_corr(); aggregated into
        # gsp_loss_step_corr_{mean,std,min,max} attrs on write_episode().
        # Empty list → no attrs written (non-GSP runs unaffected).
        self.gsp_loss_step_corr_samples: list = []
        # JEPA latent-space diagnostic scalars. Populated when GSP_JEPA_ENABLED=True.
        # jepa_pred_mse:    per-learn-step MSE between predictor output and target latent.
        # jepa_latent_var:  mean per-dim variance of the online encoder output z_t.
        # jepa_latent_rank: approximate effective rank of z_t (SVD-based threshold count).
        # Empty lists → no attrs written (non-JEPA runs unaffected).
        self.jepa_pred_mse: list = []
        self.jepa_latent_var: list = []
        self.jepa_latent_rank: list = []

    def writerow(
        self, rewards, epsilons, terminations, losses,
        force_magnitudes, force_angles, average_force_vectors,
        cyl_x_poses, cyl_y_poses, cyl_angles,
        gate_stats, obstacle_stats,
        gsp_rewards, gsp_headings,
        run_times, robots_x_poses, robots_y_poses, robot_angles,
        robot_failure, com_X_poses=0, com_Y_poses=0,
        gsp_target=None, gsp_squared_error=None, gsp_obs=None,
    ):
        """Accumulate one timestep of data. Same signature as data_logger.writerow."""
        self.reward.append(rewards)
        self.epsilon.append(epsilons)
        self.termination.append(terminations)
        self.loss.append(losses)
        self.force_magnitude.append(force_magnitudes)
        self.force_angle.append(force_angles)
        self.average_force_vector.append(average_force_vectors)
        self.cyl_x_pos.append(cyl_x_poses)
        self.cyl_y_pos.append(cyl_y_poses)
        self.cyl_angle.append(cyl_angles)
        self.gate_stats.append(gate_stats)
        self.obstacle_stats.append(obstacle_stats)
        self.gsp_reward.append(gsp_rewards)
        if isinstance(gsp_headings, np.ndarray):
            gsp_headings = gsp_headings.tolist()
        self.gsp_heading.append(gsp_headings)
        self.run_time.append(run_times)
        self.robots_x_pos.append(robots_x_poses)
        self.robots_y_pos.append(robots_y_poses)
        self.robot_angle.append(robot_angles)
        self.robot_failures.append(robot_failure)
        self.com_X_pos.append(com_X_poses)
        self.com_Y_pos.append(com_Y_poses)
        if gsp_target is not None:
            self.gsp_target.append(gsp_target)
        if gsp_squared_error is not None:
            self.gsp_squared_error.append(gsp_squared_error)
        if gsp_obs is not None:
            # gsp_obs is a per-robot list of input vectors; convert array elements to lists for
            # consistent serialization (mirrors gsp_heading handling above).
            if isinstance(gsp_obs, np.ndarray):
                gsp_obs = gsp_obs.tolist()
            self.gsp_obs.append(gsp_obs)

    def record_e2e_diagnostics(self, diag: dict) -> None:
        """Record one e2e learning step's diagnostics."""
        self.e2e_ddqn_loss.append(float(diag.get('ddqn_loss', 0)))
        self.e2e_gsp_mse_loss.append(float(diag.get('gsp_mse_loss', 0)))
        self.e2e_total_loss.append(float(diag.get('total_loss', 0)))
        self.e2e_gsp_grad_norm.append(float(diag.get('gsp_grad_norm', 0)))
        self.e2e_gsp_grad_norm_pre_clip.append(float(diag.get('gsp_grad_norm_pre_clip', 0)))
        self.e2e_ddqn_grad_norm.append(float(diag.get('ddqn_grad_norm', 0)))
        self.e2e_gsp_input_grad.append(float(diag.get('gsp_input_grad') or 0))
        self.e2e_gsp_pred_mean.append(float(diag.get('gsp_pred_mean', 0)))
        self.e2e_gsp_pred_std.append(float(diag.get('gsp_pred_std', 0)))
        self.e2e_gsp_label_mean.append(float(diag.get('gsp_label_mean', 0)))
        self.e2e_gsp_label_std.append(float(diag.get('gsp_label_std', 0)))

    def record_gsp_loss(self, loss_value: float) -> None:
        """Record one GSP prediction network training loss sample.

        Called per GSP learning step (cadence differs from per-timestep writerow).
        """
        self.gsp_loss.append(float(loss_value))

    def record_gsp_loss_step_corr(self, corr_value: float) -> None:
        """Record one per-batch Pearson correlation from the GSP MSE loss step.

        ``corr_value`` is the Pearson r between the fresh forward-pass predictions
        used to compute the MSE loss and the replay-buffer labels for that same
        batch.  Called once per GSP learn step (same cadence as record_gsp_loss).
        NaN values are silently dropped — they represent batches where one of the
        inputs had zero variance (undefined correlation); aggregation over finite
        samples is still meaningful.

        This is intentionally separate from the episode-level gsp_pred_target_corr
        attr (which measures the actor-input-path predictions over a full episode
        — a different code path with a 1-timestep lag).  Comparing the two metrics
        reveals whether the head IS learning but the actor-input measurement is
        broken, or whether the head genuinely fails to learn.
        """
        if not np.isnan(corr_value):
            self.gsp_loss_step_corr_samples.append(float(corr_value))

    def record_jepa_pred_mse(self, v: float) -> None:
        """Record one per-learn-step JEPA prediction MSE (z_pred vs z_target)."""
        self.jepa_pred_mse.append(float(v))

    def record_jepa_latent_var(self, v: float) -> None:
        """Record one per-learn-step mean per-dim variance of the online encoder z_t."""
        self.jepa_latent_var.append(float(v))

    def record_jepa_latent_rank(self, v: float) -> None:
        """Record one per-learn-step approximate effective rank of the online encoder z_t."""
        self.jepa_latent_rank.append(float(v))

    def record_stored_transition(self, label, input_vec) -> None:
        """Record one (label, input) pair at the moment it's stored in the GSP
        replay buffer. Per Phase 1 sample-quality spec — gives us label/input
        distribution summaries without having to reach into the external gsp_rl
        replay buffer.

        ``label`` is typically a scalar (delta_theta, future_prox, time_to_goal);
        vector labels (e.g. cyl_kinematics_*) are reduced to their mean as a
        proxy — the full vector-label distribution is out of scope for v4.
        ``input_vec`` is the per-robot GSP head input that produced this label.
        """
        try:
            self.stored_gsp_labels.append(float(label))
        except (TypeError, ValueError):
            arr = np.asarray(label, dtype=np.float32).ravel()
            if arr.size > 0:
                self.stored_gsp_labels.append(float(np.mean(arr)))
        try:
            self.stored_gsp_inputs.append(
                np.asarray(input_vec, dtype=np.float32).ravel()
            )
        except (TypeError, ValueError):
            pass

    def record_episode_diagnostics(self, diag: dict) -> None:
        """Record per-episode diagnostic scalars (FAU, weight norms, effective rank,
        Q-gap, pred diversity, etc.).

        Merges into the current episode's diagnostic dict; later calls with the
        same key overwrite earlier values. All keys must be prefixed with ``diag_``
        by convention — they will be persisted as HDF5 attrs on the episode group
        when ``write_episode`` is called. ``_reset`` clears the dict between
        episodes so diagnostics never leak across episodes.

        See docs/specs/2026-04-17-diagnostics-instrumentation.md for the canonical
        key list.
        """
        for k, v in diag.items():
            # Coerce to Python float for h5 attr compatibility; NaNs are allowed.
            self.episode_diagnostics[k] = float(v)

    def record_eval_batch_states(self, states) -> None:
        """Record the fixed eval-batch states used for diagnostic computations.

        Typically called once, on the episode where the eval batch is first frozen
        (e.g. episode 50 per the diagnostics spec). The states are persisted as
        a float32 dataset ``diag_eval_batch_states`` on that episode's group so
        later reanalysis can reconstruct exactly what was measured.
        """
        self.eval_batch_states = np.asarray(states, dtype=np.float32)

    def write_episode(self, episode_num: int) -> dict:
        """Write accumulated data to HDF5 and return summary dict.

        Call this instead of data_logger.write_to_file().
        Optionally notifies the ingestion worker.
        """
        group_name = f"episode_{episode_num:04d}"

        with _open_h5_writer(self.hdf5_path) as h5f:
            h5f.swmr_mode = True
            if group_name in h5f:
                del h5f[group_name]
            grp = h5f.create_group(group_name)

            # Store 2D arrays (timesteps × robots)
            twod_specs = [
                ("reward", self.reward),
                ("gsp_reward", self.gsp_reward),
                ("force_magnitude", self.force_magnitude),
                ("force_angle", self.force_angle),
                ("robot_x_pos", self.robots_x_pos),
                ("robot_y_pos", self.robots_y_pos),
                ("robot_angle", self.robot_angle),
                ("robot_failure", self.robot_failures),
                ("gsp_heading", self.gsp_heading),
            ]
            if self.gsp_target:
                twod_specs.append(("gsp_target", self.gsp_target))
            if self.gsp_squared_error:
                twod_specs.append(("gsp_squared_error", self.gsp_squared_error))
            for key, data in twod_specs:
                arr = np.array(data, dtype=np.float32)
                # Multi-dim GSP output: gsp_heading may be (T, R, K) when K>1.
                # Per plan §4 (out-of-scope for this PR): store only the FIRST dim to
                # keep the h5 dataset shape as (T, R) for backward compatibility.
                # The last-dim extraction for the Pearson metric happens above in
                # write_episode before this block (via _heading_last_dim).
                if key == "gsp_heading" and arr.ndim == 3:
                    arr = arr[:, :, 0]  # keep first dim only: (T, R)
                if arr.size > 0:
                    grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

            # Store 3D array (timesteps × robots × gsp_input_size) for the head's
            # actual input vector. Only present when gsp_obs was passed to writerow.
            if self.gsp_obs:
                arr = np.array(self.gsp_obs, dtype=np.float32)
                if arr.size > 0:
                    grp.create_dataset("gsp_obs", data=arr, compression="gzip", compression_opts=4)

            # Store 1D arrays (timesteps)
            for key, data in [
                ("epsilon", self.epsilon),
                ("loss", self.loss),
                ("cyl_x_pos", self.cyl_x_pos),
                ("cyl_y_pos", self.cyl_y_pos),
                ("cyl_angle", self.cyl_angle),
                ("run_time", self.run_time),
                ("comX", self.com_X_pos),
                ("comY", self.com_Y_pos),
            ]:
                arr = np.array(data, dtype=np.float32)
                if arr.size > 0:
                    grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

            # GSP-specific training loss — 1D but recorded at a different cadence than writerow,
            # so it lives outside the timestep-indexed 1D block above.
            if self.gsp_loss:
                gsp_loss_arr = np.array(self.gsp_loss, dtype=np.float32)
                grp.create_dataset("gsp_loss", data=gsp_loss_arr,
                                   compression="gzip", compression_opts=4)
                # Summary attrs for cheap cross-episode comparison. Finite-mask so a
                # single numerical spike doesn't poison the aggregate.
                finite_loss = gsp_loss_arr[np.isfinite(gsp_loss_arr)]
                if finite_loss.size > 0:
                    grp.attrs["gsp_loss_mean"] = float(np.mean(finite_loss))
                    grp.attrs["gsp_loss_std"] = float(np.std(finite_loss))
                    grp.attrs["gsp_loss_count"] = int(finite_loss.size)

            # E2E per-learn-step diagnostics (schema v3+).
            e2e_fields = [
                ('e2e_ddqn_loss', self.e2e_ddqn_loss),
                ('e2e_gsp_mse_loss', self.e2e_gsp_mse_loss),
                ('e2e_total_loss', self.e2e_total_loss),
                ('e2e_gsp_grad_norm', self.e2e_gsp_grad_norm),
                ('e2e_gsp_grad_norm_pre_clip', self.e2e_gsp_grad_norm_pre_clip),
                ('e2e_ddqn_grad_norm', self.e2e_ddqn_grad_norm),
                ('e2e_gsp_input_grad', self.e2e_gsp_input_grad),
                ('e2e_gsp_pred_mean', self.e2e_gsp_pred_mean),
                ('e2e_gsp_pred_std', self.e2e_gsp_pred_std),
                ('e2e_gsp_label_mean', self.e2e_gsp_label_mean),
                ('e2e_gsp_label_std', self.e2e_gsp_label_std),
            ]
            for key, data in e2e_fields:
                if data:
                    arr = np.array(data, dtype=np.float32)
                    grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
            if self.e2e_gsp_grad_norm:
                grp.attrs["e2e_gsp_grad_norm_mean"] = float(np.mean(self.e2e_gsp_grad_norm))
                grp.attrs["e2e_gsp_pred_std_mean"] = float(np.mean(self.e2e_gsp_pred_std))

            # Termination as bool
            term_arr = np.array(self.termination, dtype=bool)
            if term_arr.size > 0:
                grp.create_dataset("termination", data=term_arr)

            # Compute and store summary attributes
            rewards = np.array(self.reward, dtype=np.float32)
            gsp_rewards = np.array(self.gsp_reward, dtype=np.float32)
            timesteps = len(self.reward)
            success = bool(np.any(term_arr)) if term_arr.size > 0 else False

            if rewards.ndim == 2:
                reward_per_robot = np.sum(rewards, axis=0).tolist()
            elif rewards.size > 0:
                reward_per_robot = [float(np.sum(rewards))]
            else:
                reward_per_robot = []

            if gsp_rewards.ndim == 2 and gsp_rewards.size > 0:
                gsp_per_robot = np.sum(gsp_rewards, axis=0).tolist()
            else:
                gsp_per_robot = []

            # Sample-quality attrs: distribution of stored (label, input) pairs this
            # episode. Captured at each store_gsp_transition call site in Main.py.
            # Necessary because the gsp_rl replay buffer is external; we cannot query
            # its composition directly. Phase 1 spec: docs/specs/2026-04-21-phase1-verification.md.
            if self.stored_gsp_labels:
                lbl = np.asarray(self.stored_gsp_labels, dtype=np.float64)
                finite = lbl[np.isfinite(lbl)]
                grp.attrs["gsp_label_count"] = int(finite.size)
                if finite.size > 0:
                    grp.attrs["gsp_label_mean"] = float(np.mean(finite))
                    grp.attrs["gsp_label_std"] = float(np.std(finite))
                    grp.attrs["gsp_label_min"] = float(np.min(finite))
                    grp.attrs["gsp_label_max"] = float(np.max(finite))
            if self.stored_gsp_inputs:
                # Stack only if all captured inputs have the same dim (they should,
                # within a single episode for a single variant). Heterogeneous shapes
                # would indicate a bug worth surfacing, so log and skip rather than coerce.
                shapes = {tuple(x.shape) for x in self.stored_gsp_inputs}
                if len(shapes) == 1:
                    X = np.stack(self.stored_gsp_inputs)  # (n_stored, input_dim)
                    per_dim_std = np.nanstd(X, axis=0)
                    finite_std = per_dim_std[np.isfinite(per_dim_std)]
                    if finite_std.size > 0:
                        grp.attrs["gsp_train_input_std"] = float(np.mean(finite_std))
                        grp.attrs["gsp_train_input_count"] = int(X.shape[0])

            # Logging schema version. Bump on every change that adds/removes a field
            # or changes semantics of an existing field. Analyzer reads this attr
            # to know which metrics are computable for this run vs gaps.
            # v1 — implicit (pre-2026-04-15); treated as default in analyzer
            # v2 — 2026-04-15: explicit log_schema_version attr added
            # v3 — 2026-04-15: e2e diagnostics (11 per-learn-step metrics) added
            # v4 — 2026-04-21: sample-quality attrs (gsp_loss_mean/std/count,
            #                  gsp_label_mean/std/min/max/count, gsp_train_input_std/count)
            grp.attrs["log_schema_version"] = 4
            grp.attrs["episode_num"] = episode_num
            grp.attrs["timesteps"] = timesteps
            grp.attrs["success"] = success
            grp.attrs["reward_per_robot"] = reward_per_robot
            grp.attrs["gsp_reward_per_robot"] = gsp_per_robot

            # Per-episode diagnostic attrs (FAU / weight norms / effective rank /
            # Q-gap / pred diversity). Only written on episodes where
            # record_episode_diagnostics was called since the last _reset.
            for k, v in self.episode_diagnostics.items():
                grp.attrs[k] = float(v)

            # Fixed eval-batch states (typically only on the episode that first
            # freezes the batch — ep 50 by default per the diagnostics spec).
            if self.eval_batch_states is not None:
                grp.create_dataset(
                    "diag_eval_batch_states",
                    data=self.eval_batch_states,
                    compression="gzip",
                    compression_opts=4,
                )

            # Information-collapse summary attrs. Computed only when both prediction
            # (gsp_heading) and target (gsp_target) are present. The caller contract is
            # that gsp_target must be passed on every writerow call within an episode once
            # it's been passed on any — i.e. it tracks gsp_heading 1:1. Enforce here rather
            # than silently zipping misaligned buffers. Use NaN-aware aggregation so a
            # single physics glitch (prediction -> NaN) does not poison the summary.
            # Undefined correlations (zero-variance in predictions or targets) are written
            # as NaN so downstream analysis can distinguish "undefined" from "measured zero".
            if self.gsp_target and self.gsp_heading:
                if len(self.gsp_target) != len(self.gsp_heading):
                    raise ValueError(
                        f"gsp_target buffer length {len(self.gsp_target)} does not match "
                        f"gsp_heading buffer length {len(self.gsp_heading)}; gsp_target must "
                        "be passed on every writerow call within an episode once it's been "
                        "passed on any call."
                    )
                # Multi-dim GSP output: extract the LAST dim of gsp_heading for the
                # Δθ correlation metric. Convention: for cyl_kinematics_3d/goal_4d
                # the last dim is the cylinder Δθ component (matching env.py's convention).
                # For legacy 1d shapes (T,) or (T,R) with K=1, [-1] on the inner axis
                # is identical to the sole element — so 1d behavior is preserved exactly.
                # gsp_heading is a list of per-timestep values; each entry may be:
                #   - a scalar (legacy 1d flat)
                #   - a list/array of per-robot scalars (legacy, shape R)
                #   - a list/array of per-robot K-dim vectors (multi-dim, shape R×K)
                # We build a flat array of the last-dim of each robot's prediction across
                # all timesteps, then correlate against the ravelled target.
                _heading_last_dim = []
                for _step_entry in self.gsp_heading:
                    _entry_arr = np.asarray(_step_entry, dtype=np.float64)
                    if _entry_arr.ndim == 0:
                        # Scalar — single value
                        _heading_last_dim.append(float(_entry_arr))
                    elif _entry_arr.ndim == 1:
                        # Either (R,) scalars or (K,) dims for 1 robot
                        # Treat as per-robot scalars; take each element as-is
                        # (for K>1 single-robot case this picks [-1] per step, correct)
                        _heading_last_dim.extend(_entry_arr.ravel().tolist())
                    else:
                        # (R, K) — take last dim per robot
                        _heading_last_dim.extend(_entry_arr[:, -1].ravel().tolist())
                pred_arr = np.array(_heading_last_dim, dtype=np.float64)
                target_arr = np.array(self.gsp_target, dtype=np.float64).ravel()
                if pred_arr.size > 0:
                    pred_std = float(np.nanstd(pred_arr))
                    target_std = float(np.nanstd(target_arr))
                    # Tolerance guard: np.nanstd of a constant returns ~1e-18 due to
                    # sum-of-squares fuzz, which passes a strict `> 0` check and lets
                    # corrcoef return a garbage value. The observables here are radians
                    # clipped to [-1, 1], so 1e-12 is well below any physical variation.
                    STD_TOL = 1e-12
                    grp.attrs["gsp_output_std"] = pred_std
                    if pred_arr.size > 1 and pred_std > STD_TOL and target_std > STD_TOL:
                        mask = np.isfinite(pred_arr) & np.isfinite(target_arr)
                        if mask.sum() > 1:
                            corr = float(np.corrcoef(pred_arr[mask], target_arr[mask])[0, 1])
                        else:
                            corr = float("nan")
                    else:
                        corr = float("nan")
                    # Phase 5 metric cleanup (2026-04-29): the existing name
                    # `gsp_pred_target_corr` is misleading because pred=future_prox
                    # (head's training target at horizon=5) but target=delta_theta
                    # (cylinder's per-step heading change from calculate_gsp_reward).
                    # These are different physical quantities, so the correlation
                    # was apples-vs-oranges. The new name `production_pred_vs_deltatheta_corr`
                    # describes what's actually computed. Both names are written
                    # during a one-batch migration window; the deprecated alias
                    # will be removed in a follow-up PR after H-phase5-1 lands.
                    # See docs/superpowers/plans/2026-04-29-metric-cleanup.md.
                    grp.attrs["production_pred_vs_deltatheta_corr"] = corr
                    grp.attrs["gsp_pred_target_corr"] = corr  # DEPRECATED — see above

            # Phase 4 loss-step correlation diagnostic attrs.
            # gsp_loss_step_corr_mean: mean Pearson r between fresh forward-pass preds
            #   and replay-buffer labels, averaged over all GSP learn steps this episode.
            # gsp_loss_step_corr_std / _min / _max: spread and range across batches.
            # Written only when at least one valid (non-NaN) batch correlation was
            # recorded.  Non-GSP runs, test runs, and warm-up episodes where the replay
            # buffer is not yet full produce no attrs, preserving backward compatibility.
            if self.gsp_loss_step_corr_samples:
                _corr_arr = np.array(self.gsp_loss_step_corr_samples, dtype=np.float64)
                grp.attrs["gsp_loss_step_corr_mean"] = float(np.nanmean(_corr_arr))
                grp.attrs["gsp_loss_step_corr_std"] = float(np.nanstd(_corr_arr))
                grp.attrs["gsp_loss_step_corr_min"] = float(np.nanmin(_corr_arr))
                grp.attrs["gsp_loss_step_corr_max"] = float(np.nanmax(_corr_arr))
                grp.attrs["gsp_loss_step_corr_n_batches"] = int(len(_corr_arr))

            # JEPA latent-space diagnostic attrs.
            # Written as episode-group attrs when at least one learn step fired.
            if self.jepa_pred_mse:
                grp.attrs["jepa_pred_mse_mean"] = float(np.mean(self.jepa_pred_mse))
            if self.jepa_latent_var:
                grp.attrs["jepa_latent_var_mean"] = float(np.mean(self.jepa_latent_var))
            if self.jepa_latent_rank:
                grp.attrs["jepa_latent_rank_mean"] = float(np.mean(self.jepa_latent_rank))

        # Reset for next episode
        summary = {
            "episode_num": episode_num,
            "timesteps": timesteps,
            "success": success,
            "reward_per_robot": reward_per_robot,
            "gsp_reward_per_robot": gsp_per_robot,
        }
        self._reset()


        return summary

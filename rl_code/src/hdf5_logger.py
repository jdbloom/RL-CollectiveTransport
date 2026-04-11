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
from typing import Optional

import h5py
import numpy as np

# Notification handled by ingestion worker (optional, external)


class HDF5Logger:
    """Accumulates timestep data and writes to HDF5 at episode boundaries."""

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
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

    def writerow(
        self, rewards, epsilons, terminations, losses,
        force_magnitudes, force_angles, average_force_vectors,
        cyl_x_poses, cyl_y_poses, cyl_angles,
        gate_stats, obstacle_stats,
        gsp_rewards, gsp_headings,
        run_times, robots_x_poses, robots_y_poses, robot_angles,
        robot_failure, com_X_poses=0, com_Y_poses=0,
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

    def write_episode(self, episode_num: int) -> dict:
        """Write accumulated data to HDF5 and return summary dict.

        Call this instead of data_logger.write_to_file().
        Optionally notifies the ingestion worker.
        """
        group_name = f"episode_{episode_num:04d}"

        with h5py.File(self.hdf5_path, "a") as h5f:
            if group_name in h5f:
                del h5f[group_name]
            grp = h5f.create_group(group_name)

            # Store 2D arrays (timesteps × robots)
            for key, data in [
                ("reward", self.reward),
                ("gsp_reward", self.gsp_reward),
                ("force_magnitude", self.force_magnitude),
                ("force_angle", self.force_angle),
                ("robot_x_pos", self.robots_x_pos),
                ("robot_y_pos", self.robots_y_pos),
                ("robot_angle", self.robot_angle),
                ("robot_failure", self.robot_failures),
                ("gsp_heading", self.gsp_heading),
            ]:
                arr = np.array(data, dtype=np.float32)
                if arr.size > 0:
                    grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

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

            grp.attrs["episode_num"] = episode_num
            grp.attrs["timesteps"] = timesteps
            grp.attrs["success"] = success
            grp.attrs["reward_per_robot"] = reward_per_robot
            grp.attrs["gsp_reward_per_robot"] = gsp_per_robot

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

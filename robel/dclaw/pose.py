# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pose tasks with DClaw robots.

The DClaw is tasked to match a pose defined by the environment.
"""

import abc
import collections
from typing import Any, Dict

import numpy as np

from robel.components.robot.dynamixel_robot import DynamixelRobotState
from robel.components.robot import RobotComponentBuilder, RobotState
from robel.dclaw.base_env import BaseDClawEnv
from robel.utils.configurable import configurable
from robel.utils.resources import get_asset_path

DCLAW3_ASSET_PATH = 'robel-scenes/dclaw/dclaw3xh.xml'

# Add overlay group to show desired position.
OVERLAY_CONFIG = {
    'groups': {
        'overlay': {
            'qpos_indices': range(9, 18),
            'actuator_indices': []
        }
    }
}

# The maximum velocity for the motion task.
MOTION_VELOCITY_LIMIT = np.pi / 6  # 30deg/s

# The error margin to the desired positions to consider as successful.
SUCCESS_THRESHOLD = 10 * np.pi / 180


class BaseDClawPose(BaseDClawEnv, metaclass=abc.ABCMeta):
    """Shared logic for DClaw pose tasks."""

    def __init__(self,
                 asset_path: str = DCLAW3_ASSET_PATH,
                 frame_skip: int = 40,
                 **kwargs):
        """Initializes the environment.

        Args:
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            frame_skip: The number of simulation steps per environment step.
        """
        super().__init__(
            sim_model=get_asset_path(asset_path),
            frame_skip=frame_skip,
            **kwargs)

        self._initial_pos = np.zeros(9)
        self._desired_pos = np.zeros(9)

    def _configure_robot(self, builder: RobotComponentBuilder):
        super()._configure_robot(builder)
        # Add an overlay group to show desired joint positions.
        builder.add_group(
            'overlay', actuator_indices=[], qpos_indices=range(9, 18))

    def _reset(self):
        """Resets the environment."""
        # Mark the target position in sim.
        self.robot.set_state({
            'dclaw': RobotState(qpos=self._initial_pos, qvel=np.zeros(9)),
        })

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({'dclaw': action})

    def get_obs_dict(self) -> Dict[str, Any]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        state = self.robot.get_state('dclaw')

        obs_dict = collections.OrderedDict((
            ('qpos', state.qpos),
            ('qvel', state.qvel),
            ('qpos_error', self._desired_pos - state.qpos),
        ))
        # Add hardware-specific state if present.
        if isinstance(state, DynamixelRobotState):
            obs_dict['current'] = state.current

        return obs_dict

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        qvel = obs_dict['qvel']

        reward_dict = collections.OrderedDict((
            ('pose_error_cost', -1 * np.linalg.norm(obs_dict['qpos_error'])),
            # Penalty if the velocity exceeds a threshold.
            ('joint_vel_cost',
             -0.1 * np.linalg.norm(qvel[np.abs(qvel) >= np.pi])),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        mean_pos_error = np.mean(np.abs(obs_dict['qpos_error']), axis=1)
        score_dict = collections.OrderedDict((
            # Clip and normalize error to 45 degrees.
            ('points', 1.0 - np.minimum(mean_pos_error / (np.pi / 4), 1)),
            ('success', mean_pos_error < SUCCESS_THRESHOLD),
        ))
        score_dict.update(
            self._get_safety_scores(
                pos=obs_dict['qpos'],
                vel=obs_dict['qvel'],
                current=obs_dict.get('current'),
            ))
        return score_dict

    def _make_random_pose(self) -> np.ndarray:
        """Returns a random pose."""
        pos_range = self.robot.get_config('dclaw').qpos_range
        random_range = pos_range.copy()
        # Clamp middle joints to at most 0 (joints always go outwards) to avoid
        # entanglement.
        random_range[[1, 4, 7], 1] = 0
        pose = self.np_random.uniform(
            low=random_range[:, 0], high=random_range[:, 1])
        return pose

    def _update_overlay(self):
        """Updates the overlay in simulation to show the desired pose."""
        self.robot.set_state({'overlay': RobotState(qpos=self._desired_pos)})


@configurable(pickleable=True)
class DClawPoseStill(BaseDClawPose):
    """Track a still random initial and final pose."""

    def _reset(self):
        self._initial_pos = self._make_random_pose()
        self._desired_pos = self._make_random_pose()
        self._update_overlay()
        super()._reset()


@configurable(pickleable=True)
class DClawPoseMotion(BaseDClawPose):
    """Track a moving pose."""

    def _reset(self):
        # Choose two poses to oscillate between.
        pose_a = self._make_random_pose()
        pose_b = self._make_random_pose()
        self._initial_pos = 0.5 * (pose_a + pose_b)
        self._dynamic_range = 0.5 * np.abs(pose_b - pose_a)

        # Initialize a random oscilliation period.
        dclaw_config = self.robot.get_config('dclaw')
        self._period = self.np_random.uniform(
            low=0.5, high=2.0, size=len(dclaw_config.qpos_indices))

        # Clamp the movement range by the velocity limit.
        vel_limit = MOTION_VELOCITY_LIMIT / self._period
        self._dynamic_range = np.minimum(self._dynamic_range, vel_limit)

        self._update_desired_pose()
        super()._reset()

    def _update_desired_pose(self):
        self._desired_pos = (
            self._initial_pos +
            (self._dynamic_range * np.sin(self._period * self.robot.time)))
        self._update_overlay()

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        result = super()._step(action)
        self._update_desired_pose()
        return result

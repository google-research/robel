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

"""Unit tests for D'Claw pose tasks."""

import unittest

import gym
import numpy as np
from parameterized import parameterized_class

from robel.dclaw.pose import (DClawPoseStill, DClawPoseMotion)
# pylint: disable=no-member


@parameterized_class(('env_id', 'env_class'), [
    ('DClawPoseStill-v0', DClawPoseStill),
    ('DClawPoseMotion-v0', DClawPoseMotion),
])
class DClawPoseTest(unittest.TestCase):
    """Unit test class for RobotEnv."""

    def test_gym_make(self):
        """Accesses the sim, model, and data properties."""
        env = gym.make(self.env_id)
        self.assertIsInstance(env.unwrapped, self.env_class)

    def test_spaces(self):
        """Checks the observation, action, and state spaces."""
        env = self.env_class()
        observation_size = np.sum([
            9,  # qpos
            9,  # qvel
            9,  # qpos_error
        ])
        self.assertEqual(env.observation_space.shape, (observation_size,))
        self.assertEqual(env.action_space.shape, (9,))
        self.assertEqual(env.state_space['qpos'].shape, (9,))
        self.assertEqual(env.state_space['qvel'].shape, (9,))

    def test_reset_step(self):
        """Checks that resetting and stepping works."""
        env = self.env_class()
        env.reset()
        env.step(env.action_space.sample())


if __name__ == '__main__':
    unittest.main()

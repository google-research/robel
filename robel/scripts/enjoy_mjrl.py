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

"""Runs an MJRL policy.

Example usage:
python -m robel.scripts.enjoy_mjrl --device /dev/ttyUSB0

This runs the DClawTurnRandom-v0 environment by default. To run other
environments, pass in the environment name with `-e/--env_name`

python -m robel.scripts.enjoy_mjrl \
    --env_name DClawScrewFixed-v0 \
    --device /dev/ttyUSB0
"""

import argparse
import os
import pickle
from typing import Optional

import gym
from mjrl.utils.gym_env import GymEnv
try:
    from mjrl.samplers.core import do_rollout
except ImportError:
    from mjrl.samplers.base_sampler import do_rollout

import robel
from robel.scripts.utils import EpisodeLogger, parse_env_args

POLICY_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 'data/mjrl')

DEFAULT_ENV_NAME = 'DKittyWalkFixedOld-v0'
DEFAULT_POLICY_FORMAT = os.path.join(POLICY_DIR, '{}-policy.pkl')
DEFAULT_EPISODE_COUNT = 10


class RenderingEnvWrapper:
    """Calls render every step."""

    def __init__(self, env: gym.Env, render_mode: Optional[str] = None):
        """Initializes the wrapper."""
        self._env = env
        self._render_mode = render_mode

    def step(self, action):
        result = self._env.step(action)
        self.env.render(self._render_mode)
        return result

    def __getattr__(self, name: str):
        return getattr(self._env, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--output',
        default='output',
        help=('The directory to save job data to.'))
    parser.add_argument(
        '-p',
        '--policy',
        help='The path to the pickled softlearning policy to load.')
    parser.add_argument(
        '-n',
        '--num_episodes',
        default=DEFAULT_EPISODE_COUNT,
        type=int,
        help='The number of episodes the evaluate.')
    parser.add_argument(
        '-r',
        '--render',
        nargs='?',
        const='human',
        default=None,
        choices=['human', 'rgb_array'],
        help='The render mode for the policy.')
    env_name, env_params, args = parse_env_args(
        parser, default_env_name=DEFAULT_ENV_NAME)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    # Create the environment.
    robel.set_env_params(env_name, env_params)
    env = GymEnv(env_name)
    if args.render:
        render_env = RenderingEnvWrapper(env, args.render)
        env = lambda: render_env

    # Get default policy path from the environment name.
    policy_path = args.policy
    if not policy_path:
        policy_path = DEFAULT_POLICY_FORMAT.format(env_name)

    # load policy
    with open(policy_path, 'rb') as f:
        policy = pickle.load(f)

    csv_path = os.path.join(args.output, '{}-results.csv'.format(env_name))
    with EpisodeLogger(csv_path) as logger:
        paths = do_rollout(args.num_episodes, env=env, policy=policy)
        for path in paths:
            logger.log_path(
                path, reward_key='rewards', env_info_key='env_infos')


if __name__ == '__main__':
    main()

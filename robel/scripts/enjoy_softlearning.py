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

"""Runs a softlearning policy.

Example usage:
python -m robel.scripts.enjoy_softlearning --device /dev/ttyUSB0

This runs the DClawTurnRandom-v0 environment by default. To run other
environments, pass in the environment name with `-e/--env_name`

python -m robel.scripts.enjoy_softlearning \
    --env_name DClawScrewFixed-v0 \
    --device /dev/ttyUSB0
"""

import argparse
import os
import pickle

import gym

from softlearning.environments.adapters import gym_adapter
from softlearning.samplers import rollout

import robel
from robel.scripts.utils import EpisodeLogger, parse_env_args

gym_adapter.DEFAULT_OBSERVATION_KEY = 'obs'

POLICY_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 'data/softlearning')

DEFAULT_ENV_NAME = 'DClawTurnRandom-v0'
DEFAULT_POLICY_FORMAT = os.path.join(POLICY_DIR, '{}-policy.pkl')
DEFAULT_EPISODE_COUNT = 10

DEFAULT_EPISODE_LENGTHS = {
    'DClawTurnFixed-v0': 80,
    'DClawTurnRandom-v0': 80,
    'DClawScrewFixed-v0': 160,
    'DClawScrewRandom-v0': 160,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        '--output',
        default='output',
        help='The output directory to save evaluation data to.')
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
        '-l',
        '--episode_length',
        type=int,
        help='The number of steps in each episode.')
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

    # Load the environment.
    env = gym.make(env_name, **env_params)
    env = gym_adapter.GymAdapter(env=env, domain=None, task=None)

    # Get default policy path from the environment name.
    policy_path = args.policy
    if not policy_path:
        policy_path = DEFAULT_POLICY_FORMAT.format(env_name)
    episode_length = args.episode_length
    if not episode_length:
        episode_length = DEFAULT_EPISODE_LENGTHS[env_name]

    # Load the policy.
    with open(policy_path, 'rb') as f:
        policy_data = pickle.load(f)
        policy = policy_data['policy']
        policy.set_weights(policy_data['weights'])

    render_kwargs = {}
    if args.render:
        render_kwargs['mode'] = args.render

    csv_path = os.path.join(args.output, '{}-results.csv'.format(env_name))
    with EpisodeLogger(csv_path) as logger:
        with policy.set_deterministic(True):
            for _ in range(args.num_episodes):
                path = rollout(
                    env,
                    policy,
                    path_length=episode_length,
                    render_kwargs=render_kwargs)
                logger.log_path(
                    path, reward_key='rewards', env_info_key='infos')


if __name__ == '__main__':
    main()

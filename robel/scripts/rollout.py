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

"""Script to perform rollouts on an environment.

Example usage:
# Visualize an environment:
python -m robel.scripts.rollout -e DClawTurnFixed-v0 --render

# Benchmark offscreen rendering:
python -m robel.scripts.rollout -e DClawTurnFixed-v0 --render rgb_array
"""

import argparse
import collections
import json
import os
import time
from typing import Callable, Optional

import gym
import numpy as np

import robel
from robel.scripts.utils import parse_env_args


def do_rollouts(env,
                num_episodes: int,
                max_episode_length: Optional[int] = None,
                action_fn: Optional[Callable[[], np.ndarray]] = None,
                render_mode: Optional[str] = None):
    """Performs rollouts with the given environment.

    Args:
        num_episodes: The number of episodes to do rollouts for.
        max_episode_length: The maximum length of an episode.
        action_fn: The function to use to sample actions for steps. If None,
            uses random actions.
        render_mode: The rendering mode. If None, no rendering is performed.

    Yields:
        episode_obs: The observations for the episode.
        episode_return: The total reward during the episode.
        episode_infos: The auxiliary information during the episode.
        episode_renders: Rendered frames during the episode.
        durations: The running execution durations.
    """
    # If no action function is given, use random actions from the action space.
    if action_fn is None:
        action_fn = env.action_space.sample

    # Maintain a dictionary of execution durations.
    durations = collections.defaultdict(float)

    # Define a function to maintain a running average of durations.
    def record_duration(key: str, iteration: int, value: float):
        durations[key] = (durations[key] * iteration + value) / (iteration + 1)

    total_steps = 0
    for episode in range(num_episodes):
        episode_start = time.time()
        obs = env.reset()
        record_duration('reset', episode, time.time() - episode_start)

        done = False
        episode_obs = []
        episode_return = 0
        episode_infos = []
        episode_renders = []

        while not done:
            action = action_fn()

            step_start = time.time()
            obs, reward, done, info = env.step(action)
            step_time = time.time()
            record_duration('step', total_steps, step_time - step_start)

            if render_mode is not None:
                render_result = env.render(render_mode)
                record_duration('render', total_steps, time.time() - step_time)
                if render_result is not None:
                    episode_renders.append(render_result)

            episode_obs.append(obs)
            episode_return += reward
            episode_infos.append(info)

            total_steps += 1
            if (max_episode_length is not None
                    and len(episode_obs) >= max_episode_length):
                done = True

        yield (episode_obs, episode_return, episode_infos, episode_renders,
               dict(durations))


def smooth_action_generator(env, smoothing_factor: float):
    """Generator function that yields smoothed random actions."""
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    while True:
        action *= smoothing_factor
        action += (1.0 - smoothing_factor) * env.action_space.sample()
        yield action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--num_episodes',
        type=int,
        default=100,
        help='The number of episodes to run.')
    parser.add_argument(
        '-l',
        '--episode_length',
        type=int,
        default=None,
        help='The maximum episode length to run for.')
    parser.add_argument(
        '--seed', type=int, default=None, help='The seed for the environment.')
    parser.add_argument(
        '-r',
        '--render',
        nargs='?',
        const='human',
        default=None,
        help=('The rendering mode. If provided, renders to a window. A render '
              'mode string can be passed here.'),
    )
    env_id, params, args = parse_env_args(parser)

    robel.set_env_params(env_id, params)
    env = gym.make(env_id)
    if args.seed is not None:
        env.seed(args.seed)

    try:
        episode_num = 0
        for obs, reward, infos, renders, durations in do_rollouts(
                env,
                num_episodes=args.num_episodes,
                max_episode_length=args.episode_length,
                action_fn=None,
                render_mode=args.render,
        ):
            print('Episode {}'.format(episode_num))
            print('> Total reward: {}'.format(reward))
            print('> Execution times: {}'.format(
                json.dumps(durations, indent=2, sort_keys=True)))
            episode_num += 1
    finally:
        env.close()


if __name__ == '__main__':
    main()

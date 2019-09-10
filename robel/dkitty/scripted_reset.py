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

"""Hardware reset functions for the D'Kitty."""

import logging
import time

import numpy as np

from robel.components.robot import RobotComponentBuilder, RobotState
from robel.components.robot.dynamixel_robot import DynamixelRobotComponent
from robel.components.tracking.tracker import TrackerComponent

# Maximum values for each joint.
BASEMAX = .8
MIDMAX = 2.4
FOOTMAX = 2.5

# Common parameters for all `set_state` commands.
SET_PARAMS = dict(
    error_tol=5 * np.pi / 180,  # 5 degrees
    last_diff_tol=.1 * np.pi / 180,  # 5 degrees
)


def add_groups_for_reset(builder: RobotComponentBuilder):
    """Defines groups required to perform the reset."""
    builder.add_group('base', motor_ids=[10, 20, 30, 40])
    builder.add_group('middle', motor_ids=[11, 21, 31, 41])
    builder.add_group('feet', motor_ids=[12, 22, 32, 42])
    builder.add_group('front', motor_ids=[11, 12, 21, 22])
    builder.add_group('back', motor_ids=[31, 32, 41, 42])
    builder.add_group(
        'all', motor_ids=[10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42])


def reset_standup(robot: DynamixelRobotComponent, tracker: TrackerComponent):
    """Resets the D'Kitty to a standing position."""
    start_time = time.time()
    reset = False
    num_attempts = 0
    # Is it high, flat, and are all of it's joints at 0
    while not is_standing(robot, tracker) and not reset:
        # Splay the kitty out and let limp
        robot.set_state(
            {
                'base': RobotState(
                    qpos=np.array(
                        [-np.pi / 2.1, np.pi / 2.1, np.pi / 2.1, -np.pi / 2.1])
                ),
                'feet': RobotState(qpos=np.full(4, FOOTMAX)),
                'middle': RobotState(qpos=np.full(4, -MIDMAX))
            },
            **SET_PARAMS,
            timeout=5,
        )
        time.sleep(1)
        robot.set_motors_engaged('dkitty', engaged=False)
        time.sleep(.3)
        robot.set_motors_engaged('dkitty', engaged=True)

        if tracker.is_hardware:
            # If the robot has a tracker use it to do the reset
            # Gravity tells us whether it's on its back
            base_state = np.abs(robot.get_state('base').qpos)
            if np.all(base_state > 1.75) and np.all(base_state < 2.45):
                # Use base joints because tracker can't be seen
                back_recover(robot)
            elif (
                    is_flat(tracker) and get_height(tracker) < .17
                    and np.all(base_state < 1.75) and np.all(base_state > 1.1)
            ):  # Tracker values tells us for sure whether it's on its chest
                chest_recover(robot)
            else:
                flail(robot)
        else:
            # If the robot doesn't have a tracker, use gravity
            base_state = np.abs(robot.get_state('base').qpos)
            if np.all(base_state < 1.75) and np.all(base_state > 1.1):
                chest_recover(robot)
                reset = True
            elif np.all(base_state > 1.75) and np.all(base_state < 2.45):
                back_recover(robot)
                reset = True
            else:
                flail(robot)
        num_attempts += 1

    logging.info('Total reset time: %1.2f over %d attempts',
                 time.time() - start_time, num_attempts)


def get_rotation(tracker: TrackerComponent):
    """Returns the Euler rotation of the tracker."""
    # Data is [0]=side-side [1]=flat [2]=forward-back
    if tracker.is_hardware:
        torso_state = tracker.get_state('torso')
        return torso_state.rot_euler
    return (0, 0, 0)


def get_height(tracker: TrackerComponent):
    """Returns the height of the tracker, assuming relative to the floor."""
    if tracker.is_hardware:
        torso_state = tracker.get_state('torso')
        return torso_state.pos[2]
    return 0


def is_standing(robot: DynamixelRobotComponent, tracker: TrackerComponent):
    """Returns True if the D'Kitty is standing.

    Checks the rotation and height of the tracker and whether all joints are 0
    in case the other two fail.
    """
    if tracker.is_hardware:
        flat = is_flat(tracker)
        h1 = get_height(tracker)
        time.sleep(.2)
        h2 = get_height(tracker)
        tall = h1 != h2 and h1 > .175
        straight_legs = np.all(abs(robot.get_state('all').qpos) < .15)
        return flat and tall and straight_legs
    return False


def is_flat(tracker: TrackerComponent):
    """Returns True if the tracker is flat relative to floor."""
    if tracker.is_hardware:
        return abs(abs(get_rotation(tracker)[0]) - 1.56) < .2
    return True


def flail(robot: DynamixelRobotComponent):
    """Commands the robot to flail if it's stuck on an obstacle."""
    for _ in range(6):
        robot.set_state(
            {'all': RobotState(qpos=(np.random.rand(12) - .5) * 3)},
            **SET_PARAMS,
            timeout=.15,
        )


def chest_recover(robot: DynamixelRobotComponent):
    """Recovers the robot when it's laying flat on its chest."""
    robot.set_state({
        'middle':
            RobotState(qpos=np.array([-MIDMAX, -MIDMAX, -MIDMAX, -MIDMAX])),
        'feet': RobotState(qpos=np.array([FOOTMAX, FOOTMAX, FOOTMAX, FOOTMAX]))
    }, **SET_PARAMS)
    # Wiggle back and forth to get the feet fully underneath the dkitty.
    base_position = np.array([BASEMAX, BASEMAX, BASEMAX, BASEMAX])
    for i in range(2):
        robot.set_state(
            {
                'base': RobotState(
                    qpos=(base_position if i % 2 == 0 else -base_position))
            },
            **SET_PARAMS,
            timeout=1,
        )
        time.sleep(1)
    robot.set_state({'base': RobotState(qpos=np.zeros(4))}, **SET_PARAMS)

    straighten_legs(robot, True)


def back_recover(robot: DynamixelRobotComponent):
    """Recovers the D'Kitty from a flipped over position."""
    # Fixed legs in straight and put all base joints at same max angle.
    # Illustration: _._._ -> _‾/
    robot.set_state({
        'base': RobotState(qpos=np.full(4, -BASEMAX)),
        'middle': RobotState(qpos=np.zeros(4)),
        'feet': RobotState(qpos=np.zeros(4))
    }, **SET_PARAMS)
    time.sleep(1)
    # Use the legs to shift its CG and get onto its side. -> _‾|
    robot.set_state({'base': RobotState(qpos=np.zeros(4))}, **SET_PARAMS)
    robot.set_state({
        'middle': RobotState(qpos=np.array([0, -MIDMAX, -MIDMAX, 0])),
        'feet': RobotState(qpos=np.array([0, FOOTMAX, FOOTMAX, 0]))
    }, **SET_PARAMS)
    cbstate = np.array([BASEMAX, BASEMAX / 1.4, BASEMAX / 1.4, BASEMAX])
    robot.set_state({'base': RobotState(qpos=cbstate)}, **SET_PARAMS)
    robot.set_state({
        'middle': RobotState(qpos=np.zeros(4)),
        'feet': RobotState(qpos=np.zeros(4))
    }, **SET_PARAMS)
    # Fully fold legs causing the dkitty to tip onto its feet, doing this and
    # the previous step separately decreases the possibility of binding. -> ̷‾̷
    robot.set_state({
        'base': RobotState(qpos=np.full(4, BASEMAX)),
        'feet': RobotState(qpos=np.zeros(4)),
        'middle': RobotState(qpos=np.array([-MIDMAX, 0, 0, -MIDMAX]))
    }, **SET_PARAMS)
    time.sleep(.5)
    robot.set_state({
        'feet': RobotState(qpos=np.full(4, FOOTMAX)),
        'middle': RobotState(qpos=np.full(4, -MIDMAX))
    }, **SET_PARAMS)
    # Pivot the legs so that the kitty is now in a pouncing stance. -> |‾|
    robot.set_state({'base': RobotState(qpos=np.zeros(4))}, **SET_PARAMS)
    time.sleep(1)
    straighten_legs(robot, True)


def straighten_legs(robot: DynamixelRobotComponent, reverse: bool = False):
    """Straightens out the legs of the D'Kitty."""
    front_state, back_state = robot.get_state(['front', 'back'])
    frontpos = front_state.qpos
    backpos = back_state.qpos
    states = []
    resolution = 4
    for i in range(resolution, -1, -1):
        if reverse:
            states.append(dict(back=RobotState(qpos=backpos * i / resolution)))
            states.append(
                dict(front=RobotState(qpos=frontpos * i / resolution)))
        else:
            states.append(
                dict(front=RobotState(qpos=frontpos * i / resolution)))
            states.append(dict(back=RobotState(qpos=backpos * i / resolution)))
    for state in states:
        robot.set_state(state, **SET_PARAMS)
        time.sleep(.2)
    time.sleep(1)

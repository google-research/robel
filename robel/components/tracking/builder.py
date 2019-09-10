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

"""Builder-specific logic for creating TrackerComponents."""

from typing import Any, Dict, Optional, Union

from robel.components.builder import ComponentBuilder
from robel.components.tracking.tracker import TrackerComponent
from robel.components.tracking.vr_tracker import VrTrackerComponent


class TrackerComponentBuilder(ComponentBuilder):
    """Builds a RobotComponent."""

    def __init__(self):
        super().__init__()
        self._vr_tracker_ids = set()

    def build(self, *args, **kwargs):
        """Builds the component."""
        if self._vr_tracker_ids:
            return VrTrackerComponent(
                *args, groups=self.group_configs, **kwargs)
        return TrackerComponent(*args, groups=self.group_configs, **kwargs)

    def set_vr_tracker_id(self, group_name: str, tracker_id: Union[str, int]):
        """Sets the VR hardware tracker ID for the given group.

        Args:
            group_name: The group to set the tracker ID for.
            tracker_id: Either the device serial number string, or the device
                index of the VR tracking device.
        """
        self._check_group_exists(group_name)
        self._vr_tracker_ids.add(tracker_id)
        self.group_configs[group_name]['device_identifier'] = tracker_id

    def add_tracker_group(
            self,
            group_name: str,
            vr_tracker_id: Union[str, int],
            sim_params: Optional[Dict[str, Any]] = None,
            hardware_params: Optional[Dict[str, Any]] = None,
            mimic_sim: bool = True,
            mimic_xy_only: bool = False,
    ):
        """Convenience method for adding a tracking group.

        Args:
            group_name: The group name to create.
            vr_tracker_id: Either the device serial number string, or the device
                index of the VR tracking device.
            sim_params: The group parameters for simulation tracking.
            hardware_params: The group parameters for hardware tracking.
            mimic_sim: If True, adds parameters so that simulation mimics the
                hardware.
            mimic_xy_only: If True, adds parameters so that only the XY plane
                movement is mimicked to the simulation.
        """
        self.add_group(group_name, **sim_params)
        if vr_tracker_id is not None:
            self.set_vr_tracker_id(group_name, vr_tracker_id)

            hardware_params = hardware_params or {}
            if mimic_sim:
                hardware_params['mimic_in_sim'] = True
            if mimic_xy_only:
                hardware_params.update({
                    'mimic_ignore_z_axis': True,
                    'mimic_ignore_rotation': True,
                })
            if hardware_params:
                self.update_group(group_name, **hardware_params)

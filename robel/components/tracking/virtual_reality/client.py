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

"""Client to communicate with a VR device using OpenVR.

Example usage:
>>> client = VrClient()
>>> client.set_devices({'tracker': 1})
"""

from typing import Dict, Iterable, Optional, List, Union

import numpy as np
import openvr

from robel.components.tracking.virtual_reality.device import VrDevice
from robel.components.tracking.virtual_reality.poses import (
    VrPoseBatch, VrCoordinateSystem)
from robel.utils.math_utils import average_quaternions


class VrClient:
    """Communicates with a VR device."""

    def __init__(self):
        self._vr_system = None
        self._devices = []
        self._device_serial_lookup = {}
        self._device_index_lookup = {}
        self._coord_system = VrCoordinateSystem()
        self._last_pose_batch = None
        self._plot = None

        # Attempt to start OpenVR.
        if not openvr.isRuntimeInstalled():
            raise OSError('OpenVR runtime not installed.')

        self._vr_system = openvr.init(openvr.VRApplication_Other)

    def close(self):
        """Cleans up any resources used by the client."""
        if self._vr_system is not None:
            openvr.shutdown()
            self._vr_system = None

    def get_device(self, identifier: Union[int, str]) -> VrDevice:
        """Returns the device with the given name."""
        identifier = str(identifier)
        if identifier in self._device_index_lookup:
            return self._device_index_lookup[identifier]
        if identifier in self._device_serial_lookup:
            return self._device_serial_lookup[identifier]

        self.discover_devices()
        if (identifier not in self._device_index_lookup
                and identifier not in self._device_serial_lookup):
            raise ValueError(
                'Could not find device with name or index: {} (Available: {})'
                .format(identifier, sorted(self._device_serial_lookup.keys())))

        if identifier in self._device_index_lookup:
            return self._device_index_lookup[identifier]
        return self._device_serial_lookup[identifier]

    def discover_devices(self) -> List[VrDevice]:
        """Returns and caches all connected devices."""
        self._device_index_lookup.clear()
        self._device_serial_lookup.clear()
        devices = []
        for device_index in range(openvr.k_unMaxTrackedDeviceCount):
            device = VrDevice(self._vr_system, device_index)
            if not device.is_connected():
                continue
            devices.append(device)
            self._device_index_lookup[str(device.index)] = device
            self._device_serial_lookup[device.get_serial()] = device
        self._devices = devices
        return devices

    def get_poses(self, time_from_now: float = 0.0,
                  update_plot: bool = True) -> VrPoseBatch:
        """Returns a batch of poses that can be queried per device.

        Args:
            time_from_now: The seconds into the future to read poses.
            update_plot: If True, updates an existing plot.
        """
        pose_batch = VrPoseBatch(self._vr_system, self._coord_system,
                                 time_from_now)
        self._last_pose_batch = pose_batch
        if update_plot and self._plot and self._plot.is_open:
            self._plot.refresh()
        return pose_batch

    def update_coordinate_system(self,
                                 origin_device: Optional[VrDevice] = None,
                                 device_position_offsets: Optional[
                                     Dict[VrDevice, Iterable[float]]] = None,
                                 device_rotation_offsets: Optional[
                                     Dict[VrDevice, Iterable[float]]] = None,
                                 num_samples: int = 10):
        """Configures the position and rotation of the devices.

        Args:
            origin_device: If given, the coordinate system uses the world
                position of this device as (0, 0, 0). Additionally, the facing
                direction of the device is assumed to be the +y axis.
            device_position_offsets: The (x, y, z) Cartesian offsets for each
                device in the coordinate system space.
            device_rotation_offsets: The (w, x, y, z) quaternion offsets for
                each device in the coordinate system space.
            num_samples: The number of samples to collect to calculate the
                current position and orientation of the origin device.
        """
        raw_origin_pos = None
        raw_origin_quat = None
        if origin_device:
            # Collect samples for the devices.
            pos_samples = []
            quat_samples = []
            for _ in range(num_samples):
                pose_batch = self.get_poses()
                pos, quat = pose_batch.get_pos_quat(origin_device, raw=True)
                pos_samples.append(pos)
                quat_samples.append(quat)
            raw_origin_pos = np.mean(pos_samples, axis=0)
            raw_origin_quat = average_quaternions(quat_samples)

        self._coord_system.initialize(
            origin_device=origin_device,
            raw_origin_pos=raw_origin_pos,
            raw_origin_quat=raw_origin_quat,
            pos_offsets=device_position_offsets,
            quat_offsets=device_rotation_offsets)

    def show_plot(self, auto_update: bool = False, frame_rate: int = 10):
        """Displays a plot that shows the current tracked positions.

        Args:
            auto_update: If True, queries and updates the plot with new pose
                data at the given frame rate.
            frame_rate: The frequency at which the plot is refreshed.
        """
        if self._plot and self._plot.is_open:
            return
        from robel.utils.plotting import AnimatedPlot
        self._plot = AnimatedPlot(bounds=[-5, 5, -5, 5])

        devices = self._devices
        if not devices:
            devices = self.discover_devices()

        data = np.zeros((len(devices), 2), dtype=np.float32)
        scatter = self._plot.ax.scatter(
            data[:, 0], data[:, 1], cmap='jet', edgecolor='k')
        self._plot.add(scatter)

        # Make annotations
        labels = []
        for i, device in enumerate(devices):
            label = self._plot.ax.annotate(
                device.get_model_name(), xy=(data[i, 0], data[i, 1]))
            labels.append(label)
            self._plot.add(label)

        def update():
            if auto_update:
                pose_batch = self.get_poses(update_plot=False)
            else:
                pose_batch = self._last_pose_batch
            if pose_batch is None:
                return
            for i, device in enumerate(devices):
                pos, euler = pose_batch.get_pos_euler(device)
                data[i, 0] = pos[0]
                data[i, 1] = pos[1]
                labels[i].set_position((pos[0], pos[1] + 0.2))
                labels[i].set_text('\n'.join([
                    '{}', 'Tx:{:.2f} Ty:{:.2f} Tz:{:.2f},',
                    'Rx:{:.2f} Ry:{:.2f} Rz:{:.2f}'
                ]).format(device.get_model_name(), *pos, *euler))
            scatter.set_offsets(data)

        self._plot.update_fn = update
        self._plot.show(frame_rate=frame_rate)

    def __enter__(self):
        """Enables use as a context manager."""
        return self

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.close()

    def __del__(self):
        """Automatically disconnect on destruction."""
        self.close()

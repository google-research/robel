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

"""Pose-related logic for OpenVR devices."""

import collections
import logging
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import openvr
from transforms3d.euler import euler2mat, mat2euler, quat2euler
from transforms3d.quaternions import mat2quat, qmult, quat2mat

from robel.components.tracking.virtual_reality.device import VrDevice


class VrCoordinateSystem:
    """Stores the most recently returned values for device poses."""

    # Transform from OpenVR space (y upwards) to simulation space (z upwards).
    GLOBAL_TRANSFORM = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                                dtype=np.float32)

    # Locally rotate to preserve original orientation.
    LOCAL_TRANSFORM = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]],
                               dtype=np.float32)

    def __init__(self):
        # Per-device cached position and rotation matrix.
        self._cached_pos = collections.defaultdict(lambda: np.zeros(
            3, dtype=np.float32))
        self._cached_rot = collections.defaultdict(lambda: np.identity(
            3, dtype=np.float32))

        self._world_translation = np.zeros(3, dtype=np.float32)
        self._world_global_rotation = np.identity(3, dtype=np.float32)
        self._device_translations = {}
        self._device_local_rotations = {}

    def initialize(
            self,
            origin_device: Optional[VrDevice] = None,
            raw_origin_pos: Optional[Iterable[float]] = None,
            raw_origin_quat: Optional[Iterable[float]] = None,
            pos_offsets: Optional[Dict[VrDevice, Iterable[float]]] = None,
            quat_offsets: Optional[Dict[VrDevice, Iterable[float]]] = None):
        """Initializes the coordinate system from the given data.

        Args:
            origin_device: If given, updates the world position of the system
                with this device as (0, 0, 0). Additionally, the facing
                direction of the device is assumed to be the +y axis.
            raw_origin_pos: The raw VR-space world position of the origin
                device.
            raw_origin_quat: The raw VR-space orientation of the origin device.
            pos_offsets: The (x, y, z) Cartesian offsets for each device in the
                coordinate system space.
            quat_offsets: The (w, x, y, z) quaternion offsets for each device
                in the coordinate system space.
        """
        pos_offsets = pos_offsets or {}
        quat_offsets = quat_offsets or {}

        # Set the device translations.
        for device, position in pos_offsets.items():
            if device is origin_device or position is None:
                continue
            position = np.asarray(position)
            assert position.shape == (3,), position
            self._device_translations[device.index] = -position

        # Set the device rotations.
        for device, quat in quat_offsets.items():
            if quat is None:
                continue
            assert np.shape(quat) == (
                4,), 'Rotation offset must be a quaternion.'
            self._device_local_rotations[device.index] = quat2mat(quat)

        # If an origin device is given, measure the origin position.
        if origin_device is not None:
            # Calculate the origin position of the coordinate system.
            if raw_origin_pos is not None:
                origin_pos = np.asarray(raw_origin_pos)
                assert origin_pos.shape == (3,), origin_pos
                origin_offset = pos_offsets.get(origin_device)
                if origin_offset is not None:
                    origin_offset = np.asarray(origin_offset)
                    assert origin_offset.shape == (3,), origin_offset
                    origin_pos -= origin_offset
                self._world_translation = -origin_pos

            # Calculate the origin rotation of the coordinate system. We use the
            # current z-axis rotation as the transformation.
            if raw_origin_quat is not None:
                origin_quat = np.asarray(raw_origin_quat)
                assert origin_quat.shape == (4,), origin_quat

                origin_quat_offset = quat_offsets.get(origin_device)
                if origin_quat_offset is not None:
                    origin_quat_offset = np.asarray(origin_quat_offset)
                    origin_quat = qmult(origin_quat, origin_quat_offset)

                origin_rx, origin_ry, origin_rz = quat2euler(
                    origin_quat, axes='rxyz')
                logging.info('Have origin rotation: %1.2f %1.2f %1.2f',
                             origin_rx, origin_ry, origin_rz)
                self._world_global_rotation = euler2mat(0, 0, -origin_rz)

    def get_cached_pos_rot(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the cached program space position and rotation."""
        return self._cached_pos[index].copy(), self._cached_rot[index].copy()

    def process_from_vr(self, index: int, pos: np.ndarray,
                        rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms from VR space to program space."""
        assert pos.shape == (3,)
        assert rot.shape == (3, 3)
        # If all of the translations are 0, get from the cache.
        if not pos.any():
            return self.get_cached_pos_rot(index)
        # Apply the constant transforms.
        pos = np.matmul(self.GLOBAL_TRANSFORM, pos)
        rot = np.matmul(rot, self.LOCAL_TRANSFORM)
        rot = np.matmul(self.GLOBAL_TRANSFORM, rot)
        # Update the cache.
        self._cached_pos[index] = pos
        self._cached_rot[index] = rot
        return pos.copy(), rot.copy()

    def to_user_space(self, index: int, pos: np.ndarray, rot: np.ndarray):
        """Transforms from program space to user space."""
        pos = pos + self._world_translation
        if index in self._device_translations:
            pos += self._device_translations[index]
        pos = np.matmul(self._world_global_rotation, pos)
        rot = np.matmul(self._world_global_rotation, rot)
        if index in self._device_local_rotations:
            rot = np.matmul(rot, self._device_local_rotations[index])
        return pos, rot


class VrPoseBatch:
    """Represents a batch of poses calculated by the OpenVR system."""

    def __init__(self,
                 vr_system,
                 coord_system: VrCoordinateSystem,
                 time_from_now: float = 0.0):
        """Initializes a new pose batch."""
        self._vr_system = vr_system
        self._coord_system = coord_system
        # Query poses for all devices.
        self.poses = self._vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, time_from_now,
            openvr.k_unMaxTrackedDeviceCount)

    def get_pos_rot(self, device: VrDevice,
                    raw: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the 4x4 pose matrix of the given device."""
        if not self.poses[device.index].bPoseIsValid:
            pos, rot = self._coord_system.get_cached_pos_rot(device.index)
        else:
            vr_pose = np.ctypeslib.as_array(
                self.poses[device.index].mDeviceToAbsoluteTracking[:],
                shape=(3, 4))
            # Check that the pose is valid.
            # If all of the translations are 0, get from the cache.
            assert vr_pose.shape == (3, 4)
            pos, rot = self._coord_system.process_from_vr(
                device.index, vr_pose[:, 3], vr_pose[:, :3])
        if not raw:
            pos, rot = self._coord_system.to_user_space(device.index, pos, rot)
        return pos, rot

    def get_pos_euler(
            self,
            device: VrDevice,
            raw: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the translation and euler rotation of the given device."""
        pos, rot = self.get_pos_rot(device, raw)
        ai, aj, ak = mat2euler(rot, axes='rxyz')
        return pos, np.array([ai, aj, ak], dtype=np.float32)

    def get_pos_quat(
            self,
            device: VrDevice,
            raw: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the translation quaternion rotation of the given device."""
        pos, rot = self.get_pos_rot(device, raw)
        return pos, mat2quat(rot).astype(np.float32)

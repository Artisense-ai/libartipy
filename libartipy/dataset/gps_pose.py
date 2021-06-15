#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for working with gps pose
    Author Dmytro Bobkov, 2019
    Author Thomas Schmid
"""

import numpy as np
from typing import Optional

from libartipy.geometry import Pose
from libartipy.dataset import Constants, get_logger
logger = get_logger()


class GPSPose(object):
    def __init__(self,
                 pose: Pose,
                 timestamp_utc_ns: int,
                 valid_ts: bool,
                 translation_scale: float = None,
                 fused: int = 2):
        """

        :param pose: instance of object class pose
        :param timestamp_utc_ns: timestamp in UTC format
        :param valid_ts: is the timestamp valid?
        :param fused: flag indicating if the GPS pose information is coming from fusion result
        """
        assert pose.valid()
        # allow negative timstamps to indicate that there is no timestamp available

        self.pose = pose
        self.timestamp_utc_ns = timestamp_utc_ns
        self.valid_ts = valid_ts
        self.translation_scale = translation_scale
        self._fused = fused

    def get_timestamp_ns(self) -> int:
        return self.timestamp_utc_ns

    def set_timestamp_ns(self, ts: int):
        self.timestamp_utc_ns = ts
        self.valid_ts = True

    def has_valid_ts(self) -> bool:
        return self.valid_ts

    def get_pose(self) -> Pose:
        assert self.has_valid_timestamp()
        return self.pose

    def has_translation_scale(self) -> bool:
        return self.translation_scale is not None

    def get_translation_scale(self) -> float:
        """

        :raises AssertionError if the translation scale was not set.
        :return:
        """
        assert self.has_translation_scale(), 'No translation scale set, but you asked for it.'
        return self.translation_scale

    @classmethod
    def init_from_line(cls, line: str) -> Optional['GPSPose']:
        """

        :param line: [timestamp], tran_x, tran_y, tran_z, quat_x, quat_y, quat_z, quat_w, [scale]
        :return:
        """
        # line into pose and quaternion
        pose_info_str = list(line.strip().split(','))
        assert len(pose_info_str) >= 7, \
            'There have to be at least 7 values for tran and rot, but we have {}'.format(len(pose_info_str))

        timestamps_available = len(pose_info_str) >= 8
        if timestamps_available:
            assert '.' not in pose_info_str[0], 'Float in timestamp not supported now: {}'.format(pose_info_str[0])
            timestamp_utc_ns = int(pose_info_str[0])  # timestamps are integers
            offset = 1
            valid_ts = True
        else:
            timestamp_utc_ns = Constants.INVALID_TIMESTAMP
            valid_ts = False
            offset = 0
        #
        pose_info = [float(numb) for numb in pose_info_str]
        trans = pose_info[offset:offset + 3]
        rot_wxyz = [pose_info[offset + 6], pose_info[offset + 3], pose_info[offset + 4], pose_info[offset + 5]]  # WXYZ

        if not np.isfinite(rot_wxyz[0]):
            return None

        scale_available = len(pose_info) >= 9
        if scale_available:
            translation_scale = pose_info[offset+7]
        else:
            translation_scale = None

        fused_info_available = len(pose_info) >= 10
        if fused_info_available:
            fused = int(pose_info[9])
            assert fused in [0, 1, 2], 'Fusion flag has to be in range 0..2: {}'.format(fused)
        else:
            fused = None

        gps_pose = Pose(use_scale_in_rotation=False)
        gps_pose.rotation_quaternion = rot_wxyz
        gps_pose.translation = trans

        return cls(gps_pose, timestamp_utc_ns, valid_ts, translation_scale, fused=fused)

    def has_valid_timestamp(self) -> bool:
        return self.valid_ts

    def fused(self) -> int:
        """

        :return: 0 - non fused, 1 - interpolated, 2 - fused (higher is better quality)
        """
        if self._fused is None:
            logger.warning("Fused field not set, returning best quality")
            return 2
        else:
            return self._fused

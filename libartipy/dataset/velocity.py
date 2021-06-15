#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for working with velocities
    Author Pavel Ermakov, 2020
"""

import numpy as np

from libartipy.dataset import Constants

from typing import Optional


class Velocity(object):
    def __init__(self,
                 tangent: np.array,
                 timestamp_utc_ns: int,
                 valid_ts: bool):
        """

        :param tangent: se3 local tangent/body velocity (translational + rotational: [t, w])
        :param timestamp_utc_ns: utc timestamp
        """
        assert len(tangent) == 6

        self.body_velocity = tangent
        self.timestamp_utc_ns = timestamp_utc_ns
        self.valid_ts = valid_ts

    def get_timestamp_ns(self) -> int:
        return self.timestamp_utc_ns

    def set_timestamp_ns(self, ts: int):
        self.timestamp_utc_ns = ts
        self.valid_ts = True

    def has_valid_timestamp(self) -> bool:
        return self.valid_ts

    def get_velocity(self) -> np.array:
        assert self.has_valid_timestamp()
        return self.body_velocity

    @classmethod
    def init_from_line(cls, line: str) -> Optional['Velocity']:
        """
        :param line: [timestamp], v_x, v_y, v_z, w_x, w_y, w_z
        :return:
        """
        # line into pose and quaternion
        vel_info_str = list(line.strip().split(','))
        assert len(vel_info_str) >= 6, \
            'There have to be at least 6 values for tangent, but we have {}'.format(
                len(vel_info_str))

        timestamps_available = len(vel_info_str) >= 7
        if timestamps_available:
            assert '.' not in vel_info_str[0], 'Float in timestamp not supported now: {}'.format(
                vel_info_str[0])
            timestamp_utc_ns = int(vel_info_str[0])  # timestamps are integers
            offset = 1
            valid_ts = True
        else:
            timestamp_utc_ns = Constants.INVALID_TIMESTAMP
            valid_ts = False
            offset = 0
        #
        velocity = np.array([float(numb) for numb in vel_info_str[offset:]])
        return cls(velocity, timestamp_utc_ns, valid_ts)

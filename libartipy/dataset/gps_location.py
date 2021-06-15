#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" File for working with GPS locations that are inside keyframe.
    Author Dmytro Bobkov, 2019
"""
import numpy as np
from typing import Union, List

from libartipy.dataset import Point


class GPSLocation:
    def __init__(self, position: Point, covariance_matrix: np.ndarray):
        """

        :param position:
        :param covariance_matrix:
        """
        #assert(len(position) == 3)
        self.position = position
        assert covariance_matrix.shape == (3, 3), 'Invalid shape {}'.format(covariance_matrix.shape)
        self.covariance_matrix = covariance_matrix

    def get_position(self) -> Point:
        return self.position

    @classmethod
    def init_from_line(cls, lines: List[str], offset: int) -> 'GPSLocation':
        """

        :param lines: lines to read from
        :param offset: from which line to start, it points to header of gps covariance
        :return: class with inited members
        """
        assert len(lines) > (offset + 5), "Insufficient number of lines {}".format(len(lines))

        # read GPS covariance matrix
        line_covariance = lines[offset+1]
        gps_covariance = \
            np.array(list(map(float, line_covariance.split(',')))
                     ).reshape(3, 3)

        # read GPS point
        line_gps_location = lines[offset+4]
        gps_position = list(map(float, line_gps_location.split(',')))
        assert len(gps_position) == 3
        point = Point(gps_position)

        return cls(point, gps_covariance)

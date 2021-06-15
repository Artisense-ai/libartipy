#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for working with point in frame
    Author Dmytro Bobkov, 2019
"""

import numpy as np
from typing import List, Union


class Point(object):
    """
    This class describes 3D point.
    """

    def __init__(self, coord_xyz: Union[List[float], np.ndarray], intensity: float = None):
        """

        :param coord_xyz: iterable with 3 values
        :param intensity: float value
        """
        if type(coord_xyz) is not np.ndarray:
            coord_xyz = np.array(coord_xyz, dtype=np.float64)

        assert coord_xyz.size == 3
        self.coord_xyz = coord_xyz  # 3x,
        self.coord_xyz_homogenous = np.append(self.coord_xyz, 1.0)
        self.intensity = intensity  # scalar

    def print_info(self) -> None:
        pass

    def get_xyz(self) -> np.ndarray:
        """

        :return: 3x,
        """
        return self.coord_xyz


class FramePoint(Point):
    """
    Describes point in a keyframe that has UV coordinates, inv depth and other properties
    """

    def __init__(self,
                 coord_u: float,
                 coord_v: float,
                 inv_depth: float,
                 inv_depth_hessian: float,
                 max_rel_baseline: float,
                 num_good_res: float,
                 status: float,
                 neighb_intensity_info: Union[List[float], np.ndarray]):
        """

        :param coord_u:
        :param coord_v:
        :param inv_depth:
        :param inv_depth_hessian:
        :param max_rel_baseline:
        :param num_good_res:
        :param status:
        :param neighb_intensity_info:
        """
        assert coord_u >= 0  # in pixels starting from 0, hence no negative
        assert coord_v >= 0  # in pixels starting from 0, hence no negative
        assert len(neighb_intensity_info) >= 5
        # assert inv_depth > 0, 'Depth cannot be negative, because otherwise 3D coordinates cannot be computed.'
        # TODO double check if we allow inv_depth=0?

        self.coord_u = coord_u
        self.coord_v = coord_v
        self.inv_depth = inv_depth
        self.inv_depth_hessian = inv_depth_hessian
        self.max_rel_baseline = max_rel_baseline
        self.num_good_res = num_good_res
        self.status = status
        self.neighb_intensity_info = neighb_intensity_info

        self.intensity = neighb_intensity_info[4]  # intensity information of current point being point 5 of 8

        super(FramePoint, self).__init__([0, 0, 0], self.intensity)

    def get_array_of_fields(self) -> List[float]:
        array_fields = [self.coord_u, self.coord_v,
                        self.inv_depth, self.inv_depth_hessian,
                        self.max_rel_baseline, self.num_good_res,
                        self.status]
        array_fields.extend(self.neighb_intensity_info)
        array_fields.append(self.intensity)
        assert len(array_fields) == 16
        return array_fields

    @staticmethod
    def get_field_order() -> List[str]:
        """
        :return: list of strings containing the order of fields for this structure
        """
        fields = ['coord_u', 'coord_v',
                  'inv_depth', 'inv_depth_hessian',
                  'max_rel_baseline', 'num_good_res',
                  'status',
                  'neighb_intensity_info_0',
                  'neighb_intensity_info_1',
                  'neighb_intensity_info_2',
                  'neighb_intensity_info_3',
                  'neighb_intensity_info_4',
                  'neighb_intensity_info_5',
                  'neighb_intensity_info_6',
                  'neighb_intensity_info_7',
                  'intensity']
        return fields

    def coord_xyz_hom(self, calib_matrix: np.ndarray) -> np.ndarray:
        """
        Get homogenous coordinates of the point given calibration matrix

        :param calib_matrix: calibration matrix 3x3
        :return: homogenous coordinates 4x1
        """
        assert(calib_matrix.shape == (3, 3))

        coord_xyz = np.dot(np.linalg.inv(calib_matrix), np.array([self.coord_u, self.coord_v, 1.0]) / self.inv_depth)

        # compute 3D position
        coord_xyz_homogenous = np.append(coord_xyz, 1.0)
        return coord_xyz_homogenous

    @classmethod
    def init_from_lines(cls, lines: List[str]) -> 'FramePoint':
        """

        :param lines: list of strings
        :return: frame point
        """
        assert(len(lines) == 2)
        point_info = list(map(float, lines[0].strip().split(',')))
        assert len(point_info) > 6

        coord_u = point_info[0]
        coord_v = point_info[1]
        inv_depth = point_info[2]
        inv_depth_hessian = point_info[3]
        max_rel_baseline = point_info[4]
        num_good_res = point_info[5]
        status = point_info[6]

        intensity_strs = lines[1].strip().split(',')
        neighb_intensity_info = list(map(int, intensity_strs[:-1]))
        assert(len(neighb_intensity_info) == 8)

        return cls(coord_u, coord_v, inv_depth, inv_depth_hessian,
                   max_rel_baseline, num_good_res, status, neighb_intensity_info)

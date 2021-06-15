#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dedicated to coordinate system transformations.
# Author Nicholas Gao, Thomas Schmid, 2019

from enum import Enum, unique
import numpy as np
from libartipy.geometry import Pose, Quaternion
from typing import Callable, Union, Optional


@unique
class CoordinateSystem(Enum):
    """
    The index order here must stay fixed. It represents the transformation order to get from SLAM world to WGS84.
    If the order is changed the implementations in transformation_frame.py must be changed as well.(Just don't do it ;))
    """
    SLAM_WORLD = 0  # directly output of visual SLAM system
    WORLD = 1  # includes visual scaling (transform_S_AS) that is based on stereo data (no GPS)
    ENU = 2  # https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates#Local_east,_north,_up_(ENU)_coordinates
    ECEF = 3  # https://en.wikipedia.org/wiki/ECEF,
    # geographic and Cartesian coordinate system. It represents positions as X, Y, and Z coordinates.
    WGS84 = 4  # https://en.wikipedia.org/wiki/World_Geodetic_System#A_new_World_Geodetic_System:_WGS_84
    CAMERA = 5  # for camera local coordinate system, do not use for conversion!

    def __str__(self):
        if self == CoordinateSystem.SLAM_WORLD:
            return 'SLAM_world'
        if self == CoordinateSystem.WORLD:
            return 'world'
        if self == CoordinateSystem.ENU:
            return 'ENU'
        if self == CoordinateSystem.ECEF:
            return 'ECEF'
        if self == CoordinateSystem.WGS84:
            return 'WGS84'
        if self == CoordinateSystem.CAMERA:
            return 'Camera'
        raise ValueError()

    @staticmethod
    def from_string(cs: str) -> 'CoordinateSystem':
        cs = cs.lower()
        if cs in ['slam', 'slam_world']:
            return CoordinateSystem.SLAM_WORLD
        if cs == 'world':
            return CoordinateSystem.WORLD
        if cs == 'enu':
            return CoordinateSystem.ENU
        if cs == 'ecef':
            return CoordinateSystem.ECEF
        if cs == 'wgs84':
            return CoordinateSystem.WGS84
        if cs == 'camera':
            return CoordinateSystem.CAMERA
        raise ValueError('Invalid format')


class CoordinateSystemConverter:
    def __init__(self, transformation: Union[np.ndarray, Callable], inverse: Optional[Callable] = None):
        """

        :param transformation:
        :param inverse:
        """
        self.trans_func = None
        self.trans_mat = None

        if callable(transformation):
            self.trans_func = transformation
        if isinstance(transformation, Pose):
            self.trans_mat = transformation

        self.inverse = inverse

        assert self.trans_func is not None or self.trans_mat is not None
        assert self.trans_func is None or self.inverse is not None

    def transform(self,
                  xyz: np.ndarray,
                  reverse: bool = False,
                  transf_extra: np.ndarray = None) -> np.ndarray:
        """

        :param xyz: original points Nx3
        :param reverse: if True, perform reverse transformation (if transf_extra is given)
        :param transf_extra: apply the extra transformation, on top of the available one in
        the class (self.trans_mat or self.trans_func)
        :raises AssertionError if no transformation available
        :return: transformed points Nx3
        """
        assert xyz.shape[1] == 3

        if transf_extra is not None:
            assert transf_extra.shape == (4, 4)

        if self.trans_mat is not None:
            points_h = np.ones((xyz.shape[0], 4), dtype=np.float64)
            points_h[:, :3] = xyz

            if reverse:
                mat = self.trans_mat.inverse().transformation_matrix
            else:
                mat = self.trans_mat.transformation_matrix

            if transf_extra is not None:
                mat = mat.dot(transf_extra)

            return mat.dot(points_h.T).T[:, :3]

        if self.trans_func is not None:
            if reverse:
                assert self.inverse is not None
                return self.inverse(xyz)
            else:
                return self.trans_func(xyz)

        assert False

    def rotate_quaternion(self,
                          quaternion: Quaternion,
                          reverse: bool = False,
                          transf_extra: np.ndarray = None) -> Quaternion:
        """
        Rotates a quaternion using the set transformation matrix.
        :param quaternion: original quaternion
        :param reverse: if True, perform reverse transformation (if transf_extra is given)
        :param transf_extra: apply extra transformation on top of the available one in
        the class (self.trans_mat or self.trans_func)
        :return: rotated Quaternion
        """
        return Quaternion.from_rotation_matrix(self.rotate_orientation_by_rot_mat(quaternion.get_rotation_matrix(), reverse, transf_extra))

    def rotate_orientation_by_rot_mat(self, orientation_rot_mat: np.ndarray, reverse: bool = False, transf_extra: np.ndarray = None) -> np.ndarray:
        """
        Rotates a matrix using the set transformation matrix.
        :param orientation_rot_mat: 3x3
        :param reverse: if True, perform reverse transformation (if transf_extra is given)
        :param transf_extra: apply extra transformation on top of the available one in
        the class (self.trans_mat or self.trans_func)
        :return: rotation matrix 3x3
        """
        assert self.trans_mat is not None, "Rotating quaternions is only possible with matrix-based converter."

        assert orientation_rot_mat.shape[0:2] == (3, 3)

        if not reverse:
            transform = self.trans_mat
        else:
            transform = self.trans_mat.inverse()

        transform_rot_mat = transform.transformation_matrix[:3, :3]

        assert Quaternion.is_matrix_unitary(transform_rot_mat)

        if transf_extra is not None:
            transform_rot_mat = transform_rot_mat.dot(transf_extra)

        result = transform_rot_mat.dot(orientation_rot_mat).astype(np.float64)
        return result

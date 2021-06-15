#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dedicated to pose .
# Author Dmytro Bobkov, 2019
import numpy as np

import transforms3d

from libartipy.geometry import Quaternion
from libartipy.dataset import Constants
from typing import Union, List


class Pose(object):

    """
    Is a wrapper class that contains 6D pose (rotation quaternion, translation)
    and provides convenience functions to work with
    poses in a simple manner, like chaining transformations etc.
    """

    def __init__(self,
                 rotation: Quaternion = None,
                 translation: np.ndarray = None,
                 use_scale_in_rotation: bool = False):
        self.use_scale_in_rotation = use_scale_in_rotation
        # to keep track of SIM3 transformation which happens in one case for ASLAM output
        self.rotation_scale = 1

        self.rotation_quaternion = rotation if rotation is not None else Quaternion.init_identity()
        self.translation = translation if translation is not None else np.array([
                                                                                0., 0., 0.])

    def valid(self) -> bool:
        """
        Is this is a valid pose?
        :return: True if pose has been set, False otherwise
        """
        return self._rotation_quaternion is not None and self._translation is not None

    def set_use_scale_in_rotation(self, use_scale_in_rotation: bool):
        self.use_scale_in_rotation = use_scale_in_rotation

    @staticmethod
    def hat(so3: np.array) -> np.ndarray:
        """
        SO3 hat operator (skew-symmetric matrix)
        :param so3: so3 tangent 3-vector (rotation)
        :return: 3x3 skew-symmetric matrix
        """
        assert(so3.size == 3)
        skew_matrix = np.zeros((3, 3), dtype=float)
        skew_matrix[2, 1] = so3[0]
        skew_matrix[1, 2] = -so3[0]
        skew_matrix[0, 2] = so3[1]
        skew_matrix[2, 0] = -so3[1]
        skew_matrix[1, 0] = so3[2]
        skew_matrix[0, 1] = -so3[2]
        return skew_matrix

    @classmethod
    def identity(cls) -> 'Pose':
        return Pose(Quaternion.init_identity(), np.array([0., 0., 0.]))

    @classmethod
    def from_line(cls, line: str) -> 'Pose':
        line_parts = line.split(',')
        assert len(
            line_parts) >= 7, 'Not right number of parts in {}'.format(line)
        pos_x, pos_y, pos_z = map(float, line_parts[0:3])
        quat_x, quat_y, quat_z, quat_w = map(
            np.float64, line_parts[3:7])  # Note: XYZW order!

        a = cls(use_scale_in_rotation=True)
        a.rotation_quaternion = Quaternion([quat_w, quat_x, quat_y, quat_z])
        a.translation = [pos_x, pos_y, pos_z]
        return a

    @classmethod
    def from_transformation_matrix(cls, transformation_mat: np.array):

        a = cls(use_scale_in_rotation=True)
        a.rotation_quaternion = Quaternion.from_rotation_matrix(
            transformation_mat[:3, :3])
        a.translation = transformation_mat[:3, 3]

        return a

    @classmethod
    def exp(cls, tangent: np.array) -> 'Pose':
        """
        SE3 exponential map
        http://ethaneade.com/lie.pdf
        :param tangent: se3 tangent 6-vector [t, w] (translation + rotation)
        :return: exp(tangent)
        """
        assert len(tangent) == 6

        t = tangent[:3]
        w = tangent[3:]
        theta = np.linalg.norm(w)
        so3_exp = np.eye(3, dtype=float)
        v_matrix = np.eye(3, dtype=float)

        if theta > np.finfo(dtype=float).eps:
            omega_hat = cls.hat(w)
            omega_hat_sq = np.matmul(omega_hat, omega_hat)
            theta_sq = theta**2
            sin_theta = np.sin(theta)
            common_term = (1 - np.cos(theta)) / theta_sq
            so3_exp += sin_theta / theta * omega_hat + \
                common_term * omega_hat_sq
            v_matrix += common_term * omega_hat + \
                (theta - sin_theta) / (theta_sq * theta) * omega_hat_sq

        transformation_matrix = np.eye(4, dtype=float)
        transformation_matrix[:3, :3] = so3_exp
        transformation_matrix[:3, 3] = np.matmul(v_matrix, t)

        a = cls.from_transformation_matrix(transformation_matrix)

        # valid only for se3
        # may be extended for sim3 if necessary
        a.set_use_scale_in_rotation(False)
        return a

    @property
    def rotation_scale(self) -> float:
        return self._rotation_scale

    @rotation_scale.setter
    def rotation_scale(self, scale: float):
        self._rotation_scale = scale

    @property
    def translation(self) -> np.ndarray:
        """

        :return: numpy array with XYZ
        """
        assert self._translation is not None
        assert self._translation.size == 3
        return self._translation

    @translation.setter
    def translation(self, translation_: np.ndarray):
        if type(translation_) is not np.ndarray:
            translation_ = np.array(translation_)  # shape: (3,)

        self._translation = translation_.astype(dtype=np.float64)
        assert(self._translation.size == 3)  # 3D world

    @property
    def rotation_quaternion(self) -> Quaternion:
        """
        :return: Order WXYZ
        """
        assert(self._rotation_quaternion is not None)
        return self._rotation_quaternion

    @rotation_quaternion.setter
    def rotation_quaternion(self, quaternion_wxyz: Union[List, Quaternion, np.ndarray]):
        """
        :param quaternion_wxyz: Order WXYZ
        :return:
        """
        if isinstance(quaternion_wxyz, Quaternion):
            quaternion_wxyz = quaternion_wxyz.to_numpy()
        elif type(quaternion_wxyz) is not np.ndarray:
            quaternion_wxyz = np.array(quaternion_wxyz)

        self._rotation_quaternion = Quaternion(
            quaternion_wxyz.astype(dtype=Constants.artifloat_type))

        quaternion_norm = self._rotation_quaternion.quat_norm()
        assert np.isfinite(quaternion_norm)

        if self.use_scale_in_rotation:
            self.rotation_scale = quaternion_norm
        self._rotation_quaternion.normalize()
        assert (self._rotation_quaternion.quat_norm()-1.0) < Quaternion.EPS

    def rotation_from_rotation_matrix(self, rotation_matrix: np.ndarray):
        """

        :param rotation_matrix: 3x3
        :return:
        """
        assert np.allclose(np.eye(rotation_matrix.shape[0]), rotation_matrix.dot(rotation_matrix.T.conj())), \
            'Rotation matrix not unitary'
        quat = transforms3d.quaternions.mat2quat(rotation_matrix)
        self._rotation_quaternion = Quaternion(quat)

    @property
    def transformation_matrix(self) -> np.ndarray:
        """
        Init class from transformation matrix 4x4

        :return: 4x4
        """

        rot_matrix_4x4 = self.homogenous_rotation_matrix
        rot_matrix_4x4[0:3, 3] = self.translation
        rot_matrix_4x4[3, 0:3] = 0  # make affine
        assert rot_matrix_4x4[3, 3] == 1
        return rot_matrix_4x4

    @transformation_matrix.setter
    def transformation_matrix(self, transformation_matrix: np.array):
        """
        :param transformation_matrix

        """
        assert isinstance(transformation_matrix, np.ndarray)
        self._translation = transformation_matrix[:3, 3]
        self._rotation_quaternion = self.rotation_from_rotation_matrix(
            transformation_matrix[:3, :3]).rotation

    @property
    def homogenous_rotation_matrix(self) -> np.ndarray:
        """
        Translation not applied
        :return: 4x4
        """
        rot_matrix_4x4 = np.eye(4, dtype=Constants.artifloat_type)
        rot_matrix_4x4[0:3, 0:3] = self.rotation_matrix
        assert rot_matrix_4x4[3, 3] == 1
        return rot_matrix_4x4

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        :return: 3x3
        """
        rot_mat3x3 = self.rotation_quaternion.get_rotation_matrix()
        return rot_mat3x3

    @property
    def rotation_matrix_scaled(self) -> np.ndarray:
        """
        :return: 3x3
        """
        rot_mat3x3 = self.rotation_matrix

        if self.use_scale_in_rotation:
            rot_mat3x3 *= self.rotation_scale  # [s*R]
        return rot_mat3x3

    @property
    def euler_angles(self) -> List[float]:
        """

        :return: in radians
        """
        return self.rotation_quaternion.get_euler_angles()

    def inverse(self) -> 'Pose':
        """
        Implements inverse of the pose

        :return: pose inverse
        """
        # R = R^T
        rotat_quaternion_new = self.rotation_quaternion.inverse()

        # t = -R^T * t
        r_mult = -rotat_quaternion_new.get_rotation_matrix()
        t = np.matmul(r_mult, np.array(self.translation))

        pose_new = Pose(rotat_quaternion_new, t)
        return pose_new

    def log(self) -> np.ndarray:
        """
        SE3 logarithm map
        http://ethaneade.com/lie.pdf
        :return: se3 tangent 6-vector [t, w] (translation + rotation)
        """
        # valid only for se3
        # may be extended for sim3 if necessary
        assert not self.use_scale_in_rotation, "Log supported only for SE3 pose (no scale)"

        theta = np.arccos((self.rotation_matrix.trace() - 1) * 0.5)
        w = np.zeros(3, dtype=float)
        v_matrix_inv = np.eye(3, dtype=float)

        if theta > np.finfo(dtype=float).eps:
            theta_sq = theta**2
            a_term = np.sin(theta) / theta
            b_term = (1 - np.cos(theta)) / theta_sq
            rot_mat = self.rotation_matrix
            w = 0.5 / a_term * np.array([rot_mat[2, 1] - rot_mat[1, 2], rot_mat[0, 2] - rot_mat[2, 0],
                                         rot_mat[1, 0] - rot_mat[0, 1]])
            omega_hat = self.hat(w)
            omega_hat_sq = np.matmul(omega_hat, omega_hat)
            v_matrix_inv += (-0.5 * omega_hat + (1 - 0.5 *
                                                 a_term / b_term) * omega_hat_sq / theta_sq)

        t = np.matmul(v_matrix_inv, self.translation)
        return np.concatenate((t, w), axis=None)

    def transform_point3d(self, point3d: Union[List[float], np.ndarray]) -> np.ndarray:
        """

        :param point3d:
        :return:
        """
        if type(point3d) is not np.ndarray:
            point3d = np.array(point3d)

        assert point3d.size == 3

        t_new = np.matmul(self.rotation_matrix, point3d) + self._translation
        return t_new

    def transform_points3d(self, points3d: Union[List[float], np.ndarray]) -> np.ndarray:
        """

        :param points3d: Nx4
        :return: Nx4
        """
        if type(points3d) is not np.ndarray:
            points3d = np.array(points3d)

        assert points3d.shape[1] == 4

        t_new = np.matmul(self.transformation_matrix, points3d.T).T
        return t_new

    def transform_points3d_hom(self, points3d_hom: Union[List[float], np.ndarray]) -> np.ndarray:
        """

        :param points3d_hom: Nx4
        :return: Nx4
        """
        if type(points3d_hom) is not np.ndarray:
            points3d_hom = np.array(points3d_hom)

        assert points3d_hom.shape[1] == 4
        return np.dot(self.transformation_matrix, points3d_hom.T).T

    def deep_copy(self) -> 'Pose':
        """

        :return: deep copy of the object
        """
        import copy

        a = self.__class__()  # Is this the best way?
        tran = copy.deepcopy(self.translation)
        quat = Quaternion(copy.deepcopy(self.rotation_quaternion.to_numpy()))

        a.translation = tran
        a.rotation_quaternion = quat
        return a

    def __str__(self) -> str:
        """
        Serialize to obtain one line that can be used to write to file

        :return: string representation
        """
        delimiter = ','
        wxyz = self.rotation_quaternion.to_numpy()
        xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
        line_pose = delimiter.join([str(x) for x in self.translation]) + delimiter + \
            delimiter.join([str(x) for x in xyzw])
        return line_pose

    def __eq__(self, other: 'Pose') -> bool:
        """

        :param other: pose
        :return: true if equal
        """
        if isinstance(self, other.__class__):
            return self.rotation_quaternion == other.rotation_quaternion and \
                np.allclose(self.translation, other.translation)
        return False

    def __mul__(self, other: 'Pose') -> 'Pose':
        """
        Multiply self with other, e.g., pose_this * pose_other
        * operator reloading for chain of transformations

        :param other: pose
        :return: new pose
        """
        a = self.__class__()  # Is this the best way?
        tran = self.translation + \
            self.rotation_quaternion.transform_point3d(other.translation)

        result_quat = Quaternion(
            (self.rotation_quaternion * other.rotation_quaternion).to_numpy())
        result_quat.normalize()

        a.translation = tran
        a.rotation_quaternion = result_quat

        return a

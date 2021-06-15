#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dedicated to quaternion.
# Author Dmytro Bobkov, 2019

import numpy as np

import unittest

import transforms3d
from typing import Union, List
from libartipy.dataset import Constants


class Quaternion:
    EPS = np.finfo(float).eps * 4.0  # epsilon for testing whether a number is close to zero,
    # TODO(Dmytro) but why 4?, should not it be Constants.artifloat_type ?

    """Class for unit-norm quaternion. Scaling factor is ignored and quaternion is stored unit normalized."""

    def __init__(self, wxyz: Union[List[float], np.ndarray]):
        """
        :param wxyz: array of 4 values (list or numpy array)
        """
        if type(wxyz) is not np.ndarray:
            wxyz = np.array(wxyz, dtype=Constants.artifloat_type)
        else:
            # cast to this type to make sure
            wxyz = wxyz.astype(Constants.artifloat_type)

        assert wxyz.size == 4, 'Values {} are not of correct size'.format(wxyz.shape)
        assert np.isfinite(wxyz).all(), 'Values {} are not finite'.format(wxyz)
        assert np.any(wxyz != 0), 'Values {} are all zeros'.format(wxyz)
        self.w, self.x, self.y, self.z = wxyz[0:4]

    def to_numpy(self) -> np.ndarray:
        """

        :return: numpy array of size 4
        """
        return np.array([self.w, self.x, self.y, self.z], dtype=Constants.artifloat_type)

    @classmethod
    def init_identity(cls) -> 'Quaternion':
        """
        Get identity quaternion, coresponding to no rotation
        :return:
        """
        return cls([1., 0., 0., 0.])

    def quat_norm(self) -> float:
        """

        :return: Norm of the quaternion
        """
        return np.linalg.norm(self.to_numpy(), 2)

    def normalize(self) -> None:
        """
        In place normalization of quaternion

        :return:
        """
        quat_norm_curr = self.quat_norm()
        self.w = self.w / quat_norm_curr
        self.x = self.x / quat_norm_curr
        self.y = self.y / quat_norm_curr
        self.z = self.z / quat_norm_curr
        assert abs(self.quat_norm()-1.0) < Quaternion.EPS

    def inverse(self) -> 'Quaternion':
        """
        Compute inverse of the quaternion

        :return: inverse quaternion
        """
        quat_norm_sq_curr = self.quat_norm()**2
        w = self.w / quat_norm_sq_curr
        xyz = [-self.x / quat_norm_sq_curr,
               -self.y / quat_norm_sq_curr,
               -self.z / quat_norm_sq_curr]
        return Quaternion([w, xyz[0], xyz[1], xyz[2]])

    def vec(self) -> np.ndarray:
        """
        Get vector representation of the rotation vector XYZ axes

        :return: 3d vector
        """
        return np.array([self.x, self.y, self.z], dtype=Constants.artifloat_type)

    def conj(self) -> 'Quaternion':
        """
        Conjugate quaternion

        :return: conjugated quaternion
        """
        return Quaternion([self.w, -self.x, -self.y, -self.z])

    @staticmethod
    def is_matrix_unitary(matrix: np.ndarray) -> bool:
        """

        :param matrix: square matrix to verify
        :return: true if unitary, false otherwise
        """
        return np.allclose(np.eye(len(matrix)),
                           matrix.dot(matrix.T.conj()),
                           rtol=1.e-18, atol=1.e-18)

    def get_euler_angles(self) -> List[float]:
        """

        :return: euler angles in radians
        """
        return list(transforms3d.euler.quat2euler(self.to_numpy(), axes='sxyz'))

    def get_rotation_matrix(self) -> np.ndarray:
        """

        :return: rotation matrix 3x3
        """
        self.normalize()

        rot_matrix_3x3 = transforms3d.quaternions.quat2mat(self.to_numpy()).astype(dtype=Constants.artifloat_type)
        assert rot_matrix_3x3.shape == (3, 3), 'Not 3x3 matrix {}'.format(rot_matrix_3x3.shape)

        if not self.is_matrix_unitary(rot_matrix_3x3):
            # orthonormalize rotation matrix
            x = rot_matrix_3x3[0, :]
            y = rot_matrix_3x3[1, :]
            z = rot_matrix_3x3[2, :]

            error = x.dot(y)
            x_ort = x - (error / 2.) * y
            y_ort = y - (error / 2.) * x
            z_ort = np.cross(x_ort, y_ort)

            x_new = 1. / 2 * (3. - np.dot(x_ort, x_ort)) * x_ort
            y_new = 1. / 2 * (3. - np.dot(y_ort, y_ort)) * y_ort
            z_new = 1. / 2 * (3. - np.dot(z_ort, z_ort)) * z_ort

            rot_matrix_3x3 = np.vstack((x_new, y_new, z_new))

        assert self.is_matrix_unitary(rot_matrix_3x3), 'Not unitary'

        return rot_matrix_3x3

    def get_hom_rot_matrix(self) -> np.ndarray:
        """
        this function returns a homogeneous rotation matrix from a quaternion

        :return: homogenous rotation matrix of size 4x4, translation not used!
        """

        rot_mat_3x3 = self.get_rotation_matrix()
        rot_mat_homog_4x4 = np.identity(4)
        rot_mat_homog_4x4[0:3, 0:3] = rot_mat_3x3
        return rot_mat_homog_4x4

    def transform_point3d(self, point3d: np.ndarray) -> np.ndarray:
        """
        Transforms the 3d point to the given camera coordinate system

        Inspired by Eigen quaternion class, https://eigen.tuxfamily.org/dox/Quaternion_8h_source.html.

        :param point3d: point with 3 coordinates
        :return:
        """
        if type(point3d) is not np.ndarray:
            point3d = np.array(point3d, dtype=np.float64)  # For points we use float64 type

        uv = np.cross(self.vec(), point3d)
        uv += uv
        point3d_transformed = point3d + self.w * uv + np.cross(self.vec(), uv)
        return point3d_transformed

    def __str__(self) -> str:
        """
        Serialize to obtain one line that can be used to write to file

        :return:
        """
        delimiter = ','
        wxyz = self.to_numpy()
        xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
        line = delimiter.join(["{:.6f}".format(x) for x in xyzw])
        return line

    def __eq__(self, other: 'Quaternion') -> bool:
        if isinstance(self, other.__class__):
            # equivalent quaternions are q and -1*q
            q1 = self.to_numpy()
            q2 = other.to_numpy()

            q2_neg = q2 * -1
            cond = np.allclose(q1, q2) or np.allclose(q1, q2_neg)

            return cond
        return False

    def __mul__(self, right: 'Quaternion') -> 'Quaternion':
        """
        Multiplication operator for quaternions

        :param right: Quaternion object
        :return:
        """
        real = self.w * right.w - self.vec().dot(right.vec())
        xyz = self.w * right.vec() + right.w * self.vec() + np.cross(self.vec(), right.vec())
        return Quaternion([real, xyz[0], xyz[1], xyz[2]])

    @staticmethod
    def from_rotation_matrix(rot_mat: np.ndarray) -> 'Quaternion':
        """
        Init from rotation matrix

        :param rot_mat: 3x3
        :return:
        """
        assert rot_mat.shape == (3, 3)
        wxyz = transforms3d.quaternions.mat2quat(rot_mat)
        q = Quaternion(wxyz)
        return q

    @classmethod
    def from_xyzw_order(cls, q: np.ndarray) -> 'Quaternion':
        """
        Create quaternion from XYZW-order vector representation (SciPy convention).

        :param q: quaternion with XYZW elements order.
        :return:
        """
        wxyz = np.append(q[3], q[:3])
        return cls(wxyz=wxyz)

    def to_xyzw_order(self) -> np.ndarray:
        """
        Convert to XYZW quaternion representation (SciPy convention).

        :return: quaternion with XYZW elements order.
        """
        return np.array([self.x, self.y, self.z, self.w])

#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for working with transformtion frame
    Author Dmytro Bobkov, 2019
    Author Thomas Schmid
"""

import os
import linecache
import copy
from typing import Dict, Tuple

import numpy as np
import pymap3d as pm

from libartipy.dataset import get_logger
from libartipy.geometry import Pose, CoordinateSystem, CoordinateSystemConverter, Quaternion

logger = get_logger()


class TransformationFrame:

    def __init__(self, transformations: Dict[Tuple[CoordinateSystem, CoordinateSystem], CoordinateSystemConverter]):
        """

        :param transformations: dictionary [CoordinateSystem., CoordinateSystem.] -> Pose
        """
        self.transformations = transformations

    @staticmethod
    def init_from_file(transformations_filepath: str) -> 'TransformationFrame':
        """

        File contains lines:
        SLAM coordinate system

        :param transformations_filepath: 
        :return: 
        """
        transformations = {}

        try:
            if not os.path.exists(transformations_filepath):
                raise ValueError("Transformation file does not exist")

            offset = 2
            line_slam_scale_world = linecache.getline(transformations_filepath, offset)
            # includes visual scaling (transform_S_AS)
            pose_slam_scale_to_world = Pose.from_line(line_slam_scale_world)  # transform_S_AS
            # that is based on stereo data (no GPS)

            # Because new berlin data format has no empty lines between transformations, we need to handle this here
            line = linecache.getline(transformations_filepath, offset + 1)
            old_format_newline_between = line in ['\n', '\r\n']
            offset_between_lines = 3 if old_format_newline_between else 2

            offset += offset_between_lines
            line_imu_to_camera = linecache.getline(transformations_filepath, offset)
            pose_imu_to_camera = Pose.from_line(line_imu_to_camera)  # TS_cam_imu

            offset += offset_between_lines
            line_gps_world_to_world = linecache.getline(transformations_filepath, offset)  # transform_w_gpsw
            pose_world_to_gps_world = Pose.from_line(line_gps_world_to_world).inverse()

            offset += offset_between_lines
            line_imu_to_gps = linecache.getline(transformations_filepath, offset)
            pose_imu_to_gps = Pose.from_line(line_imu_to_gps)  # transform_gps_imu

            offset += offset_between_lines
            line_gps_world_to_earth = linecache.getline(transformations_filepath, offset)  # transform_e_gpsw
            pose_gps_world_to_earth = Pose.from_line(line_gps_world_to_earth)

            transformations[(CoordinateSystem.SLAM_WORLD, CoordinateSystem.WORLD)] = \
                CoordinateSystemConverter(pose_slam_scale_to_world)

            transformations[(CoordinateSystem.WORLD, CoordinateSystem.ENU)] = \
                CoordinateSystemConverter(pose_world_to_gps_world)

            transformations[(CoordinateSystem.ENU, CoordinateSystem.ECEF)] = \
                CoordinateSystemConverter(pose_gps_world_to_earth)

            transformations[(CoordinateSystem.ECEF, CoordinateSystem.WGS84)] = \
                CoordinateSystemConverter(TransformationFrame.transform_from_ECEF_to_WGS84,
                                          TransformationFrame.transform_from_WGS84_to_ECEF)
            a = TransformationFrame(transformations)
        except:
            logger.warning("Failed to read transformation frame from {}. Create identity transformation frame".
                           format(transformations_filepath))
            a = TransformationFrame.init_identity_transformation_frame()
        return a

    @staticmethod
    def init_identity_transformation_frame() -> 'TransformationFrame':
        transformations = {}
        pose_identity = Pose.identity()
        transformations[(CoordinateSystem.SLAM_WORLD, CoordinateSystem.WORLD)] = \
            CoordinateSystemConverter(pose_identity)

        transformations[(CoordinateSystem.WORLD, CoordinateSystem.ENU)] = \
            CoordinateSystemConverter(lambda _: TransformationFrame.raise_value_error('WORLD', 'ENU'),
                                      lambda _: TransformationFrame.raise_value_error('ENU', 'WORLD'))

        transformations[(CoordinateSystem.ENU, CoordinateSystem.ECEF)] = \
            CoordinateSystemConverter(lambda _: TransformationFrame.raise_value_error('ENU', 'ECEF'),
                                      TransformationFrame.transform_from_ECEF_to_ENU)

        transformations[(CoordinateSystem.ECEF, CoordinateSystem.WGS84)] = \
            CoordinateSystemConverter(TransformationFrame.transform_from_ECEF_to_WGS84,
                                      TransformationFrame.transform_from_WGS84_to_ECEF)

        a = TransformationFrame(transformations)
        return a

    @staticmethod
    def get_sim3_transform_mat_from_rot_scale(rotation_scale: float) -> np.ndarray:
        assert rotation_scale != 0
        transform = np.eye(4, dtype=np.float128)
        transform[0:3, 0:3] *= rotation_scale
        assert transform.shape == (4, 4)
        return transform

    def get_transform_list(self, idx_in_coord_sys: int, idx_out_coord_sys: int, rot_scale: float):
        assert idx_in_coord_sys >= CoordinateSystem.SLAM_WORLD.value, '{}'.format(idx_in_coord_sys)
        assert idx_out_coord_sys <= CoordinateSystem.ECEF.value, '{}'.format(idx_out_coord_sys)
        assert idx_in_coord_sys < idx_out_coord_sys

        transform_array = []
        for i in range(idx_in_coord_sys, idx_out_coord_sys, 1):
            j = i + 1

            assert j - i == 1, 'Cannot jump over transforms more than by 1!'

            pose = self.transformations[(CoordinateSystem(i), CoordinateSystem(j))].trans_mat
            transform_here = pose.transformation_matrix

            if i == 1 and j == i+1:  # WORLD TO ENU (or inverse)
                transf_extra = TransformationFrame.get_sim3_transform_mat_from_rot_scale(rot_scale)
                transform_array.append(transf_extra)

            transform_array.append(transform_here)
        return transform_array

    def transform_xyz_frame_specific(self,
                                     idx_in_coord_sys: int,
                                     idx_out_coord_sys: int,
                                     direction: int,
                                     xyz: np.ndarray,
                                     rotation_scales: np.ndarray) -> np.ndarray:
        """

        :param idx_in_coord_sys:
        :param idx_out_coord_sys:
        :param direction:
        :param xyz: Nx3
        :param rotation_scales: Nx1 extra transformation to be applied (GPS rotation scale uses this.)
        :return:
        """
        assert isinstance(rotation_scales, np.ndarray), 'Other type {}'.format(rotation_scales)

        assert rotation_scales.shape[0] == xyz.shape[0], '{} != {}'.format(rotation_scales.shape[0], xyz.shape[0])
        assert rotation_scales.shape[1] == 1, '{}'.format(rotation_scales.shape)
        assert xyz.shape[1] == 3, '{}'.format(xyz.shape)

        reverse = direction < 0
        un_rotation_scales = np.unique(rotation_scales)

        xyz_out = copy.deepcopy(xyz)  # make copy to keep this as intermediate data and not overwrite input data

        # Ok, two cases:
        # case 1: from WGS84 (idx_in_coord_sys==4)
        # transform all points to ECEF right away!
        # case 2: to WGS84 (idx_out_coord_sys==4)
        # first transform points to ECEF, then apply ECEF -> WGS84 transform (need to handle separately as WGS84
        # is function based trasnform, whereas others use tranformation matrix internally)

        if idx_in_coord_sys == CoordinateSystem.WGS84.value:
            xyz_out = self.transformations[(CoordinateSystem.WGS84, CoordinateSystem.ECEF)].transform(
                xyz_out,
                reverse=reverse,
                transf_extra=None)

        # Precompute chained transformations for all rotation scales
        indices_processed = set()  # keep track if we don't apply a transformation twice
        un_rotation_scales_proc = set()
        for ind_scale, rot_scale in enumerate(un_rotation_scales):
            indeces_points_here = np.where(rotation_scales == rot_scale)[0]

            for index in indeces_points_here:
                assert index not in indices_processed
                indices_processed.add(index)

            assert rot_scale not in un_rotation_scales_proc
            un_rotation_scales_proc.add(rot_scale)

            # now go through transformations themselves
            idx_in_sorted, idx_out_sorted = np.sort([idx_in_coord_sys, idx_out_coord_sys])
            idx_out_sorted = min(idx_out_sorted, CoordinateSystem.ECEF.value)

            transform_array = self.get_transform_list(idx_in_sorted, idx_out_sorted, rot_scale)

            # now unroll array of transformations
            curr_transform = np.eye(4, 4)
            for ind, tran in enumerate(transform_array):
                curr_transform = tran @ curr_transform

            if reverse:
                curr_transform = np.linalg.inv(curr_transform.astype(np.float64))

            # Now can transform the points!
            tmp = np.ones((indeces_points_here.shape[0], 4), dtype=np.float64)
            tmp[:, :3] = xyz_out[indeces_points_here, :]

            xyz_out[indeces_points_here, :] = curr_transform.dot(tmp.T).T[:, :3]

        # Time to handle case 2: to WGS84
        if idx_out_coord_sys == CoordinateSystem.WGS84.value:
            xyz_out = self.transformations[
                (CoordinateSystem.ECEF, CoordinateSystem.WGS84)].transform(
                xyz_out,
                reverse=False,
                transf_extra=None)

        return xyz_out

    def transform_xyz(self,
                      xyz: np.ndarray,
                      input_coordinate_system: CoordinateSystem,
                      output_coordinate_system: CoordinateSystem,
                      rotation_scales: np.ndarray = None) -> np.ndarray:
        """
        This method ingests XYZ points and transforms it from the input coordinate system to the output coordinate system
        :param xyz: [Nx3]
        :param input_coordinate_system:
        :param output_coordinate_system:
        :param rotation_scales extra transformation to be applied (GPS rotation scale uses this.)
        :return: [Nx3]
        """
        assert xyz.shape[1] == 3, '{}'.format(xyz.shape)

        idx_in_coord_sys = input_coordinate_system.value
        idx_out_coord_sys = output_coordinate_system.value

        assert output_coordinate_system != CoordinateSystem.CAMERA, "Transformation to camera is not supported."

        if idx_in_coord_sys == idx_out_coord_sys:
            return copy.deepcopy(xyz)

        direction = np.sign(idx_out_coord_sys - idx_in_coord_sys)

        if rotation_scales is not None:
            return self.transform_xyz_frame_specific(idx_in_coord_sys, idx_out_coord_sys, direction, xyz,
                                                     rotation_scales).astype(np.float64)
        else:
            xyz_tmp = copy.deepcopy(xyz)
            for i in range(idx_in_coord_sys, idx_out_coord_sys, direction):
                j = i + direction
                i, j = np.sort([i, j])

                assert abs(i - j) == 1, 'Cannot jump over transforms more than by 1!'

                xyz_tmp = self.transformations[(CoordinateSystem(i), CoordinateSystem(j))].transform(
                    xyz_tmp,
                    reverse=direction < 0,
                    transf_extra=None)

            return xyz_tmp.astype(np.float64)

    def rotate_quaternion_from_to_coord_system(self,
                                               quaternion: Quaternion,
                                               input_coordinate_system: CoordinateSystem,
                                               output_coordinate_system: CoordinateSystem,
                                               rotation_scale: float = None) -> Quaternion:
        """
        Rotates a given quaternion from the given coordinate system to the output coordinate system using set rotation
        matrices.
        :param quaternion:
        :param input_coordinate_system:
        :param output_coordinate_system:
        :param rotation_scale
        :return:
        """

        verbose = False
        if rotation_scale is not None and verbose:
            logger.warning("rotation scale to rotation is not applied!")

        idx_in_coord_sys = input_coordinate_system.value
        idx_out_coord_sys = output_coordinate_system.value

        if idx_in_coord_sys == idx_out_coord_sys:
            return quaternion

        assert idx_out_coord_sys < CoordinateSystem.WGS84.value, 'WGS84 and Camera are not supported.'

        direction = np.sign(idx_out_coord_sys - idx_in_coord_sys)
        reverse = direction < 0

        matrix = quaternion.get_rotation_matrix()

        for i in range(idx_in_coord_sys, idx_out_coord_sys, direction):
            j = i + direction
            i, j = np.sort([i, j])
            key = (CoordinateSystem(i), CoordinateSystem(j))
            assert key in self.transformations, "Transformation from %s to %s not available" % (key[0], key[1])
            matrix = self.transformations[key].rotate_orientation_by_rot_mat(matrix, reverse=reverse)

        quaternion = Quaternion.from_rotation_matrix(matrix)
        return quaternion

    @staticmethod
    def transform_from_ECEF_to_WGS84(xyz: np.ndarray) -> np.ndarray:
        lat, long, alt = pm.ecef2geodetic(xyz[:, 0], xyz[:, 1], xyz[:, 2], ell=pm.Ellipsoid('wgs84'))
        return np.vstack([lat, long, alt]).T

    @staticmethod
    def transform_from_WGS84_to_ECEF(wgs: np.ndarray) -> np.ndarray:
        x, y, z = pm.geodetic2ecef(wgs[:, 0], wgs[:, 1], wgs[:, 2], ell=pm.Ellipsoid('wgs84'))
        return np.vstack([x, y, z]).T

    def transform_from_ECEF_to_ENU(self, xyz: np.ndarray) -> np.ndarray:
        t = self.transformations[(CoordinateSystem.ENU, CoordinateSystem.ECEF)].trans_mat.translation
        lat, lon, h = pm.ecef2geodetic(t[0], t[1], t[2])
        return np.array(pm.ecef2enu(xyz[:, 0], xyz[:, 1], xyz[:, 2], lat, lon, h)).T

    @staticmethod
    def raise_value_error(origin: str, target: str) -> None:
        raise ValueError('Transformation from %s to %s has not been provided.' % (origin, target))

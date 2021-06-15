#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for camera operations
    Author Dmytro Bobkov, 2019
"""
from typing import Optional, Tuple
import numpy as np

from libartipy.dataset import get_logger
from libartipy.geometry import Pose, CoordinateSystem
logger = get_logger()


class Camera(object):
    def __init__(self,
                 pose: Pose,
                 calib_mat: np.ndarray,
                 image_width: int,
                 image_height: int,
                 translation_scale: float = None):
        """

        :param pose:
        :param calib_mat: 3x3
        :param image_width:
        :param image_height:
        """
        assert pose.valid(), 'Invalid pose'
        assert calib_mat.shape == (3, 3)
        assert image_height > 0
        assert image_width > 0

        self.pose = pose

        self.calib_mat = calib_mat  # 3x3
        self.image_size_wh = [int(image_width), int(image_height)]
        self.translation_scale = translation_scale

    def has_translation_scale(self) -> bool:
        return self.translation_scale is not None

    def get_translation_scale(self) -> float:
        assert self.has_translation_scale(), \
            'You have not called method dataset.set_keyframe_poses_to_gps_poses so that translation scale is set!'
        return self.translation_scale

    @property
    def pose(self) -> Pose:
        return self._pose

    @pose.setter
    def pose(self, pose: Pose):
        self._pose = pose  # camera to world transformation

    def downscale(self, scaling_factor: float):
        assert scaling_factor > 0, 'Scaling factor cannot be negative {}'.format(scaling_factor)

        # downscale all by scaling factor
        self.calib_mat[0, 0] /= scaling_factor
        self.calib_mat[1, 1] /= scaling_factor
        self.calib_mat[0, 2] /= scaling_factor
        self.calib_mat[1, 2] /= scaling_factor
        self.image_size_wh[0] /= scaling_factor
        self.image_size_wh[1] /= scaling_factor

    def project_points2d_to_3d(self,
                               points2d: np.ndarray,
                               inv_depth: np.ndarray,
                               coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD) -> Optional[np.ndarray]:
        """

        :param points2d: Nx2 [x,y]
        :param inv_depth: Nx1
        :param coordinate_system: coordinate system to project points to
        :return: points in coordinate system Nx3
        """
        assert coordinate_system in [CoordinateSystem.CAMERA, CoordinateSystem.SLAM_WORLD]
        assert points2d.shape[0] == inv_depth.shape[0]
        assert points2d.shape[1] == 2

        N = points2d.shape[0]
        ones = np.ones((N, 1))  # Nx1
        point_camera_coord_system = np.hstack((points2d, ones))  # Nx3
        point_camera_coord_system = np.divide(point_camera_coord_system, inv_depth[:, None])  # Nx3
        coord_xyz = np.dot(np.linalg.inv(self.calib_mat), point_camera_coord_system.T).T  # Nx3
        assert coord_xyz.shape[1] == 3

        if coordinate_system == CoordinateSystem.CAMERA:
            return coord_xyz
        elif coordinate_system == CoordinateSystem.SLAM_WORLD:

            # compute 3D position
            coord_xyz_homogenous = np.hstack((coord_xyz, ones))  # Nx4
            coord_xyz_world = self.pose.transform_points3d_hom(coord_xyz_homogenous)[:, :3]
            assert coord_xyz_world.shape[1] == 3
            return coord_xyz_world
        else:
            logger.warning("Coordinate system not implemented {}".format(coordinate_system))

    def project_points3d_to_points2d(self, points3d: np.ndarray) -> np.ndarray:
        """

        :param points3d: Nx3 of type float, assumed to be in world coordinate system
        :return: Nx2 of type float (convert to int yourself)
        """
        # TODO debug and test
        assert points3d.shape[1] == 3

        logger.warning("Function project_points3d_to_points2d(..) is deprecated and will be replaced with "
                       "XYZ in the next iteration!")

        ones = np.ones((points3d.shape[0], 1))  # Nx1
        points3d_hom = np.hstack((points3d, ones))  # Nx4

        coord_cam_coord = self.pose.inverse().transform_points3d(points3d_hom)[:, :3]  # Nx3
        # remove points behind the camera already
        coord_cam_coord = coord_cam_coord[coord_cam_coord[:, 2] > 0, :]
        coord_cam_coord /= coord_cam_coord[:, 2][:, None]

        coord_2d = np.matmul(self.calib_mat, coord_cam_coord.T).T  # Nx3

        mask = np.ones(coord_2d.shape[0], dtype=np.bool)
        # check if pixels inside image plane
        mask = np.logical_and(mask, coord_2d[:, 0] < self.image_size_wh[0])
        mask = np.logical_and(mask, coord_2d[:, 1] < self.image_size_wh[1])
        mask = np.logical_and(mask, coord_2d[:, 0] >= 0)
        mask = np.logical_and(mask, coord_2d[:, 1] >= 0)
        coord_2d = coord_2d[mask, :]
        return coord_2d[:, 0:2]  # Nx2

    def provide_canvas_mask(self, coords_2d: np.ndarray) -> np.ndarray:
        """
        returns a canvas mask indicating points that are on the actual image canvas

        :param coords_2d: Nx2
        :return:
        """

        mask = np.ones(coords_2d.shape[0], dtype=np.bool)

        # check if pixels inside image plane
        mask = np.logical_and(mask, coords_2d[:, 0] < self.image_size_wh[0])
        mask = np.logical_and(mask, coords_2d[:, 1] < self.image_size_wh[1])
        mask = np.logical_and(mask, coords_2d[:, 0] >= 0)
        mask = np.logical_and(mask, coords_2d[:, 1] >= 0)

        return mask

    def transform_points3d_to_image_canvas(self, points3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        transforms points from 3D slam coordinate system into image coordinates and returns
        these together with the depth values in camera coordinate system and a combined
        mask that reflects all intermediate masking steps.

        :param points3d: Nx3
        :return:
        """

        assert points3d.shape[1] == 3
        num_points3d = points3d.shape[0]
        ones = np.ones((num_points3d, 1))  # Nx1
        points3d_hom = np.hstack((points3d, ones))  # Nx4

        # project points from slam coordinate system to camera coordinate system
        coord_cam_coord = self.pose.inverse().transform_points3d(points3d_hom)[:, :3]  # Nx3

        # generate mask for positive depth values (points in front of camera), provide valid IDs
        # and filter out negative depth values
        mask_pos_cam_direction = coord_cam_coord[:, 2] > 0
        coord_cam_coord = coord_cam_coord[mask_pos_cam_direction, :]
        coord_cam_depth = coord_cam_coord[:, 2].copy()

        #  project camera coordinates to image plane
        coord_cam_coord /= coord_cam_depth[:, None]
        coord_2d = np.matmul(self.calib_mat, coord_cam_coord.T).T  # Nx3

        # generate canvas mask and filter out coordinates outside the image canvas
        mask_canvas = self.provide_canvas_mask(coord_2d)
        coord_2d = coord_2d[mask_canvas]
        coord_cam_depth = coord_cam_depth[mask_canvas]

        # combine all applied masks into one
        valid_ids = np.array(range(num_points3d))
        valid_ids = valid_ids[mask_pos_cam_direction]
        valid_ids = valid_ids[mask_canvas]

        # create boolean mask of size of input array indicating valid entries
        mask_overall = np.zeros(num_points3d, np.bool)
        mask_overall[[id for id in valid_ids]] = True

        # coord_2d: Mx2, coord_cam_Depth: Mx1, mask_overall: Nx1 (with N > M)
        return coord_2d[:, :2], coord_cam_depth, mask_overall

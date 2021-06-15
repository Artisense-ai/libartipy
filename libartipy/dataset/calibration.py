#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code for reading and working with calibration data.
    Author Thomas Schmid, 2019
    Author Dmytro Bobkov, 2019
"""

import cv2
import numpy as np
import os
from typing import Tuple, List
from enum import Enum
import yaml
import functools

from libartipy.dataset import Constants, get_logger
from libartipy.dataset import CameraType
logger = get_logger()


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        logger.warning('Call to deprecated function {}'.format(func.__name__))
        return func(*args, **kwargs)
    return new_func


class DistortionModel(Enum):
    Pinhole = 'Pinhole'
    Equidistant = 'Equidistant'

    @staticmethod
    def get_all_types() -> list:
        return [DistortionModel.Pinhole, DistortionModel.Equidistant]


LEFT_CAM_PROJ_MAT = "P1"
RIGHT_CAM_PROJ_MAT = "P2"
LEFT_CAM_ROT_MAT = "R1"
RIGHT_CAM_ROT_MAT = "R2"
DISP_DEPTH_MAP_MAT = "Q"


class CameraCalibration(object):
    """
    This class contains the information written in the calibration files.
    """

    def __init__(self, distortion_model: DistortionModel, calib_mat: np.ndarray,
                 distortion_params: List[float], img_dims: List[int],
                 cropped_img_dims: List[int]):
        """

        :param distortion_model:
        :param calib_mat: 3x3
        :param distortion_params: distortion parameters as list
        :param img_dims:
        :param cropped_img_dims:
        """
        assert calib_mat.shape == (3, 3)
        assert len(distortion_params) >= 4
        assert len(img_dims) == 2
        assert len(cropped_img_dims) == 2

        try:
            self.distortion_model = DistortionModel(distortion_model)
        except:
            logger.error('Distortion model not supported. Supported models: {}'.format(DistortionModel.get_all_types()))

        # parse calibration file
        self.calib_mat = calib_mat
        self.distortion_params = distortion_params
        self.img_dims = img_dims
        self.cropped_img_dims = cropped_img_dims


class CalibrationFactory(object):
    @staticmethod
    def get_single_cam_info(curr_cam_yaml) -> Tuple[DistortionModel, np.ndarray, np.ndarray, tuple]:
        """

        :param curr_cam_yaml:
        :return:
        """
        DIST_MODELS = {'equidistant': DistortionModel.Equidistant,
                       'pinhole': DistortionModel.Pinhole,
                       'none': DistortionModel.Pinhole}
        # 1. parse distortion model
        dist_model = DIST_MODELS[curr_cam_yaml['distortion_model'].lower()]

        # 2. parse intrinsics
        intrinsic_params = curr_cam_yaml['intrinsics']
        assert len(intrinsic_params) >= 4
        calib_mat = np.eye(3)
        calib_mat[0, 0] = intrinsic_params[0]
        calib_mat[1, 1] = intrinsic_params[1]
        calib_mat[0, 2] = intrinsic_params[2]
        calib_mat[1, 2] = intrinsic_params[3]

        # 3. parse distortion coeffs
        dist_params = curr_cam_yaml['distortion_coeffs']
        assert len(dist_params) >= 4
        dist_params = np.array(dist_params)

        # 4. parse image info, resolution
        resolution = curr_cam_yaml['resolution']
        assert len(resolution) == 2
        resolution = tuple(resolution)

        return dist_model, calib_mat, dist_params, resolution

    @staticmethod
    def get_stereo_cam_info(curr_cam_yaml: dict) -> np.ndarray:
        """
        Parse stereo transformation

        :param curr_cam_yaml:
        :return: 4x4
        """
        STEREO_FIELD = 'T_cn_cnm1'
        stereo_trans = []
        for field in curr_cam_yaml[STEREO_FIELD]:
            stereo_trans.append(field)

        stereo_trans = np.array(stereo_trans)
        assert stereo_trans.shape == (4, 4)
        return stereo_trans

    @staticmethod
    @deprecated
    def create_from_txt_files(calib_folder_path: str, distorted: bool, constants_calib: Constants) -> \
            Tuple[CameraCalibration, CameraCalibration, np.ndarray]:
        """
        Parse calibration from txt files.

        :param calib_folder_path:
        :param distorted:
        :param constants_calib:
        :return: Left camera calibration, right camera calibration and stereo calibration as transformation matrix
        """
        if distorted:
            calib_0_path = os.path.join(calib_folder_path, constants_calib.CALIBRATION_DIST_CAMERA_0)
            calib_1_path = os.path.join(calib_folder_path, constants_calib.CALIBRATION_DIST_CAMERA_1)
            calib_stereo_path = os.path.join(calib_folder_path, constants_calib.CALIBRATION_DIST_STEREO)
        else:
            calib_0_path = os.path.join(calib_folder_path, constants_calib.CALIBRATION_UNDIST_CAMERA_0)
            calib_1_path = os.path.join(calib_folder_path, constants_calib.CALIBRATION_UNDIST_CAMERA_1)
            calib_stereo_path = os.path.join(calib_folder_path, constants_calib.CALIBRATION_UNDIST_STEREO)

        assert os.path.exists(calib_0_path), 'Calibration file for left camera {} does not exist.'.format(calib_0_path)
        assert os.path.exists(calib_1_path), 'Calibration file for right camera{} does not exist.'.format(calib_1_path)
        assert os.path.exists(calib_stereo_path), 'Calibration for stereo {} does not exist.'.format(calib_stereo_path)

        calib_0 = CalibrationFactory.read_calibration_from_file(fpath=calib_0_path)
        calib_1 = CalibrationFactory.read_calibration_from_file(fpath=calib_1_path)
        calib_stereo_mat = np.loadtxt(fname=calib_stereo_path, delimiter=' ')
        assert calib_stereo_mat.shape == (4, 4)

        return calib_0, calib_1, calib_stereo_mat

    @staticmethod
    def create_from_yaml(calib_folder_path: str, distorted: bool, constants_calib: Constants) -> \
            Tuple[CameraCalibration, CameraCalibration, np.ndarray]:
        """
        Parse calibration from yaml files.

        :param calib_folder_path:
        :param distorted:
        :param constants_calib:
        :return: Left camera calibration, right camera calibration and stereo calibration as transformation matrix
        """
        assert distorted, 'So far undistorted not supported'
        yaml_file_name = os.path.join(calib_folder_path, constants_calib.CALIBRATION_YAML)

        assert os.path.exists(yaml_file_name), '{}'.format(yaml_file_name)

        with open(yaml_file_name, 'r') as yaml_file:
            # contain 2 cameras and extrinsic calibration
            yaml_entries = yaml.load(yaml_file, Loader=yaml.Loader)

        # hardcode 2 cameras
        assert 'cam0' in yaml_entries.keys() and 'cam1' in yaml_entries.keys(), '{}'.format(yaml_entries.keys())
        calibrations = []
        for ind in range(2):
            dist_model, calib_mat, dist_params, resolution = CalibrationFactory.get_single_cam_info(
                yaml_entries['cam' + str(ind)])
            camera = CameraCalibration(dist_model, calib_mat, dist_params, resolution, cropped_img_dims=resolution)
            calibrations.append(camera)
        # TODO(Dmytro) parse extrinsics into dictionary and provide generic getters
        # Note: we do not parse extrinscis to mount right now

        calib_stereo_mat = CalibrationFactory.get_stereo_cam_info(yaml_entries['cam1'])
        return calibrations[0], calibrations[1], calib_stereo_mat

    @staticmethod
    def read_calibration_from_file(fpath: str) -> CameraCalibration:
        """
        This method reads calibration files and extracts the relevant information.

        :param fpath: filepath to calibration information
        :return: camera calibration object
        """
        with open(fpath, 'r') as calib_file:
            lines = calib_file.readlines()

            # parse first line containing distortion mode, calibration matrix and distortion parameter
            distortion_model = lines[0].split()[0]
            fx, fy, cx, cy, k1, k2, k3, k4 = list(map(float, lines[0].split(' ')[1:]))

            # parse second line containing image dimensions
            img_width, img_height = list(map(int, lines[1].split(' ')))

            # parse fourth line containing image dimensions of cropped image
            c_img_width, c_img_height = list(map(int, lines[3].split(' ')))

        calib_mat = np.eye(3)
        calib_mat[0, 0] = fx
        calib_mat[1, 1] = fy
        calib_mat[0, 2] = cx
        calib_mat[1, 2] = cy

        distortion_params = np.array([k1, k2, k3, k4])

        img_dims = (img_width, img_height)
        cropped_img_dims = (c_img_width, c_img_height)

        return CameraCalibration(distortion_model, calib_mat, distortion_params, img_dims, cropped_img_dims)


class Calibration(object):
    """
    This class contains all calibration data captured for one sensor kit.
    """

    def __init__(self, calib_folder_path: str, distorted: bool = True):
        self.constants_calib = Constants()

        try:
            self.calib_0, self.calib_1, self.calib_stereo_mat = CalibrationFactory.create_from_yaml(
                calib_folder_path, distorted, self.constants_calib)
        except Exception as e:
            logger.warning("Did not succeed parsing from yaml file in {0} distorted {1} due to {2}".
                           format(calib_folder_path, distorted, str(e)))
            self.calib_0, self.calib_1, self.calib_stereo_mat = CalibrationFactory.create_from_txt_files(calib_folder_path, distorted,
                                                                                                         self.constants_calib)
        assert self.calib_stereo_mat is not None

        assert self.calib_0.distortion_model == self.calib_1.distortion_model, 'Camera distortion models differ!'
        assert self.calib_0.img_dims == self.calib_1.img_dims, 'Image dimensions differ!'

        self.img_dims = self.calib_0.img_dims

        self.distortion_model = self.calib_0.distortion_model
        assert self.distortion_model in [DistortionModel.Equidistant, DistortionModel.Pinhole],  \
            'Only {} model are implemented so far, not {}'.format([DistortionModel.Equidistant, DistortionModel.Pinhole],
                                                                  self.distortion_model)

        self.map0_x, self.map0_y = None, None
        self.map1_x, self.map1_y = None, None
        self.K0_optimized, self.K1_optimized = None, None
        self.rect_trans = None

        if self.distortion_model == DistortionModel.Equidistant:
            self._get_undistortion_to_distortion_map()
        elif self.distortion_model == DistortionModel.Pinhole:
            assert (self.calib_0.distortion_params == np.zeros(4)).all(), \
                'Calib1: Distortion parameters of Pinhole model are non-zero.'
            assert (self.calib_1.distortion_params == np.zeros(4)).all(), \
                'Calib2: Distortion parameters of Pinhole model are non-zero.'
            self._adapt_pinhole_parameters_to_equidistant_output()
        else:
            raise AssertionError

    def _get_undistortion_to_distortion_map(self) -> None:
        """
        This method performs stereo rectification and undistortion and calculates optimized calibration data as well as
        remaps that map from rectified to distorted image.

        :return:
        """

        # perform stereo rectification
        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#stereorectify
        # The function computes the rotation matrices for each camera that (virtually) make both camera image planes
        # the same plane. Consequently, this makes all the epipolar lines parallel and thus simplifies the dense
        # stereo correspondence problem. The function takes the matrices computed by stereoCalibrate() as input.
        # As output, it provides two rotation matrices and also two projection matrices in the new coordinates.
        # The function distinguishes the following two cases: Horizontal stereo and Vertical STereo
        # CALIB_ZERO_DISPARITY: the function makes the principal points of each camera have
        # the same pixel coordinates in the rectified views.
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K1=self.calib_0.calib_mat, D1=self.calib_0.distortion_params,
                                                      K2=self.calib_1.calib_mat, D2=self.calib_1.distortion_params,
                                                      imageSize=self.img_dims,
                                                      R=self.calib_stereo_mat[:3, :3],
                                                      tvec=self.calib_stereo_mat[:3, 3],
                                                      flags=cv2.CALIB_ZERO_DISPARITY)

        self.rect_trans = {LEFT_CAM_ROT_MAT: R1,  # Output 3x3 rectification transform (rotation matrix) for the first camera.
                           # Output 3x3 rectification transform (rotation matrix) for the second camera.
                           RIGHT_CAM_ROT_MAT: R2,
                           # Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera.
                           LEFT_CAM_PROJ_MAT: P1,
                           RIGHT_CAM_PROJ_MAT: P2,
                           DISP_DEPTH_MAP_MAT: Q}  # Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see reprojectImageTo3D )

        # generates maps from rectified image to distorted image
        self.map0_x, self.map0_y = cv2.fisheye.initUndistortRectifyMap(K=self.calib_0.calib_mat,
                                                                       D=self.calib_0.distortion_params,
                                                                       R=R1, P=P1, size=self.img_dims,
                                                                       m1type=cv2.CV_32FC1)

        self.map1_x, self.map1_y = cv2.fisheye.initUndistortRectifyMap(K=self.calib_1.calib_mat,
                                                                       D=self.calib_1.distortion_params,
                                                                       R=R2, P=P2, size=self.img_dims,
                                                                       m1type=cv2.CV_32FC1)

        self.K0_optimized = P1[:3, :3]
        self.K1_optimized = P2[:3, :3]

    def _adapt_pinhole_parameters_to_equidistant_output(self):

        self.rect_trans = dict()
        self.rect_trans[LEFT_CAM_PROJ_MAT] = np.zeros((3, 4))  # projection matrix of first camera
        # projection matrix of second camera (incl. baseline [in pixel] scaled by focal length)
        self.rect_trans[RIGHT_CAM_PROJ_MAT] = np.zeros((3, 4))
        self.rect_trans[LEFT_CAM_ROT_MAT] = np.eye(3)  # Identity when as no rectificatio necessary
        self.rect_trans[RIGHT_CAM_ROT_MAT] = np.eye(3)  # Identity when as no rectificatio necessary
        self.rect_trans[DISP_DEPTH_MAP_MAT] = np.eye(4)  # disparity to depth mapping, maps from ucd1 to XYZ1

        # update optimized K1 and K2
        self.K0_optimized = self.calib_0.calib_mat
        self.K1_optimized = self.calib_1.calib_mat
        focal_length_x = self.K0_optimized[0, 0]
        cx = self.K0_optimized[0, 2]
        cy = self.K0_optimized[1, 2]

        # update P1 and P2
        baseline = self.calib_stereo_mat[0, 3]
        self.rect_trans[LEFT_CAM_PROJ_MAT][:3, :3] = self.K0_optimized
        self.rect_trans[RIGHT_CAM_PROJ_MAT][:3, :3] = self.K1_optimized
        self.rect_trans[RIGHT_CAM_PROJ_MAT][0, 3] = self.K1_optimized[0, 0] * baseline

        # update Q: mapping from [u, v, disp, 1] to [X, Y, Z, 1]
        self.rect_trans[DISP_DEPTH_MAP_MAT][2, 2] = 0
        self.rect_trans[DISP_DEPTH_MAP_MAT][3, 3] = 0
        self.rect_trans[DISP_DEPTH_MAP_MAT][0, 3] = -cx
        self.rect_trans[DISP_DEPTH_MAP_MAT][1, 3] = -cy
        self.rect_trans[DISP_DEPTH_MAP_MAT][2, 3] = focal_length_x
        self.rect_trans[DISP_DEPTH_MAP_MAT][3, 2] = 1 / np.abs(baseline)

        # update maps 1 and 2 x and y
        self.map0_x = None
        self.map0_y = None
        self.map1_x = None
        self.map1_y = None


class DistortionMapper(object):
    """
    This function contains all calibration parameters and can transform images or pixel coordinates from rectified
    to distorted space.
    """

    def __init__(self, calib_data: Calibration):
        self.calib = calib_data
        assert self.calib.distortion_model == DistortionModel.Equidistant,\
            'DistorionMapper requires Equidistant distortion model.'

    def remap_rectified_image_to_distorted_image(self, rectified_image: np.ndarray,
                                                 camera_position: CameraType = CameraType.LEFT,
                                                 interpolation=None) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        This method remaps a rectified image to the corresponding distorted image.
        :param rectified_image:
        :param camera_position:
        :param interpolation:
        :return:
        """

        assert interpolation is None, 'No Interpolation implemented'

        # select correct image maps dependent on selected camera
        map_column = self.calib.map0_x if camera_position == CameraType.LEFT else self.calib.map1_x
        map_row = self.calib.map0_y if camera_position == CameraType.LEFT else self.calib.map1_y

        assert rectified_image.shape == map_column.shape, 'Image size and x/column-map size do not match.'
        assert rectified_image.shape == map_row.shape, 'Image size and y/row-map size do not match'

        # initialize distorted image with zeros
        distorted_image = np.zeros_like(rectified_image)

        # use zero image to retrieve a list of all pixel indices
        rectified_row_coords, rectified_column_coords = np.where(distorted_image == 0)

        # generate positions in distorted image using the precomputed maps
        rows, cols = self.remap_rectified_coordinates_to_distorted_coordinates(coords_row=rectified_row_coords,
                                                                               coords_column=rectified_column_coords,
                                                                               image_dims=rectified_image.shape,
                                                                               camera_position=camera_position)

        # check if transformed coordinates are still on canvas
        on_canvas_mask = (0 < rows) & (rows < rectified_image.shape[0]) & \
                         (0 < cols) & (cols < rectified_image.shape[1])

        rows, cols = rows[on_canvas_mask], cols[on_canvas_mask]
        rectified_row_coords = rectified_row_coords[on_canvas_mask]
        rectified_column_coords = rectified_column_coords[on_canvas_mask]

        # transfer value of rectified image to respective position in distorted image
        # TODO: integer casting of values not optimal. Implement interpolation mechanisms!
        distorted_image[np.floor(rows).astype(np.int), np.floor(cols).astype(np.int)] = \
            rectified_image[rectified_row_coords, rectified_column_coords]

        return distorted_image, rows, cols

    # TODO: may use fisheye::distortPoints
    def remap_rectified_coordinates_to_distorted_coordinates(self, coords_row: np.ndarray,
                                                             coords_column: np.ndarray,
                                                             image_dims: Tuple,
                                                             camera_position: CameraType = CameraType.LEFT) -> (np.ndarray, np.ndarray):
        """
        This function distorts the rectified coordinates and generates the respective coordinates in the distorted image
        :param coords_row: [N]
        :param coords_column: [N]
        :param image_dims: (H, W)
        :param camera_position:
        :return:
        """
        assert coords_column.size == coords_row.size

        # select correct image maps dependent on selected camera
        map_row = self.calib.map0_y if camera_position == CameraType.LEFT else self.calib.map1_y
        map_column = self.calib.map0_x if camera_position == CameraType.LEFT else self.calib.map1_x

        assert image_dims == map_row.shape, 'Image size and y/rows-map size do not match'
        assert image_dims == map_column.shape, 'Image size and x/column-map size do not match.'

        # get corresponding coordinates in distorted image
        result = map_row[coords_row, coords_column], map_column[coords_row, coords_column]

        return result

    def undistort_image(self, dist_image: np.ndarray,
                        camera_position: CameraType = CameraType.LEFT,
                        interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        """
        This method remaps a distorted image to the corresponding rectified image.
        :param dist_image:
        :param camera_position:
        :param interpolation:
        :return:
        """

        map1 = self.calib.map0_x if camera_position == CameraType.LEFT else self.calib.map1_x
        map2 = self.calib.map0_y if camera_position == CameraType.LEFT else self.calib.map1_y
        undist_image = cv2.remap(dist_image, map1=map1, map2=map2, interpolation=interpolation)

        return undist_image

    def undistort_coordinates(self, dist_coords: np.ndarray,
                              camera_position: CameraType = CameraType.LEFT) -> np.ndarray:
        """
        This function rectify the distorted coordinates and generates the respective coordinates in the rectified image
        :param dist_coords: [N x 2]
        :param camera_position:
        :return: v, v, u, u
        """
        # select correct parameters for the left/right camera
        camera_matrix = self.calib.calib_0.calib_mat if camera_position == CameraType.LEFT else self.calib.calib_1.calib_mat
        dist_coeffs = self.calib.calib_0.distortion_params if camera_position == CameraType.LEFT else self.calib.calib_1.distortion_params
        R = self.calib.rect_trans[LEFT_CAM_ROT_MAT] if camera_position == CameraType.LEFT else self.calib.rect_trans[RIGHT_CAM_ROT_MAT]
        P = self.calib.rect_trans[LEFT_CAM_PROJ_MAT] if camera_position == CameraType.LEFT else self.calib.rect_trans[RIGHT_CAM_PROJ_MAT]

        # transfer (row column) to (x, y) and match the input format
        dist_coords = np.array([dist_coords[1], dist_coords[0]], dtype=np.float32)
        dist_coords = np.transpose(dist_coords)
        dist_coords = np.expand_dims(dist_coords, axis=1)

        # calculate the coordinates
        undist_coords = cv2.fisheye.undistortPoints(dist_coords, K=camera_matrix, D=dist_coeffs, R=R, P=P)

        # transfer (x, y) to (row column) and match the output format
        undist_coords = undist_coords.squeeze()
        undist_coords = np.array([undist_coords[:, 1], undist_coords[:, 0]])

        return np.array(undist_coords)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" File for working with keyframes and its content. All classes in here are part of keyframes.
    Author Dmytro Bobkov, 2019
    Author Thomas Schmid, 2019
"""
import os

import pandas as pd
import numpy as np

from typing import Tuple, List

from libartipy.dataset import FramePoint, FrameMetaData, GPSLocation, FrameDataException, get_logger
from libartipy.geometry import Camera, Pose, CoordinateSystem
from libartipy.IO import get_timestamp_from_filename
logger = get_logger()


class KeyFrame(object):
    def __init__(self,
                 kf_path: str = None,
                 kf_data: List[str] = None,
                 timestamp: int = None):
        """
        :param kf_path: filename for keyframe_*.txt location
        :param kf_data: string-serialized txt file extracted from json blob
        :param timestamp: interface for timestamp when processing json blob
        """
        assert kf_path is not None or kf_data is not None, 'No Keyframe file or keyframe data provided.'
        self.kf_path = kf_path

        try:
            if kf_path is not None:
                self.metadata, self.frame_points = KeyFrame._keyframe_reader(
                    kf_file_path=self.kf_path)  # Panda dataframe
            elif kf_data is not None and timestamp is not None:
                self.metadata, self.frame_points = KeyFrame._keyframe_reader_lines(
                    lines=kf_data, timestamp_ns=timestamp)
            else:
                logger.warning("Incompatible Keyframe initialization. Provide either valid file path "
                               "or valid keyframe data and timestamp!")
                raise FrameDataException
        except FrameDataException as e:
            logger.warning("Problem while reading keyframe {}: {}".format(kf_path, e))
            # terminate early and set to None
            self.metadata, self.frame_points = None, None
            return

        self.camera = Camera(self.metadata.pose, self.metadata.calib_mat,
                             self.metadata.image_width, self.metadata.image_height)

        self.semseg_pred_image = None  # CameraImage

    @classmethod
    def from_components(cls,
                        kf_path: str,
                        meta_data: FrameMetaData,
                        frame_points: pd.DataFrame,
                        camera: Camera,
                        semseg_pred_image: np.ndarray = None) -> 'KeyFrame':

        assert meta_data.num_of_points == len(frame_points.index), \
            "Num of points {} and number of frame points {} dont match".format(
                meta_data.num_of_points, len(frame_points.index))

        # TODO: current implementation: reads keyframe from kf_path and then changes the values like metadata ..
        #   future implementation: should directly initialize from components
        kf = cls(kf_path=kf_path)
        kf.metadata = meta_data
        kf.frame_points = frame_points
        kf.camera = camera
        kf.semseg_pred_image = semseg_pred_image

        return kf

    def get_image_width_height(self) -> Tuple[int, int]:
        return self.metadata.image_width, self.metadata.image_height

    def is_valid(self) -> bool:
        return self.metadata is not None and self.frame_points is not None

    def get_frame_points(self) -> pd.DataFrame:
        return self.frame_points

    def get_timestamp(self) -> int:
        return self.metadata.get_timestamp_ns()

    def set_timestamp(self, ts: int):
        self.metadata.set_timestamp_ns(ts)

    def get_frame_metadata(self):
        return self.metadata

    def set_semseg_pred_image(self, semseg_pred_image: np.ndarray):
        self.semseg_pred_image = semseg_pred_image

    def get_camera(self) -> Camera:
        if self.camera is None:
            logger.warning("Camera for frame {} is none ".format(self.kf_path))
        return self.camera

    def compute_semseg_point_indeces(self):
        assert self.semseg_pred_image is not None
        sem_label = self.semseg_pred_image
        shape_needed = (self.metadata.image_height, self.metadata.image_width)
        assert sem_label.shape[0:2] == shape_needed, \
            "Shape {} but has to be {}".format(sem_label.shape[0:2], shape_needed)

        # now append IDs to points
        point_coord2d = self.frame_points[['coord_u', 'coord_v']].to_numpy()  # Nx2
        class_ids = sem_label[np.floor(point_coord2d[:, 1]).astype(np.int16),
                              np.floor(point_coord2d[:, 0]).astype(np.int16)]
        self.frame_points = pd.concat([self.frame_points, pd.DataFrame(class_ids, columns=['classID'])], axis=1)

        # once labels have been parsed, no need to store semseg label masks anymore, set to None
        self.set_semseg_pred_image(None)

    def has_labels(self) -> bool:
        return 'classID' in self.frame_points.columns

    @staticmethod
    def _keyframe_reader(kf_file_path: str) -> Tuple[FrameMetaData, pd.DataFrame]:
        """
        reads keyframe data
        :param kf_file_path:
        :return:
        """
        assert os.path.exists(kf_file_path), 'File {} does not exist.'.format(kf_file_path)

        timestamp_ns = get_timestamp_from_filename(kf_file_path)

        with open(kf_file_path, 'r') as kf_file:
            lines = kf_file.readlines()

            return KeyFrame._keyframe_reader_lines(lines=lines, timestamp_ns=timestamp_ns)

    @staticmethod
    def _keyframe_reader_lines(lines: List[str], timestamp_ns: int = None) -> Tuple[FrameMetaData, pd.DataFrame]:
        """

        :param lines: List of string lines containing the txt file content
        :param timestamp_ns:
        :return:
        """
        try:
            metadata = FrameMetaData.init_from_file_lines(lines)
        except FrameDataException as e:
            raise e

        # timestamp consistency check if valid timestamp is supplied
        metadata_ts_ns = metadata.get_timestamp_ns()
        if metadata_ts_ns is None:  # TODO(Dmytro) do this fix
            metadata.set_timestamp_ns(timestamp_ns)
        else:
            assert metadata_ts_ns == timestamp_ns, "Keyframes dont match! From metadata: {}\tFrom filename: {}".format(
                metadata_ts_ns, timestamp_ns)

        offset = metadata.offset_where_data_points

        number_remaining_lines = len(lines[offset::2])

        assert number_remaining_lines >= metadata.num_of_points, \
            '{}<{}'.format(number_remaining_lines, metadata.num_of_points)
        # TODO so far we ignore to check for equality because it fails for some files as there are empty lines in the end.

        number_fields = len(FramePoint.get_field_order())
        frame_point_array = np.zeros((metadata.num_of_points, number_fields))

        for point_ind in range(metadata.num_of_points):
            line_ind = 2*point_ind + offset
            lines_point = lines[line_ind:line_ind+2]
            frame_point = FramePoint.init_from_lines(lines_point)

            array_fields = frame_point.get_array_of_fields()
            frame_point_array[point_ind, :] = array_fields

        # ensure correct data type for the different data elements
        # coordinates: int16, idepth + hessian: float64, baseline: float64,
        # result + status: int8, intensity: int16
        pd_coordinates = pd.DataFrame(frame_point_array[:, :2].astype(np.int16),
                                      columns=FramePoint.get_field_order()[:2])
        pd_idepth_hessian = pd.DataFrame(frame_point_array[:, 2:4],
                                         columns=FramePoint.get_field_order()[2:4])
        pd_baseline = pd.DataFrame(frame_point_array[:, 4], columns=[FramePoint.get_field_order()[4]])
        pd_result_status = pd.DataFrame(frame_point_array[:, 5:7].astype(np.int8),
                                        columns=FramePoint.get_field_order()[5:7])
        pd_intensities = pd.DataFrame(frame_point_array[:, 7:].astype(np.int16),
                                      columns=FramePoint.get_field_order()[7:])

        frame_points = pd.DataFrame(frame_point_array, columns=FramePoint.get_field_order())

        frame_points = pd.concat([pd_coordinates, pd_idepth_hessian,
                                  pd_baseline, pd_result_status,
                                  pd_intensities], axis=1)
        return metadata, frame_points

    def assert_coords_on_canvas(self, x, y):

        assert max(x.flatten()) < self.metadata.image_width
        assert max(y.flatten()) < self.metadata.image_height

        assert min(x.flatten()) >= 0 and min(y.flatten()) >= 0

    def get_sparse_intensity_image(self, background_white: bool = True) -> np.ndarray:
        """
        This function computes a sparse image (background white) using the available points
        contained in frame points.
        :return:
        """

        x_kfs = self.frame_points['coord_u'].values
        y_kfs = self.frame_points['coord_v'].values
        intensities = self.frame_points['intensity'].values

        # create array, fill out with values
        if background_white:
            sparse_image = np.ones((self.metadata.image_height, self.metadata.image_width), dtype=np.uint8) * \
                np.iinfo(np.uint8).max
        else:
            sparse_image = np.zeros((self.metadata.image_height, self.metadata.image_width), dtype=np.uint8)

        self.assert_coords_on_canvas(x_kfs, y_kfs)

        sparse_image[y_kfs, x_kfs] = intensities

        return sparse_image

    def get_overlayed_image_black(self, image_data: np.ndarray) -> np.ndarray:
        """
        this function overlays the points contained in frame points in black over the provided input image.
        :param image_data:
        :return: WxH image with zeros at positions of available points
        """

        assert image_data.shape[0] == self.metadata.image_height
        assert image_data.shape[1] == self.metadata.image_width

        x_kfs = self.frame_points['coord_u'].values
        y_kfs = self.frame_points['coord_v'].values

        self.assert_coords_on_canvas(x_kfs, y_kfs)

        image_data_cpy = image_data.copy()
        image_data_cpy[y_kfs, x_kfs] = np.zeros(y_kfs.shape[0])

        return image_data_cpy

    def get_overlayed_image_intensity(self, image_data: np.ndarray) -> np.ndarray:
        """
        this function overlays the points contained in frame points in black over the provided input image.
        :param image_data:
        :return: WxH image with integer intensity values between 0 and 255
        """

        assert image_data.shape[0] == self.metadata.image_height
        assert image_data.shape[1] == self.metadata.image_width

        x_kfs = self.frame_points['coord_u'].values
        y_kfs = self.frame_points['coord_v'].values
        intensities = self.frame_points['intensity'].values

        self.assert_coords_on_canvas(x_kfs, y_kfs)

        image_data_cpy = image_data.copy()
        image_data_cpy[y_kfs, x_kfs] = intensities

        return image_data_cpy

    def get_idepth_image(self, cut_range: bool = False, max_value: int = 1/5, min_value: int = 1/80) -> np.ndarray:
        """
        This function computes inverser depth image from the given frame points
        :return: WxH image with floating values in inverse meters
        """

        x_kfs = self.frame_points['coord_u'].values
        y_kfs = self.frame_points['coord_v'].values
        idepths = self.frame_points['inv_depth'].values

        # create array, fill out with values
        idepth_image = np.zeros((self.metadata.image_height, self.metadata.image_width), dtype=np.float32)

        self.assert_coords_on_canvas(x_kfs, y_kfs)

        idepth_image[y_kfs, x_kfs] = idepths

        if cut_range:
            range_mask = (idepth_image > min_value) & (idepth_image < max_value)
            idepth_image[range_mask] = 0
            print("Inverse depth image: Cut {} idepth values outside range {} ... {}.".format(
                np.sum(np.logical_not(range_mask)), min_value, max_value))

        return idepth_image

    def get_disparity_image(self, baseline) -> np.ndarray:
        """
        This function computes the disparity image from the mean of the cameras focal length and the
        baseline provided upon call.
        :param baseline:
        :return: WxH image with floating values in absolute pixels
        """

        # focal length corresponds to fx
        focal_length = self.camera.calib_mat[0, 0]
        idepth_image = self.get_idepth_image()
        disparity_image = idepth_image * baseline * focal_length

        return disparity_image

    def get_normalized_disparity_image(self, baseline) -> np.ndarray:
        """
        This function computes the normalized disparity image which contains disparity values
        normalized by the image width.
        :param baseline:
        :return:
        """
        disparity_image = self.get_disparity_image(baseline)
        norm_disparity_image = disparity_image / disparity_image.shape[-1]

        return norm_disparity_image

    def get_depth_image(self,
                        cut_range: bool = False,
                        max_value: int = 80,
                        min_value: int = 5) -> np.ndarray:
        """
        This function computes depth image from the given frame points

        :return: WxH image with floating values in meters
        """

        idepth_image = self.get_idepth_image(cut_range=False)

        depth_image = np.where(idepth_image != 0,
                               1/idepth_image,
                               idepth_image)

        if cut_range:
            range_mask = (depth_image > min_value) & (depth_image < max_value)
            depth_image[range_mask] = 0
            logger.info("Depth image: Cut {} depth values outside range {} ... {}.".format(
                np.sum(np.logical_not(range_mask)), min_value, max_value))

        return depth_image

    def set_pose(self,
                 updated_pose: Pose,
                 translation_scale: float = None) -> None:
        """
        :param updated_pose:
        :param translation_scale;
        :return:
        """
        # assert updated_pose.valid(), 'Cannot set to invalid pose'
        # now we use timestamps as keys to assign poses
        self.camera.pose = updated_pose
        if translation_scale:
            self.camera.translation_scale = translation_scale

    def get_pose(self) -> Pose:
        assert self.camera.pose.valid(), 'Camera pose invalid.'
        return self.camera.pose

    def get_gps_location(self) -> GPSLocation:
        gps_loc = self.metadata.get_gps_location()
        if gps_loc is None:
            logger.warning("GPS location is not set.")
        return gps_loc

    def get_frame_points_in_cam_coord_system(self) -> np.ndarray:
        """

        :return: frame points Nx3
        """
        point_coord2d = self.frame_points[['coord_u', 'coord_v']].to_numpy()  # Nx2
        inv_depths = self.frame_points['inv_depth'].to_numpy()  # Nx1,  # [:, None]
        return self.camera.project_points2d_to_3d(point_coord2d, inv_depths, coordinate_system=CoordinateSystem.CAMERA)

    def get_frame_points_in_slam_coord_system(self) -> np.ndarray:
        """

        :return: frame points Nx3
        """
        # iterate over frame points, transform and return

        point_coord2d = self.frame_points[['coord_u', 'coord_v']].to_numpy()  # Nx2
        inv_depths = self.frame_points['inv_depth'].to_numpy()  # Nx1,  # [:, None]
        return self.camera.project_points2d_to_3d(point_coord2d, inv_depths, coordinate_system=CoordinateSystem.SLAM_WORLD)

    def export_keyframe_to_txt(self, base_path: str = None, fname: str = None):
        """
        :return:
        """
        fpath = "{}.txt".format("KeyFrame_" + str(self.get_timestamp()) if not fname else fname)
        if base_path:
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            fpath = os.path.join(base_path, fpath)

        with open(fpath, 'w') as f:
            f.write("# timestamp\n"
                    "{}\n\n".format(self.get_timestamp()))

            fx = self.metadata.calib_mat[0, 0]
            fy = self.metadata.calib_mat[1, 1]
            cx = self.metadata.calib_mat[0, 2]
            cy = self.metadata.calib_mat[1, 2]
            f.write("# fx, fy, cx, cy, width, height, npoints\n"
                    "{:.6f},{:.6f},{:.6f},{:.6f},{:d},{:d},{:d}\n\n".format(
                        fx, fy, cx, cy, self.metadata.image_width, self.metadata.image_height, self.metadata.num_of_points))

            f.write("# camToWorld: translation vector, rotation quaternion\n"
                    "{:.6f},{:.6f},{:.6f},{}\n"
                    "\n".format(*self.camera.pose.translation, str(self.camera.pose.rotation_quaternion)))

            f.write("# Exposure time (ms)\n"
                    "{}\n\n".format(self.metadata.exposure_time_ms))

            if self.metadata.get_gps_location():
                f.write("# GPS Data\n"
                        "# GPSData: Covariance matrix:\n"
                        "{:6f},{:6f},{:6f},{:6f},{:6f},{:6f},{:6f},{:6f},{:6f}\n\n".format(
                            *self.metadata.get_gps_location().covariance_matrix.reshape(-1)))
                f.write("# GPSData: GPS point\n"
                        "{:6f},{:6f},{:6f}\n\n".format(*self.metadata.get_gps_location().get_position().get_xyz()))

            f.write("# Point Cloud Data : \n"
                    "# u,v,idepth_scaled,idepth_hessian,maxRelBaseline,numGoodRes,status\n"
                    "# color information \n")
            frame_points = self.get_frame_points()
            fields = FramePoint.get_field_order()
            counter = 0
            for index, row in frame_points.iterrows():

                point_info = [row[attr] for attr in fields[:7]]
                f.write("{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.0f},{:.0f} \n".format(*point_info))

                point_neightbor_patch = [row[attr] for attr in fields[7:]]
                f.write("{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},\n".format(*point_neightbor_patch))

                counter += 1

            assert counter == self.metadata.num_of_points, "Wrote {} points to {}, but only got {} points!".format(
                counter, fpath, self.metadata.num_of_points)

    def print_info(self) -> None:
        """
        Print information on the keyframe in the defined format.

        :return:
        """
        print("Keyframe Info")
        print("Image Name:")
        print(self.kf_path)
        print("Image Resolution:")
        print("{0} x {1}".format(self.metadata.image_width, self.metadata.image_height))
        print("==========")
        print("Calibration Matrix:")
        print(self.metadata.calib_mat)
        print("==========")
        print("Keyframe pose:")
        print("Translation:")
        print(self.metadata.pose.translation)
        print("Rotation(quatenion, wxyz):")
        print(self.metadata.pose.rotation_quaternion)
        print("Transformation matrix:")
        print(self.metadata.pose.transformation_matrix)
        print("==========")
        print("Number of points:")
        print(len(self.frame_points.index))

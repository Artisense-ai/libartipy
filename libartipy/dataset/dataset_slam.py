#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for working with dataset
    Author Dmytro Bobkov, 2019
    Author Thomas Schmid, 2019
"""
import os
import glob
import linecache
import json
import time
import sys
import multiprocessing
from typing import Dict, Optional, List
import copy

import numpy as np

from libartipy.geometry import CoordinateSystem, Pose, Camera
from libartipy.pointcloud import Pointcloud3D
from libartipy.IO import input_output, get_timestamp_from_keyframe_filename
from libartipy.dataset import KeyFrame, TransformationFrame, GPSPose, CameraImage, FramePointFilter, Constants, \
    FramePointAnonymizer, DistortionMapper, CameraImageTuple, CameraType, SemsegImageLabelProvider, \
    Calibration, DistortionModel, Velocity, get_logger

logger = get_logger()


class Dataset(object):
    def __init__(self,
                 dataset_dir: str,
                 constants: Constants = Constants()):
        """

        :param dataset_dir: Directory where ASLAM dataset results are parsed from
        :param constants: constants for the dataset
        """
        assert os.path.exists(dataset_dir), 'Dataset folder {} does not exist.'.format(dataset_dir)

        self.constants_dataset = constants

        kf_dir_path = os.path.join(dataset_dir, self.constants_dataset.KF_DIR_NAME)
        gps_poses_filepath = os.path.join(dataset_dir, self.constants_dataset.GPS_POSES_FILE_NAME)
        velocity_filepath = os.path.join(dataset_dir, self.constants_dataset.VELOCITY_FILE_NAME)
        transformations_filepath = os.path.join(dataset_dir, self.constants_dataset.TRANSFORMATION_FILE_NAME)
        anonymization_bbox_dir_path = os.path.join(dataset_dir, self.constants_dataset.ANONYMIZATION_BBOX_FOLDER_NAME)
        calibration_dir_path = os.path.join(dataset_dir, self.constants_dataset.CALIBRATION_FOLDER_NAME)
        distorted_image_dir_path = os.path.join(dataset_dir, self.constants_dataset.DISTORTED_IMAGE_FOLDER_NAME)
        undistorted_image_dir_path = os.path.join(dataset_dir, self.constants_dataset.UNDISTORTED_IMAGE_FOLDER_NAME)

        assert os.path.exists(kf_dir_path), \
            'keyframe directory {} does not exist, cannot paste anything.'.format(kf_dir_path)

        self.kf_dir_path = kf_dir_path
        self.gps_poses_filepath = gps_poses_filepath
        self.velocity_filepath = velocity_filepath
        self.transformations_filepath = transformations_filepath
        self.anonymization_bbox_dir_path = anonymization_bbox_dir_path
        self.calibration_dir_path = calibration_dir_path
        self.distorted_image_dir_path = distorted_image_dir_path
        self.undistorted_image_dir_path = undistorted_image_dir_path

        ##
        self.frame_point_filter = FramePointFilter(
            constants.inv_depth_hessian_inv_pow_threshold,
            constants.inv_depth_hessian_inv_threshold,
            constants.min_rel_baseline,
            constants.sparsify_factor)

        self.frame_point_anonymizer = FramePointAnonymizer(score_threshold=0.3)

        self.gps_poses = self.parse_gps_poses(self.gps_poses_filepath)  # dictionary [timestamp->GPSpose]
        try:
            self.velocities = self.parse_velocity(self.velocity_filepath)  # dictionary [timestamp->Velocity]
        except Exception as e:
            logger.warning("({}) occurred when parsing velocities from {}".format(str(e), self.velocity_filepath))

        self.transformation_frame = TransformationFrame.init_from_file(self.transformations_filepath)

        self.keyframes = dict()

        self.distorted_stereo_image_dict = Dataset.parse_stereo_images(
            self.constants_dataset,
            self.distorted_image_dir_path)  # Dictionary with timestamp -> image mapping

        self.undistorted_stereo_image_dict = Dataset.parse_stereo_images(
            self.constants_dataset,
            self.undistorted_image_dir_path)  # Dictionary with timestamp -> image mapping
        # attach images to keyframes

        self.mapper = None
        self.calibration_distorted = None
        if os.path.exists(self.calibration_dir_path):
            try:
                self.calibration_distorted = Calibration(calib_folder_path=self.calibration_dir_path, distorted=True)
                if self.calibration_distorted.distortion_model == DistortionModel.Equidistant:
                    self.mapper = DistortionMapper(calib_data=self.calibration_distorted)
            except AssertionError:
                self.calibration_distorted = None
                self.mapper = None

            # check if undistorted calibration exists
            try:
                self.calibration_undistorted = Calibration(
                    calib_folder_path=self.calibration_dir_path, distorted=False)
            except AssertionError as e:
                self.calibration_undistorted = None
                logger.warning("Undistorted calibration not parsed from {} due to {}".
                               format(self.calibration_dir_path, str(e)))

        self.semseg_image_label_provider = SemsegImageLabelProvider(in_dataset_dir=dataset_dir,
                                                                    in_calibration_dir=self.calibration_dir_path,
                                                                    dataset_constants=self.constants_dataset)

        self.pointcloud = Pointcloud3D(transformation_frame=self.transformation_frame,
                                       constants=self.constants_dataset)
        self.pointcloud.set_semseg_image_label_provider(self.semseg_image_label_provider)

    def get_keyframe_poses_in_coordinate_system(self,
                                                coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD) -> \
            Dict[int, Pose]:
        """
        Get keyframe poses in the given coordinate system

        :param coordinate_system:
        :return:
        """
        cameras_map = self.get_cameras_in_coordinate_system(coordinate_system, update_rotation=True)
        pose_map = {ts: cam.pose for ts, cam in cameras_map.items()}
        return pose_map

    def get_rotation_scales_from_frames(self) -> Optional[np.ndarray]:
        """
        Get rotation scales for the frames

        :return: Nx1 array of rotation scales or none if no scales available.
        """
        # check if any of cameras has rotation scale
        has_tran_scales = False
        list_keyframes = list(self.keyframes.values())

        for ind, k in enumerate(list_keyframes):
            if k.get_camera().has_translation_scale():
                has_tran_scales = True
                break

        if not has_tran_scales:
            rotation_scales = None
        else:
            rotation_scales = np.ones((len(list_keyframes), 1))
            for ind, k in enumerate(list_keyframes):
                try:
                    rotation_scales[ind, :] = k.get_camera().get_translation_scale()
                except:
                    logger.warning("problem when getting rotation scale for {}".format(k.get_timestamp()))
                    rotation_scales[ind, :] = np.nan

        return rotation_scales

    def get_cameras_in_coordinate_system(self,
                                         coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD,
                                         update_rotation: bool = True) -> Dict[int, Camera]:
        """
        Get keyframe cameras, parse their information, get a deep copy and convert to the requested coordinate system

        :param coordinate_system:
        :param update_rotation:
        :return: dictionary timestamp -> Camera in requested coord. system
        """

        # create deepcopy of the object to make sure we don't change underlying dataset object
        cameras = {k.get_timestamp(): copy.deepcopy(k.get_camera()) for k in self.keyframes.values()}

        rotation_scales = self.get_rotation_scales_from_frames()

        cameras_new = {}
        if coordinate_system == CoordinateSystem.WGS84:
            logger.debug("Cannot update rotation for WGS84, setting it to false. Rotation will not be updated")
            update_rotation = False

        for ind, ts in enumerate(cameras.keys()):
            cam = cameras[ts]

            if rotation_scales is None:
                rotation_scale = None
            else:
                rotation_scale = rotation_scales[ind, :]

                if np.isnan(rotation_scale):
                    logger.warning("No scaling factor for TS {}".format(ts))
                    continue

                rotation_scale = rotation_scale.reshape((1, 1))

            new_translation = np.squeeze(self.transformation_frame.transform_xyz(
                cam.pose.translation.reshape(1, 3),
                CoordinateSystem.SLAM_WORLD,
                coordinate_system,
                rotation_scale))

            if update_rotation:
                new_rotation = self.transformation_frame.rotate_quaternion_from_to_coord_system(
                    cam.pose.rotation_quaternion,
                    CoordinateSystem.SLAM_WORLD,
                    coordinate_system,
                    rotation_scale)
            else:
                new_rotation = cam.pose.rotation_quaternion

            cameras_new[ts] = copy.deepcopy(cam)
            cameras_new[ts].pose = Pose(new_rotation, new_translation)

        return cameras_new

    def get_transformation_frame(self) -> TransformationFrame:
        return self.transformation_frame

    def get_gps_pose_with_timestamp(self, timestamp_ns: int) -> GPSPose:
        gps_pose = self.gps_poses.get(timestamp_ns, None)
        if gps_pose is None:
            logger.warning("GPS pose with TS {} not found, none".format(timestamp_ns))
        return gps_pose

    def get_keyframe_with_timestamp(self, timestamp_ns: int) -> Optional[KeyFrame]:
        """
        :param timestamp_ns:
        :return:
        """
        if timestamp_ns in self.keyframes:
            return self.keyframes[timestamp_ns]
        else:
            logger.warning("Not found keyframe with timestamp {}".format(timestamp_ns))
            return None

    def get_all_kf_timestamps(self) -> List[int]:
        """
        :return: a list of timestamps for keyframes
        """
        return sorted([kf.get_timestamp() for ts, kf in self.keyframes.items()])

    def write_to_folder(self, dataset_folder_out: str):
        """
        Write dataset into a folder
        :return:
        """
        # TODO implement writing into a folder

    def accumulate_framepoints_into_pointcloud(self) -> None:
        """

        :return:
        """
        self.pointcloud.init_from_keyframes(self.keyframes)

    def process_keyframes_filter_points(self) -> None:
        """
        Filter points of keyframes, remove them in-place.

        :return:
        """

        time_start = time.time()
        number_nans = 0
        num_kframe = 0
        num_before_filtering = 0
        num_after_filtering = 0

        for kf_timestamp, kframe in self.keyframes.items():
            num_before_filtering += len(kframe.get_frame_points().index)
            logical_mask_consider_points = self.frame_point_filter.consider_frame_points_vectorized(
                kframe.frame_points)
            points_filtered_new = kframe.frame_points.loc[logical_mask_consider_points]

            # Dropping NaN values
            number_total_with_nans = len(points_filtered_new)
            points_filtered_new = points_filtered_new.dropna()
            number_nans += number_total_with_nans - len(points_filtered_new)

            # reset indexing of rows (points to incremental sequence, otherwise our indexing has gaps)
            points_filtered_new = points_filtered_new.reset_index(drop=True)

            num_kframe += 1
            if num_kframe % 100 == 0:
                logger.debug("Frame {}/{} Points filtered, before: {}, after: {}. There are {} NaNs.".format(num_kframe,
                                                                                                             len(
                                                                                                                 self.keyframes),
                                                                                                             kframe.metadata.num_of_points,
                                                                                                             len(
                                                                                                                 points_filtered_new),
                                                                                                             number_nans))

            kframe.metadata.num_of_points = len(points_filtered_new.index)
            kframe.frame_points = points_filtered_new
            num_after_filtering += len(points_filtered_new)

        time_end = time.time()
        logger.info(
            "{} frames filtered in {} s. {}/{} total points remaining after filtering. {} NaN values removed".format(
                len(self.keyframes), time_end - time_start, num_after_filtering, num_before_filtering, number_nans))

    def _parse_keyframes(self, keyframe_files: List[str]) -> None:
        """
        Parse keyframes from the given keyframe files

        :param keyframe_files: np.ndarray
        :return:
        """
        time_start = time.time()
        use_multiprocessing = sys.version_info >= (3, 5)
        if use_multiprocessing:
            with multiprocessing.Pool() as pool:
                keyframes = pool.map(KeyFrame, keyframe_files)  # when #keyframe_files is large, better to use imap()
        else:
            keyframes = []
            for file in keyframe_files:
                keyframes.append(KeyFrame(file))

        # make it a dictionary
        for f_ind in range(len(keyframes)):
            kf = keyframes[f_ind]
            if kf.is_valid() is False:
                logger.debug("Skipping none keyframe")
                continue
            ts = kf.get_timestamp()
            if ts is None:
                ts = f_ind
            self.keyframes[ts] = keyframes[f_ind]
            self.keyframes[ts].set_timestamp(ts)
            img_data = self.semseg_image_label_provider.get_semseg_pred_for_timestamp(ts)
            if img_data is not None:
                self.keyframes[ts].set_semseg_pred_image(img_data)
                self.keyframes[ts].compute_semseg_point_indeces()

        time_end = time.time()
        logger.info("{} frames parsed in {} s".format(len(self.keyframes), time_end - time_start))

    def parse_keyframes_from_json_blob(self) -> None:
        """
        parses keyframes from single json blob containing timestamps as keys and string-serialized txtfile content as
        value. The json blob file is expected to be located in the keyframe directory.
        :return:
        """

        # get json blob if exist
        ext_keyframe = '.json'
        keyframe_json_file = [f for f in glob.glob(os.path.join(self.kf_dir_path, '*' + ext_keyframe))]
        assert len(keyframe_json_file) == 1, \
            "Single json file expected, got {}.".format(0 if len(keyframe_json_file) == 0 else keyframe_json_file)

        # extract data from blob and sort timestamps by value
        keyframe_json_file = keyframe_json_file[0]
        with open(keyframe_json_file, "r") as f:
            json_blob = json.load(f)

        time_start = time.time()

        # iterate through timestamps and create list of keyframes
        keyframes = []
        for timestamp_ns in json_blob:
            kf_data = json_blob[timestamp_ns].splitlines()
            keyframes.append(KeyFrame(kf_data=kf_data, timestamp=int(timestamp_ns)))

        # make it a dictionary
        for f_ind in range(len(keyframes)):
            kf = keyframes[f_ind]
            if kf.is_valid() is False:
                continue
            ts = kf.get_timestamp()
            if ts is None:
                ts = f_ind
            self.keyframes[ts] = keyframes[f_ind]
            self.keyframes[ts].set_timestamp(ts)
            img_data = self.semseg_image_label_provider.get_semseg_pred_for_timestamp(ts)
            if img_data is not None:
                self.keyframes[ts].set_semseg_pred_image(img_data)
                self.keyframes[ts].compute_semseg_point_indeces()

        time_end = time.time()
        logger.info("{} frames parsed in {} s \n".format(len(self.keyframes), time_end - time_start))

    def parse_keyframes_target(self,
                               target_frame_id: int,
                               neighbor_frame_distance: int = np.iinfo(np.int).max) -> None:
        """
        Note: dictionary doesn't have order, so it is hard to get the adjacent keyframes. if this function is not necessary,
         we can abandon it. TODO(Dmytro)

        :param target_frame_id:
        :param neighbor_frame_distance:
        :return:
        """
        assert target_frame_id >= 0
        assert neighbor_frame_distance >= 0

        logger.warning(
            "This function is going to be deprecated, please use parse_keyframes_with_start_end_timestamps() instead.")
        kf_ts_to_filename = Dataset.get_keyframe_files(self.kf_dir_path)

        min_frame_id = max(0, target_frame_id - neighbor_frame_distance)
        max_frame_id = min(target_frame_id + neighbor_frame_distance, len(kf_ts_to_filename))
        frame_numbers = np.arange(min_frame_id, max_frame_id).astype(np.int)

        keyframe_files = list(kf_ts_to_filename.values())
        keyframe_files = sorted(keyframe_files)
        keyframe_files_selected = [keyframe_files[k] for k in frame_numbers if k < len(keyframe_files)]
        self._parse_keyframes(keyframe_files_selected)

    def parse_keyframes(self) -> None:
        """
        Parses all keyframes
        :return:
        """
        kf_ts_to_filename = Dataset.get_keyframe_files(self.kf_dir_path)
        keyframe_filenames = list(kf_ts_to_filename.values())
        self._parse_keyframes(keyframe_filenames)

    @staticmethod
    def get_keyframe_files(kf_dir_path: str) -> Dict[int, str]:
        """

        :param kf_dir_path: directory with keyframes
        :return: dictionary timestamp to keyframe filepath
        """
        kf_dir_path = kf_dir_path if kf_dir_path[-1] == '/' else kf_dir_path + '/'
        ext_keyframe = '.txt'

        keyframe_files = [f for f in glob.glob(kf_dir_path + '*' + ext_keyframe)]
        assert len(keyframe_files) >= 1

        keyframe_files = input_output.filename_sort(keyframe_files)
        kf_ts_to_filename = {}

        for kf_filename in keyframe_files:
            ts = get_timestamp_from_keyframe_filename(kf_filename)
            kf_ts_to_filename[ts] = kf_filename

        return kf_ts_to_filename

    def parse_keyframes_with_start_end_timestamps(self, start_ts: int, end_ts: int) -> None:
        """

        :param start_ts: inclusive
        :param end_ts:  inclusive
        :return:
        """

        assert start_ts <= end_ts

        # first we get all files and get their timestamps
        kf_ts_to_filename = Dataset.get_keyframe_files(self.kf_dir_path)
        # now obtain those who are in the range
        keyframe_filenames = []
        for ts in kf_ts_to_filename.keys():
            if ts >= start_ts and ts <= end_ts:
                keyframe_filenames.append(kf_ts_to_filename[ts])

        self._parse_keyframes(keyframe_filenames)

    def set_keyframe_poses_to_gps_poses(self) -> None:
        """
        set the kframe pose as the GPS pose if there're matching time stamps otherwise leave the keyframe pose unassigned

        :return:
        """
        # check if all gps poses have timestamp not equal to invalid; if the number of gps poses is less than the number
        # of keyframes, no pose assignment.
        if len(self.gps_poses) == 0:
            logger.warning('Number of GPS poses is 0. Check the data format and the GPS text file.'
                           'not setting keyframe poses to them, doing nothing.')
            return
        elif len(self.gps_poses) < len(self.keyframes):
            logger.warning('Number of GPS poses, {}, is less than number of keyframes, {}. '
                           'Check the data format and the GPS text file.'
                           'not setting keyframe poses to them, doing nothing.'
                           .format(len(self.gps_poses), len(self.keyframes)))
            return
        else:
            logger.debug("Number of gps poses {}, number of keyframes {}".format(
                len(self.gps_poses), len(self.keyframes)))

            # if GPS poses have invalid timestamps, set them to keyframes
            all_invalid = True
            gps_poses = list(self.gps_poses.values())
            for gps_pose in self.gps_poses.values():
                if gps_pose.has_valid_timestamp():
                    all_invalid = False

            if all_invalid:
                logger.debug("GPS poses have all invalid timestamps, setting their timestamps to ones from keyframes.")
                ts = self.get_all_kf_timestamps()

                if len(gps_poses) != len(ts):
                    logger.warning("Number of GPS poses is not the same as the number of keyframes: {}<{}. Returning.".
                                   format(len(ts), len(gps_poses)))
                    return

                for ind_kf in range(len(gps_poses)):
                    gps_poses[ind_kf].set_timestamp_ns(ts[ind_kf])
                self.gps_poses = {}
                for ind_gps in range(len(gps_poses)):
                    self.gps_poses[gps_poses[ind_gps].get_timestamp_ns()] = gps_poses[ind_gps]

            for kf_timestamp in self.get_all_kf_timestamps():
                try:

                    if self.gps_poses[kf_timestamp].has_translation_scale():
                        translation_scale = self.gps_poses[kf_timestamp].get_translation_scale()
                    else:
                        translation_scale = None

                    self.keyframes[kf_timestamp].set_pose(self.gps_poses[kf_timestamp].pose, translation_scale)
                except Exception as e:
                    logger.warning('can not find the pose for timestamp {} {}'.
                                   format(kf_timestamp, e))

    @staticmethod
    def parse_gps_poses(gps_poses_filepath: str) -> Dict[int, GPSPose]:
        """
        :param gps_poses_filepath:
        :return:
        """
        gps_dict = {}
        if not os.path.exists(gps_poses_filepath):
            logger.warning("File with GPS poses {} does not exist, not using them.".format(
                gps_poses_filepath))
            return gps_dict

        # iterate over all lines
        gps_poses = []
        num_lines = sum(1 for line in open(gps_poses_filepath))

        num_header_lines = 0
        version_gps_pose_file = 2

        line_first = linecache.getline(gps_poses_filepath, 1)
        if '#' in line_first:
            num_header_lines = 1
            # TODO(Dmytro) assert on the header values if available

        for line_id in range(num_header_lines, num_lines, 1):

            # get keyframe specific line from file
            line = linecache.getline(gps_poses_filepath, line_id + 1)
            assert (len(line) > 0)

            gps_pose_this = GPSPose.init_from_line(line)
            if gps_pose_this is None:
                continue
            if not gps_pose_this.has_valid_timestamp():
                gps_pose_this.timestamp_utc_ns = line_id

            gps_poses.append(gps_pose_this)

        gps_dict = {}
        for f_ind in range(len(gps_poses)):
            f = gps_poses[f_ind]
            if not f.has_valid_timestamp():
                # setting timestamps to incremental integers as no timestamp information available
                f.set_timestamp_ns(f_ind)
                f.valid_ts = False

            gps_dict[f.get_timestamp_ns()] = f

        logger.info("Parsed {} GPS poses from {}".format(len(gps_dict), gps_poses_filepath))

        return gps_dict

    @staticmethod
    def parse_velocity(velocity_filepath: str) -> Dict[int, Velocity]:
        """
        :param velocity_filepath:
        :return:
        """
        velocity_dict = {}
        if not os.path.exists(velocity_filepath):
            logger.warning(
                "File with velocities {} does not exist, not using them.".format(
                    velocity_filepath))
            return velocity_dict

        # iterate over all lines
        velocities = []
        num_lines = sum(1 for line in open(velocity_filepath))

        num_header_lines = 0
        version_velocities_file = 1

        line_first = linecache.getline(velocity_filepath, 1)
        if '#' in line_first:
            num_header_lines = 1

        for line_id in range(num_header_lines, num_lines, 1):
            line = linecache.getline(velocity_filepath, line_id + 1)
            assert (len(line) > 0)

            velocity = Velocity.init_from_line(line)
            if velocity is None:
                continue

            velocities.append(velocity)

        for f_ind in range(len(velocities)):
            f = velocities[f_ind]
            if not f.has_valid_timestamp():
                # setting timestamps to incremental integers as no timestamp information available
                f.set_timestamp_ns(f_ind)
                f.valid_ts = False

            velocity_dict[f.get_timestamp_ns()] = f

        logger.info("Parsed {} velocities from {}".format(len(velocity_dict), velocity_filepath))

        return velocity_dict

    def process_keyframes_anonymizer_points(self,
                                            score_threshold: float = 0.3,
                                            bbox_information_is_undistorted: bool = True,
                                            save_dir: str = None) -> None:
        """
        Anonymize points given bounding boxes
        perform anonymization to each frames
        :param score_threshold:
        :param bbox_information_is_undistorted
        :param save_dir: if none, nothing is written, but trimmed on the fly
        :return:
        """
        save_images = save_dir is not None
        if save_images:
            os.makedirs(save_dir)

        time_start = time.time()
        number_anon = 0
        num_kframe = 0

        self.frame_point_anonymizer.set_score_threshold(score_threshold)

        # read the bboxes
        self.frame_point_anonymizer.read_bboxes(self.anonymization_bbox_dir_path)

        if bbox_information_is_undistorted:
            undistorter_fcn = None  # BBox info is already undistorted, no need for mapper anymore.
        else:
            mapper = DistortionMapper(calib_data=self.calibration_distorted)
            undistorter_fcn = mapper.undistort_coordinates

        for kf_timestamp in self.get_all_kf_timestamps():

            kframe = self.get_keyframe_with_timestamp(kf_timestamp)
            width, height = kframe.get_image_width_height()
            img_h_w = (height, width)

            bboxes = self.frame_point_anonymizer.get_bbox(kf_timestamp)
            if save_images:
                # Here we save images to directory
                stereo_pair = self.get_undistorted_stereo_images_with_timestamp(kf_timestamp)
                left_image = stereo_pair.images[CameraType.LEFT]
                logical_mask_consider_points = self.frame_point_anonymizer.consider_frame_points_anonymized(
                    kframe.frame_points,
                    image_size=img_h_w,
                    bboxes=bboxes,
                    rectified_fcn=undistorter_fcn,
                    verification=True,
                    save_dir=save_dir,
                    image_path=left_image.file_path)
            else:
                # do not save anything, just trim points in the keyframe
                logical_mask_consider_points = self.frame_point_anonymizer.consider_frame_points_anonymized(
                    kframe.frame_points,
                    image_size=img_h_w,
                    bboxes=bboxes,
                    rectified_fcn=undistorter_fcn,
                    verification=False)

            points_anonymized = kframe.frame_points.loc[logical_mask_consider_points]

            # reset indexing of rows (points to incremental sequence, otherwise our indexing has gaps
            points_anonymized = points_anonymized.reset_index(drop=True)

            number_anon += kframe.metadata.num_of_points - len(points_anonymized)

            num_kframe += 1
            if num_kframe % 100 == 0:
                logger.debug("Frame {}/{} Points filtered, before: {}, after: {}. ".format(num_kframe, len(self.keyframes),
                                                                                           kframe.metadata.num_of_points,
                                                                                           len(points_anonymized)))

            kframe.metadata.num_of_points = len(points_anonymized)
            kframe.frame_points = points_anonymized

        time_end = time.time()
        logger.info("Anonymization finished! {} frames anonymized in {} s. {} frame points removed".format(
            len(self.keyframes), time_end - time_start, number_anon))

    @staticmethod
    def parse_stereo_images(constants_dataset: Constants,
                            stereo_image_dir_path: str) -> Dict[int, CameraImageTuple]:
        """
        parse all the path and time stamps for left and right image folder

        :param constants_dataset: constants to use for parsing
        :param stereo_image_dir_path: directory with stereo images
        :return: dictionary timestamp to CameraImageTuple (which is a stereo image pair)
        """
        stereo_images_dict = {}

        if not os.path.exists(stereo_image_dir_path):
            logger.debug("Info: image dir {} does not exist. Not parsing any images.".format(
                stereo_image_dir_path))
            return stereo_images_dict

        left_image_dir_path = os.path.join(stereo_image_dir_path, constants_dataset.LEFT_IMAGE_FOLDER_NAME)
        right_image_dir_path = os.path.join(stereo_image_dir_path, constants_dataset.RIGHT_IMAGE_FOLDER_NAME)

        left_image_paths = []
        right_image_paths = []
        image_exts = ['.jpg', '.png']
        for ext in image_exts:
            left_image_paths += glob.glob(os.path.join(left_image_dir_path, '*' + ext))
            right_image_paths += glob.glob(os.path.join(right_image_dir_path, '*' + ext))

        left_image_paths = sorted(left_image_paths)
        right_image_paths = sorted(right_image_paths)

        stereo_images_dict = {}
        for img_path in left_image_paths:
            ts = CameraImage.get_timestamp_from_image_filename(img_path)
            img_tuple = CameraImageTuple()
            cam_img = CameraImage.init_from_file(img_path)
            img_tuple.add_image(CameraType.LEFT, cam_img)
            stereo_images_dict[ts] = img_tuple

        for img_path in right_image_paths:
            ts = CameraImage.get_timestamp_from_image_filename(img_path)
            # TODO: is the assertion too strict? since it's possible there's a new ts in the right folder
            if ts not in stereo_images_dict.keys():
                logger.warning('New timestamp in right image {} that is not available for left image dir {}. Ignoring.'
                               .format(ts, stereo_image_dir_path))
                continue

            cam_img = CameraImage.init_from_file(img_path)
            stereo_images_dict[ts].add_image(CameraType.RIGHT, cam_img)

        logger.info("{} pairs of stereo images in {} has been parsed.".format(
            len(stereo_images_dict), stereo_image_dir_path))

        return stereo_images_dict

    def get_distorted_stereo_images_with_timestamp(self, timestamp_s: int) -> Optional[CameraImageTuple]:
        """
        :param timestamp_s:
        :return: left image, right image pairs or None
        """
        if timestamp_s in self.distorted_stereo_image_dict:
            return self.distorted_stereo_image_dict[timestamp_s]
        else:
            logger.warning("Not found the image with timestamp {}".format(timestamp_s))
            return None

    def _get_undistorted_stereo_images_from_distorted(self, distorted_tuple: CameraImageTuple) -> CameraImageTuple:
        undistorted_pair = CameraImageTuple()
        for type_img in CameraType.get_all_types():
            distorted_cam_img = distorted_tuple.get_image(type_img)
            if distorted_cam_img is None:
                logger.warning("Image is none in tuple.")
                continue

            undistorted_img_data = self.mapper.undistort_image(distorted_cam_img.get_image_data(), type_img)
            undistorted_cam_img = CameraImage(timestamp_ns=distorted_cam_img.get_timestamp(),
                                              file_path=None,  # when loading with data, no file path needed
                                              img_data=undistorted_img_data)
            undistorted_pair.add_image(type_img, undistorted_cam_img)
        return undistorted_pair

    def get_undistorted_stereo_images_with_timestamp(self, timestamp_ns: int) -> Optional[CameraImageTuple]:
        """
        :param timestamp_ns:
        :return: left image, right image pairs or None
        """

        if timestamp_ns in self.undistorted_stereo_image_dict:
            return self.undistorted_stereo_image_dict[timestamp_ns]
        else:
            if self.mapper is None:
                logger.warning("Returning none as mapper is not initialized and no "
                               "undistorted image available for ts {}".format(timestamp_ns))
                return None
            else:
                distorted_pair = self.get_distorted_stereo_images_with_timestamp(timestamp_ns)
                if distorted_pair is None:
                    logger.warning("Returning none as distorted and undistorted images NOT available for ts {}".
                                   format(timestamp_ns))
                    return None
                assert len(distorted_pair) > 0, "Distorted images are empty for ts {}".format(timestamp_ns)
                undistorted_pair = self._get_undistorted_stereo_images_from_distorted(distorted_pair)
                return undistorted_pair

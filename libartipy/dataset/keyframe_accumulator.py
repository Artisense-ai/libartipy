""" Class for keyframe accumulation
    Author Thomas Schmid, 2019
"""

from typing import List, Dict, Tuple
import numpy as np
import os
import pandas as pd
import shutil

from libartipy.dataset import CameraType, CameraImageTuple, CameraImage, KeyFrame, Calibration, FrameMetaData, \
    DepthDisparityImageType, get_logger, RIGHT_CAM_PROJ_MAT
from libartipy.geometry import Pose, Camera
from libartipy.IO import data_to_16bit_png, encode_data_image_to_16bit
logger = get_logger()


class KeyFrameAccumulator (object):
    """ This class ist used to fuse a target keyframe and several other keyframes together into one keyframe object
    in the target keyframes coordinate system and with its metadata and camera. """

    def __init__(self, constants, calibration_path: str, output_path: str):

        self.constants = constants

        # ensure calibration data is provided
        assert os.path.exists(calibration_path)
        assert os.path.exists(output_path)

        # delete output folder if exists and recreate new folder
        self.left_output_folder = os.path.join(output_path, self.constants.LEFT_IMAGE_FOLDER_NAME)
        self.right_output_folder = os.path.join(output_path, self.constants.RIGHT_IMAGE_FOLDER_NAME)

        for folder_path in [self.left_output_folder, self.right_output_folder]:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)

        # get undistorted right camera calibration data
        calib = Calibration(calibration_path, distorted=True)
        self.right_camera_calib = calib.K1_optimized
        self.right_img_width = calib.img_dims[0]
        self.right_img_height = calib.img_dims[1]

        # P2: projection matrix in the new rectified coordinate system for the right camera
        # K0: optimized Calibration matrix of the left camera
        P2 = calib.rect_trans[RIGHT_CAM_PROJ_MAT]
        K1 = calib.K0_optimized

        baseline_pixel = P2[0, 3]
        focal_length_x = K1[0, 0]
        stereo_transform = np.eye(4)
        stereo_transform[0, 3] = baseline_pixel / focal_length_x
        self.left_to_right_pose = Pose.from_transformation_matrix(stereo_transform)

        self.acc_frame_path_dict = {}

    def _get_right_camera(self, left_camera: Camera) -> Camera:
        """
        this method creates a right camera object that contains the respective meta data for the right camera
        including right camera pose and image size.
        :param left_camera:
        :return:
        """

        # use stereo transformation to transform left pose to right pose
        left_camera_pose = left_camera.pose
        right_camera_pose = (self.left_to_right_pose * left_camera_pose.inverse()).inverse()

        # right_camera = Camera(right_camera_pose, self.right_camera_calib,
        #                       self.right_img_width, self.right_img_height)

        right_camera = Camera(right_camera_pose, left_camera.calib_mat,
                              self.right_img_width, self.right_img_height)

        return right_camera

    def get_accumulated_frame(self, target_frame: KeyFrame, list_of_frames: List[KeyFrame],
                              cam_type_list: List[CameraType],
                              undist_stereo_dict: Dict[int, CameraImageTuple] = None, intensity_thres: int = 10) \
            -> Dict[CameraType, KeyFrame]:
        """
        this method projects a list of keyframes into the target frame, creates a new keyframe objects with the
        resulting points and filters these points given the original greyscale image by checking the difference
        in intensity values.
        :param target_frame:
        :param list_of_frames:
        :param cam_type_list:
        :param undist_stereo_dict
        :param intensity_thres
        :return:
        """

        # add keyframe to list
        list_of_frames.append(target_frame)

        # get all involved points in slam coordinate system
        frame_slam_coord_list = [frame.get_frame_points_in_slam_coord_system()
                                 for frame in list_of_frames]

        # vertically stack list of coordinates to stacked numpy array
        assert frame_slam_coord_list[0].shape[-1] == 3
        coords_in_slam_world = np.vstack(frame_slam_coord_list)

        frame_dict = {}
        for cam_type in cam_type_list:
            # select left or generated right camera
            cur_target_camera = target_frame.camera if cam_type == CameraType.LEFT \
                else self._get_right_camera(target_frame.camera)

            # return coordinates, depth and applied mask contained in current target frame
            pixel_coords, pixel_depth, mask = self._transform_kf_list_to_target_kf(
                coords_in_slam_world, cur_target_camera)

            # generate new frame object
            updated_kf = self.update_target_frame_with_other_frames(target_frame, list_of_frames, cur_target_camera,
                                                                    pixel_coords, pixel_depth, mask, cam_type)

            # filter out intensities that are differing from target frame greyscale image too much
            # TODO: implement proper inconsistency check, the images have to be exposure-optimized first
            #  following, the neighboring pixels have to be projected into the target frame and their gradient to
            #  be compared to the target image neighboring gradient.
            if undist_stereo_dict is not None:
                updated_kf = self.filter_inconsistent_intensities(updated_kf, undist_stereo_dict, cam_type,
                                                                  intensity_thres)

            frame_dict[cam_type] = updated_kf

        return frame_dict

    @staticmethod
    def _transform_kf_list_to_target_kf(coords_in_slam_world: np.ndarray, target_camera: Camera) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        this function transforms a list of keyframes into the coordinate system of the target camera.
        :param coords_in_slam_world:
        :param target_camera:
        :return:
        """

        # convert coordinates from slam world coordinate system to target keyframe camera coordinate system
        pixel_coords, pixel_depth, projection_mask = \
            target_camera.transform_points3d_to_image_canvas(coords_in_slam_world)

        # cast float coordinates to integer coordinates using NN interpolation
        pixel_coords = np.round(pixel_coords + 0.5).astype(np.int)
        # TODO: perform bilinear interpolation!

        # create canvas mask and filter coordinates and depths
        canvas_mask = target_camera.provide_canvas_mask(pixel_coords)
        pixel_coords = pixel_coords[canvas_mask]
        pixel_depth = pixel_depth[canvas_mask]

        # combine all applied masks to one
        num_points = coords_in_slam_world.shape[0]
        valid_ids = np.array(range(num_points))
        valid_ids = valid_ids[projection_mask]
        valid_ids = valid_ids[canvas_mask]

        mask_overall = np.zeros(num_points, np.bool)
        mask_overall[[id for id in valid_ids]] = True

        return pixel_coords, pixel_depth, mask_overall

    @staticmethod
    def update_target_frame_with_other_frames(target_frame: KeyFrame, list_of_frames: List[KeyFrame], target_camera: Camera,
                                              pixel_coords: np.ndarray, pixel_depth: np.ndarray, mask: np.ndarray,
                                              camera_type: CameraType) -> KeyFrame:
        """
        this function consumes a list of keyframes and updated pixel and depth values by returning a new keyframe
        object.
        :param target_frame:
        :param list_of_frames:
        :param target_camera:
        :param pixel_coords:
        :param pixel_depth:
        :param mask:
        :param camera_type:
        :return:
        """

        # get all frame point and filter them according to the filtering performed on the projected
        # foreign camera points
        frame_points_list = [frame.get_frame_points() for frame in list_of_frames]

        frame_points_pd = pd.concat(frame_points_list, axis=0, ignore_index=True)
        frame_points_pd_filtered = frame_points_pd[mask]

        # assign projected camera points to initial points in respective camera coordinate system
        frame_points_pd_filtered.loc[:, "coord_u"] = pixel_coords[:, 0]
        frame_points_pd_filtered.loc[:, "coord_v"] = pixel_coords[:, 1]
        frame_points_pd_filtered.loc[:, "inv_depth"] = 1. / pixel_depth

        # generate a new meta data object
        calib_mat = target_camera.calib_mat
        image_width = target_camera.image_size_wh[0]
        image_height = target_camera.image_size_wh[1]
        num_of_points = len(frame_points_pd_filtered.index)
        timestamp_utc_ns = target_frame.metadata.get_timestamp_ns()
        pose = target_camera.pose
        gps_location = None if camera_type == CameraType.RIGHT else target_frame.metadata.get_gps_location()
        exposure_time_ms = target_frame.metadata.get_exposure_time_ms()
        new_metadata = FrameMetaData(calib_mat, image_width, image_height,
                                     num_of_points, timestamp_utc_ns, pose,
                                     gps_location, exposure_time_ms)

        # generate new keyframe object
        new_kf = KeyFrame.from_components(target_frame.kf_path, new_metadata, frame_points_pd_filtered,
                                          target_camera, None)

        return new_kf

    @staticmethod
    def filter_inconsistent_intensities(frame: KeyFrame, undist_stereo_image_dict: Dict[int, CameraImageTuple],
                                        camera_type: CameraType, intensity_threshold: int) -> KeyFrame:
        """
        this function checks photometric consistency of all pixel and intensity values contained in the keyframe
        object by applying an absolute threshold to the intensities from the original greyscale image and the
        intensities stored in the keyframe object.
        :param frame:
        :param undist_stereo_image_dict:
        :param camera_type:
        :param intensity_threshold:
        :return:
        """
        time_stamp = frame.get_timestamp()

        # this catches occasions when a keyframe.txt is available, but not an actual image
        try:
            img_object = undist_stereo_image_dict[time_stamp].get_image(camera_type)
        except KeyError:
            return frame

        if not img_object:
            img_fpath = undist_stereo_image_dict[time_stamp].get_image(camera_type).get_image_data()
            img_object = CameraImage.init_from_file(img_fpath)

        assert img_object, "Image object not successfully loaded!"

        img_data = img_object.get_image_data()

        # extract the intensity values from keyframe object and corresponding image respectively
        x_coord = frame.frame_points['coord_u'].values
        y_coord = frame.frame_points['coord_v'].values
        frame_intensities = frame.frame_points['intensity'].values
        image_intensity = img_data[y_coord, x_coord]
        intensity_abs_diff = np.abs(frame_intensities - image_intensity)

        # filter frame points according to the intensity difference
        num_before_filtering = len(frame.get_frame_points().index)
        logical_mask_consistent_intensity = intensity_abs_diff < intensity_threshold
        points_filtered = frame.frame_points.loc[logical_mask_consistent_intensity]
        points_filtered = points_filtered.reset_index(drop=True)

        # update keyframe object with filtered points
        frame.metadata.num_of_points = len(points_filtered.index)
        frame.frame_points = points_filtered
        num_after_filtering = len(points_filtered)

        logger.info("Filtered out {} of {} total points".format(num_before_filtering - num_after_filtering,
                                                                num_before_filtering))

        return frame

    def to_file(self, frame: KeyFrame, camera_type: CameraType, export_type: DepthDisparityImageType,
                fname: str = None, max_data_value: int = 120) -> str:
        """
        this method saves the selected export_type to file. The file name consists of a name as well as the timestamp of
        the current frame. For the export type "depth image", "idepth image" and "disparity image",
        the maximum data value can be selected
        :param frame:
        :param camera_type:
        :param export_type:
        :param fname:
        :param max_depth:
        :return:
        """
        ts_target = frame.get_timestamp()
        output_folder = self.left_output_folder if camera_type == CameraType.LEFT else self.right_output_folder
        output_path = os.path.join(output_folder, "{}{}".format(fname + '_' if fname else '', ts_target))

        # fill dictionary of accumulated data output for filenames_file export
        # (try) if timestamp registered in dict, add camera type and path
        # (except) else, assign new CameraTuple
        try:
            img_obj = CameraImage(ts_target, output_path + ".png")
            self.acc_frame_path_dict[ts_target].add_image(camera_type, img_obj)

        # except:
        except KeyError:
            img_obj = CameraImage(ts_target, output_path + ".png")
            cam_tuple = CameraImageTuple()
            cam_tuple.add_image(camera_type, img_obj)
            self.acc_frame_path_dict.update({ts_target: cam_tuple})

        if export_type == DepthDisparityImageType.DEPTH:
            data_image = frame.get_depth_image()

        elif export_type == DepthDisparityImageType.I_DEPTH:
            data_image = frame.get_idepth_image()

        elif export_type == DepthDisparityImageType.DISPARITY:
            baseline = np.abs(self.left_to_right_pose.transformation_matrix[0, 3])
            data_image = frame.get_disparity_image(baseline)

        elif export_type == DepthDisparityImageType.NORMALIZED_DISPARITY:
            baseline = np.abs(self.left_to_right_pose.transformation_matrix[0, 3])
            data_image = frame.get_normalized_disparity_image(baseline)

            assert (data_image < 1.).all(), "Normalized disparity contains values greater 1!"
            assert (data_image >= 0.).all(), "Normalized disparity contains values lower 0!"

            pass
        else:
            raise NotImplementedError()

        data_image = encode_data_image_to_16bit(data_image, max_data_value)
        data_to_16bit_png(output_path, data_image)

        return output_path

    def generate_filenames_file(self, fpath_filenames_file: str, list_of_cams: List[CameraType],
                                undist_stereo_img_dict: Dict[int, CameraImageTuple] = None):
        """
        given the filepath dictionary of undistorted images, this method generates a txt file containing the following information:
        left image path, left data path, right image path, right data path
        where data can be either depth image, inverse depth image, disparity map or normalized dsparity map
        :param fpath_filenames_file:
        :param list_of_cams:
        :param undist_stereo_img_dict:
        """

        with open(fpath_filenames_file, 'w') as f:

            for ts in self.acc_frame_path_dict:

                fpath_list = []
                for cam_type in list_of_cams:

                    # if available, include undistorted stereo images into filenames file
                    if undist_stereo_img_dict:
                        try:
                            img_path = undist_stereo_img_dict[ts].get_image(cam_type).file_path
                        except KeyError:
                            continue

                        assert os.path.exists(img_path), "Image path {} does not exist.".format(img_path)
                        fpath_list.append(img_path)

                    # include accumulated data maps into filenames file
                    acc_img_path = self.acc_frame_path_dict[ts].get_image(cam_type).file_path
                    assert os.path.exists(
                        acc_img_path), "Accumulated image path {} does not exist.".format(acc_img_path)
                    fpath_list.append(acc_img_path)

                if len(list_of_cams) == 2 and undist_stereo_img_dict:
                    f.write("{} {} {} {}\n".format(*fpath_list))
                elif (len(list_of_cams) == 2 and not undist_stereo_img_dict) or \
                        (len(list_of_cams) == 1 and undist_stereo_img_dict):
                    f.write("{} {}\n".format(*fpath_list))
                elif len(list_of_cams) == 1 and not undist_stereo_img_dict:
                    f.write("{}\n".format(*fpath_list))
                else:
                    logger.warning("Sample can not be written to file!")

            logger.info("Filenames file wrote to {}.".format(fpath_filenames_file))
            if len(list_of_cams) == 2 and undist_stereo_img_dict:
                logger.info("Filenames file: \n"
                            "Left image path, left data path, right image path, right data path.")
            elif len(list_of_cams) == 2 and not undist_stereo_img_dict:
                logger.info("Filenames file: \n"
                            "Left data path, right data path.")
            elif len(list_of_cams) == 1 and undist_stereo_img_dict:
                logger.info("Filenames file: \n"
                            "{} image path, {}} data path or \n".format(list_of_cams[0], list_of_cams[0]))
            elif len(list_of_cams) == 1 and not undist_stereo_img_dict:
                logger.info("Filenames file: \n"
                            "1. {} data path.".format(list_of_cams[0]))

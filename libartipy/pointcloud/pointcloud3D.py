#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for working with point clouds
    Author Dmytro Bobkov, 2019
    Author Thomas Schmid
"""

import os
import time
from typing import List, Dict

import numpy as np
import pandas as pd

from laspy import header as LASHeader
from laspy import file as LASFile
import open3d

# PYPCD
from libartipy.pointcloud import pypcd
from libartipy.dataset import TransformationFrame, KeyFrame, SemsegImageLabelProvider, Constants, get_logger
from libartipy.geometry import Pose, CoordinateSystem
logger = get_logger()


class Pointcloud3D(object):
    ROTATION_SCALE_FIELD = 's'
    CLASS_FIELD = 'c'
    INTENSITY_FIELD = 'i'

    def __init__(self,
                 transformation_frame: TransformationFrame,
                 coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD,
                 constants: Constants = Constants()):
        """
        :param transformation_frame: transformation frame to use, if none, then we use identity transformation
        :param coordinate_system: in which coordinate system is given
        :param constants: used for label mapping, point filtering etc.
        """

        self.transformation_frame = transformation_frame

        self.points = None  # Panda dataframe

        self.coordinate_system = coordinate_system

        self.semseg_provider = None
        self.constants = constants

    def init_from_keyframes(self, keyframes_to_consider: Dict[int, KeyFrame]) -> None:
        """
        Init pointcloud data from keyframe points and their poses

        :param keyframes_to_consider: dict(kf_timestamp, KeyFrame)
        :return:
        """
        start_time = time.time()

        # Set coordinate system to SLAM_WORLD if initialized by keyframes
        assert self.coordinate_system == CoordinateSystem.SLAM_WORLD, '{}'.format(self.coordinate_system)

        # now iterate over keyframes, apply transformation and accumulate points
        # treat all key frames except target keyframe here

        n_total = sum([len(kframe.get_frame_points().index) for kf_timestamp, kframe in keyframes_to_consider.items()])
        points_xyz_all_frames = np.zeros((n_total, 3))
        intensities = np.zeros((n_total,), dtype=np.int)
        translation_scales = np.ones((n_total,), dtype=np.float)

        invalid_label = self.constants.get_semseg_class_mapping().get("Static Void")
        # it is defined in semantic_lcass_mapping.json
        # it denotes static_void and is used as a placeholder to indicate invalid label
        class_indices = np.full((n_total,), fill_value=invalid_label, dtype=np.int)

        n = 0
        ind_k = 0
        num_keyframes_with_classes = 0
        for kf_timestamp, kframe in keyframes_to_consider.items():
            if ind_k % 100 == 0:
                logger.debug("Accumulating KF {}/{} keyframes into PC".format(ind_k,
                                                                              len(keyframes_to_consider)))
            ind_k += 1

            # project and append keyframe points
            # obtain numpy array of points from this frame

            number_points = len(kframe.get_frame_points().index)
            points_xyz_all_frames[n: n+number_points] = kframe.get_frame_points_in_slam_coord_system()

            intensities[n: n+number_points] = np.squeeze(kframe.get_frame_points()[['intensity']].to_numpy())

            if kframe.get_camera().has_translation_scale():
                translation_scales[n: n+number_points] = kframe.get_camera().get_translation_scale()

            if kframe.has_labels():
                num_keyframes_with_classes += 1
                class_indices[n:  n + number_points] = np.squeeze(kframe.get_frame_points()[['classID']].to_numpy())

            n = n + number_points

        column_order = ['x', 'y', 'z', Pointcloud3D.INTENSITY_FIELD, Pointcloud3D.ROTATION_SCALE_FIELD]
        point_dict = {'x': points_xyz_all_frames[:, 0],
                      'y': points_xyz_all_frames[:, 1],
                      'z': points_xyz_all_frames[:, 2],
                      Pointcloud3D.INTENSITY_FIELD: intensities,
                      Pointcloud3D.ROTATION_SCALE_FIELD: translation_scales}
        if num_keyframes_with_classes > 0:
            column_order.append(Pointcloud3D.CLASS_FIELD)
            point_dict[Pointcloud3D.CLASS_FIELD] = class_indices

        self.points = pd.DataFrame(point_dict, columns=column_order)

        number_total_with_nans = len(self.points)
        self.points = self.points.dropna()
        number_nans = number_total_with_nans - len(self.points)
        assert number_nans == 0, 'Number of NaN values in points {} is > 0.'.format(number_nans)

        end_time = time.time()
        logger.info("Accumulated {} points in {} s".format(len(self.points), end_time-start_time))

    def init_from_numpy(self,
                        points: np.ndarray,
                        fmt: str = "xyzi") -> None:
        """
        Initialize pointcloud data frame from numpy array.

        :param points: N_points x N_fields
        :param fmt: specification of fields for given array (XYZ are cartesian coordinates, I intensity, rgb colors)
        :return:
        """
        N_features = points.shape[1]
        N_format = len(fmt)
        assert N_features == N_format, '{} != {}'.format(N_features, N_format)

        point_dict = {fmt[p]: points[:, p] for p in range(N_features)}

        self.points = pd.DataFrame(point_dict, columns=list(fmt))

    def get_numpy_array(self, fmt: str = "xyz") -> np.ndarray:
        """

        :param fmt: specification of fields to consider (XYZ are cartesian coordinates, I intensity, rgb colors)
        :return: N_points x N_features
        """
        # split fmt into elements
        columns = list(fmt)
        assert set(fmt).issubset(list(self.points)), 'Point DataFrame does not contain {} columns'.format(fmt)
        array = self.points[columns].to_numpy()
        assert array.shape[1] == len(columns), 'Number of columns {}'.format(array.shape[1])
        return array

    def set_semseg_image_label_provider(self, semseg_provider: SemsegImageLabelProvider):
        self.semseg_provider = semseg_provider

    def get_point_cloud_in_coordinate_system(self, coordinate_system: CoordinateSystem) -> 'Pointcloud3D':
        """
        Returns a copy of this pointcloud in another coordinate system
        :param coordinate_system: Target coordinate system
        :return:
        """
        result = Pointcloud3D(transformation_frame=self.transformation_frame, coordinate_system=coordinate_system)
        fmt = self.get_column_fmt_string()
        points = self.get_points_in_needed_coordinate_system(coordinate_system, fmt)
        result.init_from_numpy(points.to_numpy(), fmt)
        return result

    def get_points_in_needed_coordinate_system(self,
                                               coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD,
                                               fmt: str = None,
                                               use_classes_for_coloring: bool = True) -> pd.DataFrame:
        """
        Returns a copy of points in the specified coordinate system and given format.

        :param coordinate_system:
        :param fmt: specification of fields to consider (XYZ are cartesian coordinates, I intensity, rgb colors)
        :param use_classes_for_coloring
        :return: panda dataframe containing points
        """

        dataframe_points = self.points.copy()

        assert {'x', 'y', 'z'}.issubset(list(dataframe_points)), 'Point DataFrame does not contain XYZ columns'

        points = dataframe_points[['x', 'y', 'z']].to_numpy()

        # transform only XYZ
        if self.transformation_frame is None:
            dataframe_points.loc[:, ['x', 'y', 'z']] = points
        else:
            assert {Pointcloud3D.ROTATION_SCALE_FIELD}.issubset(list(dataframe_points)), \
                'Point DataFrame does not contain rotation scale'
            rotation_scales = dataframe_points[[Pointcloud3D.ROTATION_SCALE_FIELD]].to_numpy()
            dataframe_points.loc[:, ['x', 'y', 'z']] = \
                self.transformation_frame.transform_xyz(
                    points,
                    self.coordinate_system,
                    coordinate_system,
                    rotation_scales=rotation_scales)

        if fmt is not None:
            # get subset
            columns_available = list(dataframe_points)  # put it to str

            columns_requested = list(fmt)
            if {'r', 'g', 'b'}.issubset(columns_requested):
                rgb_present = {'r', 'g', 'b'}.issubset(columns_available)
                classes_present = Pointcloud3D.CLASS_FIELD in columns_available
                intensity_present = Pointcloud3D.INTENSITY_FIELD in columns_available

                if rgb_present:
                    logger.info("All good, we have RGB.")
                else:
                    if use_classes_for_coloring and classes_present:
                        logger.info("PC does not have RGB information, but you requested it! Hence, "
                                    "painting points based on class ids")
                        class_id = dataframe_points[[Pointcloud3D.CLASS_FIELD]].to_numpy().squeeze()
                        colors, classnames = self.constants.convert_classid_to_classcolor(class_id)  # TODO
                        dataframe_points['r'] = colors[:, 0]
                        dataframe_points['g'] = colors[:, 1]
                        dataframe_points['b'] = colors[:, 2]
                    elif intensity_present:
                        dataframe_points['r'] = dataframe_points[Pointcloud3D.INTENSITY_FIELD]
                        dataframe_points['g'] = dataframe_points[Pointcloud3D.INTENSITY_FIELD]
                        dataframe_points['b'] = dataframe_points[Pointcloud3D.INTENSITY_FIELD]
                    else:
                        logger.info("You requested RGB, but we do not have it. "
                                    "Neither do we have classes to use for coloring. Hence, Not showing RGB.")
                        columns_requested.remove('r')
                        columns_requested.remove('g')
                        columns_requested.remove('b')

            columns_available = list(dataframe_points)  # because updated
            assert set(columns_requested).issubset(columns_available), \
                'Selected format {} does not have corresponding ' \
                'data entries in dataframe {}'.format(columns_requested, columns_available)
            dataframe_points = dataframe_points[columns_requested]

        return dataframe_points

    def read_from_file(self, file_name: str) -> None:
        """
        Read point cloud data from appropriate format

        :param file_name: Filename to read from (pcd, ply etc.)
        :raise AssertionError if file does not exist.
        :return:
        """
        assert os.path.exists(file_name), 'File {} does not exist'.format(file_name)

        ext = os.path.splitext(file_name)[1]
        start_time = time.time()

        if ext == '.csv':
            self.read_from_csv_file(file_name, delimiter_type=',')
        elif ext == '.las':
            self.read_from_las_file(file_name)
        elif ext == '.ply':
            self.read_from_ply_file(file_name)
        elif ext == '.pcd':
            self.read_from_pcd_file(file_name)
        elif ext == '.h5':
            self.read_from_hdf_file(file_name)
        else:
            logger.error("Reading not implemented for extension {}".format(ext))

        end_time = time.time()
        logger.info("Read {} points from file {} with fmt {} in {} s".format(
            len(self.points), file_name, self.get_column_fmt_string(), end_time-start_time))

    def read_from_hdf_file(self, file_name: str, delimiter_type: str = ',') -> None:
        """
        :param file_name: filename to read from
        :param delimiter_type:
        :return:
        """
        self.points = pd.read_hdf(file_name, 'pointcloud')

    def read_from_csv_file(self, file_name: str, delimiter_type: str = ',') -> None:
        """
        :param file_name: filename to read from
        :param delimiter_type:
        :return:
        """
        # Solution: pure panda reading is much faster
        self.points = pd.read_csv(file_name, delimiter=delimiter_type, header=0)

    def read_from_las_file(self, file_name: str) -> None:
        """

        :param file_name: filename to read from
        :return:
        """

        assert os.path.splitext(file_name)[1] == '.las', 'Not las {}'.format(file_name)

        in_file = LASFile.File(file_name, mode='r')
        las_header = in_file.header

        las_offset = las_header.offset
        las_scale = las_header.scale

        x = in_file.X * las_scale[0] + las_offset[0]
        y = in_file.Y * las_scale[1] + las_offset[1]
        z = in_file.Z * las_scale[2] + las_offset[2]

        intensity = in_file.intensity

        point_dict = {'x': x,
                      'y': y,
                      'z': z}
        fmt = list(point_dict.keys())

        # out_file.Raw_Classification = class_id

        has_classes = hasattr(in_file, 'Raw_Classification')  # 'classification'
        logger.debug("Points have classes set to {}".format(has_classes))
        if has_classes:
            classification = in_file.Raw_Classification  # classification

            # TODO(Dmytro) investigate differences between raw_classification and classification
            # see https://buildmedia.readthedocs.org/media/pdf/laspy/1.2.5/laspy.pdf
            # classif = in_file.classification
            # print("Compare {} vs {}".format(classification, classif))

            point_dict[Pointcloud3D.CLASS_FIELD] = classification.astype(np.uint8)
            fmt.append(Pointcloud3D.CLASS_FIELD)

        has_intensity = hasattr(in_file, 'intensity')
        if has_intensity:
            point_dict[Pointcloud3D.INTENSITY_FIELD] = intensity.astype(np.uint8)
            fmt.append(Pointcloud3D.INTENSITY_FIELD)

        has_rgb = hasattr(in_file, 'red')
        if has_rgb:
            # check if the values are not same
            red = in_file.red
            green = in_file.green
            blue = in_file.blue
            # TODO(Dmytro) this is hacky, but laspy has this field
            #  even if RGB information not present, find a robust solution
            if min(red.flatten()) == max(red.flatten()):
                logger.debug("RGB values are same, no rgb actually")
                has_rgb = False
            else:
                logger.debug("Parsed rgb, min {} max {}".format(min(red.flatten()), max(red.flatten())))
                point_dict['r'] = red.astype(np.uint8)
                fmt.append('r')

                point_dict['g'] = green.astype(np.uint8)
                fmt.append('g')

                point_dict['b'] = blue.astype(np.uint8)
                fmt.append('b')

        logger.debug("Has rgb? {}".format(has_rgb))

        self.points = pd.DataFrame(point_dict, columns=fmt)

    def read_from_ply_file(self, file_name: str) -> None:
        """
        :param file_name: filename to read from
        :return:
        """
        plyfile = open3d.io.read_point_cloud(file_name)
        xyz = np.asarray(plyfile.points)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        colors = np.asarray(plyfile.colors) * 255
        colors = colors.astype(np.uint8)
        r = colors[:, 0]
        g = colors[:, 1]
        b = colors[:, 2]

        assert np.allclose(r, g) and np.allclose(
            r, b), 'All three RGB channels should holds same intensity value'  # TODO(Thomas) why?

        point_dict = {'x': x,
                      'y': y,
                      'z': z,
                      Pointcloud3D.INTENSITY_FIELD: r}

        self.points = pd.DataFrame(point_dict, columns=['x', 'y', 'z', Pointcloud3D.INTENSITY_FIELD])

    def read_from_pcd_file(self, file_name: str) -> None:
        """
        :param file_name: filename to read from
        :return:
        """
        cloud = pypcd.PointCloud.from_path(file_name)
        # get which columns we have
        fmt_to_field_name = {'x': 'x',
                             'y': 'y',
                             'z': 'z',
                             'intensity': Pointcloud3D.INTENSITY_FIELD,
                             'label': Pointcloud3D.CLASS_FIELD,
                             'rgb': ['r', 'g', 'b']}
        point_dict = {}
        # step through fields in PointCloud object and see which ones are available
        for field in cloud.fields:
            field_name_in_pandataframe = fmt_to_field_name.get(field)
            if field == 'rgb':
                rgb_float = cloud.pc_data[field]
                rgb_uint8 = pypcd.decode_rgb_from_pcl(rgb_float)
                for ind in range(len(fmt_to_field_name[field])):
                    point_dict[fmt_to_field_name[field][ind]] = rgb_uint8[:, ind]
            else:
                point_dict[field_name_in_pandataframe] = cloud.pc_data[field]
            logger.debug("Append {} from {}".format(field_name_in_pandataframe, field))

        columns = list(point_dict.keys())
        self.points = pd.DataFrame(point_dict, columns=columns)

    def write_to_file(self,
                      file_name: str,
                      fmt: str = "XYZI",
                      coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD,
                      use_classes_for_coloring: bool = True) -> None:
        """
        :param file_name: filename to write to (pcd, ply, las etc.)
        :param fmt: specification of fields to consider (XYZ are cartesian coordinates, I intensity, rgb colors)
        :param coordinate_system:
        :param use_classes_for_coloring only relevant if you requested rgb in fmt.
        :return:
        """
        assert file_name is not None
        ext = os.path.splitext(file_name)[1]
        start_time = time.time()

        # 1. Obtain panda dataframe in the needed coordinate system
        fmt = fmt.lower()
        dataframe_pointcloud = self.get_points_in_needed_coordinate_system(
            coordinate_system=coordinate_system, fmt=fmt, use_classes_for_coloring=use_classes_for_coloring)

        if ext == '.csv':
            Pointcloud3D.write_to_csv_file(dataframe_pointcloud,
                                           file_name,
                                           fmt=fmt,
                                           coordinate_system=coordinate_system,
                                           delimiter_type=',')
        elif ext == '.las':
            Pointcloud3D.write_to_las_file(dataframe_pointcloud,
                                           file_name=file_name,
                                           fmt=fmt,
                                           coordinate_system=coordinate_system)
        elif ext == '.ply':
            Pointcloud3D.write_to_ply_file(dataframe_pointcloud,
                                           file_name,
                                           fmt=fmt,
                                           coordinate_system=coordinate_system)
        elif ext == '.pcd':
            Pointcloud3D.write_to_pcd_file(dataframe_pointcloud,
                                           file_name,
                                           fmt=fmt,
                                           coordinate_system=coordinate_system,
                                           binary=False)
        elif ext == '.h5':
            Pointcloud3D.write_to_hdf_file(dataframe_pointcloud,
                                           file_name,
                                           fmt=fmt,
                                           coordinate_system=coordinate_system)
        else:
            logger.error("Writing not implemented for extension {}".format(ext))

        end_time = time.time()
        logger.info("Written {} points with fmt {} to {} in {} s".format(
            len(self.points), list(fmt), file_name, end_time-start_time))

    @staticmethod
    def write_to_las_file(dataframe_pointcloud: pd.DataFrame,
                          file_name: str,
                          fmt: str = "XYZI",
                          coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD) -> None:
        """
        :param dataframe_pointcloud:
        :param file_name: filename to write to
        :param fmt: specification of fields to consider (XYZ are cartesian coordinates, I intensity, rgb colors)
        :param coordinate_system:
        :return:
        """
        fmt = fmt.lower()
        assert os.path.splitext(file_name)[1] == '.las'
        supported_formats = "xyzicrgb"
        assert set(fmt).issubset(list(supported_formats)), 'Fmt {} not supported yet'.format(fmt)

        assert set(list(fmt)).issubset(list(dataframe_pointcloud)), 'Selected format {} does not have corresponding ' \
                                                                    'data entries in dataframe {}'.format(
                                                                        list(fmt), list(dataframe_pointcloud))

        # xyz filename extensions (eg. coordinates_xyz) refer to xyz as used by the LAS file and do not refer to the
        # xyz coordinates as stored in the dataframe columns xyz. In case fmt == 'yxzi', coordinates_xyz already
        # contains swapped x and y values.
        if 'yx' in fmt:
            coordinates_xyz = dataframe_pointcloud[['y', 'x', 'z']].to_numpy()
        else:
            coordinates_xyz = dataframe_pointcloud[['x', 'y', 'z']].to_numpy()

        # center data around zero to exploit negative values of long integer too
        las_offset = np.min(coordinates_xyz, axis=0) + \
            np.abs(np.max(coordinates_xyz, axis=0) - np.min(coordinates_xyz, axis=0)) / 2
        coordinates_xyz -= las_offset

        # per XYZ element, LAS point can only hold 4 byte long number: range (-214748368, +214748367) = +/- (2**31 - 1),
        # hence it has to be scaled to represent respective precision and not overflow
        range_long_int = 2**31 - 1
        las_scale = np.ceil(np.max(np.abs(coordinates_xyz), axis=0)) / (range_long_int - 1)
        coordinates_xyz /= las_scale

        # assert correct of input data to long
        assert np.max(coordinates_xyz).astype(
            np.int32) <= range_long_int, 'Value overflow: Error occurred upon scaling'
        assert np.min(coordinates_xyz).astype(np.int32) >= -range_long_int - \
            1, 'Value underflow: Error occurred upon scaling'

        # explicitly cast coordinates to long/int32/4byte long
        coordinates_xyz = coordinates_xyz.astype(np.int32)

        # Point Data Record Format as per https://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf
        las_header = LASHeader.Header(point_format=2)
        las_header.x_offset, las_header.y_offset, las_header.z_offset = las_offset
        las_header.x_scale, las_header.y_scale, las_header.z_scale = las_scale

        out_file = LASFile.File(file_name, mode='w', header=las_header)
        out_file.X, out_file.Y, out_file.Z = coordinates_xyz[:, 0], coordinates_xyz[:, 1], coordinates_xyz[:, 2]

        if "i" in fmt:
            intensities = dataframe_pointcloud[[Pointcloud3D.INTENSITY_FIELD]].to_numpy(dtype=np.uint8).squeeze()
            out_file.intensity = intensities

        if Pointcloud3D.CLASS_FIELD in fmt:
            class_id = dataframe_pointcloud[[Pointcloud3D.CLASS_FIELD]].to_numpy().squeeze()
            # cast to byte
            assert (np.max(class_id.flatten()) <= 255) and (np.min(class_id.flatten()) >= 0), \
                'Cannot be outside of range 0..255 as it will overflow'
            class_id = class_id.astype(np.uint8)
            logger.debug("Labels max {} shape {}".format(np.max(class_id.flatten()), class_id.shape))
            # out_file.classification = class_id  # integer
            out_file.Raw_Classification = class_id  # Cannot use classification as it is limited to 32

        if 'rgb' in fmt:
            colors = dataframe_pointcloud[['r', 'g', 'b']].to_numpy().squeeze()
            out_file.red, out_file.green, out_file.blue = colors[:, 0], colors[:, 1], colors[:, 2]  # integer

        out_file.close()

    @staticmethod
    def write_to_hdf_file(dataframe_pointcloud: pd.DataFrame,
                          file_name: str,
                          fmt: str = "XYZI",
                          coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD,
                          delimiter_type: str = ',') -> None:
        """
        :param file_name:
        :param fmt: specification of fields to consider (XYZ are cartesian coordinates, I intensity, rgb colors)
        :param coordinate_system:
        :param delimiter_type:
        :return:
        """
        assert os.path.splitext(file_name)[1] == '.h5'

        dataframe_pointcloud.to_hdf(file_name, key='pointcloud', mode='w')

    @staticmethod
    def write_to_csv_file(dataframe_pointcloud: pd.DataFrame,
                          file_name: str,
                          fmt: str = "XYZI",
                          coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD,
                          delimiter_type: str = ',') -> None:
        """
        :param dataframe_pointcloud:
        :param file_name:
        :param fmt: specification of fields to consider (XYZ are cartesian coordinates, I intensity, rgb colors)
        :param coordinate_system:
        :param delimiter_type:
        :return:
        """
        fmt = fmt.lower()
        assert os.path.splitext(file_name)[1] == '.csv'
        dataframe_pointcloud[list(fmt)].to_csv(file_name, sep=',', index=False)

    @staticmethod
    def write_to_ply_file(dataframe_pointcloud: pd.DataFrame,
                          file_name: str,
                          fmt: str = "XYZI",
                          coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD) -> None:
        """
        :param dataframe_pointcloud:
        :param file_name:
        :param fmt: specification of fields to consider (XYZ are cartesian coordinates, I intensity, rgb colors)
        :param coordinate_system:
        :return:
        """
        fmt = fmt.lower()
        assert os.path.splitext(file_name)[1] == '.ply'

        supported_formats = "xyzi"
        assert set(fmt).issubset(list(supported_formats)), 'Fmt {} not supported yet'.format(fmt)

        points = dataframe_pointcloud[list(fmt)[:3]].to_numpy(np.float32)  # numpy
        intensities = dataframe_pointcloud[[Pointcloud3D.INTENSITY_FIELD]].to_numpy().squeeze()

        ply = open3d.geometry.PointCloud()
        ply.points = open3d.utility.Vector3dVector(points)
        intensities = intensities.astype(np.float) / 255.
        colors = np.tile(intensities, (3, 1)).T

        ply.colors = open3d.utility.Vector3dVector(colors)
        open3d.io.write_point_cloud(file_name, ply)

    @staticmethod
    def write_to_pcd_file(dataframe_pointcloud: pd.DataFrame,
                          file_name: str,
                          fmt: str = "XYZI",
                          coordinate_system: CoordinateSystem = CoordinateSystem.SLAM_WORLD,
                          binary: bool = False) -> None:
        """
        :param dataframe_pointcloud:
        :param file_name:
        :param fmt: specification of fields to consider (XYZ are cartesian coordinates, I intensity, rgb colors)
        :param coordinate_system:
        :param binary:
        :return:
        """
        fmt = fmt.lower()
        assert os.path.splitext(file_name)[1] == '.pcd', 'Not pcd {}'.format(file_name)

        supported_formats = "xyzicrgb"
        assert set(fmt).issubset(list(supported_formats)), 'Fmt {} not supported yet'.format(fmt)

        assert not binary, 'Binary format not supported yet.'

        if coordinate_system == CoordinateSystem.WGS84:
            logger.info("WGS84 coordinate system is not meaningful for PCD files.")

        number_points = len(dataframe_pointcloud)

        data_field = 'binary' if binary else 'ascii'

        fmt_values = list(fmt)

        # if has r, g, b
        if 'r' in fmt_values:
            # join together
            values = ['r', 'g', 'b']
            for value in values:
                fmt_values.remove(value)
            fmt_values.append('rgb')

            # now update rgb values inside
            rgb_values = dataframe_pointcloud[['r', 'g', 'b']].to_numpy(dtype=np.uint8)
            rgb_values = pypcd.encode_rgb_for_pcl(rgb_values)
            dataframe_pointcloud['rgb'] = rgb_values

        fmt_to_field_name = {'x': 'x',
                             'y': 'y',
                             'z': 'z',
                             Pointcloud3D.INTENSITY_FIELD: 'intensity',
                             Pointcloud3D.CLASS_FIELD: 'label',
                             'rgb': 'rgb'}
        fields_list = [fmt_to_field_name[fmt_i] for fmt_i in fmt_values]

        fmt_to_size = {'x': 4,
                       'y': 4,
                       'z': 4,
                       Pointcloud3D.INTENSITY_FIELD: 4,
                       Pointcloud3D.CLASS_FIELD: 4,
                       'rgb': 4}
        size_list = [fmt_to_size[fmt_i] for fmt_i in fmt_values]

        fmt_to_type = {'x': 'F',
                       'y': 'F',
                       'z': 'F',
                       Pointcloud3D.INTENSITY_FIELD: 'F',
                       Pointcloud3D.CLASS_FIELD: 'I',
                       'rgb': 'F'}
        type_list = [fmt_to_type[fmt_i] for fmt_i in fmt_values]

        fmt_to_count = {'x': 1,
                        'y': 1,
                        'z': 1,
                        Pointcloud3D.INTENSITY_FIELD: 1,
                        Pointcloud3D.CLASS_FIELD: 1,
                        'rgb': 1}
        count_list = [fmt_to_count[fmt_i] for fmt_i in fmt_values]

        metadata = {
            'version': .7,
            'fields': fields_list,
            'size': size_list,
            'type': type_list,
            'count': count_list,
            'width': number_points,
            'height': 1,
            'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            'points': number_points,
            'data': data_field  # binary
        }

        pc_pcd_format = pypcd.PointCloud(
            metadata=metadata,
            pc_data=dataframe_pointcloud[fmt_values].to_numpy(dtype=np.float32),
        )

        pypcd.point_cloud_to_path(pc_pcd_format, file_name)

    def get_number_points(self) -> int:
        return len(self.points.index)

    def get_column_fmt_string(self) -> str:
        """

        :return: "xyzi" or similar
        """
        columns = "".join(list(self.points))
        return columns

    def __add__(self, other: 'Pointcloud3D') -> 'Pointcloud3D':
        """
        Add points of both point clouds and returns a new object of pointcloud.
        The points of the resulting pointcloud are in ECEF coordinate system.

        :param other:
        :return: pointcloud which is the sum of two objects
        """
        assert isinstance(other, Pointcloud3D)

        # TODO(Dmytro) use panda dataframe for this, not convert to numpy

        # Note: if one of them has no points, we still go through the standard flow anyway.

        columns_this = self.get_column_fmt_string()
        columns_other = other.get_column_fmt_string()
        if columns_this != columns_other:
            other_bigger = columns_this.find(columns_other)
            this_bigger = columns_other.find(columns_this)
            logger.warning(
                "Adding clouds with different fields {} and {}. First is subset? {}. Using format of first".
                format(columns_this, columns_other, other_bigger))

        self_points = self.get_points_in_needed_coordinate_system(coordinate_system=CoordinateSystem.ECEF,
                                                                  fmt=columns_this).to_numpy()
        other_points = other.get_points_in_needed_coordinate_system(coordinate_system=CoordinateSystem.ECEF,
                                                                    fmt=columns_this).to_numpy()

        points = np.append(self_points, other_points, axis=0)

        result = Pointcloud3D(self.transformation_frame, CoordinateSystem.ECEF)
        result.init_from_numpy(points, fmt=columns_this)
        return result

    def __eq__(self, other: 'Pointcloud3D') -> bool:
        """

        :param other: Pointcloud3D
        :return: true if two objects are equal, false otherwise
        """
        assert isinstance(other, Pointcloud3D)

        columns_this = self.get_column_fmt_string()
        columns_other = other.get_column_fmt_string()
        if columns_this != columns_other:
            return False

        # check if number of points same
        if self.get_number_points() != other.get_number_points():
            return False

        self_points = self.get_points_in_needed_coordinate_system(
            coordinate_system=CoordinateSystem.ECEF,
            fmt=columns_this).to_numpy()
        other_points = other.get_points_in_needed_coordinate_system(
            coordinate_system=CoordinateSystem.ECEF,
            fmt=columns_this).to_numpy()

        rtol = 1.e-5
        atol = 1.e-5

        return np.allclose(self_points, other_points, rtol=rtol, atol=atol)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" File for working with keyframes and its content. All classes in here are part of keyframes.
    Author Dmytro Bobkov, Thomas Schmid, 2019
"""
import numpy as np
from typing import List, Optional

from libartipy.geometry import Pose
from libartipy.dataset import Constants, GPSLocation, get_logger
from libartipy.IO import get_timestamp_from_filename
logger = get_logger()


class FrameDataException(Exception):
    """
    Exception to indicate that there is a problem when parsing keyframe data from keyframe_ts.txt
    """
    pass


class FrameMetaData:
    def __init__(self,
                 calib_mat: np.ndarray,
                 image_width: int,
                 image_height: int,
                 num_of_points: int = 0,
                 timestamp_utc_ns: int = Constants.INVALID_TIMESTAMP,
                 pose: Pose = None,
                 gps_location: GPSLocation = None,
                 exposure_time_ms: float = Constants.INVALID_TIMESTAMP):

        assert calib_mat.shape == (3, 3)
        assert image_width > 0
        assert num_of_points <= image_width*image_height, \
            'number of points {} cannot be larger than image size'.format(num_of_points)

        self.calib_mat = calib_mat
        self.image_width = image_width
        self.image_height = image_height
        self.num_of_points = num_of_points

        self.timestamp_utc_ns = timestamp_utc_ns  # UTC timestamp this is the one used in GNSSposes.txt and provided in headers

        self.pose = pose  # camera to world coordinate system transformation
        self.gps_location = gps_location
        self.offset_where_data_points = 0  # when reading from file by lines
        self.exposure_time_ms = exposure_time_ms

    def get_exposure_time_ms(self):
        if self.exposure_time_ms >= 0:
            return self.exposure_time_ms
        else:
            logger.warning("Exposure unknown.")
            return None

    def init_timestamp_from_filename(self, file_name: str) -> None:
        """

        :param file_name:
        :return:
        """
        ts = get_timestamp_from_filename(file_name)
        self.set_timestamp_ns(ts)

    def get_gps_location(self) -> GPSLocation:
        return self.gps_location

    def get_timestamp_ns(self) -> Optional[int]:
        if self.timestamp_utc_ns == Constants.INVALID_TIMESTAMP:
            return None
        else:
            return self.timestamp_utc_ns

    def set_timestamp_ns(self, ts: int):
        self.timestamp_utc_ns = ts

    @classmethod
    def init_from_file_lines(cls, lines: List[str]) -> 'FrameMetaData':

        offset = 0
        if 'timestamp' in lines[offset]:
            try:
                # get time stamps in seconds
                str_timestamp = lines[offset + 1].split(',')
                assert len(str_timestamp) == 1, 'Timestamp str {}'.format(str_timestamp)
                timestamp_s = list(map(int, str_timestamp))
                assert len(timestamp_s) == 1, 'Timestamp not one value {}'.format(len(timestamp_s))
                timestamp_utc_ns = timestamp_s[0]  # Otherwise we get array with 1 element
            except AssertionError as e:
                raise FrameDataException("Timestamp information is wrong. {}".format(e))
            offset += 3
        else:
            timestamp_utc_ns = Constants.INVALID_TIMESTAMP  # get timestamp from the filename

        # get calibration information

        frame_info = list(map(float, lines[offset+1].strip().split(',')))
        try:
            assert len(frame_info) >= 6
            fx, fy, cx, cy = frame_info[0], frame_info[1], frame_info[2], frame_info[3]
            assert np.isfinite([fx, fy, cx, cy]).all(), 'Frame info not finite: {}'.format(frame_info)
        except AssertionError as e:
            raise FrameDataException("Camera intrinsics are wrong. {}".format(e))

        calib_mat = np.zeros([3, 3])
        calib_mat[0, 0] = fx  # fx
        calib_mat[1, 1] = fy  # fy
        calib_mat[0, 2] = cx  # cx
        calib_mat[1, 2] = cy  # cy
        calib_mat[2, 2] = 1.0

        image_width = int(frame_info[4])
        image_height = int(frame_info[5])
        num_of_points = int(frame_info[6])

        offset += 3

        # read pose information
        try:
            pose_this = Pose.from_line(lines[offset + 1])
        except AssertionError:
            raise FrameDataException("Pose cannot be initialized")
        offset += 3

        # Check for exposure time information
        if 'Exposure time (ms)' in lines[offset]:
            exposure_time_ms = float(lines[offset + 1])
            offset += 3
        else:
            exposure_time_ms = Constants.INVALID_TIMESTAMP

        # check for GPS information
        if 'GPS' in lines[offset]:
            try:
                gps_location = GPSLocation.init_from_line(lines, offset=offset + 1)
            except AssertionError as e:
                # do not raise exception, can keep going
                logger.warning("GPS location data problem {}".format(e))
                gps_location = None

            offset += 7
            # 7 lines because we have the following info
            """
            # GPS Data
            # GPSData: Covariance matrix
            0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000001

            # GPSData: GPS point
            66.702263, 26.950417, 0.588476"""
        else:
            gps_location = None
            offset += 0  # Do not change

        # the remaining lines are points!
        offset += 3  # Skip PC data header lines
        # Point Cloud Data :
        # u,v,idepth_scaled,idepth_hessian,maxRelBaseline,numGoodRes,status
        # color information

        a = cls(calib_mat, image_width, image_height, num_of_points,
                timestamp_utc_ns=timestamp_utc_ns,
                pose=pose_this,
                gps_location=gps_location,
                exposure_time_ms=exposure_time_ms)
        a.offset_where_data_points = offset
        return a

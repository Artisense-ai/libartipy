#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for defining constants
    Author Dmytro Bobkov, 2019
    Author Thomas Schmid, 2019
"""

import pkg_resources
from typing import Dict, List, Tuple
import json
from matplotlib import colors as mcolors
import numpy as np
import sys
import logging

from libartipy.IO import convert_color_from_hex_to_rgb


# keep it for backwards compatibility
class PrintColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""
    debug = "\x1b[34m"
    info = "\033[92m"
    warning = "\x1b[33m"
    error = "\x1b[31;21m"
    fatal = "\x1b[31;1m"
    reset = "\x1b[0m"

    # make format consistent with artislam log format
    format = "%(levelname)s %(filename)s:%(lineno)d %(asctime)s : %(message)s"
    date_format = '%H:%M:%S'

    FORMATS = {
        logging.DEBUG: debug + format + reset,
        logging.INFO: info + format + reset,
        logging.WARNING: warning + format + reset,
        logging.ERROR: error + format + reset,
        logging.CRITICAL: fatal + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, CustomFormatter.date_format)
        return formatter.format(record)


# global logger to avoid double prints, shared across libartipy
class Logger:
    logger = logging.getLogger("LibArtiPy")

    logger.setLevel(logging.INFO)  # set info level by default
    logger.propagate = False  # if you do not do this, then log messages will appear twice
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)  # it handles all levels, hence debug
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)


def get_logger():
    return Logger.logger


if sys.version_info < (3, 6):
    Logger.logger.fatal("Our official development version is Python3.6! You are using {}".format(sys.version_info))
    sys.exit()


class Constants:
    artifloat_type = np.longdouble

    def __init__(self, verbosity_level=None):
        """

        :param verbosity_level: keep it for backwards compatibility
        """
        # ASLAM output
        self.KF_DIR_NAME = 'KeyFrameData'
        self.GPS_POSES_FILE_NAME = 'GNSSPoses.txt'
        self.VELOCITY_FILE_NAME = 'Velocity.txt'
        self.TRANSFORMATION_FILE_NAME = 'Transformations.txt'

        # Calibration data, assumed that all other calib. files are inside CALIBRATION_FOLDER_NAME
        self.CALIBRATION_FOLDER_NAME = 'Calibration'
        self.CALIBRATION_YAML = 'camchain.yaml'
        self.CALIBRATION_DIST_CAMERA_0 = 'calib_0.txt'
        self.CALIBRATION_DIST_CAMERA_1 = 'calib_1.txt'
        self.CALIBRATION_DIST_STEREO = 'calib_stereo.txt'

        self.CALIBRATION_UNDIST_CAMERA_0 = 'undistorted_calib_0.txt'
        self.CALIBRATION_UNDIST_CAMERA_1 = 'undistorted_calib_1.txt'
        self.CALIBRATION_UNDIST_STEREO = 'undistorted_calib_stereo.txt'

        # image directories
        self.DISTORTED_IMAGE_FOLDER_NAME = 'distorted_images'
        self.UNDISTORTED_IMAGE_FOLDER_NAME = 'undistorted_images'
        self.LEFT_IMAGE_FOLDER_NAME = 'cam0'
        self.RIGHT_IMAGE_FOLDER_NAME = 'cam1'

        # For working with anonymization and distorted/undistorted images
        self.ANONYMIZATION_BBOX_FOLDER_NAME = 'AnonymBBoxes'

        # Semantic
        self.SEMSEG_LABEL_DIR_NAME = 'semseg_labels'
        self.SEMSEG_LABEL_DIR_DISTORTED_NAME = 'distorted_semseg_labels'
        self.SEMSEG_CLASS_MAPPING_FILE_NAME = 'semantic_class_mapping.json'
        self.SEMSEG_COLOR_MAPPING_FILE_NAME = 'color_map.json'

        self.classid_to_name_mapping = self.prepare_classid_to_name_mapping()  # Dict[str, int]
        self.classname_to_color_mapping = self.prepare_color_mapping()  # Dict[str, str] class to color mapping

        # FramePointFilter
        self.inv_depth_hessian_inv_pow_threshold = 0.001
        self.inv_depth_hessian_inv_threshold = 0.001
        self.min_rel_baseline = 0.1
        self.sparsify_factor = 1

    def get_semseg_class_mapping(self) -> Dict[str, int]:
        """
        Parse semseg class mapping dictionary
        https://stackoverflow.com/questions/37151414/how-to-load-json-from-resource-stream-in-python-3
        :return: dictionary
        """
        resource_package = __package__  # Could be any module/package name
        resource_path = '/'.join(("..", "IO", self.SEMSEG_CLASS_MAPPING_FILE_NAME))
        json_stream = pkg_resources.resource_stream(resource_package, resource_path)
        json_string = json_stream.read().decode()
        loaded_json = json.loads(json_string)
        return loaded_json

    def get_semseg_color_mapping(self) -> Dict[str, str]:
        """
        Parse semseg color mapping dictionary
        :return:
        """
        resource_package = __package__  # Could be any module/package name
        resource_path = '/'.join(("..", "IO", self.SEMSEG_COLOR_MAPPING_FILE_NAME))
        json_stream = pkg_resources.resource_stream(resource_package, resource_path)
        json_string = json_stream.read().decode()
        loaded_json = json.loads(json_string)
        return loaded_json

    def prepare_color_mapping(self) -> Dict[str, List[int]]:
        """
        This method creates the color mapping from class name to RGB value.
        :return:
        """
        color_mapping_data = self.get_semseg_color_mapping()
        assert len(color_mapping_data) > 0
        colors = dict(**mcolors.CSS4_COLORS)

        # create dictionary with color names and RGB values
        for color_name in colors:
            hex_color = colors[color_name]
            colors[color_name] = convert_color_from_hex_to_rgb(hex_color)

        # generate dictionary with class name and RGB values
        color_map = {}
        for key, val in color_mapping_data.items():
            color_map[key] = colors[val]
        return color_map

    def prepare_classid_to_name_mapping(self) -> Dict[int, str]:
        """
        This method creates the class mapping from class id to class name
        :return:
        """
        classname_to_classid_mapping = self.get_semseg_class_mapping()
        assert len(classname_to_classid_mapping) > 0
        # invert dictionary
        classid_to_classname_mapping = {v: k for k, v in classname_to_classid_mapping.items()}
        return classid_to_classname_mapping

    def convert_classid_to_classname(self, point_classes: np.ndarray) -> np.ndarray:
        """
        This method converts from Class ID to Class names
        :param point_classes
        :return:
        """

        assert self.classid_to_name_mapping is not None and len(
            self.classid_to_name_mapping) > 0, 'No Class mapping available'
        assert point_classes.size > 0

        classnames = np.vectorize(self.classid_to_name_mapping.get)(point_classes).squeeze()

        assert all(v is not None for v in classnames), 'Class mapping and color mapping do not match. ' \
            'ID exists is invalid or does not have a corresponding color'

        return classnames

    def convert_classid_to_classcolor(self, point_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method converts from class ID to RGB class colors and returns both,
        :param point_classes
        :return: RGB colors as well as class names. what dimensions?
        """

        classnames = self.convert_classid_to_classname(point_classes)

        assert self.classname_to_color_mapping is not None, 'No Color mapping available'

        class_ids = np.array([self.classname_to_color_mapping[classname] for classname in classnames], dtype=np.int16)

        return class_ids, classnames

    # static member
    INVALID_TIMESTAMP = -1

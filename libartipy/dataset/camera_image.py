#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for working with images
    Author Dmytro Bobkov, 2019
    Author Thomas Schmid
"""

import os
from enum import Enum, unique
import matplotlib.image as mpimg
import numpy as np
import imageio

from typing import Optional, List
from libartipy.dataset import Constants, get_logger

logger = get_logger()


class CameraImage(object):
    """
    Only loads the images if needed.
    """

    def __init__(self,
                 timestamp_ns: int,
                 file_path: Optional[str],
                 img_data: np.ndarray = None):
        if img_data is not None:
            self.img_data = img_data
        else:
            self.img_data = None  # lazy loading

        self.file_path = file_path

        self.timestamp_ns = timestamp_ns

    @classmethod
    def init_from_file(cls, file_path: str) -> 'CameraImage':
        """
        :param file_path:
        """
        assert os.path.exists(file_path)
        filename = os.path.split(file_path)[1]
        # get ext
        timestamp = CameraImage.get_timestamp_from_image_filename(filename)
        assert timestamp >= 0
        a = CameraImage(timestamp, file_path)
        return a

    def get_image_data(self, mode: str = "float") -> Optional[np.ndarray]:
        """

        :return: WxH (no 3 channels as our images are monochrome)
        """
        assert mode in ["float", "int"]
        if self.img_data is None:  # if img_data not loaded yet, load it now
            assert os.path.exists(self.file_path), "File does not exist {}".format(self.file_path)
            if mode == 'float':
                try:
                    img = mpimg.imread(self.file_path)  # loads it as HxWx3
                except:
                    logger.warning(" Failed to read from {}".format(self.file_path))
                    return None
            elif mode == 'int':
                try:
                    img = imageio.imread(self.file_path, pilmode='L')  # (8-bit pixels, black and white)
                except:
                    logger.warning(" Failed to read from {}".format(self.file_path))
                    return None
            else:
                logger.warning("Mode {} not implemented".format(mode))
                return None
            return img
        else:
            assert len(self.img_data.shape) >= 2, "Incorrect shape {}".format(self.img_data.shape)
            return self.img_data

    def set_image_data(self, img_data: np.ndarray):
        assert len(img_data.shape) >= 2, "Incorrect shape {}".format(img_data.shape)
        self.img_data = img_data

    @staticmethod
    def get_timestamp_from_image_filename(filename: str) -> int:
        basename = os.path.basename(filename)
        ts = int(os.path.splitext(basename)[0])  # get without extension
        assert ts > 0
        return ts

    def get_timestamp(self) -> int:
        return self.timestamp_ns


@unique
class CameraType(Enum):
    LEFT = 0
    RIGHT = 1

    @staticmethod
    def get_type_from_image_path(image_path: str, constants: Constants) -> 'CameraType':
        folder_name = os.path.basename(os.path.dirname(image_path))
        # constant = Constants()

        if folder_name == constants.LEFT_IMAGE_FOLDER_NAME:
            cam_type = CameraType.LEFT
        elif folder_name == constants.RIGHT_IMAGE_FOLDER_NAME:
            cam_type = CameraType.RIGHT
        else:
            cam_type = -1
            logger.warning("Not implemented for {}".format(folder_name))

        return cam_type

    @staticmethod
    def get_all_types() -> List['CameraType']:
        return [CameraType.LEFT, CameraType.RIGHT]


class CameraImageTuple(object):
    """
    Describes collection of camera images and their file paths that are taken at the same time.
    """

    def __init__(self):
        self.images = dict()  # Dict Type to Camera Image
        self.image_paths = dict()

    def __len__(self) -> int:
        return len(self.images)

    def add_image(self, type_cam: CameraType, image: CameraImage) -> None:
        self.images[type_cam] = image

    def get_image(self, type_cam: CameraType) -> Optional[CameraImage]:
        if type_cam in self.images.keys():
            return self.images[type_cam]
        else:
            logger.warning("Image of type {} does not exist for this tuple".format(type_cam))
            return None

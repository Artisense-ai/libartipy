#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class to provide mask predictions from semseg for a given timestamp
    Author Dmytro Bobkov, 2019
    Author Thomas Schmid
"""

import glob
import os
from typing import Dict, Optional

import cv2
import numpy as np

from libartipy.IO import get_timestamp_from_filename
from libartipy.dataset import Constants, CameraType, DistortionMapper, CameraImage, Calibration, \
    DistortionModel, get_logger

logger = get_logger()


class SemsegImageLabelProvider(object):
    def __init__(self,
                 in_dataset_dir: str,
                 in_calibration_dir: str,
                 dataset_constants: Constants):

        # We only use left camera image for predictions
        semseg_pred_dir = os.path.join(in_dataset_dir, dataset_constants.SEMSEG_LABEL_DIR_NAME,
                                       dataset_constants.LEFT_IMAGE_FOLDER_NAME)

        distorted_images_used_semseg = False  # this is the default setting in our pipeline.

        if not os.path.exists(semseg_pred_dir):
            semseg_pred_dir = os.path.join(in_dataset_dir, dataset_constants.SEMSEG_LABEL_DIR_DISTORTED_NAME,
                                           dataset_constants.LEFT_IMAGE_FOLDER_NAME)
            distorted_images_used_semseg = True

        self.semseg_pred_dir = semseg_pred_dir
        if os.path.exists(self.semseg_pred_dir):
            self.semseg_pred_images = SemsegImageLabelProvider.parse_pred_images(self.semseg_pred_dir)
        else:
            logger.debug("Directory {} does not exist. No labels to parse.".format(semseg_pred_dir))
            self.semseg_pred_images = {}

        self.mapper = None
        self.calibration = None
        if os.path.exists(in_calibration_dir) and distorted_images_used_semseg:
            self.calibration = Calibration(calib_folder_path=in_calibration_dir)

            if self.calibration.distortion_model == DistortionModel.Equidistant:
                self.mapper = DistortionMapper(calib_data=self.calibration)
                logger.info("Initialized mapper from {}.".format(in_calibration_dir))
            else:
                logger.debug("Not initialiazing mapper.")

        # Init class mapping and colormap
        self.class_mapping = dataset_constants.prepare_classid_to_name_mapping()
        self.color_mapping = dataset_constants.prepare_color_mapping()

    @staticmethod
    def parse_pred_images(semseg_pred_dir: str) -> Dict[int, CameraImage]:
        exts_semseg_label = ['.png', '.jpg']
        semseg_pred_images = {}

        semseg_label_files = []
        for ext_semseg_label in exts_semseg_label:
            semseg_label_files_ext = [f for f in glob.glob(os.path.join(semseg_pred_dir, '*' + ext_semseg_label))]
            for semseg_label_file in semseg_label_files_ext:
                semseg_label_files.append(semseg_label_file)

        assert (len(semseg_label_files) >= 1), 'No files in {}'.format(semseg_pred_dir)

        for img_filepath in semseg_label_files:

            img_filepath = os.path.split(img_filepath)[1]  # remove parent folders
            filename = os.path.join(semseg_pred_dir, img_filepath)
            assert os.path.exists(filename), "{} does not exist".format(filename)
            ts = get_timestamp_from_filename(img_filepath)
            semseg_pred_images[ts] = CameraImage(timestamp_ns=ts, file_path=filename)

        logger.info("Parsed {} semseg images".format(len(semseg_pred_images)))
        return semseg_pred_images

    def get_semseg_pred_for_timestamp(self, timestamp_ns: int) -> Optional[np.ndarray]:
        """

        :param timestamp_ns:
        :return: np.ndarray with WxH
        """
        if timestamp_ns not in self.semseg_pred_images.keys():
            return None
        else:
            img = self.semseg_pred_images[timestamp_ns]
            img_data = img.get_image_data(mode="int")
            if self.mapper is not None:
                # nearest interpolation because we do not want to interpolate between classes
                # as this would create new classes
                img_data = self.mapper.undistort_image(img_data, CameraType.LEFT, interpolation=cv2.INTER_NEAREST)
            return img_data

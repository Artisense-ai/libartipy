#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for working with frame point anonymization
    Author Qing Cheng, 2019
    referring to libartipy/dataset/frame_point_filter.py
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import json
import glob
import cv2
import os

from libartipy.dataset import CameraType, get_logger
logger = get_logger()


class FramePointAnonymizer:
    def __init__(self, score_threshold: float = 0.3):
        self.BBoxes = {}  # Dict[int, List[Dict[str, float]]]
        self.score_threshold = score_threshold

    def set_score_threshold(self, score_threshold: float):
        self.score_threshold = score_threshold

    def read_bboxes(self, bboxes_dir_path: str) -> None:
        assert os.path.exists(bboxes_dir_path), "Dir with bounding boxes {} does not exist.".format(bboxes_dir_path)
        bbox_files = glob.glob(bboxes_dir_path + '/*.json')
        logger.debug("Found {} Bboxes".format(len(bbox_files)))

        for bbox_file in bbox_files:
            timestamp = int(os.path.basename(bbox_file).split('.')[0])  # timestamp given in the filename
            with open(bbox_file, 'r') as f:
                data = json.load(f)
            self.BBoxes[timestamp] = data

        logger.debug('{} anonymization bounding boxes have been parsed successfully'.format(len(self.BBoxes)))

    def get_bbox(self, timestamp: int):
        return self.BBoxes.get(timestamp, None)

    def consider_frame_points_anonymized(self,
                                         frame_point_dataframe: pd.DataFrame,
                                         image_size: tuple,
                                         bboxes: List[Dict[str, float]],
                                         rectified_fcn,
                                         verification: bool = False,
                                         save_dir: str = None,
                                         image_path: str = None) -> np.ndarray:
        """
        This function removes points from framepoints of keyframes that fall into bounding boxes given by anonymizer.
        :param frame_point_dataframe:
        :param image_size:
        :param bboxes:
        :param rectified_fcn: rectified_fcn accepts 2d numpy array of size [row, col]
        :param verification: if true, we save images, otherwise we don't save anything to file.
        :param save_dir:
        :param image_path:
        :return:
        """

        if verification:
            assert os.path.exists(image_path), "Image {} does not exist".format(image_path)
            img = cv2.imread(image_path)

        height, width = image_size

        coord_u = frame_point_dataframe['coord_u'].to_numpy()
        coord_v = frame_point_dataframe['coord_v'].to_numpy()
        mask = np.zeros(shape=image_size)
        n = len(frame_point_dataframe.index)

        # idx = np.arange(n)
        idx = np.arange(1, n+1)

        # use coordinate as array indexes, and idx as array values
        mask[coord_v, coord_u] = idx

        points_to_be_deleted = set()

        if bboxes is not None:
            for bbox in bboxes:

                if bbox['x_min'] < self.score_threshold:
                    continue
                u_min = bbox['x_min']
                u_max = bbox['x_max']
                v_min = bbox['y_min']
                v_max = bbox['y_max']

                if u_min >= u_max or v_min >= v_max:
                    logger.warning("Invalid box values {}".format(bbox))
                    continue

                rows = [v_min, v_max]
                cols = [u_min, u_max]
                dist_coords = np.array([rows, cols])
                if rectified_fcn is not None:
                    rectified_coords = rectified_fcn(dist_coords=dist_coords, camera_position=CameraType.LEFT)
                else:
                    rectified_coords = dist_coords
                v_min, v_max, u_min, u_max = rectified_coords.flatten()

                # make int
                shift = 0.5
                u_min = min(max(0, int(u_min+shift)), width-1)
                u_max = min(max(0, int(u_max+shift)), width-1)
                v_min = min(max(0, int(v_min+shift)), height-1)
                v_max = min(max(0, int(v_max+shift)), height-1)

                points = mask[v_min:v_max, u_min:u_max]
                points_to_be_deleted.update((points[points > 0]))

                if verification:
                    img = cv2.rectangle(img, (u_min, v_min), (u_max, v_max), (255, 255, 0), 2)

        points_to_be_deleted = list(points_to_be_deleted)
        points_to_be_deleted = [x-1 for x in points_to_be_deleted]

        # to delete points
        considered = np.ones(n, dtype=np.bool)
        considered[np.array(points_to_be_deleted, dtype=int)] = False

        # to save the anonymized points
        # considered = np.zeros(n, dtype=np.bool)
        # considered[np.array(deleted, dtype=int)] = True

        if verification:
            assert os.path.isdir(save_dir), '{} does not exist'.format(save_dir)
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, img)
            logger.debug('Anonymized image is written to {}'.format(save_path))

        return considered

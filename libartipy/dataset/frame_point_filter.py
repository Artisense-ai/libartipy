#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for working with frame point filtering
    Author Dmytro Bobkov, 2019
    Author Thomas Schmid
"""

import numpy as np
import pandas as pd
from libartipy.dataset import get_logger
logger = get_logger()


class FramePointFilter:
    def __init__(self,
                 inv_depth_hessian_inv_pow_threshold: float = 0.001,
                 inv_depth_hessian_inv_threshold: float = 0.001,
                 min_rel_baseline: float = 0.1,
                 sparsify_factor: int = 1):
        """

        :param inv_depth_hessian_inv_pow_threshold:
        :param inv_depth_hessian_inv_threshold:
        :param min_rel_baseline:
        :param sparsify_factor:
        """
        # threshold values as per pointcloudviewer/browse/src/Pangolin/KeyFrameDisplay.cpp ll. 49 - 52
        self.inv_depth_hessian_inv_pow_threshold = inv_depth_hessian_inv_pow_threshold  # 0.001 (default)
        self.inv_depth_hessian_inv_threshold = inv_depth_hessian_inv_threshold  # 0.001 (default)
        self.min_rel_baseline = min_rel_baseline  # 0.1 (default)
        self.sparsify_factor = sparsify_factor  # 1 (default)

    def consider_frame_points_vectorized(self, frame_point_dataframe: pd.DataFrame) -> np.ndarray:
        """

        :param frame_point_dataframe: pandas dataframe
        :return: Nx1 with True if to consider, False otherwise
        """

        # to make division by zero not throw exception, we follow
        # https://stackoverflow.com/questions/10011707/how-to-get-nan-when-i-divide-by-zero
        # and cast to np.float64 so that we get NaN which is properly handled
        inv_depth = frame_point_dataframe['inv_depth'].to_numpy().astype(np.float64)
        inv_depth_hessian = frame_point_dataframe['inv_depth_hessian'].to_numpy()
        max_rel_baseline = frame_point_dataframe['max_rel_baseline'].to_numpy()
        n = len(frame_point_dataframe.index)
        considered = np.ones(n, dtype=np.bool)

        # don't consider points with negative depth values
        considered = considered & (inv_depth > 0)

        # Supress warning as we correctly handle division by 0 using mask considered
        with np.errstate(divide='ignore'):
            depth = 1. / inv_depth
        depth4 = depth ** 4

        #
        inv_depth_hessian_inverted = 1. / (inv_depth_hessian + 0.01)

        considered = considered & \
            np.logical_not(inv_depth_hessian_inverted * depth4 > self.inv_depth_hessian_inv_pow_threshold)
        N_original = inv_depth.size
        N1 = N_original - sum(considered)
        logger.debug("FramePointFilter: inv_depth_hessian_inv_pow_threshold, {}/{}".format(N1, N_original))

        # remove points
        considered = considered & \
            np.logical_not(inv_depth_hessian_inverted > self.inv_depth_hessian_inv_threshold)
        N2 = N_original - sum(considered) - N1
        logger.debug("FramePointFilter: inv_depth_hessian_inv_threshold, {}/{}".format(N2, N_original))

        # remove points where max_rel_baseline < min_rel_baseline
        considered = considered & \
            np.logical_not(max_rel_baseline < self.min_rel_baseline)
        N3 = N_original - sum(considered) - N2
        logger.debug("FramePointFilter: min_rel_baseline, {}/{}".format(N3, N_original))

        # randomly removes points. Does not do anything if less or equal to 1.
        considered = considered & \
            np.logical_not((self.sparsify_factor > 1) &
                           (np.random.rand(n) % self.sparsify_factor != 0))
        N4 = N_original - sum(considered) - N3
        logger.debug("FramePointFilter: sparsify_factor, {}/{}".format(N4, N_original))

        return considered

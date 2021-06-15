#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for keyframe utilities
    Author Thomas Schmid, 2019
"""

from typing import List, Iterator, Dict, Union
from enum import Enum

from libartipy.dataset import Dataset, KeyFrame


class DepthDisparityImageType(Enum):

    DEPTH = 0
    I_DEPTH = 1
    DISPARITY = 2
    NORMALIZED_DISPARITY = 3

    def __str__(self):
        if self == DepthDisparityImageType.DEPTH:
            return 'depth_image'
        if self == DepthDisparityImageType.I_DEPTH:
            return 'inverse_depth_image'
        if self == DepthDisparityImageType.DISPARITY:
            return 'disparity_image'
        if self == DepthDisparityImageType.NORMALIZED_DISPARITY:
            return 'normalized_disparity_image'
        raise ValueError()


class FrameDiffer(Enum):
    TARGET_FRAME = 0
    OTHER_FRAMES_LIST = 1

    def __str__(self):
        if self == FrameDiffer.TARGET_FRAME:
            return "target_frame"
        if self == FrameDiffer.OTHER_FRAMES_LIST:
            return "other_frames_list"
        raise ValueError()


class KeyFrameProvider(object):
    """ This class reads provides a target keyframe as well as a keyframe list according to the number of keyframes 
    before and after which can then be used to be accumulated. """

    def __init__(self, num_frames_before: int = 7, num_frames_after: int = 0):

        assert num_frames_before >= 0, "Index before {} smaller than 0".format(num_frames_before)
        assert num_frames_after >= 0, "Index after {} smaller than 0".format(num_frames_after)

        # number of frames before and after keyframe
        self.num_frames_before = num_frames_before
        self.num_frames_after = num_frames_after

    def sample_keyframes(self, dataset: Dataset, timestamps_list: List[int], sample_frequency: int) -> \
            Iterator[Dict[FrameDiffer, Union[KeyFrame, List[KeyFrame]]]]:
        """
        samples timestamp from a list of timestamps and returns target keyframe and a list of other keyframes
        :param dataset:
        :param timestamps_list:
        :param sample_frequency:
        :return:
        """

        num_target_frames = len(timestamps_list)

        for idx_target in range(num_target_frames)[::sample_frequency]:

            # gaher target and other keyframe timestamps
            ts_dict = self._accumulate_timestamps(timestamps_list, idx_target=idx_target)

            # get keyframes from dataset by timestamp
            target_frame = dataset.get_keyframe_with_timestamp(ts_dict[FrameDiffer.TARGET_FRAME])
            other_frames = [dataset.get_keyframe_with_timestamp(ts)
                            for ts in ts_dict[FrameDiffer.OTHER_FRAMES_LIST]]

            yield {FrameDiffer.TARGET_FRAME: target_frame,
                   FrameDiffer.OTHER_FRAMES_LIST: other_frames}

    def select_keyframes(self, dataset: Dataset, timestamps_list: List[int], preselected_ts: List[int]) \
            -> Iterator[Dict[FrameDiffer, Union[KeyFrame, List[KeyFrame]]]]:
        """
        returns preselected target keyframe and a list of other keyframes
        :param dataset:
        :param timestamps_list:
        :param preselected_ts:
        :return:
        """

        for ts_target_frame in preselected_ts:

            # get index of target timestamp and calculate neighboring timestamps
            idx_target = timestamps_list.index(ts_target_frame)
            ts_dict = self._accumulate_timestamps(timestamps_list, idx_target=idx_target)

            # select target keyframe and other keyframes from dataset
            target_frame = dataset.get_keyframe_with_timestamp(ts_target_frame)
            other_frames = [dataset.get_keyframe_with_timestamp(ts)
                            for ts in ts_dict[FrameDiffer.OTHER_FRAMES_LIST]]

            yield {FrameDiffer.TARGET_FRAME: target_frame,
                   FrameDiffer.OTHER_FRAMES_LIST: other_frames}

    def _accumulate_timestamps(self, timestamp_list: List[int], idx_target: int) \
            -> Dict[FrameDiffer, Union[int, List[int]]]:
        """
        calculates the timestamps that shall be accumulated into the target frame
        :param timestamp_list:
        :param idx_target:
        :return:
        """
        num_target_frames = len(timestamp_list)

        num_frames_before = self.num_frames_before
        num_frames_after = self.num_frames_after

        # perform handling of keyframe timestamps at list boundary
        num_frames_before = num_frames_before if (idx_target - num_frames_before) > 0 else idx_target
        num_frames_after = num_frames_after if (idx_target + num_frames_after) < (num_target_frames - 1) \
            else (num_target_frames - 1 - idx_target)

        # get target and other timestamps
        ts_target_frame = timestamp_list[idx_target]
        ts_other_frames = timestamp_list[idx_target - num_frames_before:idx_target] + \
            timestamp_list[idx_target + 1:idx_target + 1 + num_frames_after]

        return {FrameDiffer.TARGET_FRAME: ts_target_frame,
                FrameDiffer.OTHER_FRAMES_LIST: ts_other_frames}

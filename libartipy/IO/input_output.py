#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dedicated to reading/writing to files.
# Author Dmytro Bobkov, 2019

import argparse
import os
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import cv2


def get_timestamp_from_keyframe_filename(filename: str) -> int:
    """
    Extract timestamp from keyframe filename based on some naming conventions

    :param filename:
    :return:
    """

    _, ext = os.path.splitext(filename)
    assert ext == ".txt"

    ts = get_timestamp_from_filename(filename)

    return ts


def filename_sort(file_list: List[str]) -> List[str]:
    """
    checks first two elements for same word prefix and sorts keyframe files by their
    timestamps whilst maintaining their absolute path
    :param file_list: list of filenames
    :return: sorted list of filenames
    """
    assert len(file_list) > 1, 'for one element, cannot sort.'

    # split filename in the form KeyFrame_123.txt into KeyFrame and 123.txt
    filename = os.path.basename(file_list[0])
    word_prefix = filename.split('_')[0]
    assert word_prefix in file_list[1]

    sorted_list = sorted(file_list,
                         key=lambda fname: get_timestamp_from_keyframe_filename(fname))

    assert len(sorted_list) == len(file_list)

    return sorted_list


def convert_color_from_hex_to_rgb(value: str) -> List[int]:
    """
    converts a hex encoded colors to rgb encoded colors

    :param value: hex encoded color
    :return: rgb value
    """
    value = value.lstrip('#')
    lv = len(value)

    return list(tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))


def get_timestamp_from_filename(filename: str) -> int:
    """

    :param filename: filepath
    :return: timestamp
    :raise AssertionError if conversion fails
    """
    base = os.path.basename(filename)
    file_only, _ = os.path.splitext(base)
    parts = file_only.split("_")
    assert len(parts) > 0, "No TS in {}".format(file_only)
    numerical_part = parts[-1]  # Timestamp on the last position in filename
    assert numerical_part.isdigit(), 'File {} does not contain number on the last position'.format(file_only)
    ts = int(numerical_part)
    # TODO(Dmytro) handle with exception except ValueError:
    return ts


def data_to_16bit_png(filepath: str, img_array: np.array) -> str:
    """
    Saves arrays to file as 16 bit png

    :param filepath:
    :param img_array: expects img_array to be np.uint16
    :return: path to the file where it is written
    """

    assert img_array.dtype == np.uint16, "Input array type is not np.uint16!"
    filepath = filepath + ".png"
    cv2.imwrite(filepath, img_array.astype(np.uint16))
    return filepath


def encode_data_image_to_16bit(data_image: np.ndarray, max_data_value: int = 120) -> np.ndarray:
    """
    this method sets all data values above max_data_value to zero, scales it by the max_data_value
    and rescales the depth image to the uint16 range.

    :param data_image:
    :param max_data_value:
    :return: image data in 16-bit format
    """
    # only consider depth values within max distance and normalize on that
    data_image[data_image > max_data_value] = 0
    data_image = data_image / max_data_value

    # scale depth image to uint16 range
    data_image = data_image * np.iinfo(np.uint16).max
    return data_image.astype(np.uint16)


def decode_data_image_from_16bit_file(fpath: str, max_data_value: int = 120) -> np.ndarray:
    """
    this method performs inverse operation of encode_data_image, by unscaling uint16 range,
    and rescaling the data_image with max_data_value.

    :param fpath: filename
    :param max_data_value: maximum value of range
    :return: image data as array
    """

    # read uint16 depht values unchanged from file
    data_image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)

    # undo uint16 range and scale with max depth
    data_image = data_image / np.iinfo(np.uint16).max
    data_image *= max_data_value

    return data_image


def image_to_plasma_png(fname: str, img: np.array) -> None:
    """
    saves images to file using the plasma color scheme

    :param fname:
    :param img:
    """
    plt.imsave(fname + '.png', img, cmap='plasma')


def str2bool(v: Any) -> bool:
    """
    Converts multiple possible boolean string encryptions to pythonic True and False.

    :param v: string or boolean
    :return: true or false depending on the given value
    :raise ArgumentTypeError if conversion fails
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" File for preprocessing keyframe files.
    Author Thomas Schmid, 2019
"""

import glob
import json
import os

from libartipy.IO import get_timestamp_from_filename


class JsonBlobCreator(object):
    """ Converts keyframe txt files to json blob (key: timestamp, value: txt file content) and vice versa. """

    @staticmethod
    def json_blob_from_keyframe_txts(keyframe_folder: str, json_blob_path: str) -> None:
        """
        gets all txt files from folder and combines them into single json blob with keys being timestamp and value
        being the file content.
        :param keyframe_folder:
        :param json_blob_path:
        :return:
        """

        txt_file_list = [file for file in glob.glob(os.path.join(keyframe_folder, "KeyFrame_*.txt"))]

        assert len(txt_file_list) >= 1, "No Keyframe txt files provided"

        tmp_json_dump = {}

        for filepath in txt_file_list:

            file_name = os.path.basename(filepath)
            ts = get_timestamp_from_filename(file_name)

            with open(filepath, "r") as f:
                content = f.readlines()

                # merge all content items to one string
                content = "".join(content)
                # content = content.split("\n")
                tmp_json_dump[ts] = content

        with open(json_blob_path, 'w') as f:
            json.dump(tmp_json_dump, f, sort_keys=True)

    @staticmethod
    def keyframe_txts_from_json_blob(json_blob_path: str, keyframe_folder: str) -> None:
        """
        inverse operation to 'create_json_blob_from_keyframe_txts'. Recreates all keyframe txt files from json blob.
        :param json_blob_path:
        :param keyframe_folder:
        :return:
        """

        assert os.path.join(json_blob_path), "Json blob file does not exist!"

        with open(json_blob_path, 'r') as f:
            json_dict = json.load(f)

        for timestamp_key in json_dict:

            # create filepath and write txt content to file
            cur_filename = os.path.join(keyframe_folder, "KeyFrame_{}.txt".format(timestamp_key))
            with open(cur_filename, "w") as f:
                f.write(json_dict[timestamp_key])

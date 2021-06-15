#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Class for SOR on pointcloud
    Author Nicholas Gao, 2019
    http://www.pointclouds.org/documentation/tutorials/statistical_outlier.php

"""
from libartipy.pointcloud import Pointcloud3D
from libartipy.geometry import CoordinateSystem
from scipy.spatial import cKDTree


class SorFilter(object):

    def __init__(self, num_neighbors: int = 6, n_sigma: float = 1.0):
        self.num_neighbors = num_neighbors
        self.n_sigma = n_sigma

    def filter(self,
               pointcloud: Pointcloud3D) -> Pointcloud3D:
        """
        Returns a new filtered point cloud in the same coordinate system
        :param pointcloud:
        :return:
        """

        fmt = pointcloud.get_column_fmt_string()

        # Use any cartesian coordinate system, if WGS84 convert to ECEF
        input_cs = pointcloud.coordinate_system
        internal_cs = input_cs if input_cs != CoordinateSystem.WGS84 else CoordinateSystem.ECEF

        pc = pointcloud.get_points_in_needed_coordinate_system(internal_cs, fmt=fmt).to_numpy()
        points = pc[:, :3]

        tree = cKDTree(points, balanced_tree=False)

        distances = tree.query(points, self.num_neighbors + 1, n_jobs=-1)[0][:, 1:]
        mean_distances = distances.mean(1)
        mean = mean_distances.mean()
        std = mean_distances.std()

        max_distance = mean + self.n_sigma * std

        mask = mean_distances <= max_distance

        result = Pointcloud3D(pointcloud.transformation_frame, internal_cs)
        result.init_from_numpy(pc[mask], fmt=fmt)
        # Ensure output and input coordinate system correspond
        result = result.get_point_cloud_in_coordinate_system(input_cs)
        return result

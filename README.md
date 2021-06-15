# libartipy

## Dependencies

Most of dependencies defined in `requirements.txt`. Install as follows:

```bash
# 1. Correct python3.6 version.
# make sure you have python3.6 installed as this is the supported version!
python3.6 --version
# if not installed, see https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get
# to install python3 required system packages
sudo apt-get install python3-tk

# 2. Virtualenv. Install virtualenv if not installed
pip3 install --user virtualenv

# 3. create virtual environment
virtualenv --python=python3.6 --no-site-packages venv36

# 4. activate virtual environment
source venv36/bin/activate

# 5. install all dependencies of this module
pip3 install -r requirements.txt
```

**Note**: we use python 3.6 together with virtualenv.

After dependencies are installed, install the python library as follows by running inside this repo folder:

```bash
pip3 install .
```

## Structure

Modules are as follows:
* **dataset**. It provides functionality to work the data provided as output by DSO. It includes dataset_slam that is a high-level API to instantiatate dataset from folder. Dataset contains transformation frame, GPS poses, images and keyframes.
* **geometry**. Provides functionality to work with poses and geometric transformations.
* **IO**. Provides functionality to input/output to/from files.
* **pointcloud**. Functionality to work point cloud class, read/write to file.

## Dataset folder structure

We assume that the dataset folder contains the following files/folders:
* KeyFrameData: this folder contains keyframe_*.txt keyframe information
* GNSSPoses.txt: this file contains GPS poses that correspond to the keyframes
* Transformation.txt: this file describes all the transformations between SLAM world, GPS, IMU and camera coordinate systems.
* (optional) distorted_images: this folder contains distorted images captured with cameras correspondingly in folders _cam0_ and _cam1_ for both cameras.
* (optional) undistorted_images: this folder contains undistorted (rectified) images captured with cameras correspondingly in folders _cam0_ and _cam1_ for both cameras.
* (optional) Calibration: this folder contains calibration information for left and right images intrinsics and extrinsics (files _calib_0.txt, calib_1.txt, calib_stereo.txt_).

Fore more information on the dataset structure, see [4Seasons dataset structure](https://www.4seasons-dataset.com/documentation)

If you have a different naming, simply change the corresponding fields in dataset/constants.py.

## Example how to use

```python
from libartipy.dataset import Dataset, CameraType
from libartipy.geometry import CoordinateSystem

berlin_new_format_dir = '/tmp/'
dataset_new = Dataset(berlin_new_format_dir)

dataset_new.parse_keyframes() # parse keyframes and their points
dataset_new.set_keyframe_poses_to_gps_poses() # if you want to use GPS poses (if available)
dataset_new.process_keyframes_filter_points() # filter frame points in keyframes according to certain criteria

# To iterate over the keyframes, we first obtain timestamps. By using timestamps we get a specific keyframe.
for ts in dataset_new.get_all_kf_timestamps():
    print("Dataset frame has {} points".format(len(dataset_new.get_keyframe_with_timestamp(ts).get_frame_points().index)))

dataset_new.accumulate_framepoints_into_pointcloud()

file_las = '/tmp/pcd_out/out.las' # LAS, PLY, H5, PCD, CSV file formats supported.
dataset_new.pointcloud.write_to_file(file_las, coordinate_system=CoordinateSystem.ECEF, fmt="xyzi")

# to get distorted or undistorted stereo images, query them by timestamps
for ts in dataset_new.get_all_kf_timestamps():
    image_tuple = dataset_new.get_undistorted_stereo_images_with_timestamp(ts)
    img_left = image_tuple.get_image(CameraType.LEFT)
    img_left_data = img_left.get_image_data() # this is numpy array
    img_right = image_tuple.get_image(CameraType.RIGHT)
```

## License

This python library was developed at Artisense and is licensed under the MIT License.

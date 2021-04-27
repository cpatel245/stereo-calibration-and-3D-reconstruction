"""utility functions to obtain disparity map and depth from clicked stereo imaegs and stored camera calibration parameters

 - reads the camera calibration parameters from firectory
 - computes disparity maps
 - implements various utility functions to compute depth and location of the pixel in an image

"""

import math
import pickle
import cv2
import numpy as np
import glob


class StereoParameters:
    """Implements the methods to read the various calibration parameters, which are used for triangulation
    """

    def __init__(self, calib_param_dir):
        """Initiates the instance of an class

        Args:
            calib_param_dir (str): path of the directory where calibration parameters are stored in a pickle (.p) file format
        """
        self.parameters_dir = str(glob.glob(calib_param_dir + '/*')[-1])
        self.stereo_parametrs()

    def stereo_parametrs(self):
        """Reads the parameters from the saved pickle files
        """
        try:
            print(f"\n\n[INFO] Loading Calibration parameters from '{self.parameters_dir}' directory...")
            with open(self.parameters_dir + '/left_undistort_map.p', 'rb') as file:
                self.Left_Stereo_Map = pickle.load(file)
            with open(self.parameters_dir + '/right_undistort_map.p', 'rb') as file:
                self.Right_Stereo_Map = pickle.load(file)
            with open(self.parameters_dir + '/reprojection_matrix.p', 'rb') as file:
                self.Q = pickle.load(file)

        except:
            raise ValueError('provide valid directory path. provided path: {} is not valid'.format(self.parameters_dir))

    def get_focal_length(self):
        """Returns the focal length of the stereo camera from reprojection metrics
        Returns:
            focal length of the stereo camera system in a pixel units

        """
        self._focal_length = self.Q[2, 3]
        return self._focal_length

    def left_camera_center(self):
        """
        Returns:
            left camera principal point coordinates obtained from reprojection metrics

        """
        left_camera_center = self.Q[:2, 3]
        return left_camera_center

    def rectified_pair(self, imgL, imgR):
        """Rectifies the captured stereo image pair using un-distortion maps

        Args:
            imgL (array): left stereo image in a numpy array format
            imgR (array): right stereo image in a numpy array format

        Returns:
            rectified image pair

        """
        imgL_rect = cv2.remap(imgL, self.Left_Stereo_Map[0], self.Left_Stereo_Map[1], cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT, 0)
        imgR_rect = cv2.remap(imgR, self.Right_Stereo_Map[0], self.Right_Stereo_Map[1], cv2.INTER_LANCZOS4,
                              cv2.BORDER_CONSTANT, 0)
        return imgL_rect, imgR_rect

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
        self.parameters_dir = calib_param_dir
        self.stereo_parametrs()

    def stereo_parametrs(self):
        """Reads the parameters from the saved pickle files
        """
        try:
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


class Disparity(StereoParameters):
    """Implements the methods to compute disparity map. Inherits the StereoParameters class to obtain the calibration parameters
    - computes raw disparity map which is used for depth calculation
    - computes filtered disparity maps for visualization purpose
    """

    def __init__(self, imgL, imgR, calib_param_dir):
        """Initiates the instance of the class when methos is called

        Args:
            imgL (array): left image in the array format (usually read using OpenCV)
            imgR (array): right image in the array format (usually read using OpenCV)
            calib_param_dir (str):   path of the directory where calibration parameters are stored in a pickle (.p) file format
        """
        super().__init__(calib_param_dir)
        self._img_l = imgL
        self._img_r = imgR
        self.block_matcher()

    def block_matcher(self):
        """Initiate the instance of the block matching method from OpenCV to compute disparity
        """

        # Create StereoSGBM class instance and prepare all parameters
        window_size = 7
        min_disp = 1
        num_disp = 129 - min_disp
        self._blockmatcher = cv2.StereoSGBM_create(minDisparity=min_disp,
                                                   numDisparities=num_disp,
                                                   blockSize=window_size,
                                                   uniquenessRatio=10,
                                                   speckleWindowSize=50,
                                                   speckleRange=1,
                                                   preFilterCap=9,
                                                   disp12MaxDiff=-1,
                                                   mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                                                   P1=1 * 8 * 3 * window_size,
                                                   P2=1 * 32 * 3 * window_size)

        lmbda = 80000
        sigma = 3.2

        # create instances and set parameters of the WLS filter for postprocessing of the disparity map
        # WLS is disparity map filter based on Weighted Least Squares filter
        # Used to refine the results in half-occlusions and uniform areas.
        self._BMright = cv2.ximgproc.createRightMatcher(self._blockmatcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self._blockmatcher)
        self.wls_filter.setLambda(lmbda)
        self.wls_filter.setSigmaColor(sigma)

    def get_rectified_pair(self):
        """returns rectified stereo image pair using un-distortion maps
        Returns:
            rectified stereo image pair

        """
        imgL_rect, imgR_rect = self.rectified_pair(imgL=self._img_l, imgR=self._img_r)
        return (imgL_rect, imgR_rect)

    def get_raw_disparity(self):
        """Computes raw disaprity map from rectified stereo image pair. this disparity map is used for triangulation

        Returns:
            raw-disparity map obtained from block matching algorithm, without any post processing

        """
        # obtain rectified image pair from original captured stereo image pair
        imgL_rect, imgR_rect = self.rectified_pair(imgL=self._img_l, imgR=self._img_r)

        # convert rectified images to gray from BGR(color)
        grayR = cv2.cvtColor(imgR_rect, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2GRAY)

        # Compute the disparity map from rectified stereo images
        raw_disp = self._blockmatcher.compute(grayL, grayR).astype(np.float32) / 16

        return raw_disp

    def get_disparity(self):
        """Returns raw disparity map and also disparity maps with postprocessing for better visualization
        adopted from : https://github.com/LearnTechWithUs/Stereo-Vision
        Returns:
            raw_disparity map without any post processing and filtered greyscale and colored disparity map for better visualization

        """
        # rectify stereo image pair before computing disparity map
        imgL_rect, imgR_rect = self.rectified_pair(imgL=self._img_l, imgR=self._img_r)

        # convert rectified images to gray from BGR(color)
        grayR = cv2.cvtColor(imgR_rect, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2GRAY)

        # Compute the disparity map from rectified stereo images
        raw_disp = self._blockmatcher.compute(grayL, grayR).astype(np.float32) / 16

        dispL = raw_disp
        dispR = self._BMright.compute(grayR, grayL)
        dispL = np.int16(dispL)
        dispR = np.int16(dispR)

        # Filtering the Results with a closing filter
        # In the computed disparity map there is still a lot of noise
        # A closing filter is used from OpenCV to remove the small black dots present in the disparity map.
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(raw_disp, cv2.MORPH_CLOSE, kernel)
        dispc = (closing - closing.min()) * 255
        dispC = dispc.astype(np.uint8)
        disp_gray = cv2.applyColorMap(dispC, cv2.COLORMAP_BONE)

        # Using the WLS filter
        filteredImg = self.wls_filter.filter(dispL, grayL, None, dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
        filteredImg = np.uint8(filteredImg)
        filt_disp_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)

        return raw_disp, filteredImg, filt_disp_Color


def point_location(point, translation_vector):
    """Obtains the location of the pixel

            ---------------------------------
            .               .               .
            .               .               .
            .               .               .
            .               .               .
            .--------------------------------
            .               .               .
            .               .               .
            .               .               .
            .               .               .
            .--------------------------------
     the image can be tiled in four quadrents as shown in abouve figure
     the pixel location is decided with reference to left camera's principle point

    Args:
        point : x and y coordinate of the pixel
        translation_vector : vector consisting the coordinates of left camera's principal point.
                            It can be obtained from reprojection matrix

    Returns:
        translated point with the location

    """
    # translate pixel from image coordinate system to camera coordinate system
    translated_point = np.add(point, translation_vector)

    # obtain the signs of the translated point to decide the point's location(quadrant)
    signs = np.sign(translated_point)

    # decide location according to value of the translated point
    if np.array_equal(signs, [-1, -1]):
        quad = 'up_left'
    elif np.array_equal(signs, [1, -1]):
        quad = 'up_right'
    elif np.array_equal(signs, [-1, 1]):
        quad = 'down_left'
    elif np.array_equal(signs, [1, 1]):
        quad = 'down_right'
    else:
        quad = 'Invalid entry received'

    # absolute location of the point
    abs_point_loc = np.abs(translated_point)
    point = (translated_point, abs_point_loc, quad)

    return point


def depth_of_roi(corners=(None, None, None, None), focal_length=None, baseline=None, disparity=None,
                 left_cam_center=None):
    """
    calculates the depth at the center of the rectangle region

    Args:
        corners (tuple): tuple of the corners of the rectangular box in the sequence [xmin, xmax, ymin, ymax]
        focal_length (float): focal length in a pixel units, obtained from reprojection matrix
        baseline (float): baseline distance in cm, obtained from geometry of stereo system
        disparity : raw disparity map
        left_cam_center: left cameras principal point coordinates, obtained from reprojection matrix

    Returns:
        depth: depth at the center of rectangular area in cm
        point_loc: location (quadrant) of the point e.g., up_left or down_right
        side: location of the point with respect to left camera center , either left or right
        msg: formatted string message.
            format of the message is  'Point is about <y-distance> centimeter <up or down>  <x-distance> centimeter <left or right> and <depth(z)> centimeters away from center'

    """
    # calculate center of the rectangular region
    c_x = int(corners[0] + corners[1]) // 2
    c_y = int(corners[2] + corners[3]) // 2
    # print('box centers: ', c_x, c_y)
    center = np.array([c_x, c_y])

    # compute the average disparity value in the 3*3 region around the obtained center of the rectangular region
    average_disp = 0
    for u in range(-1, 2):
        for v in range(-1, 2):
            average_disp += disparity[c_y + u, c_x + v]

    average_disp = average_disp / 9

    # calculate the depth using triangulation formula
    depth = (focal_length * baseline) / average_disp
    # limit the value upto two decimal points
    depth = np.around(depth, decimals=2)

    # Obtain the point coordinates in camera coordinate system and point location
    transformed_pt, abs_pt_loc, quadrant = point_location(center, left_cam_center)

    # calculate x and y distance of the point in cm
    point_loc = ((depth / focal_length) * abs_pt_loc).astype(np.int16)

    # obtain location of the point w.r.t left camera's center
    side = quadrant.split('_')[1]

    # if the point is in hollow-pixel region in disparity map, set the distance to zero
    if math.isinf(depth):
        depth = 0

    msg = 'Point is about {} centimeter {}  {} centimeter {} and {} centimeters away from center' \
        .format(point_loc[1], quadrant.split('_')[0], point_loc[0], quadrant.split('_')[1], int(depth))

    return depth, point_loc, side, msg


def depth_at_pixel(coordinate=(None, None), focal_length=None, baseline=None, disparity=None, left_cam_center=None):
    """calculates the depth at the pixel location

    Args:
        coordinate (tuple): (x,y) coordinate of the pixel where depth value must be calculated
        focal_length (float): focal length in a pixel units, obtained from reprojection matrix
        baseline (float): baseline distance in cm, obtained from geometry of stereo system
        disparity : raw disparity map
        left_cam_center: left cameras principal point coordinates, obtained from reprojection matrix

    Returns:
        depth: depth at the pixel location in cm
        point_loc: location (quadrant) of the point e.g., up_left or down_right
        side: location of the point with respect to left camera center , either left or right
        msg: formatted string message.
            format of the message is  'Point is about <y-distance> centimeter <up or down>  <x-distance> centimeter <left or right> and <depth(z)> centimeters away from center'

    """
    x, y = coordinate
    point = np.array([x, y])

    # compute the average disparity value in the 3*3 region around the obtained center of the rectangular region
    average_disp = 0
    for u in range(-1, 2):
        for v in range(-1, 2):
            average_disp += disparity[y + u, x + v]
    average_disp = average_disp / 9

    # calculate the depth using triangulation formula
    depth = (focal_length * baseline) / (average_disp)

    # limit the value upto two decimal points
    depth = np.around(depth, decimals=2)

    # Obtain the point coordinates in camera coordinate system and point location
    transformed_pt, abs_pt_loc, quadrent = point_location(point, left_cam_center)
    # calculate x and y distance of the point in cm
    point_loc = ((depth / focal_length) * abs_pt_loc).astype(np.int16)
    # obtain location of the point w.r.t left camera's center
    side = quadrent.split('_')[1]

    # if the point is in hollow-pixel region in disparity map, set the distance to zero
    if math.isinf(depth):
        depth = 0

    msg = 'Point is about {} cemtimeters {}  {} centimeters {} and {} centimeters away from center' \
        .format(point_loc[1], quadrent.split('_')[0], point_loc[0], quadrent.split('_')[1], int(depth))

    return depth, point_loc, side, msg

"""
Calibrate the stereo camera from captured calibration images and save calibration parameters to the directory

usage: CalibrateStereocamera.py [-h] [-images CALIBRATION_IMAGES] [-param CALIBRATION_PARAMETERS]

arguments:

  -h, --help            show this help message and exit
  -images calibration_images, --calibration_Images_dir CALIBRATION_IMAGES
                        path to the dir where captured calibration images are stored.
  -param Calibration_parameters, --calibration_Param CALIBRATION_PARAMETERS
                        dir path where calibration parameters will be saved.
"""
import numpy as np
import cv2
import os
import pickle
import glob
import argparse
from datetime import datetime
from tqdm import tqdm

def Main():
    parser = argparse.ArgumentParser(
        description="Calibrate stereo camera using captured chessboard patterns")

    parser.add_argument("-images",
                        "--calibration_Images_dir",
                        help="Path to the dir where captured calibration images are stored",
                        type=str, default=None)
    parser.add_argument("-param",
                        "--calibration_Param",
                        help="Dir path where calibration parameters will be saved",
                        type=str, default=os.getcwd())

    args = parser.parse_args()

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    now = datetime.now()
    dateTime = now.strftime("%d-%m-%Y")

    img_dir = args.calibration_Images_dir

    # print('\n[INFO]: Calibration parameters will be saved at "{}" dir'.format(parameters_dir))

    # numbers of columns and raws of internal corner points in the chessboard pattern
    # these are the corners points where black and white squares intersects
    n_c = 9  # number of columns
    n_r = 6  # number of raws

    # Prepare object points
    objp = np.zeros((n_c * n_r, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_c, 0:n_r].T.reshape(-1, 2)

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d points in real world space
    imgpointsR = []  # 2d points in image plane
    imgpointsL = []

    # Start calibration from the captured images
    print('\n\nStarting calibration for the both cameras... ')
    # Call all saved images from the directory
    left_images = glob.glob(img_dir + 'left_*.jpg')
    right_images = glob.glob(img_dir + 'right_*.jpg')

    print('\n[INFO] No of calibration images: left:{} , Right:{}'.format(len(left_images), len(right_images)))

    for left, right in tqdm(zip(left_images, right_images)):  # detect the chessboard corners and store it in the arrays
        ChessImageR = cv2.imread(right, 0)  # Right image
        ChessImageL = cv2.imread(left, 0)  # Left image

        retR, cornersR = cv2.findChessboardCorners(ChessImageR, (n_c, n_r),
                                                   None)  # Define the number of chess corners we are looking for in the image
        retL, cornersL = cv2.findChessboardCorners(ChessImageL, (n_c, n_r), None)

        if (True == retR) & (True == retL):
            objpoints.append(objp)
            cv2.cornerSubPix(ChessImageR, cornersR, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(ChessImageL, cornersL, (11, 11), (-1, -1), criteria)
            imgpointsR.append(cornersR)
            imgpointsL.append(cornersL)

    # perfrom mono camera calibration and extract intrinsic and extrinsic parameters for both cameras

    # Right camera
    retR, mtxR, distR, rvecR, tvecR = cv2.calibrateCamera(objpoints, imgpointsR, ChessImageR.shape[::-1], None, None)

    # Left camera
    retL, mtxL, distL, rvecL, tvecL = cv2.calibrateCamera(objpoints,
                                                          imgpointsL,
                                                          ChessImageL.shape[::-1], None, None)

    # Calibrate the stereo camera

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    print('\n\n[INFO] starting stereo calibration.....')

    # StereoCalibrate function
    retS, mtxLS, distLS, mtxRS, distRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                         imgpointsL,
                                                                         imgpointsR,
                                                                         mtxL,
                                                                         distL,
                                                                         mtxR,
                                                                         distR,
                                                                         ChessImageR.shape[::-1],
                                                                         criteria,
                                                                         flags)

    print('Reprojection error Stereo calibrate',
          retS)  # prints the reprojection error. It should be minimum for better calibration

    rectify_scale = 0  # if 0 image is cropped and if 1 image is not croped

    # rectify both the stereo cameras
    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(mtxLS, distLS, mtxRS, distRS, ChessImageR.shape[::-1], R, T,
                                                rectify_scale, (0, 0))

    # find un-distortion map for both the cameras and save it for rectification of images later

    left_undistort_map = cv2.initUndistortRectifyMap(mtxLS, distLS, RL, PL,
                                                     ChessImageR.shape[::-1],
                                                     cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
    right_undistort_map = cv2.initUndistortRectifyMap(mtxRS, distRS, RR, PR,
                                                      ChessImageR.shape[::-1], cv2.CV_16SC2)

    parameters = {
        'left camera matrix': mtxLS,
        'right camera matrix': mtxRS,
        'DistortionCoeff Left': distLS,
        'DistortionCoeff Right': distRS,
        'rotation matrix': R,
        'Translation Matrix': T,
        'Essential Matrix': E,
        'Fundamental MAtrix': F,
        'Reprojection Matrix': Q,
        'Reprojection Error': retS
    }

    stereo_rectify_maps = {'image_size': [left_undistort_map[0].shape[1], left_undistort_map[0].shape[0]],
                           'leftMapX': left_undistort_map[0],
                           'leftMapY': left_undistort_map[1],
                           'rightMapX': right_undistort_map[0],
                           'rightMapY': right_undistort_map[1],
                           'disparityToDepthMap': Q
                           }

    if not os.path.isdir(args.calibration_Param + '/calibration_parameters'):
        os.mkdir(args.calibration_Param + '/calibration_parameters')

    parameters_dir = os.path.join(args.calibration_Param + '/calibration_parameters/', 'param_' + dateTime)
    os.mkdir(parameters_dir)

    print('\n[INFO] Writing calibration parametrs at {} ...'.format(parameters_dir))
    # save all the parameters to pickle (.p) files
    with open(parameters_dir + '/left_undistort_map.p', 'wb') as file:
        pickle.dump(left_undistort_map, file)
    with open(parameters_dir + '/right_undistort_map.p', 'wb') as file:
        pickle.dump(right_undistort_map, file)
    with open(parameters_dir + '/reprojection_matrix.p', 'wb') as file:
        pickle.dump(Q, file)
    with open(parameters_dir + '/stereo_matrices.p', 'wb') as file:
        pickle.dump(parameters, file)
    with open(parameters_dir + '/stereo_rectification_maps.p', 'wb') as file:
        pickle.dump(stereo_rectify_maps, file)

    file.close()


if __name__ == '__main__':
    Main()

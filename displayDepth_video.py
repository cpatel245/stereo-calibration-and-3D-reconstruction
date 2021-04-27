"""
Display depth at the pixel on the image with the mouse-click
usage: displayDepth_video.py [-h] [-param DIR_PATH]

arguments:

  -h, --help            show this help message and exit

  -param Dir_path, --calibration_parameters DIR_PATH
                        Path to the folder where calibration parameters are stored.
"""

import sys
import os
import cv2
import argparse

from misc.depth_utils import *
from misc.utils import set_image_res
import matplotlib.pyplot as plt


def Main():
    parser = argparse.ArgumentParser(
        description="Capture stereo image pair of chessboard pattern for calibration")

    parser.add_argument("-param",
                        "--calibration_param_dir",
                        help="Path to the dir where calibration parameters are stored.",
                        type=str, default=os.path.join(os.getcwd(), 'calibration_parameters'))

    args = parser.parse_args()

    def mouse_event(event, x, y, flags, param):
        """
        Prints the depth at (x,y) pixel of an image
        Args:
            event ():  type of an event on a displayed image window  e.g., left or right mouse button clicks
            x (): x-coordinate of an image where event is perfromed
            y (): y-coordinate of an image where event is perfromed
            flags (): None
            param (): None
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            depth, spatial_loc, side, _ = depth_at_pixel((x, y), focal_length, baseline, raw_disp, left_camera_center)

            print('at pixel:{} , depth: {} cm, location:{} '.format([x, y], depth, side))

    calibrationParameters = str(glob.glob(args.calibration_param_dir + '/*')[-1])
    print(f"\n\n[INFO] Loading Calibration parameters from '{calibrationParameters}' directory...")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    CamR = cv2.VideoCapture(1)  # i -> Right Camera
    CamL = cv2.VideoCapture(0)  # j -> Left Camera

    set_image_res(pair=(CamL, CamR))  # default (width,height) = (640,480)

    calib_parameters = StereoParameters(calibrationParameters)
    focal_length = calib_parameters.get_focal_length()
    baseline = 9  # cm #TODO: Change the value of the parameters according to camera geometry
    left_camera_center = calib_parameters.left_camera_center()

    while True:
        retR, frameR = CamR.read()
        retL, frameL = CamL.read()

        disparity = Disparity(frameL, frameR, calibrationParameters)
        raw_disp = disparity.get_raw_disparity()

        raw_disp, filteredImg, filt_disp_Color = disparity.get_disparity()

        cv2.namedWindow('Disparity map', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Disparity map', filt_disp_Color)

        # displays the left image
        win_name = 'Left Camera'
        cv2.namedWindow(win_name)
        cv2.imshow(win_name, frameL)
        cv2.setMouseCallback(win_name, mouse_event, 0)  # perform mouse click at the location to know the depth

        # End the Programme
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press and hold 'q' to exit this programm
            break

    # Release the video capture object and close the display windows
    CamR.release()
    CamL.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Main()

"""
visualize the efficiency of rectification by saved calibration parameters and save the comparision to image file

"""

import os
import glob
from misc.depth_utils import StereoParameters
import numpy as np
import cv2
import ntpath
import argparse

# load the test images and saved calibration parameters from the directory
# TODO: Change the directories according to use
testImageDir = os.path.join(os.getcwd(), 'test_images')
Parameters = os.path.join(os.getcwd(), 'calibration_parameters')
calibrationParameters = str(glob.glob(Parameters + '/*')[-1])
print(f"\n\n[INFO] Loading Calibration parameters from '{calibrationParameters}' directory...")

left_image_list = glob.glob(testImageDir + '/left_*.jpg')
right_iamge_list = glob.glob(testImageDir + '/right_*.jpg')

show_rectified = True  # if 'True' , shows comparision between original and rectified images
write_images_to_dir = True  # if 'True' , saves the images to directory

# read the saved calibration parameters files.
calibParam = StereoParameters(calibrationParameters)

assert left_image_list
assert right_iamge_list

for left, right in zip(left_image_list, right_iamge_list):  # read all test images in directory one-by-one

    frameL = cv2.imread(left, None)
    frameR = cv2.imread(right, None)
    height, width, _ = frameL.shape

    left_rect, right_rect = calibParam.rectified_pair(frameL, frameR)  # rectify images with stereo parameters

    # print original and rectified on images for identification
    font_color = (150, 20, 80)
    cv2.putText(frameL, "Left Original", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 4, cv2.LINE_AA)
    cv2.putText(frameR, "Right Original", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 4)
    cv2.putText(left_rect, "Left Rectified", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 4)
    cv2.putText(right_rect, "Right Rectified", (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 4)

    # plot horizontal lines (epilines) in the image at the intermediate distance to compare the quality of rectification
    sp = 30  # TODO: Increase or decrease value to change the spacing between horizontal lines
    for line in range(0, int(left_rect.shape[
                                 0] / sp)):  # Draw the Lines on the images Then numer of line is defines by the image Size/20
        frameL[line * sp, :] = (0, 0, 255)
        frameR[line * sp, :] = (0, 0, 255)
    for line in range(0, int(left_rect.shape[
                                 0] / sp)):  # Draw the Lines on the images Then numer of line is defines by the image Size/20
        left_rect[line * sp, :] = (0, 0, 255)
        right_rect[line * sp, :] = (0, 0, 255)

    # Combine the images for comparision
    imgConcat = np.hstack((frameL, frameR))
    imgConcat_rect = np.hstack((left_rect, right_rect))
    compareImg = np.vstack((imgConcat, imgConcat_rect))
    compareImg = cv2.resize(compareImg, (width, height))

    if show_rectified:  # If true, displays the comparision
        cv2.imshow('img', compareImg)

    if write_images_to_dir:  # if True, writes the image to the directory
        index = (ntpath.basename(left).split('.')[0])[-2:]

        if not os.path.isdir(os.path.join(os.getcwd(), 'recified_images')):
            os.mkdir(os.path.join(os.getcwd(), 'recified_images'))
        # TODO: Change the directory path to write the images
        cv2.imwrite(str(os.path.join(os.getcwd(), 'recified_images')) + f'/test_{index}_rectified.jpg', compareImg)

    k = cv2.waitKey(0)

    if k == ord(' '):  # press spacebar to display the next image
        cv2.destroyAllWindows()
    elif k == ord('q'):  # press 'q' to terminate the program
        break

# on exit close all the display windows
cv2.destroyAllWindows()

"""
Capture stereo image pairs with chessboard pattern for calibration

usage: inference_captured_video.py [-h] [-n NO_OF_IMAGES] [-dir DIR_PATH]

arguments:

  -h, --help            show this help message and exit
  -n No_Of_Images, --no_of_calibration_images NO_OF_IMAGES
                        Number of calibration images to be captured.
  -dir Dir_path, --Dir_path DIR_PATH
                        Path to the folder where captured images will be stored.
"""

from datetime import datetime
import argparse
import os

import cv2
from tqdm import tqdm
from misc.utils import *

print('\n' + '-' * 90)
print('| Starting the capture process. Press and hold (q) to exit the script.')
print('| Push (s) to save the image you want and push (c) to see next frame without saving the image.')
print('-' * 90 + '\n')


def Main():
    parser = argparse.ArgumentParser(
        description="Capture stereo image pair of chessboard pattern for calibration")

    parser.add_argument("-n",
                        "--no_of_images",
                        help="Defind how much images needs to be clicked for calibration",
                        type=int, default=20)
    parser.add_argument("-dir",
                        "--output_dir",
                        help="Path to the dir where captured stereo images will be stored.",
                        type=str, default=os.path.join(os.getcwd(), 'calibration_images'))

    args = parser.parse_args()

    imageCount = 1
    noOfClickedImages = args.no_of_images
    now = datetime.now()
    dateTime = now.strftime("%d-%m-%Y_%H.%M.%S")

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    calibrationImadeDir = os.path.join(args.output_dir, dateTime)
    os.mkdir(calibrationImadeDir)

    print('[INFO]: clicked images will be saved at "{}" dir'.format(calibrationImadeDir))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    CamR = cv2.VideoCapture(1)  # 1 -> Right Camera
    CamL = cv2.VideoCapture(0)  # 2 -> Left Camera

    # validate_im_size(stereopair=(CamL,CamR))
    set_image_res(pair=(CamL, CamR))  # default (width,height) = (640,480)

    with tqdm(total=noOfClickedImages) as pbar:
        while imageCount <= noOfClickedImages:
            # read frame-by-frame from the video feed
            retR, frameR = CamR.read()
            retL, frameL = CamL.read()

            # display the image pair
            cv2.imshow('right image', frameR)
            cv2.imshow('left image', frameL)

            # convert images to greyscale
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            retR, cornersR = cv2.findChessboardCorners(grayR, (9, 6),
                                                       None)  # Define the number of chess corners (here 9 by 6) we are looking for with the right Camera
            retL, cornersL = cv2.findChessboardCorners(grayL, (9, 6), None)  # Same with the left camera

            # If chessboard corners are found found, add object points, image points (after refining them)
            if (retR == True) & (retL == True):
                corners2R = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)  # Refining the Position
                corners2L = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)

                # Draw and display the corners
                cv2.drawChessboardCorners(grayR, (9, 6), corners2R, retR)
                cv2.drawChessboardCorners(grayL, (9, 6), corners2L, retL)

                cv2.imshow('Right image corners', grayR)
                cv2.imshow('Left image corners', grayL)

                if cv2.waitKey(0) & 0xFF == ord('s'):  # press "s" to save the images and "c" if you don't want to
                    pbar.update()
                    str_id_image = '{:02d}'.format(imageCount)
                    cv2.imwrite(calibrationImadeDir + '/right_' + str_id_image + '.jpg',
                                frameR)  # Save the image in the file at the provided image directory
                    cv2.imwrite(calibrationImadeDir + '/left_' + str_id_image + '.jpg', frameL)
                    # print('images saved')
                    imageCount = imageCount + 1

                else:
                    pass

            # End the Programme
            if cv2.waitKey(1) & 0xFF == ord('q'):  # press and hold 'q' to exit this programm
                break
    print('\n[INFO] Clicked total {} calibration images\n\n'.format(imageCount - 1))

    # When everything done, release the video capture object and close the display windows
    CamR.release()
    CamL.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Main()

"""
Capture stereo image pairs for test

usage: CaptureTestImages.py [-h] [-n NO_OF_IMAGES] [-dir DIR_PATH]

arguments:

  -h, --help            show this help message and exit
  -n No_Of_Images, --no_of_test_images NO_OF_IMAGES
                        Number of test images to be captured.
  -dir Dir_path, --Dir_path DIR_PATH
                        Path to the folder where test images will be stored.
"""

from datetime import datetime
import argparse
import os
import glob
import cv2
from tqdm import tqdm
from misc.utils import *

print('\n' + '-' * 90)
print('| Starting the capture process. Press and hold (q) to exit the script.')
print('| Push (s) to save the image you want and push (c) to see next frame without saving the image.')
print('-' * 90 + '\n')


def Main():
    parser = argparse.ArgumentParser(
        description="Capture stereo image pair")

    parser.add_argument("-n",
                        "--no_of_images",
                        help="Defind how much stereo image pairs needs to be clicked",
                        type=int, default=5)
    parser.add_argument("-dir",
                        "--output_dir",
                        help="Path to the dir where captured stereo images will be stored.",
                        type=str, default=os.path.join(os.getcwd(), 'test_images'))

    args = parser.parse_args()

    imageCount = 1
    noOfClickedImages = args.no_of_images

    now = datetime.now()
    dateTime = now.strftime("%d-%m-%Y_%H.%M.%S")
    testImageDir = args.output_dir
    testImages = os.listdir(testImageDir)

    if testImages:
        img_id = int(((testImages[-1]).strip('.jpg')).split("_")[-1]) + 1
    else:
        img_id = 1

    print('[INFO]: clicked images will be saved at "{}" dir'.format(testImageDir))

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

            # If chessboard corners are found found, add object points, image points (after refining them)
            if (retR == True) & (retL == True):
                if cv2.waitKey(0) & 0xFF == ord('s'):  # press "s" to save the images and "c" if you don't want to
                    pbar.update()
                    str_id_image = '{:02d}'.format(img_id)
                    cv2.imwrite(testImageDir + '/right_' + str_id_image + '.jpg',
                                frameR)  # Save the image in the file at the provided image directory
                    cv2.imwrite(testImageDir + '/left_' + str_id_image + '.jpg', frameL)
                    # print('images saved')
                    imageCount += 1
                    img_id += 1
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

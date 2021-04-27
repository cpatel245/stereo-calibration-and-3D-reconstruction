"""
Compute the disparity map and calculate the depth value at particular pixel on the image with the mouse-click
displayes the raw and filtered disparity maps

"""
import os
import cv2
import glob
from misc.depth_utils import *
import matplotlib.pyplot as plt


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


# TODO:Edit the image and calibration parameters directory

imageDir = os.path.join(os.getcwd(), 'test_images')

Parameters = os.path.join(os.getcwd(), 'calibration_parameters')
calibrationParameters = str(glob.glob(Parameters + '/*')[-1])
print(f"\n\n[INFO] Loading Calibration parameters from '{calibrationParameters}' directory...")

left_image_list = glob.glob(imageDir + 'left_*.jpg')
right_iamge_list = glob.glob(imageDir + 'right_*.jpg')

# select the first image in the directory # TODO: chage the selection of image according to use
frameL = cv2.imread(left_image_list[0])
frameR = cv2.imread(right_iamge_list[0])

# read the saved calibration parameters files.
stereo_parameters = StereoParameters(calibrationParameters)
# obtain focal length and left camera principal point for triangulation
focal_length = stereo_parameters.get_focal_length()
left_camera_center = stereo_parameters.left_camera_center()
baseline = 9  # cm

# compute disparity map
disparity = Disparity(frameL, frameR, calibrationParameters)
raw_disp, filteredImg, filt_disp_Color = disparity.get_disparity()

# displays the disparity map
plt.imshow(raw_disp, cmap='gray')
plt.colorbar(orientation='vertical')
plt.show()

plt.imshow(filteredImg)
plt.colorbar(orientation='vertical')
plt.show()

plt.imshow(filt_disp_Color, cmap='jet')
plt.colorbar(orientation='vertical')
plt.show()

# displays the left image
win_name = 'left image'
cv2.namedWindow(win_name)
cv2.imshow(win_name, frameL)
cv2.setMouseCallback(win_name, mouse_event, 0)  # perform mouse click at the location to know the depth

# End the Programme
k = cv2.waitKey(0)
if k == ord('q'):  # press 'q' to exit the program
    cv2.destroyAllWindows()

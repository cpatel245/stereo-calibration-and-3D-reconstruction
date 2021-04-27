# stereo-calibration-and-3D-reconstruction

This developed project contains files ...
- to capture calibration images / test images using stereo cameras
- calibrate cameras using captured calibration imags
- Rectify the test images using saved calibration parameters
- Calculate depth at the pixel of the image/live video feed using mouse click

The chessboard pattern `pattern.png` is included in the repository. or download it from:

`https://github.com/opencv/opencv/blob/master/doc/pattern.png`


# Requirements

### Intstall below packages using pip. 

- Two webcams or stereo camera
- numpy==1.16.6
- opencv-python==4.4.0.46
- matplotlib


# Steps

1. Print the chessboard pattern and paste it on a hard surface  
2. Fabricate stereo camera pair and fix it on a hard board, so it is fixed and does not move while using
3. Capture multiple images of the chessboard pattern from the different views for calibration using  `CaptureCalibrationImages.py`
4. After capturing sufficient amount of images, calibrate stereo camera pair using `CalibrateStereocamera.py`

5. To check the efficiency of the calibration, click some test images using `CaptureTestImages.py` and check the rectification efficiency using `Rectification.py`.

6. Use `displayDepth_image.py` or `displayDepth_video.py` to compute disparity and depth maps. The depth values are calculated at the pixel location with the mouse click. 



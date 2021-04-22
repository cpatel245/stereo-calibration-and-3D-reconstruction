import cv2


def validate_im_size(stereopair=(None, None)):
    """
    The width nad height of left and right image should be the same.varifies the width and height of the stereo image pair.

    @param stereopair: Stereoi camera pair (left, right)
    @type stereopair: tuple
    """
    cam1 = stereopair[0]
    cam2 = stereopair[1]

    if not cam1 or not cam2:
        raise AttributeError("Provided values for stereo camera pairs are incorrect")

    width1 = cam1.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height1 = cam1.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    width2 = cam2.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height2 = cam2.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    if width1 != width2 or height1 != height2:
        raise ValueError("Images have different sizes")
    else:
        print('\n [INFO]  Image width:{} , height:{} \n'.format(width1, height1))


def set_image_res(pair=(None, None), width=640, height=480, fps=25):
    """Configures the Frame size and FPS for stereo camera pair for inference

    Args:
        pair (tuple): tuple (left,right) camera pair of video capture objects
        width (int): preferred image width
        height (int): preferred image height
        fps (int): preferred FPS for the captured video
    """
    # set width, height and fps for each camera in stereo pair
    if not pair[0] or not pair[1]:
        raise AttributeError("Provided values for stereo camera pairs are incorrect")

    for cam in pair:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cam.set(cv2.CAP_PROP_FPS, fps)

    # obtain the configured frame size and FPS
    width = pair[0].get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = pair[0].get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = pair[0].get(cv2.CAP_PROP_FPS)

    validate_im_size(pair)

    # print('\n[INFO]  Capturing Images of Width:{} , Height:{}  at {} FPS'.format(width, height, fps))

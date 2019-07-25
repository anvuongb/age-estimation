import cv2
import os
import numpy as np
import sys

def scale_image(img, target=(128,128), by_ratio=None, interpolation=cv2.INTER_LINEAR):
    '''
    scale an cv2 image to prefered resolution or by ratio
    img: cv2 array
    target: prefered resolution, defaulted to 128x128
    by_ratio: scale by a ratio, multiply all dimestions by this ratio, defaulted to None
    interpolation: defaulted to cv2.INTER_LINEAR
    '''
    if by_ratio is None:
        return cv2.resize(img, target, interpolation=interpolation)
    else:
        return cv2.resize(img, (img.shape[0]*by_ratio, img.shape[1]*by_ratio), interpolation=interpolation)
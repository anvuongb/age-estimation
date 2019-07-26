import cv2
import os
import dlib
import numpy as np
import sys

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x1 = rect.left()
    y1 = rect.top()
    x2 = rect.right()
    y2 = rect.bottom()

    # return a tuple of (x, y, w, h)
    return (x1, y1, x2, y2)

def get_cnn_face_detector(mmod_model='mmod_human_face_detector.dat'):
    cnn_face_detector = dlib.cnn_face_detection_model_v1(mmod_model)
    return cnn_face_detector

def face_detect(image_path, cnn_face_detector, upsample=1):
    '''
    image_path: path to the image
    cnn_face_detector: mmod model
    upsample: upsize the image to detect smaller faces
    '''
    img = dlib.load_rgb_image(image_path)
    # img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = cnn_face_detector(img, upsample)
    
    faces_list = []
    for det in dets:
        faces_list.append(rect_to_bb(det.rect))
    return faces_list

def square_bbox(x1, y1, x2, y2, im_shape):
    height = y2 - y1
    width = x2 - x1 

    if width > height:
        delta = width - height
        y1_square = int(y1 - delta/2.0) if int(y1 - delta/2.0) > 0 else 0
        y2_square = int(y2 + delta/2.0) if int(y2 + delta/2.0) < im_shape[0] else im_shape[0]-1
        return x1, y1_square, x2, y2_square
    else:
        delta = height - width
        x1_square = int(x1 - delta/2.0) if int(x1 - delta/2.0) > 0 else 0
        x2_square = int(x2 + delta/2.0) if int(x2 + delta/2.0) < im_shape[1] else im_shape[1]-1
        return x1_square, y1, x2_square, y2

def expand_bbox(face, im_shape, ratio=0.2):
    '''
    expand bouding box by ratio each side
    face: face coordinate from dlib mmod
    im_shape: shape of input image
    ratio: ratio to expand
    '''
    
    delta_y = np.abs(face[1]-face[3])*ratio
    delta_x = np.abs(face[0]-face[2])*ratio
    
    x1 = int(face[0]-delta_x) if int(face[0]-delta_x) > 0 else 0
    y1 = int(face[1]-delta_y) if int(face[1]-delta_y) > 0 else 0

    x2 = int(face[2]+delta_x) if int(face[2]+delta_x) < im_shape[1] else im_shape[1]-1
    y2 = int(face[3]+delta_y) if int(face[3]+delta_y) < im_shape[0] else im_shape[0]-1
    
    return (x1,y1,x2,y2)

def face_crop(im, faces_list, ratio=0.2):
    '''
    crop out faces detected in image
    im: cv2 image
    faces_list: list of face coordinates detected after calling face_detect
    ratio: ratio to expand bounding box before cropping since mmod dlib bb is small
    faces_crop: a list of face cropped
    '''
    faces_crop = []
    for face in faces_list:
        bigger_box = expand_bbox(face, im.shape, ratio=ratio)
        crop_img = im[bigger_box[1]:bigger_box[3], bigger_box[0]:bigger_box[2]].copy()
        faces_crop.append(crop_img)
        
    return faces_crop

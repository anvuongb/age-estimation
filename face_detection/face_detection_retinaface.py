import sys
import os
import mxnet as mx
import glob
import argparse
import numpy as np
import cv2
import json
import imutils
import pickle
import gc
import matplotlib.pyplot as plt
import pandas as pd

insight_face_path = "./insightface"
sys.path.append(os.path.join(insight_face_path, 'src', 'common'))
sys.path.append(os.path.join(insight_face_path, 'deploy'))
sys.path.append(os.path.join(insight_face_path, 'RetinaFace'))
from retinaface import RetinaFace

from misc_utils import scale_image
from rotate_utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, draw_5_point
from detect_utils import expand_bbox, square_bbox

def read_image(img_path):
    try:
        img_raw = cv2.imread(img_path)
        if img_raw.shape[0]>1 and img_raw.shape[1]>1:
            return img_raw
        return None
    except:
        return None

def get_id(x, idx=0):
    s = x.split('/')[-1]
    s = s.split('.')[0]
    s = s.split('_')
    return s[idx]

def detect(input_df_path, output_folder_path, start_idx=0, end_idx=None, gpu=0):
    # this script reads all the *.jpg files in a given folder, extract all the faces

    model_path = os.path.join(insight_face_path, "RetinaFace/models/R50")
    retina_detector = RetinaFace(model_path, epoch=0, ctx_id=gpu, network='net3')

    # Create necessary folders
    if os.path.exists(output_folder_path) is False:
        os.makedirs(output_folder_path)
        
    detection_cropped_output = os.path.join(output_folder_path, "detection_cropped_output")
    if os.path.exists(detection_cropped_output) is False:
        os.mkdir(detection_cropped_output)

    print("loading input df")
    input_df = pd.read_csv(input_df_path)
    if end_idx is None:
        end_idx = len(input_df)
        
    print("start_idx={}, end_idx={}, start detection for {} images".format(start_idx, end_idx, end_idx-start_idx))
    
    if os.path.exists(output_folder_path + '/detection_metadata.csv') is False:
        with open(output_folder_path + '/detection_metadata.csv', 'w') as f:
            f.write('img_path,k,x1,y1,x2,y2,confidence,rotation,rotation_center_x,rotation_center_y,error\n')

    # Iterate all files
    for idx, row in input_df.iloc[start_idx:end_idx,:].iterrows():
        error_code = 0
        f = open(output_folder_path + '/detection_metadata.csv', 'a')
        img_path = row["img_path"]
        print('processing {}'.format(img_path))
        print('at idx {}, progress {:.4f}%\n'.format(idx, 100*idx/end_idx))
        # perform face detection
        img_raw = read_image(img_path)

        profile_id = row["profile_id"]
        photo_id = row["photo_id"]
        created_date = row["created_date"]
        birthyear = row["birthyear"]

        if img_raw is None:
            error_code = 1

        else:
            bboxes, points = retina_detector.detect(img_raw, threshold=0.5, scales=[1.0], do_flip=True)
            if bboxes is None:
                print('detection error'.format(len(bboxes)))
                error_code = 2
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(img_path, 0, 0, 0, 0, 0, 0, 0, 0, 0, error_code))
                
            elif len(bboxes) > 0:
                error_code = 0
                print('detected {} faces'.format(len(bboxes)))

                for k, d in enumerate(bboxes):
                    # perform 5-point landmark detection
                    rect = np.array(d[:4], dtype=np.int)
                    confidence = d[4]
                    face_landmark = points[k,:].copy() # left eye, right eye, nose, mouth left, mouth right

                    ## Get rotation matrix
                    left_eye = face_landmark[0, :]
                    right_eye = face_landmark[1, :]
                    _, angle, center = get_rotation_matrix(left_eye, right_eye)

                    ## Rotate image
                    image_rot = imutils.rotate(img_raw.copy(), angle, center)

                    # Save cropped faces for insight face input
                    bbox = rect

                    ## Crop rotated image with 20% box expansion
                    bbox_square = square_bbox(bbox[0], bbox[1], bbox[2], bbox[3], image_rot.shape)
                    bbox_square_20 = expand_bbox(bbox_square, image_rot.shape, ratio=0.20)
                    img_crop = image_rot[bbox_square_20[1]:bbox_square_20[3], bbox_square_20[0]:bbox_square_20[2]]

                    ## Save
                    cropped_save_path = os.path.join(detection_cropped_output, "{}_{}_{}_{}_{}.jpg".format(profile_id, photo_id, created_date, birthyear, k))
                    plt.imsave(cropped_save_path, cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))

                    # img_path, x1,y1,x2,y2, rotation, rotation_center, error
                    f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(img_path, k, bbox[0], bbox[1], bbox[2], bbox[3], confidence, angle, center[0], center[1], error_code))
            else:
                print('detected {} faces'.format(len(bboxes)))
                error_code = 3
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(img_path, 0, 0, 0, 0, 0, 0, 0, 0, 0, error_code))
        f.close()
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input df", required=True)
    parser.add_argument("--output", help="path to output images folder", required=True)
    parser.add_argument("--start", help="start idx", type=int, default=0)
    parser.add_argument("--end", help="end idx", type=int, default=None)
    parser.add_argument("--gpu", help="use gpu no 0 or 1", type=int, default=0)
    args = parser.parse_args()
    
    python_version = sys.version_info
    if python_version.major==3 and python_version.minor>=6:
        detect(args.input, args.output, args.start, args.end, args.gpu)
    else:
        print("This script cannot run correctly with this Python version: {}".format('.'.join(python_version.major, python_version.minor))) 
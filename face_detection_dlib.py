import sys
import os
import dlib
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

from misc_utils import scale_image
from rotate_utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, draw_5_point, face_align_dlib
from detect_utils import expand_bbox, square_bbox

def read_image(img_path):
    try:
        img_raw = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
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

def detect(input_df_path, output_folder_path, start_idx=0, end_idx=None, gpu=0, dlib_org=0, align=0):
    # this script reads all the *.jpg files in a given folder, extract all the faces
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(gpu)

    if dlib_org == 1:
        face_detector_path = './models/dlib/face_detector/mmod_human_face_detector.dat'
    else:
        face_detector_path = './models/dlib/face_detector/facebook_mmod_network_finetune_8000_92_50_flipped_rotated180.dat'
    landmark_predictor_path = './models/dlib/landmark_estimator/shape_predictor_5_face_landmarks.dat'

    # Load neccessary models
    print('loading face detector from {}'.format(face_detector_path))
    cnn_face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)

    print('loading landmark predictor from {}'.format(landmark_predictor_path))
    landmark_predictor = dlib.shape_predictor(landmark_predictor_path)

    # Create necessary folders
    if os.path.exists(output_folder_path) is False:
        os.makedirs(output_folder_path)
        
    detection_cropped_output = os.path.join(output_folder_path, "detection_cropped_output")
    if os.path.exists(detection_cropped_output) is False:
        os.mkdir(detection_cropped_output)
        
    if align:
        detection_aligned_output = os.path.join(output_folder_path, "detection_aligned_output")
        if os.path.exists(detection_aligned_output) is False:
            os.mkdir(detection_aligned_output)
        

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
            faces = cnn_face_detector(img_raw, 1)
            if faces is None:
                print('detection error'.format(len(faces)))
                error_code = 2
                f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(img_path, 0, 0, 0, 0, 0, 0, 0, 0, 0, error_code))
                
            elif len(faces) > 0:
                error_code = 0
                print('detected {} faces'.format(len(faces)))

                for k, d in enumerate(faces):
                    # perform 5-point landmark detection
                    rect = d.rect
                    confidence = d.confidence
                    face_landmark = landmark_predictor(img_raw, rect)

                    ## Get rotation matrix
                    left_eye = extract_left_eye_center(face_landmark)
                    right_eye = extract_right_eye_center(face_landmark)
                    _, angle, center = get_rotation_matrix(left_eye, right_eye)

                    ## Rotate image
                    image_rot = imutils.rotate(img_raw.copy(), angle, center)

                    # Save cropped faces for insight face input
                    bbox = (d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom())

                    ## Crop rotated image with 20% box expansion
                    bbox_square = square_bbox(bbox[0], bbox[1], bbox[2], bbox[3], image_rot.shape)
                    bbox_square_20 = expand_bbox(bbox_square, image_rot.shape, ratio=0.20)
                    img_crop = image_rot[bbox_square_20[1]:bbox_square_20[3], bbox_square_20[0]:bbox_square_20[2]]

                    ## Save
                    cropped_save_path = os.path.join(detection_cropped_output, "{}_{}_{}_{}_{}.jpg".format(profile_id, photo_id, created_date, birthyear, k))
                    plt.imsave(cropped_save_path, img_crop)
                    
                    if align:
                        img_aligned = face_align_dlib(img_raw, face_landmark, rect)
                        aligned_save_path = os.path.join(detection_aligned_output, "{}_{}_{}_{}_{}.jpg".format(profile_id, photo_id, created_date, birthyear, k))
                        plt.imsave(aligned_save_path, img_aligned)

                    # img_path, x1,y1,x2,y2, rotation, rotation_center, error
                    f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(img_path, k, bbox[0], bbox[1], bbox[2], bbox[3], confidence, angle, center[0], center[1], error_code))
            else:
                print('detected {} faces'.format(len(faces)))
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
    parser.add_argument("--dlib-org", help="use default dlib model or finetuned, 0=finetuned", type=int, default=0)
    parser.add_argument("--align", help="align cropped", type=int, default=0)
    args = parser.parse_args()
    
    python_version = sys.version_info
    if python_version.major==3 and python_version.minor>=6:
        detect(args.input, args.output, args.start, args.end, args.gpu, args.dlib_org, args.align)
    else:
        print("This script cannot run correctly with this Python version: {}".format('.'.join(python_version.major, python_version.minor))) 
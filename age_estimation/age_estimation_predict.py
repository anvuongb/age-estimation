import sys
import os
import glob
import argparse
import numpy as np
import pandas as pd
import time
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

from model import get_model
from generator import FacePredictGenerator

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input csv", required=True)
    parser.add_argument("--output", help="path to output csv", required=True)
    parser.add_argument("--model", help="model name: ResNet50 or InceptionResNetV2 or InceptionV3", required=True)
    parser.add_argument("--batch-size" ,help="batch size for batch prediction", required=False, type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0, help="gpu to train on")

    args = parser.parse_args()
    return args

def main():

    args = get_args()

    print("Prediction will be done on GPU {}".format(args.gpu))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)

    input_csv_path = args.input
    output_pkl_path = args.output
    model_name = args.model
    batch_size = args.batch_size

    # Load age predictor
    if model_name == 'InceptionResNetV2':
        # path to trained InceptionResNetV2
        weight_file = '/home/anvuong/working/anvuong/fb_age_estimation/models/age_estimator/inception-resnet/weights.027-2.720-2.835.hdf5'
    
    if model_name == 'ResNet50':
        # path to trained ResNet50
        weight_file = '/Users/anvuong/Desktop/hw3/models/age_estimator/resnet50/weights.028-2.757-3.419.hdf5'

    if model_name == 'InceptionV3':
        # path to trained InceptionV3
        weight_file = '/home/anvuong/working/anvuong/fb_age_estimation/models/age_estimator/resnet50/weights.028-2.757-3.419.hdf5'

    start = time.time()
    print('loading {} model and corresponding weights from {}'.format(model_name, weight_file))
    model = get_model(model_name=model_name)
    model.load_weights(weight_file)
    image_size = model.input.shape.as_list()[1]
    end = time.time()
    print('load weights took {:.4f}s\n'.format(end-start))

    # Prediction data generator
    start = time.time()
    print('start prediction')
    pred_gen = FacePredictGenerator(input_csv_path, path_col="img_path",
                                    batch_size=batch_size, image_size=image_size)
    predictions = model.predict_generator(pred_gen, verbose=1)
    end = time.time()
    print('prediction took {:.4f}s\n'.format(end-start))
    
    # Save
    print('saving results into {}'.format(output_pkl_path))
    input_df = pd.read_csv(input_csv_path)
    img_path_list = input_df["img_path"].tolist()
    output_df = pd.DataFrame(list(zip(img_path_list, predictions)), columns=["img_path", "pred_age_dist"])
    output_df.to_pickle(output_pkl_path, protocol=2)

if __name__ == '__main__':
    main()

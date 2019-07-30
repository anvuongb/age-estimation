import sys
import os
import dlib
import glob
import argparse
import numpy as np
import pandas as pd
import time
from keras import backend as K

import matplotlib.pyplot as plt

from model import get_model
from generator import FaceValGenerator

def main(input_csv_path, output_csv_path, model_name, batch_size):
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
    print('load weights took {:.4f}s\n\n'.format(end-start))

    # Prediction data generator
    start = time.time()
    print('start prediction')
    pred_gen = FaceValGenerator(input_csv_path, batch_size=batch_size, image_size=image_size)
    predictions = model.predict_generator(pred_gen)
    end = time.time()
    print('prediction took {:.4f}s\n\n'.format(end-start))
    
    # Save
    print('saving results into {}'.format(output_csv_path))
    input_df = pd.read_csv(input_csv_path)
    output_df = input_df[["img_path"]]
    output_df["pred_age_dist"] = list(predictions)
    output_df.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input csv", required=True)
    parser.add_argument("--output", help="path to output csv", required=True)
    parser.add_argument("--model", help="model name: ResNet50 or InceptionResNetV2 or InceptionV3", required=True)
    parser.add_argument("--batch-size" ,help="batch size for batch prediction", required=False, type=int, default=32)

    args = parser.parse_args()
    main(args.input, args.output, args.model, args.batch_size)

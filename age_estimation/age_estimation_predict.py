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
    parser.add_argument("--model-name", type=str, default="ResNet50",
                        help="model name: ResNet50 or InceptionResNetV2 or InceptionV3 or SEInceptionV3")
    parser.add_argument("--weight-file", type=str, required=True, 
                        help="continue to train from a pretrained model")
    parser.add_argument("--batch-size" ,help="batch size for batch prediction", required=False, type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0, help="gpu to predict on")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="number of cpu threads for generating batches")
    parser.add_argument("--queue-size", type=int, default=10,
                        help="number of batches prepared in advance")
    parser.add_argument("--provider", type=str, default="nvidia",
                        help="use nvidia or amd gpu")
    args = parser.parse_args()
    return args

def main():

    args = get_args()

    if args.provider == "amd":
        os.environ["HIP_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["HIP_VISIBLE_DEVICES"]="{}".format(args.gpu)
    elif args.provider == "nvidia":
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)
        
    print("Prediction will be done on GPU {}".format(args.gpu))

    input_csv_path = args.input
    output_pkl_path = args.output
    model_name = args.model_name
    weight_file = args.weight_file
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_queue_size = args.queue_size

    start = time.time()
    print('loading {} model and corresponding weights from {}'.format(model_name, weight_file))
    model = get_model(model_name=model_name, weights=None)
    model.load_weights(weight_file)
    image_size = model.input.shape.as_list()[1]
    end = time.time()
    print('load weights took {:.4f}s\n'.format(end-start))

    # Prediction data generator
    start = time.time()
    print('start prediction')
    pred_gen = FacePredictGenerator(input_csv_path, path_col="img_path",
                                    batch_size=batch_size, image_size=image_size)
    predictions = model.predict_generator(pred_gen, verbose=1,
                                          workers=num_workers, max_queue_size=max_queue_size,
                                          use_multiprocessing=False)
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

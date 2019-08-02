import better_exceptions
import random
import math
import os
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from keras.utils import Sequence, to_categorical
from keras_vggface import utils
from keras.preprocessing import image

###########################################################
# GENERATOR WILL REQUIRE AN CSV WITH THE FOLLOWING FORMAT #
# img_path                                                #
# abs_path_1                                              #
# abs_path_2                                              #
###########################################################

class FaceValGenerator(Sequence):
    def __init__(self, meta_csv_path, batch_size=32, image_size=224, version=2):
        self.image_path_and_age = []
        self._load_meta_csv(meta_csv_path)
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size
        self.version = version

    def __len__(self):
        return int(np.ceil(self.image_num/self.batch_size))

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        
        idx_min = idx*batch_size
        idx_max = min(idx_min + batch_size, len(self.image_path_and_age))

        current_batch_size = idx_max - idx_min

        x = np.zeros((current_batch_size, image_size, image_size, 3), dtype=np.uint8)

        for i, image_path in enumerate(self.image_path_and_age[idx_min:idx_max]):
            image = self._load_and_preprocess(str(image_path), self.image_size, self.version)
            x[i] = image

        return x

    def _load_meta_csv(self, meta_csv_path):
        meta_csv = pd.read_csv(meta_csv_path)

        for idx, row in meta_csv.iterrows():
            self.image_path_and_age.append(str(row["img_path"]))
    
    def _load_and_preprocess(self, img_path, target_size, version):
        # version 1 for vgg
        # version 2 for resnet50 and senet50
        img = image.load_img(img_path, target_size=(target_size, target_size))
        x = image.img_to_array(img)
        x = utils.preprocess_input(x, version=version) 
        return x

    
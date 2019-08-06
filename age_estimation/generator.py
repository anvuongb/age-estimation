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
import Augmentor


def get_transform_func():
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.zoom_random(probability=0.5, percentage_area=0.95)
    p.random_distortion(probability=0.5, grid_width=2, grid_height=2, magnitude=8)
    p.random_color(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=1, min_factor=0.8, max_factor=1.2)
    p.random_erasing(probability=0.5, rectangle_area=0.2)

    def transform_image(image):
        image = [Image.fromarray(image)]
        for operation in p.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return image[0]
    return transform_image

###########################################################
# GENERATOR WILL REQUIRE AN CSV WITH THE FOLLOWING FORMAT #
# img_path,age                                            #
# abs_path_1,age1                                         #
# abs_path_2,age2                                         #
###########################################################

class FaceGenerator(Sequence):
    def __init__(self, meta_csv_path, batch_size=32, image_size=224):
        self.image_path_and_age = []
        self._load_meta_csv(meta_csv_path)

        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size
        self.indices = np.random.permutation(self.image_num)
        self.transform_image = get_transform_func()

    def __len__(self):
        return int(np.ceil(self.image_num/self.batch_size))

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size

        idx_min = idx*batch_size
        idx_max = min(idx_min + batch_size, len(self.image_path_and_age))

        current_batch_size = idx_max - idx_min

        x = np.zeros((current_batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((current_batch_size, 1), dtype=np.int32)

        sample_indices = self.indices[idx_min:idx_max]

        for i, sample_id in enumerate(sample_indices):
            image_path, age = self.image_path_and_age[sample_id]
            image = cv2.imread(str(image_path))
            x[i] = self.transform_image(cv2.resize(image, (image_size, image_size)))
            age += math.floor(np.random.randn() * 2 + 0.5)
            y[i] = np.clip(age, 0, 69)

        return x, to_categorical(y, 70)

    def on_epoch_end(self):
        self.indices = np.random.permutation(self.image_num)

    def _load_meta_csv(self, meta_csv_path):
        meta_csv = pd.read_csv(meta_csv_path)

        for idx, row in meta_csv.iterrows():
            self.image_path_and_age.append([str(row["img_path"]), int(row["age"])])


class FaceValGenerator(Sequence):
    def __init__(self, meta_csv_path, batch_size=32, image_size=224):
        self.image_path_and_age = []
        self._load_meta_csv(meta_csv_path)
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(self.image_num/self.batch_size))

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        
        idx_min = idx*batch_size
        idx_max = min(idx_min + batch_size, len(self.image_path_and_age))

        current_batch_size = idx_max - idx_min

        x = np.zeros((current_batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((current_batch_size, 1), dtype=np.int32)

        for i, (image_path, age) in enumerate(self.image_path_and_age[idx_min:idx_max]):
            image = cv2.imread(str(image_path))
            x[i] = cv2.resize(image, (image_size, image_size))
            y[i] = age

        return x, to_categorical(y, 70)

    def _load_meta_csv(self, meta_csv_path):
        meta_csv = pd.read_csv(meta_csv_path)

        for idx, row in meta_csv.iterrows():
            self.image_path_and_age.append([str(row["img_path"]), int(row["age"])])

class FacePredictGenerator(Sequence):
    # this function only take "img_path" columng
    def __init__(self, meta_csv_path, path_col="img_path", batch_size=32, image_size=224):
        self.image_path_and_age = []
        self._load_meta_csv(meta_csv_path)
        self.image_num = len(self.image_path_and_age)
        self.batch_size = batch_size
        self.image_size = image_size
        self.path_col = path_col

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
            image = cv2.imread(str(image_path))
            x[i] = cv2.resize(image, (image_size, image_size))

        return x

    def _load_meta_csv(self, meta_csv_path):
        meta_csv = pd.read_csv(meta_csv_path)

        for idx, row in meta_csv.iterrows():
            self.image_path_and_age.append(str(row[self.path_col]))

    
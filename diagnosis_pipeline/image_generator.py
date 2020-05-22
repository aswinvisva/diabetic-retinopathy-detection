import os
import random
import re
from collections import Counter

import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.applications import resnet50, inception_v3
from tensorflow.python.keras.applications import vgg16


class Generator:

    def __init__(self, split):
        self.train = []
        self.test = []
        self.split = split

        paths = []

        train_df = pd.read_csv('train.csv')
        data_dict = {}
        data_dict[0] = []
        data_dict[1] = []
        data_dict[2] = []
        data_dict[3] = []
        data_dict[4] = []

        for root, dirs, files in os.walk('train_images'):
            random.shuffle(files)
            random.shuffle(dirs)

            for file in files:
                file_path = os.path.join(root, file)

                id = re.search("(.*?)\.", os.path.basename(file_path)).group(1)
                data = train_df[train_df['id_code'] == id]['diagnosis'].values[0]
                data_dict[data].append(file_path)
                paths.append(file_path)

        random.shuffle(paths)

        split_index = int((1-self.split) * len(paths))

        self.train = paths[split_index:]
        self.test = paths[:split_index]

    def under_sampe(self, data_dict):
        max_samples = max(len(data_dict[0]), len(data_dict[1]), len(data_dict[2]), len(data_dict[3]), len(data_dict[4]))

        data_dict[0] = random.choices(data_dict[0], k=max_samples)
        data_dict[1] = random.choices(data_dict[1], k=max_samples)
        data_dict[2] = random.choices(data_dict[2], k=max_samples)
        data_dict[3] = random.choices(data_dict[3], k=max_samples)
        data_dict[4] = random.choices(data_dict[4], k=max_samples)

        return data_dict

    def generate(self, batch_size=10):
        train_df = pd.read_csv('train.csv')

        batch_data = []
        batch_label = []

        for path in self.train:
            mat = cv2.imread(path)
            id = re.search("(.*?)\.", os.path.basename(path)).group(1)

            data = train_df[train_df['id_code'] == id]['diagnosis'].values[0]

            mat = cv2.resize(mat, (224, 224))
            mat = inception_v3.preprocess_input(mat)

            # mat = mat/255

            batch_data.append(mat)
            batch_label.append(data)

            batch_data_np = np.array(batch_data)
            batch_label_np = np.array(batch_label)

            if len(batch_data) == batch_size:
                # print(batch_label_np)

                yield (batch_data_np.reshape(batch_size, 224, 224, 3),
                       batch_label_np.reshape(batch_size, 1))
                batch_data = []
                batch_label = []

            # yield (np.array(mat).reshape(1, 224, 224, 1), np.array(data).reshape(-1, 1))

    def sp_noise(self, image, prob):
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def evaluate(self):
        train_df = pd.read_csv('train.csv')

        for path in self.test:
            mat = cv2.imread(path)
            id = re.search("(.*?)\.", os.path.basename(path)).group(1)

            data = train_df[train_df['id_code'] == id]['diagnosis'].values[0]

            mat = cv2.resize(mat, (224, 224))
            mat = inception_v3.preprocess_input(mat)

            yield (np.array(mat).reshape(1, 224, 224, 3), np.array(data).reshape(-1, 1))

    def image_preprocessing(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 224 / 10), -4,
                                128)  # the trick is to add this line
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image



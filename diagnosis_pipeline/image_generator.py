import os
import random
import re
import zipfile
from collections import Counter

import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.applications import resnet50, inception_v3, xception
from tensorflow.python.keras.applications import vgg16


class Generator:

    def __init__(self, split):
        self.train = []
        self.test = []
        self.split = split

        paths = []

        train_df = pd.read_csv('train_images/diabetic-retinopathy-detection/trainLabels.csv')
        data_dict = {}
        data_dict[0] = []
        data_dict[1] = []
        data_dict[2] = []
        data_dict[3] = []
        data_dict[4] = []

        for root, dirs, files in os.walk('train_images/diabetic-retinopathy-detection'):
            random.shuffle(files)
            random.shuffle(dirs)

            for file in files:

                file_path = os.path.join(root, file)

                id = re.search("(.*?)\.", os.path.basename(file_path)).group(1)

                if id is not None:
                    if len(train_df[train_df['image'] == id]['level'].values) > 0:
                        data = train_df[train_df['image'] == id]['level'].values[0]
                        data = self.label_processing(data)


                        data_dict[data].append(file_path)
                        paths.append(file_path)

        random.shuffle(paths)

        for key in data_dict.keys():
            print(key, ":", len(data_dict[key]))

        data_dictionary = self.under_sampe(data_dict)
        paths = []
        for k in data_dictionary.keys():
            paths = paths + data_dictionary[k]

        random.shuffle(paths)

        print(len(paths))

        self.train = paths

        self.test = []

        for root, dirs, files in os.walk('train_images/APTOS'):
            random.shuffle(files)
            random.shuffle(dirs)

            for file in files:

                file_path = os.path.join(root, file)

                self.test.append(file_path)


    def under_sampe(self, data_dict):
        max_samples = max(len(data_dict[0]), len(data_dict[1]), len(data_dict[2]), len(data_dict[3]), len(data_dict[4]))

        data_dict[0] = random.choices(data_dict[0], k=max_samples)
        data_dict[1] = random.choices(data_dict[1], k=max_samples)
        data_dict[2] = random.choices(data_dict[2], k=max_samples)

        return data_dict

    def generate(self, batch_size=7):
        train_df = pd.read_csv('train_images/diabetic-retinopathy-detection/trainLabels.csv')

        batch_data = []
        batch_label = []

        for path in self.train:
            mat = cv2.imread(path)
            id = re.search("(.*?)\.", os.path.basename(path)).group(1)

            if id is not None:
                if len(train_df[train_df['image'] == id]['level'].values) > 0:
                    data = train_df[train_df['image'] == id]['level'].values[0]

                    data = self.label_processing(data)

                    mat = self.image_preprocessing(mat)

                    batch_data.append(mat)
                    batch_label.append(data)

                    batch_data_np = np.array(batch_data)
                    batch_label_np = np.array(batch_label)

                    if len(batch_data) == batch_size:
                        # print(batch_label_np)

                        yield (batch_data_np.reshape(batch_size, 512, 512, 3),
                               batch_label_np.reshape(batch_size, 1))
                        batch_data = []
                        batch_label = []

            # yield (np.array(mat).reshape(1, 512, 512, 1), np.array(data).reshape(-1, 1))

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

            if id is not None:
                if len(train_df[train_df['id_code'] == id]['diagnosis'].values) > 0:
                    data = train_df[train_df['id_code'] == id]['diagnosis'].values[0]
                    data = self.label_processing(data)
                    mat = self.image_preprocessing(mat)

                    cv2.imshow("ASD", mat)
                    cv2.waitKey(0)

                    yield (np.array(mat).reshape(1, 512, 512, 3), np.array(data).reshape(-1, 1))

    def image_preprocessing(self, mat):
        mat = cv2.resize(mat, (512, 512))
        mat = self.rotate_image(mat, np.random.randint(-10, 10))

        mat = self.improve_contrast_image_using_clahe(mat)
        mat = cv2.GaussianBlur(mat, (5, 5), 0)
        mat = mat/255
        #
        # cv2.imshow("ASD", mat)
        # cv2.waitKey(0)

        # mat = xception.preprocess_input(mat)

        return mat

    def label_processing(self, label):
        if label >= 3:
            label = 2
        elif label >= 1:
            label = 1
        else:
            label = 0

        return label

    def improve_contrast_image_using_clahe(self, bgr_image):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        hsv_planes = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv_planes[2] = clahe.apply(hsv_planes[2])
        hsv = cv2.merge(hsv_planes)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def rotate(self, image, angle=90, scale=1.0):
        '''
        Rotate the image
        :param image: image to be processed
        :param angle: Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).
        :param scale: Isotropic scale factor.
        '''
        w = image.shape[1]
        h = image.shape[0]
        # rotate matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        # rotate
        image = cv2.warpAffine(image, M, (w, h))
        return image

    def flip(self, image, vflip=False, hflip=False):
        '''
        Flip the image
        :param image: image to be processed
        :param vflip: whether to flip the image vertically
        :param hflip: whether to flip the image horizontally
        '''
        if hflip or vflip:
            if hflip and vflip:
                c = -1
            else:
                c = 0 if vflip else 1
            image = cv2.flip(image, flipCode=c)
        return image

    def add_GaussianNoise(self, image):
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy

    def image_augment(self, image):
        '''
        Create the new image with imge augmentation
        :param path: the path to store the new image
        '''
        img = image.copy()
        img_flip = self.flip(img, vflip=True, hflip=False)
        img_rot = self.rotate(img_flip)
        # img_gaussian = self.add_GaussianNoise(img_rot)

        return img_rot


    




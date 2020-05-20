import os
import random
import re
from collections import Counter

import pandas as pd
import numpy as np
import cv2


class Generator:

    def __init__(self):
        self.train = []
        self.test = []
        self.split = 0.85

        paths = []

        for root, dirs, files in os.walk('gaussian_filtered_images'):
            random.shuffle(files)
            random.shuffle(dirs)

            for file in files:
                file_path = os.path.join(root, file)
                paths.append(file_path)

        random.shuffle(paths)
        split_index = int((1-self.split) * len(paths))

        self.train = paths[split_index:]
        self.test = paths[:split_index]

    def generate(self):
        train_df = pd.read_csv('train.csv')

        for path in self.train:
            mat = cv2.imread(path)
            id = re.search("(.*?)\.", os.path.basename(path)).group(1)

            data = train_df[train_df['id_code'] == id]['diagnosis'].values[0]
            mat = mat/255

            yield (np.array(mat).reshape(1, 224, 224, 3), np.array(data).reshape(-1, 1))

    def evaluate(self):
        train_df = pd.read_csv('train.csv')

        for path in self.test:
            mat = cv2.imread(path)
            id = re.search("(.*?)\.", os.path.basename(path)).group(1)

            data = train_df[train_df['id_code'] == id]['diagnosis'].values[0]

            yield (np.array(mat).reshape(1, 224, 224, 3), np.array(data).reshape(-1, 1))



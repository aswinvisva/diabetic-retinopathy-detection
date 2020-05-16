import os
import re

import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import numpy as np
from tensorflow.python.keras.layers import Flatten


import image_generator


class Detector:

    def __init__(self, input_size=(224,224,3)):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        self.gen = image_generator.Generator()
        self.input_size = input_size
        self.model = Sequential()
        self.model.add(Conv2D(8, 3, input_shape=self.input_size, activation='relu', padding='same', kernel_initializer='he_normal'))
        self.model.add(Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal'))

        self.model.add(MaxPooling2D((2,2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(16, 3, activation='relu', padding='same',
                              kernel_initializer='he_normal'))
        self.model.add(Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'))

        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(32, 3, activation='relu', padding='same',
                              kernel_initializer='he_normal'))
        self.model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'))

        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, 3, activation='relu', padding='same',
                              kernel_initializer='he_normal'))
        self.model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'))

        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Dropout(0.2))
        self.model.add(Flatten())

        self.model.add(Dense(5, activation='softmax'))

        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.model.summary()

    def train(self):
        self.model.fit(x=self.gen.generate(), epochs=100, steps_per_epoch=75)

    def evaluate(self):
        return self.model.evaluate(x=self.gen.evaluate())

    def show_predictions(self):
        train_df = pd.read_csv('train.csv')

        for path in self.gen.test:
            mat = cv2.imread(path)
            id = re.search("(.*?)\.", os.path.basename(path)).group(1)

            label = train_df[train_df['id_code'] == id]['diagnosis'].values[0]

            prediction = self.model.predict(np.array(mat).reshape(1, 224, 224, 3))

            print("Prediction", prediction)
            print("Label", label)

            cv2.imshow("Image", mat)
            cv2.waitKey(0)

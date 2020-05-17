import os
import re

import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import numpy as np
from tensorflow.python.keras.layers import Flatten


import image_generator


class Detector:

    def __init__(self, input_size=(224,224,3), should_load_model=False):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        self.gen = image_generator.Generator()
        self.input_size = input_size

        if not should_load_model:
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
        else:
            self.model = load_model('detector_model.h5')

        self.model.summary()

    def train(self):
        self.model.fit(x=self.gen.generate(), epochs=100, steps_per_epoch=75)
        save_model(self.model, 'detector_model.h5')

    def evaluate(self):
        self.gen.generate()
        return self.model.evaluate(x=self.gen.evaluate())

    def show_predictions(self):
        train_df = pd.read_csv('train.csv')

        for path in self.gen.test:
            mat = cv2.imread(path)
            id = re.search("(.*?)\.", os.path.basename(path)).group(1)

            label = train_df[train_df['id_code'] == id]['diagnosis'].values[0]

            prediction = self.model.predict(np.array(mat).reshape(1, 224, 224, 3))

            index = np.where(prediction == prediction.max())[1][0]

            if index == 0:
                print("No Diabetic Retinopathy detected!")
            elif index == 1:
                print("Mild Diabetic Retinopathy detected!")
            elif index == 2:
                print("Moderate Diabetic Retinopathy detected!")
            elif index == 3:
                print("Severe Diabetic Retinopathy detected!")
            elif index == 4:
                print("Proliferate Diabetic Retinopathy detected!")

            print("Label", label)

            if index != label:
                print("False detection")

            cv2.imshow("Image", mat)
            cv2.waitKey(0)

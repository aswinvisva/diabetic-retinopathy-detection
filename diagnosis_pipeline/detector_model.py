import os
import re

import cv2
import pandas as pd
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras import applications
from tensorflow.keras import Model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np
from tensorflow.python.keras.layers import Flatten, GlobalAveragePooling2D

from diagnosis_pipeline import image_generator


class Detector:

    def __init__(self, input_size=(224,224,3), should_load_model=False, use_resnet=False):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        self.gen = image_generator.Generator()
        self.input_size = input_size

        if not should_load_model:

            if not use_resnet:

                inputs = Input(input_size)

                conv1 = Conv2D(8, 3, input_shape=self.input_size, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
                conv2 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
                merge1 = concatenate([conv1, conv2], axis=3)

                mp1 = MaxPooling2D((2,2))(merge1)
                drop1 = Dropout(0.2)(mp1)

                conv3 = Conv2D(8, 3, activation='relu', padding='same',
                                      kernel_initializer='he_normal')(drop1)
                conv4 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
                merge2 = concatenate([conv3, conv4], axis=3)


                mp2 = MaxPooling2D((2, 2))(merge2)
                drop2 = Dropout(0.35)(mp2)
                conv5 = Conv2D(16, 3, activation='relu', padding='same',
                                      kernel_initializer='he_normal')(drop2)
                conv6 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
                merge3 = concatenate([conv5, conv6], axis=3)

                mp3 = MaxPooling2D((2, 2))(merge3)

                conv7 = Conv2D(16, 3, activation='relu', padding='same',
                                      kernel_initializer='he_normal')(mp3)
                conv8 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
                merge3 = concatenate([conv7, conv8], axis=3)

                conv9 = Conv2D(32, 3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(merge3)
                conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
                merge4 = concatenate([conv9, conv10], axis=3)

                mp4 = MaxPooling2D((2, 2))(merge4)

                conv11 = Conv2D(32, 3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(mp4)
                conv12 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
                merge5 = concatenate([conv11, conv12], axis=3)

                conv13 = Conv2D(64, 3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(merge5)
                conv14 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)
                merge6 = concatenate([conv13, conv14], axis=3)

                conv15 = Conv2D(64, 3, activation='relu', padding='same',
                                kernel_initializer='he_normal')(merge6)
                conv16 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv15)
                merge7 = concatenate([conv15, conv16], axis=3)

                mp5 = MaxPooling2D((2, 2))(merge7)

                drop4 = Dropout(0.45)(mp5)
                flat = Flatten()(drop4)

                dense1 = Dense(5, activation='softmax')(flat)
                self.model = Model(inputs, dense1)
            else:
                base_model = applications.resnet50.ResNet50(weights=None, include_top=False,
                                                            input_shape=(224, 224, 3))
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = Dropout(0.7)(x)
                predictions = Dense(5, activation='softmax')(x)
                self.model = Model(inputs=base_model.input, outputs=predictions)

            self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model = load_model('diagnosis_pipeline/detector_model.h5')

        self.model.summary()

    def train(self):
        self.model.fit(x=self.gen.generate(), epochs=100, steps_per_epoch=75)
        save_model(self.model, 'diagnosis_pipeline/detector_model.h5')

    def evaluate(self):
        self.gen.generate()
        return self.model.evaluate(x=self.gen.evaluate())

    def image_preprocessing(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 224 / 10), -4,
                                128)  # the trick is to add this line

        return image
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

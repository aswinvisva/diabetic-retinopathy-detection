import os
import re

import cv2
import pandas as pd
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras import applications
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet50, inception_v3
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import numpy as np
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.layers import Flatten, GlobalAveragePooling2D
from tensorflow.keras.metrics import SparseCategoricalAccuracy

from diagnosis_pipeline import image_generator


class Detector:

    def __init__(self, input_size=(512, 512, 1), should_load_model=False, use_transfer_learning_ensemble=False,
                 train_split=0.85):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        self.gen = image_generator.Generator(train_split)
        self.input_size = input_size
        self.models = []

        if not should_load_model:

            if not use_transfer_learning_ensemble:

                inputs = Input(input_size)

                conv1 = Conv2D(8, 3, input_shape=self.input_size, activation='relu', padding='same',
                               kernel_initializer='he_normal')(inputs)
                conv2 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
                merge1 = concatenate([conv1, conv2], axis=3)

                mp1 = MaxPooling2D((2, 2))(merge1)
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
                self.models.append(Model(inputs, dense1))
            else:
                xception_base_model = Xception(include_top=False, weights="imagenet", input_shape=(512, 512, 3))

                # for layer in xception_base_model.layers:
                #     layer.trainable = False

                x = Flatten()(xception_base_model.output)
                x = Dropout(0.25)(x)
                xception_predictions = Dense(3, activation='softmax')(x)

                self.models.append(Model(inputs=xception_base_model.input, outputs=xception_predictions))
                #
                # inception_base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=(512, 512, 3))
                #
                # for layer in inception_base_model.layers:
                #     layer.trainable = False
                #
                # x = Flatten()(inception_base_model.output)
                # x = Dropout(0.25)(x)
                # inception_predictions = Dense(3, activation='softmax')(x)
                #
                # self.models.append(Model(inputs=inception_base_model.input, outputs=inception_predictions))

                # resnet_base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(512, 512, 3))
                #
                # for layer in resnet_base_model.layers:
                #     layer.trainable = False
                #
                # x = Flatten()(resnet_base_model.output)
                # x = Dropout(0.25)(x)
                # resnet_predictions = Dense(3, activation='softmax')(x)

                # self.models.append(Model(inputs=resnet_base_model.input, outputs=resnet_predictions))

            for model in self.models:
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                              metrics=['sparse_categorical_accuracy'])
        else:
            for root, dirs, files in os.walk('diagnosis_pipeline/saved_models'):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.models.append(load_model(file_path))

        for model in self.models:
            model.summary()

    def train(self):

        for i, model in enumerate(self.models):
            model.fit(x=self.gen.generate(), epochs=100, steps_per_epoch=75)
            save_model(model, 'diagnosis_pipeline/saved_models/model_%s.h5' % str(i))

    def evaluate(self):

        results = []
        for x_test, y_test in self.gen.evaluate():
            m = SparseCategoricalAccuracy()

            predictions = []

            predictions.append(self.models[0].predict(np.array(x_test)))
            predictions.append(self.models[0].predict(np.array(x_test)))

            # calculate average
            outcomes = average(predictions).numpy()
            print(outcomes)
            print(y_test)

            if len(results) > 0:
                curr_avg = sum(results) / len(results)
                print("Accuracy:", curr_avg)

            m.update_state(
                # We have changed y_true = [[2], [1], [3]] to the following
                y_true=y_test,
                y_pred=outcomes,
                sample_weight=[1, 1, 1]
            )

            results.append(m.result().numpy())

        avg = sum(results) / len(results)
        return avg

    def show_predictions(self):
        train_df = pd.read_csv('train.csv')

        for path in self.gen.test:
            mat = cv2.imread(path)
            mat = cv2.resize(mat, (512, 512))
            mat = inception_v3.preprocess_input(mat)

            id = re.search("(.*?)\.", os.path.basename(path)).group(1)

            label = train_df[train_df['id_code'] == id]['diagnosis'].values[0]

            predictions = []

            for model in self.models:
                # make predictions
                predictions.append(model.predict(np.array(mat).reshape(1, 512, 512, 3)))

            # calculate average
            outcomes = average(predictions).numpy()

            print(outcomes)
            print(id)

            index = np.where(outcomes == outcomes.max())[1][0]

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

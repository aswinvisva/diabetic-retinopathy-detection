import io

import flask
from flask import jsonify, request, Response
from tensorflow.python.keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Diabetic Retinotherapy API</h1><p>Under Construction</p>"

@app.route('/api/v1/predictor/diagnosis', methods=['POST'])
def api_diagnosis():
    # initialize the data dictionary that will be returned from the
    # view
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    data = {"success": False}
    model = load_model('../diagnosis_pipeline/detector_model.h5')
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            prediction = model.predict(image)

            # loop over the results and add them to the list of
            # returned predictions
            index = np.where(prediction == prediction.max())[1][0]
            r = {"label": float(index), "probability distribution:": prediction[0].tolist()}
            data["prediction"] = r

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

app.run()
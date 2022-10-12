import os
from i2_client import I2Client
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from uuid import uuid4
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)


# Initialize the isquare client.
client = ...


def get_image(buffer, flags=cv2.IMREAD_ANYCOLOR):
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(bytes_as_np_array, flags)


# route for prediction
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return "there is no file1 in form!"
        file = request.files["file"]
        image = get_image(file)
        image = cv2.resize(image, (200, 200))
        img_name1 = "static/" + str(uuid4()) + ".png"
        cv2.imwrite(img_name1,image)
        success, output = client.inference(image)[0]
        if not success:
            raise RuntimeError(output)

        img_name2 = "static/" + str(uuid4()) + ".png"
        cv2.imwrite(img_name2, output)
        # return the path of the saved image
        return jsonify(image_paths=[img_name1,img_name2])
    return render_template("index.html")

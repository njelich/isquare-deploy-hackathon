import os
from i2_client import I2Client
from flask import Flask, request, jsonify, render_template
# from flask_debugtoolbar import DebugToolbarExtension
from uuid import uuid4
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# UPLOAD_FOLDER = './uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['SECRET_KEY'] = 'some random string'
# app.debug = True
# toolbar = DebugToolbarExtension(app)

# Initialize the isquare client.
client = I2Client(access_key="2be984f3-5616-4675-9546-bd5352b8c5b4", url="url wss://prod.archipel.isquare.ai/3a431d96-4679-4659-b539-6bc51335e109") # add your model access here



# route for prediction
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if 'file' not in request.files:
            return 'there is no file1 in form!'
        file = request.files['file']
        #obj=file.read()
        # inference the model with my_text string variable
        #output_image = client.inference([obj])
        #output_image = output_image[0][1] # index the output
        # choose a random name for the picture then save it
        img_name = "static/" + str(uuid4()) + ".png"
        file.save(img_name)
        # return the path of the saved image
        return jsonify(image_path=img_name)
    return render_template("index.html")
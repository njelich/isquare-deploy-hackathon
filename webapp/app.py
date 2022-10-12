from i2_client import I2Client
from flask import Flask, request, jsonify, render_template
from uuid import uuid4

app = Flask(__name__)

# Initialize the isquare client.
client = I2Client(...) # add your model access here



# route for prediction
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # get the text from the input form
        my_text = request.get_data().decode()
        # inference the model with my_text string variable
        output_image = client.inference([my_text])
        output_image = output_image[0][1] # index the output
        # choose a random name for the picture then save it
        img_name = "static/" + str(uuid4()) + ".png"
        output_image.save(img_name)
        # return the path of the saved image
        return jsonify(image_path=img_name)
    return render_template("index.html")
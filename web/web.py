from flask import Flask, render_template, request, jsonify
import re
import base64
from io import BytesIO

from test import MLP, preprocess_image, load_and_train_model

app = Flask(__name__)

# Placeholder for the MLP model
mlp = None

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/train_model", methods=["POST"])
def train_model():
    global mlp
    mlp = load_and_train_model()
    return jsonify({'status': 'Model trained successfully'})

@app.route("/predict", methods=["POST"])
def predict():
    if mlp is None:
        return jsonify({'error': 'Model is not trained yet'}), 400

    data = request.get_json()
    image_data = data['image']
    processed_image = preprocess_image(image_data)
    prediction = mlp.predict(processed_image)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)

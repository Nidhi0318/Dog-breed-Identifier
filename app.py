import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join('models', 'dog_breed_classifier.h5')
model = tf.keras.models.load_model(MODEL_PATH)

BREED_NAMES = np.load(os.path.join("models", "breed_names.npy"), allow_pickle=True)

IMAGE_SIZE = (224, 224)

def preprocess_image(img_data):
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


@app.route('/')
def home():
    return "Dog Breed Predictor API is running! Use POST /predict"


@app.route('/predict', methods=['POST'])
def predict_breed():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        img_data = request.files['file'].read()
        processed_image = preprocess_image(img_data)

        probabilities = model.predict(processed_image)[0]

        best_index = np.argmax(probabilities)
        breed_name = BREED_NAMES[best_index].replace("_", " ").title()

        confidence = float(probabilities[best_index] * 100)

        behavior = f"{breed_name}s are usually friendly and intelligent pets."

        return jsonify({
            "breed": breed_name,
            "confidence": round(confidence, 2),
            "behavior": behavior
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500



if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

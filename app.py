"""Step 2: Run the LeafLens web app.
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""
import io
import json
import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

app = Flask(__name__, template_folder='template')
CORS(app)

MODEL_PATH = 'leaflens_model.keras'
CLASSES_PATH = 'class_names.json'

if not (os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH)):
    raise SystemExit('Model files not found. Run "python train_model.py" first.')

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH) as f:
    class_names = json.load(f)


def prepare_image(file_storage):
    img = Image.open(io.BytesIO(file_storage.read())).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype='float32')
    return np.expand_dims(arr, axis=0)


def clean_name(name):
    return name.replace('___', ' - ').replace('_', ' ')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'classes': len(class_names)})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    try:
        batch = prepare_image(request.files['image'])
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400

    probs = model.predict(batch, verbose=0)[0]
    top3 = np.argsort(probs)[::-1][:3]
    results = [{'label': clean_name(class_names[i]),
                'confidence': round(float(probs[i]) * 100, 2)} for i in top3]
    return jsonify({'prediction': results[0], 'top3': results})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

from flask import Flask, request, jsonify, send_file, send_from_directory
import numpy as np
import joblib
from flask_cors import CORS
import json
import os
import csv

app = Flask(__name__)
CORS(app)

# Serve the frontend
@app.route("/")
def index():
    return send_file("index.html")

@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory("assets", filename)

# Load model and labels
MODEL_PATH = "model.joblib"
LABELS_PATH = "labels.json"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
else:
    model = None
    print("Model file not found. Please run train_tf_model.py first.")

if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH) as f:
        labels = json.load(f)
else:
    labels = []

def normalize_landmarks(sample):
    sample = sample.reshape(21, 3)
    wrist = sample[0]
    sample = sample - wrist
    distances = np.linalg.norm(sample, axis=1)
    max_dist = np.max(distances)
    if max_dist > 0:
        sample = sample / max_dist
    return sample.flatten()

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.json.get("landmarks")
        if not data:
            return jsonify({"error": "No landmarks provided"}), 400
        
        sample = np.array(data)
        sample = normalize_landmarks(sample)
        
        # Random Forest expectation: (n_samples, n_features)
        X = sample.reshape(1, -1)
        
        # Get probability to show confidence
        probs = model.predict_proba(X)[0]
        index = np.argmax(probs)
        confidence = float(np.max(probs))
        
        return jsonify({
            "label": labels[index],
            "confidence": confidence
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
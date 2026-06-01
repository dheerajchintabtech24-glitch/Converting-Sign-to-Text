# VisionSign Pro: Real-time ISL Translator

VisionSign Pro is a state-of-the-art Sign Language recognition system using MediaPipe landmarks and a deep learning model augmented with 30x synthetic data variations for extreme accuracy.

## ✨ Features

- **Extreme Accuracy**: Data augmentation (3D rotation, noise, scaling) makes the model robust to any hand angle.
- **Premium UI**: Stunning dark-mode interface with glassmorphism and real-time visualization.
- **Instant Inference**: Flask-powered backend for lightning-fast predictions.
- **Text-to-Speech**: Built-in voice synthesis to read your translations aloud.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install tensorflow mediapipe pandas numpy scikit-learn flask flask-cors opencv-python
```

### 2. Train the "Master" Model
If you want to use the existing dataset or after collecting new data:
```bash
python train_tf_model.py
```
*Note: This script now generates 30 variants for every single sample, significantly boosting model robustness.*

### 3. Run the Application
Start the backend:
```bash
python app.py
```

Then, serve the frontend (in a new terminal):
```bash
python -m http.server 8000
```
Open `http://localhost:8000` in your browser.

## 📂 Project Structure

- `index.html` - The premium VisionSign Pro frontend.
- `app.py` - Flask API for real-time gesture prediction.
- `train_tf_model.py` - Advanced training script with massive data augmentation.
- `data_collection.py` - Tool to record your own custom hand gestures.
- `landmarks.csv` - The training dataset (landmark coordinates).
- `model.h5` - The trained deep learning brain.

## 🛠️ Data Collection Tips
- Use `data_collection.py` to add new letters.
- Move your hand around slightly while recording to capture different angles.
- The training script will automatically multiply your samples to make the model "unbreakable".

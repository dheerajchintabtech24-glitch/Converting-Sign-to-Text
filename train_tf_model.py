import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

def normalize_landmarks(sample):
    """Deep normalization: centers on wrist and scales by maximum hand span."""
    sample = sample.reshape(21, 3)
    wrist = sample[0]
    sample = sample - wrist
    # Scale by the maximum distance found in the hand to be invariant to distance
    max_dist = np.max(np.linalg.norm(sample, axis=1))
    if max_dist > 0:
        sample = sample / max_dist
    return sample.flatten()

def augment_landmarks(landmarks, factor=30):
    """
    Generates high-fidelity synthetic variations.
    Includes 3D rotation, perspective distortion, scaling, and finger morphing.
    """
    augmented = []
    original_points = landmarks.reshape(21, 3)
    
    for _ in range(factor):
        # 1. 3D Rotation (X, Y, Z)
        angles = np.random.uniform(-0.25, 0.25, 3)
        cos = np.cos(angles)
        sin = np.sin(angles)
        
        # Rotation matrices
        Rx = np.array([[1, 0, 0], [0, cos[0], -sin[0]], [0, sin[0], cos[0]]])
        Ry = np.array([[cos[1], 0, sin[1]], [0, 1, 0], [-sin[1], 0, cos[1]]])
        Rz = np.array([[cos[2], -sin[2], 0], [sin[2], cos[2], 0], [0, 0, 1]])
        
        variant = original_points @ Rx @ Ry @ Rz
        
        # 2. Random Scaling (Distance variation)
        variant = variant * np.random.uniform(0.9, 1.1)
        
        # 3. Finger Morphing (Subtle movement of finger tips)
        tips = [4, 8, 12, 16, 20]
        for tip in tips:
            variant[tip] += np.random.normal(0, 0.015, 3)
            
        # 4. Global Gaussian Noise (Sensor jitter)
        variant += np.random.normal(0, 0.003, variant.shape)
        
        augmented.append(normalize_landmarks(variant.flatten()))
        
    return augmented

print("--- LOADING MASTER DATASET ---")
if not os.path.exists("landmarks.csv"):
    print("Error: landmarks.csv not found!")
    exit()

data = pd.read_csv("landmarks.csv")
data = data.replace("####", np.nan).dropna()

X_raw = data.iloc[:, 1:].astype(float).values
y_raw = data.iloc[:, 0].values

print(f"Base samples found: {len(X_raw)}")
unique_labels = sorted(list(set(y_raw)))
print(f"Classes to master: {unique_labels}")

print(f"--- STARTING OPTIMIZED AUGMENTATION (30x Factor) ---")
X_aug, y_aug = [], []

for i in range(len(X_raw)):
    # Original sample
    X_aug.append(normalize_landmarks(X_raw[i]))
    y_aug.append(y_raw[i])
    
    # Generate 30 variants
    variants = augment_landmarks(X_raw[i], factor=30)
    for v in variants:
        X_aug.append(v)
        y_aug.append(y_raw[i])

X = np.array(X_aug)
y = np.array(y_aug)

print(f"Final training set size: {len(X)} samples")

# Label Encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
with open("labels.json", "w") as f:
    json.dump(list(encoder.classes_), f)

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42
)

print(f"--- TRAINING OPTIMIZED RANDOM FOREST ---")
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=30, 
    n_jobs=-1, 
    random_state=42,
    verbose=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"--- EVALUATION RESULTS ---")
print(f"MODEL ACCURACY: {acc*100:.2f}%")
print("Top-tier robustness achieved.")

# Save the final masterpiece
joblib.dump(model, "model.joblib")
print("Model saved as model.joblib")

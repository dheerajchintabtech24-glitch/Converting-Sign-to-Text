import csv
import os
import urllib.request
from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
OUTPUT_FILE = Path("landmarks.csv")
MODEL_FILE = Path("hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SAMPLES_PER_GESTURE = 100
# Each row: label + 21 landmarks * 3 (x,y,z) = 64 columns
FEATURE_COLUMNS = [f"{axis}{index}" for index in range(21) for axis in ("x", "y", "z")]
CSV_COLUMNS = ["label", *FEATURE_COLUMNS]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

def ensure_csv_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_COLUMNS)

def ensure_model_file() -> None:
    if MODEL_FILE.exists():
        return
    print(f"Downloading MediaPipe hand model to {MODEL_FILE}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)

def flatten_landmarks(hand_landmarks) -> list[float]:
    values = []
    for landmark in hand_landmarks:
        values.extend([landmark.x, landmark.y, landmark.z])
    return values

def draw_landmarks(frame, landmarks) -> None:
    height, width = frame.shape[:2]
    # Draw connections
    for start, end in HAND_CONNECTIONS:
        a, b = landmarks[start], landmarks[end]
        cv2.line(frame, (int(a.x * width), int(a.y * height)), (int(b.x * width), int(b.y * height)), (55, 211, 153), 2)
    # Draw points
    for landmark in landmarks:
        cv2.circle(frame, (int(landmark.x * width), int(landmark.y * height)), 4, (242, 193, 78), -1)

def append_sample(path: Path, label: str, features: list[float]) -> None:
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([label, *features])

def main() -> None:
    ensure_csv_header(OUTPUT_FILE)
    ensure_model_file()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    recording_label = None
    recorded_count = 0
    frame_index = 0

    base_options = python.BaseOptions(model_asset_path=os.fspath(MODEL_FILE))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    print("\n--- ISL DATA COLLECTION READY ---")
    print("Press A-Z to record 100 samples for that letter.")
    print("Keep your hand in the frame and move it slightly for better variety.")
    print("Press ESC to exit.\n")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_index * 1000 / 30)
            frame_index += 1
            
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_detected = bool(results.hand_landmarks)

            if hand_detected:
                landmarks = results.hand_landmarks[0]
                draw_landmarks(frame, landmarks)
                if recording_label:
                    features = flatten_landmarks(landmarks)
                    append_sample(OUTPUT_FILE, recording_label, features)
                    recorded_count += 1
                    if recorded_count >= SAMPLES_PER_GESTURE:
                        print(f"DONE: Recorded {SAMPLES_PER_GESTURE} samples for {recording_label}")
                        recording_label = None
                        recorded_count = 0

            # UI Overlay
            cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
            msg = f"Mode: IDLE | Press A-Z to start"
            if recording_label:
                msg = f"Recording {recording_label}: {recorded_count}/{SAMPLES_PER_GESTURE}"
            
            color = (55, 211, 153) if hand_detected else (100, 100, 255)
            cv2.putText(frame, msg, (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.imshow("Sign-to-Text Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            if ord('a') <= key <= ord('z') or ord('A') <= key <= ord('Z'):
                recording_label = chr(key).upper()
                recorded_count = 0
                print(f"Started recording {SAMPLES_PER_GESTURE} samples for {recording_label}...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

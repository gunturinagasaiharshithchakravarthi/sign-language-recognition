"""Interactive Real-Time Data Collection Script using Webcam.

Collects labeled 3D hand landmark samples for sign language gesture classes
and saves them directly into a structured CSV dataset.
"""

import sys
import time
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import pandas as pd
import numpy as np

from config import (
    DATASET_PATH,
    GESTURE_CLASSES,
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    USE_3D_LANDMARKS
)
from src.hand_detection import HandDetector
from src.landmark_extraction import generate_feature_names


def run_data_collection(
    gesture_label: str,
    target_samples: int = 150,
    camera_index: int = CAMERA_INDEX
):
    """Run interactive webcam recording for a specific gesture label."""
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_names = generate_feature_names(include_z=USE_3D_LANDMARKS)
    columns = feature_names + ["label"]

    detector = HandDetector()
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[Error] Could not open webcam at index {camera_index}.")
        print("Please check camera connections or privacy permissions in Windows Settings.")
        return

    print(f"\n=======================================================")
    print(f" DATA COLLECTION: Gesture Class '{gesture_label}'")
    print(f" Target Samples: {target_samples}")
    print(f" Press 's' to START recording (with a 3-second countdown)")
    print(f" Press 'q' to QUIT at any time")
    print(f"=======================================================\n")

    started = False
    samples_collected = 0
    collected_rows = []

    countdown_start = None
    countdown_duration = 3.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab webcam frame.")
            break

        # Flip horizontally for natural mirror preview
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Detect hands
        landmarks_list, _ = detector.process_frame(frame)
        display_frame = detector.draw_landmarks(frame, landmarks_list)

        # UI Overlay
        cv2.rectangle(display_frame, (0, 0), (w, 70), (20, 20, 20), -1)
        cv2.putText(
            display_frame,
            f"Class: '{gesture_label}'  |  Collected: {samples_collected}/{target_samples}",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nCollection aborted by user.")
            break
        elif key == ord('s') and not started and countdown_start is None:
            countdown_start = time.time()

        # Handle Countdown
        if countdown_start is not None and not started:
            elapsed = time.time() - countdown_start
            remaining = int(countdown_duration - elapsed) + 1
            if elapsed < countdown_duration:
                cv2.putText(
                    display_frame,
                    f"GET READY: {remaining}",
                    (w // 2 - 140, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.4,
                    (0, 165, 255),
                    3,
                    cv2.LINE_AA
                )
            else:
                started = True
                countdown_start = None

        # Capture Samples
        if started:
            if landmarks_list:
                # Extract invariant normalized coordinates
                features = detector.extract_features(landmarks_list[0], include_z=USE_3D_LANDMARKS)
                row_data = list(features) + [gesture_label]
                collected_rows.append(row_data)
                samples_collected += 1

                # Progress bar
                progress = samples_collected / target_samples
                cv2.rectangle(display_frame, (20, h - 35), (w - 20, h - 15), (50, 50, 50), -1)
                cv2.rectangle(display_frame, (20, h - 35), (int(20 + progress * (w - 40)), h - 15), (0, 255, 0), -1)

                if samples_collected >= target_samples:
                    print(f"\n[Success] Collected {samples_collected} samples for class '{gesture_label}'!")
                    break
            else:
                cv2.putText(
                    display_frame,
                    "NO HAND DETECTED - PLEASE SHOW HAND",
                    (w // 2 - 200, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )

        cv2.imshow("Sign Language Data Collector", display_frame)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    # Save to CSV
    if collected_rows:
        df_new = pd.DataFrame(collected_rows, columns=columns)
        if DATASET_PATH.exists():
            df_existing = pd.read_csv(DATASET_PATH)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(DATASET_PATH, index=False)
            print(f"[Dataset] Appended {len(df_new)} rows. Total samples in dataset: {len(df_combined)}")
        else:
            df_new.to_csv(DATASET_PATH, index=False)
            print(f"[Dataset] Created new dataset at {DATASET_PATH} with {len(df_new)} rows.")


if __name__ == "__main__":
    print("\n--- ASL Static Alphabet Gesture Collector ---")
    print(f"Configured Gestures: {', '.join(GESTURE_CLASSES)}")
    label_input = input(f"Enter gesture letter to collect [{GESTURE_CLASSES[0]}]: ").strip().upper()
    if not label_input:
        label_input = GESTURE_CLASSES[0]

    count_str = input("Enter number of samples to collect [150]: ").strip()
    target = int(count_str) if count_str.isdigit() else 150

    run_data_collection(gesture_label=label_input, target_samples=target)

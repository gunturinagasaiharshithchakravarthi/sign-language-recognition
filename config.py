"""Configuration settings for Real-Time Sign Language Recognition System."""

import os
from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "gestures.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "gesture_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
ASSETS_DIR = ROOT_DIR / "assets"
CONFUSION_MATRIX_PATH = ASSETS_DIR / "confusion_matrix.png"

# Supported Static Gestures (ASL letters + special controls)
GESTURE_CLASSES = [
    "A", "B", "C", "D", "E",
    "L", "O", "V", "W", "Y"
]

# MediaPipe Hand Detection Hyperparameters
STATIC_IMAGE_MODE = False
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.6

# Landmark Specifications
NUM_LANDMARKS = 21  # 21 3D points (x, y, z) per hand
NUM_COORDINATES = 42  # Normalized (x, y) relative coordinates (or 63 for x, y, z)
USE_3D_LANDMARKS = True  # Using x, y, z gives 63 feature dimensions

if USE_3D_LANDMARKS:
    FEATURE_COUNT = NUM_LANDMARKS * 3  # 63 features
else:
    FEATURE_COUNT = NUM_LANDMARKS * 2  # 42 features

# Real-Time Recognition & Stabilization
CONFIDENCE_THRESHOLD = 0.75
STABILIZATION_BUFFER_SIZE = 8  # consecutive matching predictions required for auto-confirm
PREDICTION_COOLDOWN_SECONDS = 1.0

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Text-to-Speech Settings
TTS_RATE = 150  # Words per minute
TTS_VOLUME = 1.0  # 0.0 to 1.0

# Model Training Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20
RF_N_ESTIMATORS = 120
RF_MAX_DEPTH = 15

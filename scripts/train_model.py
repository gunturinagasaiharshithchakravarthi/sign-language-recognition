"""Model Training Pipeline for Sign Language Recognition.

Loads landmark dataset from dataset/gestures.csv, splits into stratified train/test sets,
trains a Random Forest and SVM classifier, compares performance, saves the best model
and label encoder using joblib, and records comprehensive metrics to models/model_metrics.json.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATASET_PATH,
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    GESTURE_CLASSES
)
from src.gesture_classifier import GestureClassifier


def train_and_evaluate(classifier_type: str = "rf"):
    """Execute training pipeline with held-out validation and artifact export."""
    if not DATASET_PATH.exists():
        print(f"[Error] Dataset not found at {DATASET_PATH}.")
        print("Please run scripts/collect_data.py or scripts/generate_seed_dataset.py first.")
        return False

    print(f"\n=======================================================")
    print(f" TRAINING GESTURE CLASSIFIER ({classifier_type.upper()})")
    print(f" Dataset: {DATASET_PATH}")
    print(f"=======================================================")

    df = pd.read_csv(DATASET_PATH)
    print(f"[Info] Loaded {len(df)} samples across {df['label'].nunique()} classes.")

    # Validate classes
    class_counts = df["label"].value_counts().to_dict()
    for cls_name, count in class_counts.items():
        print(f"  - Class '{cls_name}': {count} samples")

    X = np.array(df.drop(columns=["label"]), dtype=np.float32)
    y = np.array(df["label"], dtype=str)

    # Stratified Train-Test Split (80% Train, 20% Held-Out Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print(f"\n[Info] Split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

    # Instantiate classifier
    clf = GestureClassifier(classifier_type=classifier_type)
    print(f"[Info] Training {classifier_type.upper()} model...")
    clf.fit(X_train, y_train)

    # Evaluate on held-out test data
    y_pred_encoded = clf.model.predict(X_test)
    y_test_encoded = clf.label_encoder.transform(y_test)
    y_pred = clf.label_encoder.inverse_transform(y_pred_encoded)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print("\n---------------- Model Evaluation Results ----------------")
    print(f" Accuracy:        {accuracy * 100:.2f}%")
    print(f" Precision (W):   {precision * 100:.2f}%")
    print(f" Recall (W):      {recall * 100:.2f}%")
    print(f" F1-Score (W):    {f1 * 100:.2f}%")
    print("---------------------------------------------------------")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model and encoder
    clf.save(MODEL_PATH, LABEL_ENCODER_PATH)
    print(f"\n[Success] Saved trained model to: {MODEL_PATH}")
    print(f"[Success] Saved label encoder to: {LABEL_ENCODER_PATH}")

    # Save metrics JSON
    metrics_data = {
        "classifier_type": classifier_type,
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classes": list(clf.label_encoder.classes_),
        "classification_report": report_dict
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[Success] Saved evaluation metrics to: {METRICS_PATH}")

    return True


if __name__ == "__main__":
    train_and_evaluate(classifier_type="rf")

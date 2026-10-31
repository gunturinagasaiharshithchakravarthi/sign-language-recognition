"""Independent Model Evaluation and Visualization Script.

Generates a publication-grade confusion matrix heatmap and saves it
to assets/confusion_matrix.png for presentation and dashboard embedding.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    DATASET_PATH,
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    CONFUSION_MATRIX_PATH,
    RANDOM_STATE,
    TEST_SIZE
)
from src.gesture_classifier import GestureClassifier


def evaluate_and_plot():
    """Evaluate model and save styled confusion matrix plot."""
    if not DATASET_PATH.exists() or not MODEL_PATH.exists():
        print("[Error] Model or dataset missing. Please run scripts/train_model.py first.")
        return False

    df = pd.read_csv(DATASET_PATH)
    X = np.array(df.drop(columns=["label"]), dtype=np.float32)
    y = np.array(df["label"], dtype=str)

    # Same split for consistency
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    clf = GestureClassifier()
    if not clf.load(MODEL_PATH, LABEL_ENCODER_PATH):
        print("[Error] Failed to load trained model.")
        return False

    y_pred_encoded = clf.model.predict(X_test)
    y_pred = clf.label_encoder.inverse_transform(y_pred_encoded)
    classes = list(clf.label_encoder.classes_)

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    # Style: Modern Dark Theme matching application aesthetic
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    # Custom cyan/blue colormap
    cmap = sns.dark_palette("#00d4ff", as_cmap=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=1.0,
        linecolor="#1f2937",
        ax=ax,
        annot_kws={"size": 11, "weight": "bold", "color": "#ffffff"}
    )

    ax.set_title("Real-Time Gesture Classifier - Confusion Matrix", fontsize=14, weight="bold", pad=16, color="#00d4ff")
    ax.set_xlabel("Predicted Gesture Class", fontsize=12, labelpad=10, color="#e5e7eb")
    ax.set_ylabel("True Ground-Truth Class", fontsize=12, labelpad=10, color="#e5e7eb")
    ax.tick_params(colors="#9ca3af", labelsize=10)

    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    print(f"\n[Success] Confusion matrix plot generated and saved to: {CONFUSION_MATRIX_PATH}")
    print("\nSummary Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return True


if __name__ == "__main__":
    evaluate_and_plot()

"""Machine Learning Gesture Classification Engine using Scikit-Learn."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from config import (
    MODEL_PATH,
    LABEL_ENCODER_PATH,
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    CONFIDENCE_THRESHOLD
)


class GestureClassifier:
    """Classifier for static hand gestures using Scikit-Learn with probability estimates."""

    def __init__(
        self,
        classifier_type: str = "rf",  # 'rf' for Random Forest, 'svm' for Support Vector Machine
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        self.classifier_type = classifier_type
        self.confidence_threshold = confidence_threshold
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False

        if classifier_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        elif classifier_type == "svm":
            self.model = SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                random_state=RANDOM_STATE
            )
        else:
            raise ValueError(f"Unknown classifier type '{classifier_type}'. Use 'rf' or 'svm'.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GestureClassifier":
        """Fit the classifier on landmark feature vectors and categorical labels."""
        y_encoded = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encoded)
        self.is_trained = True
        return self

    def predict(
        self,
        feature_vector: np.ndarray,
        return_raw: bool = False
    ) -> Any:
        """Predict gesture for a single feature vector.
        
        Args:
            feature_vector: 1D array of normalized hand landmarks.
            return_raw: If True, also returns the top candidate label regardless of threshold.
            
        Returns:
            Tuple of:
            - predicted_label (str or None if below confidence threshold)
            - confidence (float between 0.0 and 1.0)
            - probabilities_dict (map from class name to confidence probability)
            - raw_label (optional 4th item if return_raw=True)
        """
        if not self.is_trained or self.model is None:
            return (None, 0.0, {}, None) if return_raw else (None, 0.0, {})

        # Ensure 2D shape (1, num_features)
        if feature_vector.ndim == 1:
            features = feature_vector.reshape(1, -1)
        else:
            features = feature_vector

        # Predict probabilities
        probabilities = self.model.predict_proba(features)[0]
        max_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[max_idx])

        # Get class label
        classes = self.label_encoder.classes_
        raw_label = str(classes[max_idx])

        # Map all probabilities
        prob_dict = {
            str(cls_name): float(prob)
            for cls_name, prob in zip(classes, probabilities)
        }

        # Apply confidence threshold filtering
        confirmed_label = raw_label if confidence >= self.confidence_threshold else None

        if return_raw:
            return confirmed_label, confidence, prob_dict, raw_label
        return confirmed_label, confidence, prob_dict

    def save(
        self,
        model_path: Path = MODEL_PATH,
        encoder_path: Path = LABEL_ENCODER_PATH
    ):
        """Save the trained model and label encoder to disk via joblib."""
        model_path = Path(model_path)
        encoder_path = Path(encoder_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        encoder_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)

    def load(
        self,
        model_path: Path = MODEL_PATH,
        encoder_path: Path = LABEL_ENCODER_PATH
    ) -> bool:
        """Load trained model and label encoder from disk.
        
        Returns True if successful, False if files are missing or corrupted.
        """
        model_path = Path(model_path)
        encoder_path = Path(encoder_path)

        if not model_path.exists() or not encoder_path.exists():
            return False

        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"[GestureClassifier] Error loading model: {e}")
            self.is_trained = False
            return False

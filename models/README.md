# Serialized Machine Learning Models & Evaluation Artifacts

## Directory Contents
- **`gesture_model.pkl`**: Serialized Scikit-Learn `RandomForestClassifier` (120 estimators, max depth 15) trained on normalized 3D hand landmark coordinates.
- **`label_encoder.pkl`**: Serialized Scikit-Learn `LabelEncoder` mapping integer class indices to ASL gesture letters.
- **`hand_landmarker.task`**: Official Google MediaPipe Hand Landmarker task bundle for deep-learning-based 21 3D joint regression.
- **`model_metrics.json`**: Quantitative evaluation results (accuracy, precision, recall, F1-score, sample counts, and full classification report) computed on the held-out test set.

## Model Retraining
To retrain and update these serialized model files:
```powershell
python scripts/train_model.py
```

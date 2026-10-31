"""Landmark extraction and invariant coordinate normalization module."""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


# Hand landmark connection topology (21 keypoints)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base connection
    (0, 17)
]

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


def extract_raw_coordinates(landmarks) -> np.ndarray:
    """Extract raw 21 x 3 numpy array from MediaPipe landmarks.
    
    Supports both MediaPipe Tasks (list with .x, .y, .z) and legacy NormalizedLandmarkList.
    """
    coords = []
    # If landmarks is an iterable of landmark objects
    for lm in landmarks:
        coords.append([lm.x, lm.y, getattr(lm, "z", 0.0)])
    return np.array(coords, dtype=np.float32)


def normalize_landmarks(raw_coords: np.ndarray, include_z: bool = True) -> np.ndarray:
    """Normalize hand landmark coordinates to achieve translation and scale invariance.
    
    Mathematical Formulation:
    1. Translation Invariance: Shift all coordinates relative to the wrist (landmark 0):
       P'_i = P_i - P_wrist
    2. Scale Invariance: Scale by the maximum Euclidean distance from the wrist:
       d_max = max(||P'_i||)
       P''_i = P'_i / d_max
       
    Args:
        raw_coords: (21, 3) numpy array of landmark coordinates.
        include_z: Whether to retain normalized Z coordinates.
        
    Returns:
        1D numpy array of normalized coordinates (length 63 if include_z else 42).
    """
    if raw_coords.shape[0] != 21:
        raise ValueError(f"Expected 21 landmarks, got {raw_coords.shape[0]}")

    coords = raw_coords.copy()

    # Step 1: Translation Invariance (reference to wrist point 0)
    wrist = coords[0].copy()
    coords = coords - wrist

    # Step 2: Scale Invariance
    # Compute Euclidean distances from origin (wrist)
    distances = np.linalg.norm(coords, axis=1)
    max_distance = np.max(distances)

    if max_distance > 1e-6:
        coords = coords / max_distance
    else:
        # Fallback if hand is a single point (degenerate case)
        pass

    # Step 3: Dimensionality Selection & Flattening
    if include_z:
        normalized_vector = coords.flatten()  # 63 features
    else:
        normalized_vector = coords[:, :2].flatten()  # 42 features

    return normalized_vector.astype(np.float32)


def generate_feature_names(include_z: bool = True) -> List[str]:
    """Generate descriptive column names for CSV storage and DataFrame columns."""
    names = []
    axes = ["x", "y", "z"] if include_z else ["x", "y"]
    for idx, lm_name in enumerate(LANDMARK_NAMES):
        for axis in axes:
            names.append(f"{lm_name}_{axis}")
    return names

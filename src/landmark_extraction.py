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


def compute_kinematic_features(coords: np.ndarray) -> np.ndarray:
    """Compute translation- and scale-invariant geometric finger relationships.
    
    Extracts 22 invariant kinematic metrics:
    - 5 Fingertip-to-Wrist distances (extension length)
    - 5 Fingertip-to-MCP distances (curl degree)
    - 4 Fingertip-to-Thumb distances (pinch / contact measures)
    - 3 Adjacent fingertip distances (finger spread / separation)
    - 5 Extension differentials (Tip vs PIP distance to wrist)
    
    Args:
        coords: (21, 3) normalized hand coordinates centered at wrist.
        
    Returns:
        1D numpy array of 22 kinematic distance features.
    """
    wrist = coords[0]  # (0, 0, 0)
    
    # Key landmark indices
    tips = [4, 8, 12, 16, 20]      # Thumb, Index, Middle, Ring, Pinky
    mcps = [2, 5, 9, 13, 17]       # Base knuckles
    pips = [3, 6, 10, 14, 18]      # Mid-joints
    
    # 1. Fingertip to wrist distance
    d_tip_wrist = [np.linalg.norm(coords[t] - wrist) for t in tips]
    
    # 2. Fingertip to MCP distance (curl degree: small = tightly curled)
    d_tip_mcp = [np.linalg.norm(coords[t] - coords[m]) for t, m in zip(tips, mcps)]
    
    # 3. Fingertip to thumb tip distance (pinch measure: ~0 in O/F, large in L/D)
    thumb_tip = coords[4]
    d_thumb_tips = [np.linalg.norm(coords[t] - thumb_tip) for t in tips[1:]]
    
    # 4. Adjacent fingertip distances (spread: small in U, large in V/W)
    d_spread = [
        np.linalg.norm(coords[8] - coords[12]),   # Index - Middle
        np.linalg.norm(coords[12] - coords[16]),  # Middle - Ring
        np.linalg.norm(coords[16] - coords[20])   # Ring - Pinky
    ]
    
    # 5. Extension differentials (Tip distance to wrist minus PIP distance)
    # Positive = straight finger; Negative = curled finger
    d_ext = [
        np.linalg.norm(coords[t] - wrist) - np.linalg.norm(coords[p] - wrist)
        for t, p in zip(tips, pips)
    ]
    
    kinematic_vec = np.array(
        d_tip_wrist + d_tip_mcp + d_thumb_tips + d_spread + d_ext,
        dtype=np.float32
    )
    return kinematic_vec


def normalize_landmarks(raw_coords: np.ndarray, include_z: bool = True) -> np.ndarray:
    """Normalize hand landmark coordinates to achieve translation and scale invariance.
    
    Mathematical Formulation:
    1. Translation Invariance: Shift all coordinates relative to the wrist (landmark 0):
       P'_i = P_i - P_wrist
    2. Scale Invariance: Scale by the maximum Euclidean distance from the wrist:
       d_max = max(||P'_i||)
       P''_i = P'_i / d_max
    3. Kinematic Feature Augmentation: Compute 22 invariant relative distances and angles
       between fingertips, knuckles, and the wrist to resolve fine-grained gesture ambiguities.
       
    Args:
        raw_coords: (21, 3) numpy array of landmark coordinates.
        include_z: Whether to retain normalized Z coordinates.
        
    Returns:
        1D numpy array of 85 features (63 normalized 3D coordinates + 22 kinematic metrics).
    """
    if raw_coords.shape[0] != 21:
        raise ValueError(f"Expected 21 landmarks, got {raw_coords.shape[0]}")

    coords = raw_coords.copy()

    # Step 1: Translation Invariance (reference to wrist point 0)
    wrist = coords[0].copy()
    coords = coords - wrist

    # Step 2: Scale Invariance
    distances = np.linalg.norm(coords, axis=1)
    max_distance = np.max(distances)

    if max_distance > 1e-6:
        coords = coords / max_distance

    # Step 3: Dimensionality Selection & Flattening
    if include_z:
        coord_vector = coords.flatten()  # 63 features
    else:
        coord_vector = coords[:, :2].flatten()  # 42 features

    # Step 4: Augment with Invariant Kinematic Features
    kinematic_vector = compute_kinematic_features(coords)

    # Combined feature vector (63 + 22 = 85 features)
    feature_vector = np.concatenate([coord_vector, kinematic_vector]).astype(np.float32)
    return feature_vector


KINEMATIC_FEATURE_NAMES = [
    # Tip to wrist distances (5)
    "dist_wrist_thumb", "dist_wrist_index", "dist_wrist_middle", "dist_wrist_ring", "dist_wrist_pinky",
    # Curl distances (5)
    "curl_thumb", "curl_index", "curl_middle", "curl_ring", "curl_pinky",
    # Thumb to fingertip distances (4)
    "dist_thumb_index", "dist_thumb_middle", "dist_thumb_ring", "dist_thumb_pinky",
    # Finger spread distances (3)
    "spread_index_middle", "spread_middle_ring", "spread_ring_pinky",
    # Extension differentials (5)
    "ext_diff_thumb", "ext_diff_index", "ext_diff_middle", "ext_diff_ring", "ext_diff_pinky"
]


def generate_feature_names(include_z: bool = True) -> List[str]:
    """Generate descriptive column names for CSV storage and DataFrame columns."""
    names = []
    axes = ["x", "y", "z"] if include_z else ["x", "y"]
    for idx, lm_name in enumerate(LANDMARK_NAMES):
        for axis in axes:
            names.append(f"{lm_name}_{axis}")
    # Add kinematic feature names
    names.extend(KINEMATIC_FEATURE_NAMES)
    return names

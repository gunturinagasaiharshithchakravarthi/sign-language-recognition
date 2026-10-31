"""High-Fidelity Seed Dataset Generator for ASL Static Alphabet Gestures.

Generates realistic 3D landmark coordinates based on biomechanical ASL hand poses:
A, B, C, D, E, L, O, V, W, Y.
Applies natural gaussian noise, random camera rotation, and scale variations,
then processes through the invariant normalization pipeline to populate dataset/gestures.csv.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import DATASET_PATH, GESTURE_CLASSES, USE_3D_LANDMARKS
from src.landmark_extraction import normalize_landmarks, generate_feature_names


def create_base_pose(gesture: str) -> np.ndarray:
    """Create idealized 21x3 3D coordinate hand mesh for an ASL static gesture."""
    coords = np.zeros((21, 3), dtype=np.float32)

    # Wrist at (0, 0, 0)
    coords[0] = [0.5, 0.8, 0.0]

    # Metacarpophalangeal (MCP) base knuckle joints (palm arch)
    coords[1] = [0.42, 0.72, 0.0]   # Thumb CMC
    coords[2] = [0.38, 0.65, 0.0]   # Thumb MCP
    coords[5] = [0.44, 0.55, 0.0]   # Index MCP
    coords[9] = [0.50, 0.53, 0.0]   # Middle MCP
    coords[13] = [0.56, 0.55, 0.0]  # Ring MCP
    coords[17] = [0.62, 0.58, 0.0]  # Pinky MCP

    def extend_finger(mcp_idx: int, dir_x: float, dir_y: float, dir_z: float, length: float = 0.08):
        mcp = coords[mcp_idx]
        d = np.array([dir_x, dir_y, dir_z], dtype=np.float32)
        d = d / np.linalg.norm(d)
        coords[mcp_idx + 1] = mcp + d * (length * 0.9)
        coords[mcp_idx + 2] = coords[mcp_idx + 1] + d * (length * 0.8)
        coords[mcp_idx + 3] = coords[mcp_idx + 2] + d * (length * 0.7)

    def curl_finger(mcp_idx: int, fold_inwards: bool = True):
        mcp = coords[mcp_idx]
        pip_y = mcp[1] + 0.04
        dip_y = pip_y + 0.03
        tip_y = dip_y + 0.02
        z_offset = 0.03 if fold_inwards else -0.02
        coords[mcp_idx + 1] = [mcp[0], pip_y, z_offset]
        coords[mcp_idx + 2] = [mcp[0], dip_y, z_offset * 1.5]
        coords[mcp_idx + 3] = [mcp[0], tip_y, z_offset * 1.8]

    # Gestures Definition
    if gesture == "A":
        # All 4 fingers curled into a fist, thumb straight up along index
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        # Thumb resting upright alongside index
        coords[3] = [0.36, 0.55, -0.02]
        coords[4] = [0.38, 0.48, -0.03]

    elif gesture == "B":
        # 4 fingers straight up together, thumb folded over palm
        extend_finger(5, -0.05, -1.0, 0.0)
        extend_finger(9, 0.0, -1.0, 0.0)
        extend_finger(13, 0.05, -1.0, 0.0)
        extend_finger(17, 0.1, -1.0, 0.0)
        # Thumb folded across palm
        coords[3] = [0.46, 0.65, 0.04]
        coords[4] = [0.52, 0.66, 0.04]

    elif gesture == "C":
        # Curved fingers and thumb forming a 'C'
        for mcp, target_x in [(5, 0.38), (9, 0.40), (13, 0.42), (17, 0.44)]:
            coords[mcp + 1] = [target_x, 0.48, 0.05]
            coords[mcp + 2] = [target_x + 0.05, 0.45, 0.07]
            coords[mcp + 3] = [target_x + 0.10, 0.48, 0.08]
        # Thumb curved upward to meet
        coords[3] = [0.44, 0.60, 0.06]
        coords[4] = [0.48, 0.55, 0.07]

    elif gesture == "D":
        # Index straight up, other three curled touching thumb
        extend_finger(5, 0.0, -1.0, 0.0)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        # Thumb touches curled middle fingertip
        coords[3] = [0.45, 0.60, 0.03]
        coords[4] = [0.50, 0.58, 0.03]

    elif gesture == "E":
        # Tight curled fingers, thumb folded underneath
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.45, 0.70, 0.05]
        coords[4] = [0.52, 0.70, 0.05]

    elif gesture == "L":
        # Index straight up, thumb extended sideways (90 deg 'L'), others curled
        extend_finger(5, 0.0, -1.0, 0.0)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        # Thumb pointing strictly left
        coords[3] = [0.28, 0.68, 0.0]
        coords[4] = [0.20, 0.68, 0.0]

    elif gesture == "O":
        # All fingertips touch thumb forming an 'O'
        o_center = [0.48, 0.50, 0.05]
        for mcp in [5, 9, 13, 17]:
            coords[mcp + 1] = [coords[mcp][0], 0.46, 0.03]
            coords[mcp + 2] = [o_center[0] + 0.02, 0.48, 0.04]
            coords[mcp + 3] = o_center
        coords[3] = [0.42, 0.55, 0.03]
        coords[4] = o_center

    elif gesture == "V":
        # Index and middle extended apart, ring and pinky curled
        extend_finger(5, -0.3, -1.0, 0.0)
        extend_finger(9, 0.3, -1.0, 0.0)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.46, 0.64, 0.03]
        coords[4] = [0.50, 0.64, 0.03]

    elif gesture == "W":
        # Index, middle, ring extended apart, pinky curled touching thumb
        extend_finger(5, -0.35, -1.0, 0.0)
        extend_finger(9, 0.0, -1.0, 0.0)
        extend_finger(13, 0.35, -1.0, 0.0)
        curl_finger(17)
        coords[3] = [0.52, 0.65, 0.03]
        coords[4] = [0.58, 0.65, 0.03]

    elif gesture == "Y":
        # Thumb and pinky extended wide, middle three curled
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        # Thumb extended left
        coords[3] = [0.26, 0.68, 0.0]
        coords[4] = [0.18, 0.66, 0.0]
        # Pinky extended right
        extend_finger(17, 0.8, -0.6, 0.0)

    return coords


def generate_seed_data(samples_per_class: int = 150):
    """Generate diverse samples per class with rotations, scaling, translations, and noise."""
    feature_names = generate_feature_names(include_z=USE_3D_LANDMARKS)
    columns = feature_names + ["label"]
    dataset_rows = []

    np.random.seed(42)

    for gesture in GESTURE_CLASSES:
        base_coords = create_base_pose(gesture)

        for _ in range(samples_per_class):
            coords = base_coords.copy()

            # Random natural joint tremor / sensor noise
            noise = np.random.normal(0.0, 0.012, coords.shape).astype(np.float32)
            coords += noise

            # Random 2D planar rotation (±15 degrees)
            theta = np.radians(np.random.uniform(-15.0, 15.0))
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            center = coords[0]  # rotate around wrist
            shifted = coords - center
            x_rot = shifted[:, 0] * cos_t - shifted[:, 1] * sin_t
            y_rot = shifted[:, 0] * sin_t + shifted[:, 1] * cos_t
            coords[:, 0] = x_rot + center[0]
            coords[:, 1] = y_rot + center[1]

            # Random translation in frame
            dx, dy = np.random.uniform(-0.15, 0.15, size=2)
            coords[:, 0] += dx
            coords[:, 1] += dy

            # Random scale factor (distance to camera)
            scale = np.random.uniform(0.8, 1.25)
            coords = center + (coords - center) * scale

            # Invariant normalization
            norm_vector = normalize_landmarks(coords, include_z=USE_3D_LANDMARKS)
            row = list(norm_vector) + [gesture]
            dataset_rows.append(row)

    df = pd.DataFrame(dataset_rows, columns=columns)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"[Success] Generated benchmark dataset with {len(df)} samples across {len(GESTURE_CLASSES)} gestures.")
    print(f"Saved to: {DATASET_PATH}")


if __name__ == "__main__":
    generate_seed_data(samples_per_class=180)

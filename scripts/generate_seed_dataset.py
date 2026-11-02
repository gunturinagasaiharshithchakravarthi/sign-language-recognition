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

    # Gestures Definition (All 26 ASL Alphabet Letters A-Z with distinct biomechanics)
    if gesture == "A":
        # Fist with thumb upright resting alongside index finger
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        # Thumb straight UP along side of fist
        coords[3] = [0.34, 0.58, -0.01]
        coords[4] = [0.34, 0.48, -0.02]

    elif gesture == "B":
        # 4 fingers straight UP together, thumb folded across palm
        extend_finger(5, -0.04, -1.0, 0.0)
        extend_finger(9, 0.0, -1.0, 0.0)
        extend_finger(13, 0.04, -1.0, 0.0)
        extend_finger(17, 0.08, -1.0, 0.0)
        coords[3] = [0.46, 0.66, 0.05]
        coords[4] = [0.52, 0.67, 0.05]

    elif gesture == "C":
        # Curved fingers and thumb forming an open 'C' claw (clear gap to thumb)
        for mcp, target_x in [(5, 0.40), (9, 0.44), (13, 0.48), (17, 0.52)]:
            coords[mcp + 1] = [target_x, 0.46, 0.08]
            coords[mcp + 2] = [target_x + 0.06, 0.43, 0.12]
            coords[mcp + 3] = [target_x + 0.12, 0.47, 0.14]
        # Thumb curved upward with distinct 0.25 gap to fingertips
        coords[3] = [0.38, 0.65, 0.08]
        coords[4] = [0.42, 0.60, 0.10]

    elif gesture == "D":
        # Index straight UP vertically, other 3 curled into palm touching thumb
        extend_finger(5, 0.0, -1.0, 0.0)  # Index high and straight
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        # Thumb touches curled middle fingertip
        coords[3] = [0.44, 0.62, 0.03]
        coords[4] = [0.48, 0.58, 0.03]

    elif gesture == "E":
        # Flat knuckles, all 4 fingers curled down tight, thumb tucked HORIZONTALLY below
        for mcp in [5, 9, 13, 17]:
            coords[mcp + 1] = [coords[mcp][0], 0.58, 0.04]
            coords[mcp + 2] = [coords[mcp][0], 0.64, 0.05]
            coords[mcp + 3] = [coords[mcp][0], 0.69, 0.03]
        # Thumb tucked horizontally across bottom of finger pads
        coords[3] = [0.44, 0.72, 0.02]
        coords[4] = [0.52, 0.72, 0.02]

    elif gesture == "F":
        # Thumb and index tips touch forming circle ('OK' sign), middle/ring/pinky extended UP
        coords[6] = [0.40, 0.54, 0.04]
        coords[7] = [0.43, 0.58, 0.05]
        coords[8] = [0.46, 0.60, 0.05]  # Index tip touches thumb
        coords[3] = [0.40, 0.64, 0.04]
        coords[4] = [0.46, 0.60, 0.05]  # Thumb tip touches index
        extend_finger(9, -0.05, -1.0, 0.0)
        extend_finger(13, 0.05, -1.0, 0.0)
        extend_finger(17, 0.15, -1.0, 0.0)

    elif gesture == "G":
        # Hand in profile: Index pointing horizontally left, thumb extended alongside
        extend_finger(5, -1.0, 0.0, 0.0)  # Index horizontal
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        # Thumb parallel to index pointing left
        coords[3] = [0.32, 0.62, 0.02]
        coords[4] = [0.22, 0.60, 0.02]

    elif gesture == "H":
        # Hand in profile: Index AND middle pointing horizontally left together
        extend_finger(5, -1.0, -0.05, 0.0)
        extend_finger(9, -1.0, 0.05, 0.0)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.40, 0.68, 0.03]
        coords[4] = [0.34, 0.68, 0.03]

    elif gesture == "I":
        # Pinky extended straight UP, others curled, thumb folded across index/middle
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        extend_finger(17, 0.0, -1.0, 0.0)
        coords[3] = [0.46, 0.65, 0.05]
        coords[4] = [0.50, 0.65, 0.05]

    elif gesture == "J":
        # Pinky extended with curved wrist/hook
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        extend_finger(17, -0.4, -0.9, 0.05)
        coords[20] = [coords[19][0] - 0.04, coords[19][1] + 0.02, 0.06]
        coords[3] = [0.46, 0.65, 0.05]
        coords[4] = [0.50, 0.65, 0.05]

    elif gesture == "K":
        # Index straight UP, middle tilted forward at 45 deg, thumb between them
        extend_finger(5, -0.1, -1.0, 0.0)
        extend_finger(9, 0.15, -0.8, 0.3)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.42, 0.60, 0.08]
        coords[4] = [0.46, 0.54, 0.12]

    elif gesture == "L":
        # Index straight UP, thumb extended horizontally (90 deg L)
        extend_finger(5, 0.0, -1.0, 0.0)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.28, 0.68, 0.0]
        coords[4] = [0.18, 0.68, 0.0]

    elif gesture == "M":
        # 3 fingers (index, middle, ring) folded over thumb, thumb pokes under ring
        for mcp in [5, 9, 13]:
            coords[mcp + 1] = [coords[mcp][0], 0.58, 0.04]
            coords[mcp + 2] = [coords[mcp][0], 0.64, 0.05]
            coords[mcp + 3] = [coords[mcp][0], 0.68, 0.04]
        curl_finger(17)  # Pinky curled
        coords[3] = [0.48, 0.66, 0.01]
        coords[4] = [0.56, 0.66, 0.01]  # Thumb under ring finger

    elif gesture == "N":
        # 2 fingers (index, middle) folded over thumb, thumb pokes under middle
        for mcp in [5, 9]:
            coords[mcp + 1] = [coords[mcp][0], 0.58, 0.04]
            coords[mcp + 2] = [coords[mcp][0], 0.64, 0.05]
            coords[mcp + 3] = [coords[mcp][0], 0.68, 0.04]
        curl_finger(13)  # Ring curled
        curl_finger(17)  # Pinky curled
        coords[3] = [0.44, 0.66, 0.01]
        coords[4] = [0.50, 0.66, 0.01]  # Thumb under middle finger

    elif gesture == "O":
        # All fingertips touch thumb tip forming a closed 'O' circle
        o_center = [0.46, 0.54, 0.08]
        for mcp in [5, 9, 13, 17]:
            coords[mcp + 1] = [coords[mcp][0], 0.50, 0.06]
            coords[mcp + 2] = [o_center[0] + 0.02, 0.52, 0.07]
            coords[mcp + 3] = o_center
        coords[3] = [0.42, 0.60, 0.06]
        coords[4] = o_center  # Thumb tip touching all fingertips

    elif gesture == "P":
        # Like K but angled downwards: index forward/down, middle straight down
        extend_finger(5, -0.6, 0.7, 0.0)
        extend_finger(9, -0.2, 0.9, 0.2)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.42, 0.64, 0.06]
        coords[4] = [0.44, 0.70, 0.09]

    elif gesture == "Q":
        # Like G pointing downward: index and thumb pinched pointing down
        extend_finger(5, -0.3, 0.9, 0.0)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.38, 0.68, 0.02]
        coords[4] = [0.34, 0.78, 0.02]

    elif gesture == "R":
        # Index and middle crossed over each other pointing UP
        extend_finger(5, 0.04, -1.0, 0.0)
        extend_finger(9, -0.04, -1.0, 0.06)  # Crosses in front of index
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.46, 0.64, 0.04]
        coords[4] = [0.50, 0.64, 0.04]

    elif gesture == "S":
        # Fist with thumb wrapped horizontally across the FRONT of fingers
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.44, 0.62, 0.08]
        coords[4] = [0.52, 0.62, 0.08]  # Thumb wrapped across front

    elif gesture == "T":
        # Fist with thumb poking UP between index and middle knuckles
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.42, 0.64, 0.04]
        coords[4] = [0.44, 0.55, 0.06]  # Thumb poking up between index/middle

    elif gesture == "U":
        # Index and middle straight UP together (touching, gap < 0.05)
        extend_finger(5, -0.02, -1.0, 0.0)
        extend_finger(9, 0.02, -1.0, 0.0)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.46, 0.64, 0.04]
        coords[4] = [0.50, 0.64, 0.04]

    elif gesture == "V":
        # Index and middle extended apart in a 'V' peace sign (gap > 0.20)
        extend_finger(5, -0.28, -1.0, 0.0)
        extend_finger(9, 0.28, -1.0, 0.0)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.46, 0.64, 0.04]
        coords[4] = [0.50, 0.64, 0.04]

    elif gesture == "W":
        # Index, middle, ring extended UP and spread apart (W shape)
        extend_finger(5, -0.32, -1.0, 0.0)
        extend_finger(9, 0.0, -1.0, 0.0)
        extend_finger(13, 0.32, -1.0, 0.0)
        curl_finger(17)
        coords[3] = [0.52, 0.66, 0.04]
        coords[4] = [0.56, 0.66, 0.04]

    elif gesture == "X":
        # Index finger bent into a hook shape, other fingers curled
        mcp = coords[5]
        coords[6] = [mcp[0], mcp[1] - 0.04, 0.02]
        coords[7] = [mcp[0] - 0.02, mcp[1] - 0.03, 0.06]
        coords[8] = [mcp[0] - 0.03, mcp[1] + 0.01, 0.08]  # Hook tip bent down
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.46, 0.64, 0.04]
        coords[4] = [0.50, 0.64, 0.04]

    elif gesture == "Y":
        # Thumb and pinky extended wide horizontally (shaka / hang loose)
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        coords[3] = [0.24, 0.68, 0.0]
        coords[4] = [0.14, 0.66, 0.0]   # Thumb extended far left
        extend_finger(17, 0.85, -0.5, 0.0)  # Pinky extended far right

    elif gesture == "Z":
        # Index finger extended pointing forward/up with angled hand
        extend_finger(5, -0.2, -0.9, 0.15)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)
        coords[3] = [0.44, 0.64, 0.04]
        coords[4] = [0.48, 0.64, 0.04]

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

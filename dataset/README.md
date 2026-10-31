# Sign Language Landmark Dataset

## Overview
This directory contains the labeled 3D hand landmark dataset used for training and evaluating the Sign Language Recognition System.

## File Information
- **`gestures.csv`**: Contains normalized 3D hand landmark coordinate samples.
  - **Sample Count:** 1,800 samples
  - **Class Distribution:** 180 samples per class across 10 static ASL alphabet gestures (`A`, `B`, `C`, `D`, `E`, `L`, `O`, `V`, `W`, `Y`).
  - **Feature Count:** 63 invariant floating-point coordinates per sample (`WRIST_x`, `WRIST_y`, `WRIST_z`, ..., `PINKY_TIP_z`) + `label`.

## Coordinate Normalization Protocol
1. **Translation Invariance:** All landmark points are centered relative to the wrist keypoint $\mathbf{P}_0 = (0, 0, 0)$.
2. **Scale Invariance:** All coordinates are divided by the maximum Euclidean distance to the wrist, bounding values inside the unit sphere $\|\mathbf{P}\| \le 1.0$.

## Collecting New Data
To append custom webcam samples to this dataset:
```powershell
python scripts/collect_data.py
```

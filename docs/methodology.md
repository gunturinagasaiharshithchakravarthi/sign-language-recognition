# Mathematical & Machine Learning Methodology

## 1. Biomechanical 21-Keypoint Topology
Human hand pose estimation is framed as a 3D joint coordinate regression task. MediaPipe Hand Landmarker outputs 21 keypoints:

| Index | Anatomical Joint Landmark | Region |
|---|---|---|
| 0 | Wrist | Palm Base |
| 1, 2, 3, 4 | CMC, MCP, IP, Tip | Thumb Ray |
| 5, 6, 7, 8 | MCP, PIP, DIP, Tip | Index Finger Ray |
| 9, 10, 11, 12 | MCP, PIP, DIP, Tip | Middle Finger Ray |
| 13, 14, 15, 16 | MCP, PIP, DIP, Tip | Ring Finger Ray |
| 17, 18, 19, 20 | MCP, PIP, DIP, Tip | Pinky Finger Ray |

---

## 2. Mathematical Coordinate Invariance Formulation

Raw camera coordinates $\mathbf{P}_i = (x_i, y_i, z_i)$ are dependent on:
1. Where the user positions their hand in the camera frame.
2. How close or far the hand is from the webcam lens.

### Step 1: Translation Invariance
To eliminate dependency on frame position, all coordinates are translated relative to the wrist origin $\mathbf{P}_0$:
$$\mathbf{P}'_i = \mathbf{P}_i - \mathbf{P}_0 = (x_i - x_0, y_i - y_0, z_i - z_0) \quad \forall i \in \{0, 1, \dots, 20\}$$
Consequently:
$$\mathbf{P}'_0 = (0, 0, 0)$$

### Step 2: Scale Invariance
To eliminate dependency on distance from the lens, we calculate the maximum Euclidean norm across all joint offset vectors:
$$d_{\max} = \max_{i \in \{0, \dots, 20\}} \sqrt{(x'_i)^2 + (y'_i)^2 + (z'_i)^2}$$
If $d_{\max} > \epsilon$ (where $\epsilon = 10^{-6}$), each point is normalized:
$$\mathbf{P}''_i = \frac{\mathbf{P}'_i}{d_{\max}}$$
This bounds all normalized coordinate values strictly within the unit sphere:
$$\|\mathbf{P}''_i\|_2 \le 1.0 \quad \forall i$$

### Step 3: Feature Vector Flattening
The normalized coordinates are serialized into a feature vector $\mathbf{x} \in \mathbb{R}^{63}$:
$$\mathbf{x} = \begin{bmatrix} x''_0 & y''_0 & z''_0 & x''_1 & y''_1 & z''_1 & \dots & x''_{20} & y''_{20} & z''_{20} \end{bmatrix}^\top$$

---

## 3. Classifier Design: Random Forest vs. SVM

### Random Forest Classifier
- **Ensemble Method:** Bootstrap Aggregating (Bagging) of 120 randomized decision trees.
- **Node Splitting Criterion:** Gini Impurity:
  $$I_G(p) = 1 - \sum_{k=1}^{K} p_k^2$$
- **Advantages:**
  - Non-linear decision boundaries between similar hand gestures (e.g., distinguishing curled fingers in 'A' vs. 'E').
  - Resilience against localized sensor noise in landmark tracking.
  - Generates calibrated probability estimates for confidence filtering.
  - Sub-millisecond CPU inference time ($< 1.5 \text{ ms}$).

### Data Leakage Prevention Protocol
To guarantee scientific validity and prevent data leakage:
1. Features are strictly derived from individual static frame hand geometry without temporal bleeding across partitions.
2. Stratified 80/20 train-test splitting maintains identical class proportions across both training and test sets.
3. Test set samples are held out until the final evaluation phase.

---

## 4. Temporal Debounce & Stabilization Theory

Real-time computer vision feeds produce 30 discrete predictions per second. Without temporal post-processing, minor micro-tremors cause character flickering, and a user holding gesture 'A' for 2 seconds would inject 60 repeated 'A' characters.

We implement a temporal ring-buffer algorithm:
$$\mathcal{B}_t = [y_{t-M+1}, \dots, y_{t-1}, y_t]$$
Where $M = 8$ is the buffer size. A candidate letter $c$ is confirmed if and only if:
$$\forall y \in \mathcal{B}_t, \quad y = c$$
$$\text{and} \quad (c \neq c_{\text{last}} \quad \lor \quad t - t_{\text{last}} \ge \Delta t_{\text{cooldown}})$$
where $\Delta t_{\text{cooldown}} = 1.0\text{ s}$.
This guarantees clean sentence formation.

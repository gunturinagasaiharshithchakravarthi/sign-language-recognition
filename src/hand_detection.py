"""Real-Time Hand Detection and Visual Landmark Annotation using MediaPipe."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Any
import cv2
import numpy as np
import mediapipe as mp

from config import (
    MODELS_DIR,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    USE_3D_LANDMARKS
)
from src.landmark_extraction import (
    HAND_CONNECTIONS,
    extract_raw_coordinates,
    normalize_landmarks
)


class HandDetector:
    """Robust hand detector supporting both modern MediaPipe Tasks (1.2+) and legacy APIs."""

    def __init__(
        self,
        max_num_hands: int = MAX_NUM_HANDS,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        model_path: Optional[Path] = None
    ):
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.backend = None
        self.detector = None

        # Check for model bundle for MediaPipe Tasks
        if model_path is None:
            model_path = MODELS_DIR / "hand_landmarker.task"
        self.model_path = Path(model_path)

        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize MediaPipe Hand Landmarker."""
        # 1. Try MediaPipe Tasks API (modern 1.2+)
        if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            try:
                from mediapipe.tasks.python import vision, BaseOptions
                if not self.model_path.exists():
                    self._download_task_model()
                
                options = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.model_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=self.max_num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.backend = "tasks"
                return
            except Exception as e:
                print(f"[HandDetector] MediaPipe Tasks initialization note: {e}")

        # 2. Try legacy solutions API (0.10.x fallback)
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self.backend = "solutions"
                return
            except Exception as e:
                print(f"[HandDetector] Solutions initialization note: {e}")

        raise RuntimeError("Could not initialize MediaPipe hand detector via either Tasks or Solutions API.")

    def _download_task_model(self):
        """Automatically retrieve the official MediaPipe hand landmarker model if missing."""
        import urllib.request
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        print(f"[HandDetector] Downloading hand landmarker task model to {self.model_path}...")
        urllib.request.urlretrieve(url, self.model_path)
        print("[HandDetector] Task model downloaded successfully.")

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[List[Any], List[str]]:
        """Process an OpenCV BGR frame and return detected hand landmarks and handedness labels.
        
        Args:
            frame_bgr: Input BGR image from OpenCV / webcam.
            
        Returns:
            Tuple of (detected_hands_landmarks, handedness_labels)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return [], []

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "tasks":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.detector.detect(mp_image)
            hands_landmarks = result.hand_landmarks if result.hand_landmarks else []
            handedness = []
            if result.handedness:
                for h in result.handedness:
                    handedness.append(h[0].category_name if h else "Hand")
            return hands_landmarks, handedness

        elif self.backend == "solutions":
            result = self.detector.process(frame_rgb)
            if not result.multi_hand_landmarks:
                return [], []
            hands_landmarks = result.multi_hand_landmarks
            handedness = []
            if result.multi_handedness:
                for h in result.multi_handedness:
                    handedness.append(h.classification[0].label)
            return hands_landmarks, handedness

        return [], []

    def extract_features(self, landmarks, include_z: bool = USE_3D_LANDMARKS) -> np.ndarray:
        """Extract invariant normalized feature vector from a single hand's landmarks."""
        raw_coords = extract_raw_coordinates(landmarks)
        return normalize_landmarks(raw_coords, include_z=include_z)

    def draw_landmarks(
        self,
        frame_bgr: np.ndarray,
        hands_landmarks: List[Any],
        accent_color: Tuple[int, int, int] = (255, 200, 0),  # Cyan in BGR
        joint_color: Tuple[int, int, int] = (0, 255, 128)   # Green-cyan in BGR
    ) -> np.ndarray:
        """Draw futuristic cybernetic landmarks and connections over the frame."""
        annotated = frame_bgr.copy()
        h, w, _ = annotated.shape

        for hand in hands_landmarks:
            pixel_points = {}
            # Extract pixel coordinates
            for idx, lm in enumerate(hand):
                px = int(lm.x * w)
                py = int(lm.y * h)
                pixel_points[idx] = (px, py)

            # Draw futuristic skeleton connection lines
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx in pixel_points and end_idx in pixel_points:
                    pt1 = pixel_points[start_idx]
                    pt2 = pixel_points[end_idx]
                    # Semi-glow effect: thicker darker line, then sharper bright line
                    cv2.line(annotated, pt1, pt2, (40, 40, 40), 4, cv2.LINE_AA)
                    cv2.line(annotated, pt1, pt2, accent_color, 2, cv2.LINE_AA)

            # Draw keypoint joint nodes
            for idx, pt in pixel_points.items():
                if idx in (4, 8, 12, 16, 20):  # Fingertips (highlighted)
                    cv2.circle(annotated, pt, 7, (0, 140, 255), -1, cv2.LINE_AA)  # Orange core
                    cv2.circle(annotated, pt, 9, (255, 255, 255), 1, cv2.LINE_AA)  # White ring
                elif idx == 0:  # Wrist root
                    cv2.circle(annotated, pt, 8, (255, 0, 255), -1, cv2.LINE_AA)  # Magenta wrist
                    cv2.circle(annotated, pt, 10, (255, 255, 255), 1, cv2.LINE_AA)
                else:  # Finger joints
                    cv2.circle(annotated, pt, 4, joint_color, -1, cv2.LINE_AA)
                    cv2.circle(annotated, pt, 5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def close(self):
        """Release underlying detector resources."""
        if hasattr(self.detector, "close"):
            self.detector.close()

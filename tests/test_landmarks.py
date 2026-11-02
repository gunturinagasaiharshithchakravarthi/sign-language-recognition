"""Unit tests for landmark extraction and normalization algorithms."""

import unittest
import numpy as np

from src.landmark_extraction import (
    normalize_landmarks,
    generate_feature_names,
    HAND_CONNECTIONS,
    LANDMARK_NAMES
)


class TestLandmarkExtraction(unittest.TestCase):
    """Tests for translation and scale invariant landmark normalization."""

    def setUp(self):
        # Create synthetic 21 x 3 hand landmarks
        self.raw_coords = np.random.uniform(0.1, 0.9, size=(21, 3)).astype(np.float32)

    def test_landmark_shape(self):
        """Verify vector shape matches 85 for 3D and 64 for 2D (including 22 kinematic metrics)."""
        vec_3d = normalize_landmarks(self.raw_coords, include_z=True)
        self.assertEqual(vec_3d.shape, (85,))

        vec_2d = normalize_landmarks(self.raw_coords, include_z=False)
        self.assertEqual(vec_2d.shape, (64,))

    def test_translation_invariance(self):
        """Shifting hand position anywhere in the frame must produce identical normalized vectors."""
        shift = np.array([0.35, -0.25, 0.15], dtype=np.float32)
        shifted_coords = self.raw_coords + shift

        norm_original = normalize_landmarks(self.raw_coords, include_z=True)
        norm_shifted = normalize_landmarks(shifted_coords, include_z=True)

        # Difference should be essentially zero
        np.testing.assert_allclose(norm_original, norm_shifted, atol=1e-5)

    def test_scale_invariance(self):
        """Scaling hand distance (hand size in pixels) must produce identical normalized vectors."""
        scale_factor = 2.5
        wrist = self.raw_coords[0]
        scaled_coords = wrist + (self.raw_coords - wrist) * scale_factor

        norm_original = normalize_landmarks(self.raw_coords, include_z=True)
        norm_scaled = normalize_landmarks(scaled_coords, include_z=True)

        np.testing.assert_allclose(norm_original, norm_scaled, atol=1e-5)

    def test_feature_names(self):
        """Ensure feature name generator matches dimensions."""
        names_3d = generate_feature_names(include_z=True)
        self.assertEqual(len(names_3d), 85)
        self.assertIn("WRIST_x", names_3d)
        self.assertIn("INDEX_FINGER_TIP_z", names_3d)
        self.assertIn("dist_wrist_index", names_3d)

        names_2d = generate_feature_names(include_z=False)
        self.assertEqual(len(names_2d), 64)

    def test_connections_count(self):
        """Verify hand skeletal topology contains standard 21 connections."""
        self.assertEqual(len(HAND_CONNECTIONS), 21)


if __name__ == "__main__":
    unittest.main()

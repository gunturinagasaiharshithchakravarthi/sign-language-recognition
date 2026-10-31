"""Unit tests for TextGenerator debouncing, stabilization, and sentence formatting."""

import unittest
import time
from src.text_generator import TextGenerator


class TestTextGenerator(unittest.TestCase):
    """Tests for temporal buffer stabilization and text editing."""

    def setUp(self):
        self.generator = TextGenerator(buffer_size=4, cooldown_seconds=0.5)

    def test_stabilization_requires_consecutive_frames(self):
        """Single sporadic prediction must not trigger confirmation."""
        result = self.generator.update_prediction("A")
        self.assertIsNone(result)
        result = self.generator.update_prediction("A")
        self.assertIsNone(result)
        result = self.generator.update_prediction("B")  # Interrupted
        self.assertIsNone(result)
        result = self.generator.update_prediction("A")
        self.assertIsNone(result)

    def test_stabilization_confirmation(self):
        """Buffer filled with identical consecutive letters triggers confirmed letter."""
        self.generator.update_prediction("L")
        self.generator.update_prediction("L")
        self.generator.update_prediction("L")
        confirmed = self.generator.update_prediction("L")
        self.assertEqual(confirmed, "L")

    def test_text_manipulation_operations(self):
        """Test append, space, backspace, and clear operations."""
        self.generator.add_letter("H")
        self.generator.add_letter("I")
        self.assertEqual(self.generator.get_text(), "HI")

        self.generator.add_space()
        self.assertEqual(self.generator.get_text(), "HI ")

        # Prevent double spaces
        self.generator.add_space()
        self.assertEqual(self.generator.get_text(), "HI ")

        self.generator.add_letter("Y")
        self.generator.add_letter("O")
        self.generator.add_letter("U")
        self.assertEqual(self.generator.get_text(), "HI YOU")

        # Backspace
        self.generator.delete_last_character()
        self.assertEqual(self.generator.get_text(), "HI YO")

        # Clear
        self.generator.clear()
        self.assertEqual(self.generator.get_text(), "")


if __name__ == "__main__":
    unittest.main()

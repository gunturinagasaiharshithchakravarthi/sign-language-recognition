"""Text Generation and Prediction Stabilization Engine."""

import time
from collections import deque
from typing import Optional, List

from config import (
    STABILIZATION_BUFFER_SIZE,
    PREDICTION_COOLDOWN_SECONDS
)


class TextGenerator:
    """Manages sentence construction, prediction debouncing, and temporal stabilization."""

    def __init__(
        self,
        buffer_size: int = STABILIZATION_BUFFER_SIZE,
        cooldown_seconds: float = PREDICTION_COOLDOWN_SECONDS
    ):
        self.buffer_size = buffer_size
        self.cooldown_seconds = cooldown_seconds
        self.prediction_buffer = deque(maxlen=buffer_size)
        self.last_added_time = 0.0
        self.last_added_letter: Optional[str] = None
        self.current_text: str = ""

    def update_prediction(self, predicted_letter: Optional[str]) -> Optional[str]:
        """Feed real-time prediction into rapid stabilization buffer.
        
        Optimized for ultra-fast response:
        - Confirms a new letter immediately in 2-3 frames (~60-90ms).
        - Enforces a short 0.4s cooldown only when repeating the exact same letter.
        """
        if predicted_letter is None:
            self.prediction_buffer.append(None)
            return None

        self.prediction_buffer.append(predicted_letter)

        required_matches = self.buffer_size
        if len(self.prediction_buffer) >= required_matches:
            # Check if all recent buffer items match the predicted letter
            recent_items = list(self.prediction_buffer)[-required_matches:]
            if all(x == predicted_letter for x in recent_items):
                now = time.time()
                is_new_letter = (predicted_letter != self.last_added_letter)
                cooldown_passed = (now - self.last_added_time >= self.cooldown_seconds)

                if is_new_letter or cooldown_passed:
                    self.last_added_time = now
                    self.last_added_letter = predicted_letter
                    self.prediction_buffer.clear()
                    return predicted_letter

        return None

    def add_letter(self, letter: str) -> str:
        """Manually or automatically append a recognized letter to current sentence."""
        if letter:
            self.current_text += letter
        return self.current_text

    def add_space(self) -> str:
        """Append a space delimiter between words (avoiding double spaces)."""
        if self.current_text and not self.current_text.endswith(" "):
            self.current_text += " "
        return self.current_text

    def delete_last_character(self) -> str:
        """Remove the most recent character or backspace."""
        if self.current_text:
            self.current_text = self.current_text[:-1]
        return self.current_text

    def clear(self) -> str:
        """Reset text buffer."""
        self.current_text = ""
        self.prediction_buffer.clear()
        self.last_added_letter = None
        return self.current_text

    def get_text(self) -> str:
        """Return the current constructed sentence."""
        return self.current_text

    def set_text(self, new_text: str):
        """Update or overwrite the text buffer."""
        self.current_text = new_text

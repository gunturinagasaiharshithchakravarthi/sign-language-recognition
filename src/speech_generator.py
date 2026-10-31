"""Text-to-Speech (TTS) Engine supporting threaded local audio and error resilience."""

import threading
from typing import Optional
from config import TTS_RATE, TTS_VOLUME


class SpeechGenerator:
    """Thread-safe Text-to-Speech synthesizer utilizing pyttsx3."""

    def __init__(self, rate: int = TTS_RATE, volume: float = TTS_VOLUME):
        self.rate = rate
        self.volume = volume
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_speaking = False

    def _speak_worker(self, text: str):
        """Worker executed in background thread with proper COM initialization."""
        try:
            # On Windows, COM must be initialized per thread for pyttsx3 / SAPI5
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass

            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            self.is_speaking = True
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        except Exception as e:
            print(f"[SpeechGenerator] Audio synthesis error: {e}")
        finally:
            self.is_speaking = False
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def speak(self, text: str) -> bool:
        """Speak the provided text asynchronously.
        
        Args:
            text: Sentence or word to synthesize.
            
        Returns:
            bool: True if speech was queued/started, False if text was empty or busy.
        """
        clean_text = text.strip() if text else ""
        if not clean_text:
            return False

        # If already speaking, wait or don't re-enter
        if self.is_speaking and self._thread and self._thread.is_alive():
            return False

        self._thread = threading.Thread(
            target=self._speak_worker,
            args=(clean_text,),
            daemon=True
        )
        self._thread.start()
        return True

    def stop(self):
        """Request immediate speech stop."""
        self._stop_event.set()
        self.is_speaking = False

"""
NeoNetra V2.0 — gTTS Hindi Text-to-Speech Engine

Thread-safe, queue-based TTS using Google Text-to-Speech (Hindi).
Plays audio via pygame mixer. Uses unique temp files to avoid Windows file locks.
"""

import pygame
import tempfile
import threading
import queue
import os
import uuid
import config


class TTSEngine:
    """Non-blocking Hindi TTS engine with priority queue support."""

    def __init__(self):
        self.tts_queue = queue.Queue()
        self.is_speaking = False
        self.last_message = ""

        # Initialize pygame mixer with explicit settings for MP3 compatibility
        pygame.mixer.quit()  # Reset if already initialized
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=2048)
        print(f"[TTS] Mixer initialized: {pygame.mixer.get_init()}")

        # Start background worker thread
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("[TTS] gTTS Hindi engine initialized")

    def speak(self, text: str):
        """Add text to the speech queue (non-blocking)."""
        self.tts_queue.put(("normal", text))

    def speak_priority(self, text: str):
        """Flush the queue and speak this immediately (for urgent warnings)."""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break
        self.tts_queue.put(("priority", text))

    def _worker(self):
        """Background thread that consumes the TTS queue."""
        while True:
            try:
                priority, text = self.tts_queue.get()
                self.is_speaking = True
                self.last_message = text
                self._synthesize_and_play(text)
                self.is_speaking = False
            except Exception as e:
                print(f"[TTS] Worker error: {e}")
                self.is_speaking = False

    def _synthesize_and_play(self, text: str):
        """Generate Hindi speech with gTTS and play via pygame."""
        tmp_path = None
        try:
            from gtts import gTTS

            # Unique filename to avoid Windows file-lock collisions
            tmp_name = f"neonetra_tts_{uuid.uuid4().hex[:8]}.mp3"
            tmp_path = os.path.join(tempfile.gettempdir(), tmp_name)

            # Synthesize
            print(f"[TTS] Synthesizing: '{text[:50]}...'")
            tts = gTTS(text=text, lang=config.TTS_LANG)
            tts.save(tmp_path)
            print(f"[TTS] Saved to {tmp_path}")

            # Play
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            print("[TTS] Playing...")

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)

            # Release the file handle
            if hasattr(pygame.mixer.music, "unload"):
                pygame.mixer.music.unload()

        except Exception as e:
            print(f"[TTS] Playback error: {e}")

        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

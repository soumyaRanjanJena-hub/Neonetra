"""
NeoNetra V2.0 — Gemini Vision-Language Engine

Handles scene description and OCR via Google Gemini API.
Responds in pure Hindi for the visually impaired user.
"""

from google import genai
from google.genai import types
import cv2
import PIL.Image
import numpy as np
import os
import config
from dotenv import load_dotenv


class GeminiEngine:
    """Wrapper for Gemini 2.5 Flash Lite — scene description and OCR."""

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file!")

        self.client = genai.Client(api_key=api_key)
        self.model = config.GEMINI_MODEL
        print(f"[Gemini] Initialized with model: {self.model}")

    def _frame_to_pil(self, frame):
        """Convert an OpenCV BGR frame to a PIL Image."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(rgb)

    def describe_scene(self, frame) -> str:
        """Describe the scene in front of the user in Hindi."""
        try:
            pil_image = self._frame_to_pil(frame)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[config.SCENE_PROMPT, pil_image],
                config=types.GenerateContentConfig(
                    max_output_tokens=200,
                    temperature=0.5,
                ),
            )
            return response.text.strip()

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                raise  # Let caller handle rate limits
            print(f"[Gemini] Scene error: {e}")
            return "Scene describe nahi ho paya. Dobara try karein."

    def read_text(self, frame) -> str:
        """OCR mode: read visible text in the frame, respond in Hindi."""
        try:
            pil_image = self._frame_to_pil(frame)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[config.OCR_PROMPT, pil_image],
                config=types.GenerateContentConfig(
                    max_output_tokens=300,
                    temperature=0.2,
                ),
            )
            return response.text.strip()

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                raise
            print(f"[Gemini] OCR error: {e}")
            return "Text padh nahi paya. Dobara try karein."

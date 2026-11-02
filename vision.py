"""
NeoNetra V2.0 — SegFormer Floor Detection + Centroid Navigation Pipeline

5-Step Navigation Pipeline:
  1. CLAHE Enhancement → brighten low-light frames
  2. SegFormer Inference → pixel-level semantic segmentation (ADE20K)
  3. Floor Mask → merge walkable classes, build binary mask
  4. Contour + Centroid → cv2.findContours + cv2.moments
  5. Error Vector → centroid offset from frame center → PWM command

NOTE: Uses PyTorch directly on GPU — bypasses ONNX Runtime cuDNN issues entirely.
"""

import cv2
import numpy as np
import torch
import os
from dataclasses import dataclass
from typing import Optional

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

import config


@dataclass
class NavigationResult:
    command: str                           # "LEFT_120", "RIGHT_80", "CLEAR_0", "NO_FLOOR_0"
    centroid: Optional[tuple]              # (cx, cy) or None
    floor_contour: Optional[np.ndarray]    # Largest floor polygon for rendering
    floor_area_pct: float                  # % of frame that is walkable
    detected_classes: list                 # Which ADE20K floor classes were found
    floor_mask: Optional[np.ndarray]       # Binary floor mask (255=floor, 0=not)


class VisionPipeline:
    """SegFormer-based floor detection using PyTorch GPU directly."""

    def __init__(self):
        # ── Init lightweight objects FIRST ───────────────────────────────────
        self.clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP,
            tileGridSize=config.CLAHE_GRID
        )
        # Stronger CLAHE for very dark frames (auto-selected in _enhance)
        self.clahe_aggressive = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        
        # Pre-compute gamma LUT cache for performance (avoids recomputing every frame)
        self._gamma_lut_cache = {}
        
        self._smooth_cx = None
        self._smooth_cy = None

        # ── Device selection ──────────────────────────────────────────────────
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"[Vision] Using PyTorch GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("[Vision] Using PyTorch CPU")

        # ── Load SegFormer via HuggingFace Transformers ───────────────────────
        print("[Vision] Loading SegFormer-B0 (ADE20K)...")
        self.processor = SegformerImageProcessor.from_pretrained(config.SEGFORMER_MODEL)
        self.model = SegformerForSemanticSegmentation.from_pretrained(config.SEGFORMER_MODEL)
        self.model.to(self.device)
        self.model.eval()

        # ── Warmup — one dummy inference to init CUDA kernels ─────────────────
        print("[Vision] Warming up GPU (one-time, ~5 seconds)...")
        dummy = np.zeros((128, 128, 3), dtype=np.uint8)
        inputs = self.processor(images=dummy, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = self.model(**inputs)
        print("[Vision] Pipeline ready!")

    # ── Step 1: Preprocessing & Enhancement ──────────────────────────────────
    def _get_gamma_lut(self, gamma: float) -> np.ndarray:
        """Get or create a cached gamma LUT for the given gamma value."""
        # Round to 1 decimal to limit cache size
        key = round(gamma, 1)
        if key not in self._gamma_lut_cache:
            inv_gamma = 1.0 / key
            self._gamma_lut_cache[key] = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
            ).astype("uint8")
        return self._gamma_lut_cache[key]

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        """Adaptive low-light enhancement tuned for ESP32 OV2640 indoors.
        
        Pipeline:
          1. Adaptive gamma (scales with darkness: 1.2 → 2.5)
          2. Fast Gaussian denoise (replaces slow NLMeans — ~0.5ms vs ~300ms)
          3. Adaptive CLAHE (aggressive for very dark, normal otherwise)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        # ── Adaptive Gamma Correction ─────────────────────────────────────
        # Brightness < 60  → very dark  → gamma ~2.5 (strong boost)
        # Brightness 60-90 → dark       → gamma ~1.8
        # Brightness 90-120→ dim        → gamma ~1.3 (gentle boost)
        # Brightness > 120 → adequate   → no gamma needed
        if mean_brightness < 120:
            gamma = np.clip(2.8 - (mean_brightness / 80.0), 1.2, 2.5)
            frame = cv2.LUT(frame, self._get_gamma_lut(gamma))

        # ── Fast Denoise (Gaussian blur — ~0.5ms instead of ~300ms) ───────
        # OV2640 produces heavy grain in low light; a small blur cleans it
        # without destroying edges that SegFormer needs to see
        if mean_brightness < 100:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # ── CLAHE on L channel ────────────────────────────────────────────
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Use aggressive CLAHE for very dark frames, normal for the rest
        if mean_brightness < 70:
            l = self.clahe_aggressive.apply(l)
        else:
            l = self.clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # ── Step 2: SegFormer Inference ───────────────────────────────────────────
    def _segment(self, frame: np.ndarray) -> np.ndarray:
        """Run SegFormer on GPU via PyTorch and return per-pixel class map."""
        small = cv2.resize(frame, config.SEGFORMER_INPUT_SIZE)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Upsample logits to full frame size then argmax
        logits = outputs.logits  # shape: (1, num_classes, H/4, W/4)
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=(config.FRAME_HEIGHT, config.FRAME_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return seg_map

    # ── Step 3: Floor Mask ────────────────────────────────────────────────────
    def _build_floor_mask(self, seg_map: np.ndarray):
        """Merge all walkable ADE20K classes into a single smoothed binary mask."""
        floor_mask = np.zeros(
            (config.FRAME_HEIGHT, config.FRAME_WIDTH), dtype=np.uint8
        )
        detected = []

        for class_id, class_name in config.FLOOR_CLASSES.items():
            match = (seg_map == class_id)
            if match.any():
                floor_mask[match] = 255
                detected.append(class_name)

        if detected:
            floor_mask = cv2.medianBlur(floor_mask, 15)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)

        return floor_mask, detected

    # ── Step 4: Contour + Centroid ────────────────────────────────────────────
    def _find_floor_centroid(self, floor_mask: np.ndarray):
        """Find the largest floor polygon and compute its centroid via cv2.moments."""
        contours, _ = cv2.findContours(
            floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            self._smooth_cx = self._smooth_cy = None
            return None, None, 0.0

        largest  = max(contours, key=cv2.contourArea)
        area     = cv2.contourArea(largest)
        area_pct = (area / (config.FRAME_WIDTH * config.FRAME_HEIGHT)) * 100

        if area < config.MIN_FLOOR_AREA:
            self._smooth_cx = self._smooth_cy = None
            return None, None, area_pct

        M = cv2.moments(largest)
        if M["m00"] == 0:
            self._smooth_cx = self._smooth_cy = None
            return largest, None, area_pct

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Exponential Moving Average — 15% new, 85% history
        if self._smooth_cx is None:
            self._smooth_cx, self._smooth_cy = cx, cy
        else:
            alpha = 0.15
            self._smooth_cx = alpha * cx + (1.0 - alpha) * self._smooth_cx
            self._smooth_cy = alpha * cy + (1.0 - alpha) * self._smooth_cy

        return largest, (int(self._smooth_cx), int(self._smooth_cy)), area_pct

    # ── Step 5: Steering Command ──────────────────────────────────────────────
    def _compute_command(self, centroid) -> str:
        """Convert centroid X offset into a haptic PWM command string."""
        if centroid is None:
            return "NO_FLOOR_0"

        cx, _ = centroid
        error = cx - config.FRAME_CENTER_X

        if error < -config.DEADBAND:
            pwm = min(config.PWM_MAX, int(config.PWM_MIN + abs(error) * 0.5))
            return f"RIGHT_{pwm}"
        elif error > config.DEADBAND:
            pwm = min(config.PWM_MAX, int(config.PWM_MIN + error * 0.5))
            return f"LEFT_{pwm}"
        else:
            return "CLEAR_0"

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> NavigationResult:
        """Run the full 5-step pipeline on a single frame."""
        enhanced                     = self._enhance(frame)
        seg_map                      = self._segment(enhanced)
        floor_mask, detected_classes = self._build_floor_mask(seg_map)
        contour, centroid, area_pct  = self._find_floor_centroid(floor_mask)
        command                      = self._compute_command(centroid)

        return NavigationResult(
            command          = command,
            centroid         = centroid,
            floor_contour    = contour,
            floor_area_pct   = area_pct,
            detected_classes = detected_classes,
            floor_mask       = floor_mask,
        )

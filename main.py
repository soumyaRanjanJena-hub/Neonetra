"""
NeoNetra V2.0 — Edge Hub Main Application

Simulation mode (--sim):   Laptop webcam + OpenCV window + keyboard shortcuts.
ESP32 mode (--esp32):      Single-frame HTTP capture from XIAO ESP32S3 Sense.
Production mode:           FastAPI + WebSocket for ESP32 + Web Dashboard.

ESP32 Capture Flow (zero lag):
  1. Python sends GET /capture to ESP32
  2. ESP32 captures one JPEG frame, returns it
  3. Python decodes and runs SegFormer
  4. Repeat — every frame is perfectly fresh

Keyboard shortcuts:
  S = Scene Description (Gemini)
  R = Read/OCR (Gemini)
  M = Mute/Unmute Motors
  Q = Quit
"""

import argparse
import cv2
import numpy as np
import time
import threading
import sys
import urllib.request
import json

import config
from vision import VisionPipeline, NavigationResult
from tts_engine import TTSEngine
from websocket_bridge import ESP32Bridge


def render_display(
    frame: np.ndarray,
    result: NavigationResult,
    fps: float,
    is_muted: bool,
    tts_last: str,
) -> np.ndarray:
    """Render the floor detection + centroid overlay on the frame."""
    display = frame.copy()
    w = config.FRAME_WIDTH

    # 1. Green floor overlay
    if result.floor_mask is not None:
        green = np.zeros_like(display)
        green[:, :, 1] = result.floor_mask
        display = cv2.addWeighted(display, 0.7, green, 0.3, 0)

    # 2. Floor contour outline
    if result.floor_contour is not None:
        cv2.drawContours(display, [result.floor_contour], -1, config.COLOR_FLOOR, 2)

    # 3. Center reference line + deadband zone
    h = config.FRAME_HEIGHT
    cv2.line(
        display,
        (config.FRAME_CENTER_X, 0),
        (config.FRAME_CENTER_X, h),
        (100, 100, 100),
        1,
    )
    cv2.rectangle(
        display,
        (config.FRAME_CENTER_X - config.DEADBAND, 0),
        (config.FRAME_CENTER_X + config.DEADBAND, h),
        (50, 50, 50),
        1,
    )

    # 4. Centroid dot + steering arrow
    if result.centroid:
        cx, cy = result.centroid
        cv2.circle(display, (cx, cy), 10, config.COLOR_CENTROID, -1)
        cv2.circle(display, (cx, cy), 12, (255, 255, 255), 2)

        arrow_color = (
            config.COLOR_CLEAR if "CLEAR" in result.command else config.COLOR_ARROW
        )
        cv2.arrowedLine(
            display, (config.FRAME_CENTER_X, h // 2), (cx, cy), arrow_color, 3
        )

    # 5. HUD Panel (top-left)
    cv2.rectangle(display, (0, 0), (370, 175), (0, 0, 0), -1)
    cv2.rectangle(display, (0, 0), (370, 175), (50, 50, 50), 1)

    def hud(text, y, scale=0.6, color=config.COLOR_TEXT):
        cv2.putText(
            display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2
        )

    hud(f"FPS: {fps:.1f}", 22, color=(0, 255, 0))

    cmd_color = (0, 255, 0) if "CLEAR" in result.command else (0, 165, 255)
    hud(f"CMD: {result.command}", 48, scale=0.7, color=cmd_color)

    hud(f"Floor: {result.floor_area_pct:.1f}%", 72)

    classes_str = ", ".join(result.detected_classes) if result.detected_classes else "none"
    hud(f"Classes: {classes_str}", 95, scale=0.5, color=(180, 180, 180))

    if result.centroid:
        hud(
            f"Centroid: ({result.centroid[0]}, {result.centroid[1]})",
            115,
            scale=0.5,
            color=(180, 180, 180),
        )

    mute_color = (0, 0, 255) if is_muted else (0, 255, 0)
    mute_text = "MUTED" if is_muted else "ACTIVE"
    hud(f"Motors: {mute_text}", 138, scale=0.55, color=mute_color)

    if tts_last:
        # OpenCV can't render Devanagari — strip non-ASCII, show romanized hint
        ascii_msg = "".join(c for c in tts_last if ord(c) < 128)[:45]
        display_msg = ascii_msg.strip() if ascii_msg.strip() else "[Speaking Hindi...]"
        hud(f"TTS: {display_msg}", 160, scale=0.4, color=(150, 150, 150))

    # 6. Keyboard hints (bottom bar)
    cv2.rectangle(display, (0, h - 28), (config.FRAME_WIDTH, h), (0, 0, 0), -1)
    cv2.putText(
        display,
        "S=Scene  R=Read  M=Mute  Q=Quit",
        (10, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (140, 140, 140),
        1,
    )

    return display


def _check_esp32_status(url: str) -> bool:
    """Ping the ESP32 status endpoint to verify it's reachable."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            print(f"[ESP32] Device: {data.get('device', '?')}")
            print(f"[ESP32] Uptime: {data.get('uptime_ms', 0) / 1000:.0f}s")
            print(f"[ESP32] WiFi RSSI: {data.get('rssi', '?')} dBm")
            return data.get("status") == "ok"
    except Exception as e:
        print(f"[ESP32] Status check failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
#  ESP32 Single-Frame Capture
# ═══════════════════════════════════════════════════════════════

def _capture_esp32_frame(capture_url: str, timeout: float = 3.0):
    """Fetch ONE JPEG frame from ESP32's /capture endpoint.

    Returns (True, frame) on success, (False, None) on failure.
    Each call = one fresh frame. No buffering, no streaming, zero lag.
    """
    try:
        req = urllib.request.Request(capture_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            jpeg_bytes = resp.read()
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return True, frame
            return False, None
    except Exception:
        return False, None


# ═══════════════════════════════════════════════════════════════
#  Shared Processing Logic
# ═══════════════════════════════════════════════════════════════

def _process_frame(frame, vision, tts, is_muted):
    """Run the vision pipeline."""
    if frame.shape[1] != config.FRAME_WIDTH or frame.shape[0] != config.FRAME_HEIGHT:
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

    t0 = time.time()
    result = vision.process(frame)
    fps = 1.0 / max(time.time() - t0, 0.001)

    if result.command not in ("CLEAR_0", "NO_FLOOR_0"):
        print(
            f"[NAV] {result.command} | Floor: {result.floor_area_pct:.0f}% "
            f"| Centroid: {result.centroid}"
        )

    return frame, result, fps


def _handle_keys(key, gemini, latest_frame, tts, is_muted, mode_tag):
    """Handle keyboard input. Returns (should_quit, is_muted)."""
    if key == config.KEY_QUIT:
        print(f"[{mode_tag}] Quit requested.")
        return True, is_muted

    elif key == config.KEY_SCENE:
        if gemini and latest_frame is not None:
            print("[CMD] Scene description requested...")

            def _scene(f):
                try:
                    r = gemini.describe_scene(f)
                    print(f"[Gemini] Scene: {r}")
                    tts.speak(r)
                except Exception as e:
                    print(f"[Gemini] Scene error: {e}")
                    tts.speak("Scene describe nahi ho paya.")

            threading.Thread(
                target=_scene, args=(latest_frame.copy(),), daemon=True
            ).start()
        else:
            print("[CMD] Gemini not available")
            tts.speak("Gemini available nahi hai.")

    elif key == config.KEY_OCR:
        if gemini and latest_frame is not None:
            print("[CMD] OCR requested...")

            def _ocr(f):
                try:
                    r = gemini.read_text(f)
                    print(f"[Gemini] OCR: {r}")
                    tts.speak(r)
                except Exception as e:
                    print(f"[Gemini] OCR error: {e}")
                    tts.speak("Text padh nahi paya.")

            threading.Thread(
                target=_ocr, args=(latest_frame.copy(),), daemon=True
            ).start()
        else:
            print("[CMD] Gemini not available")
            tts.speak("Gemini available nahi hai.")

    elif key == config.KEY_MUTE:
        is_muted = not is_muted
        status = "MUTED" if is_muted else "ACTIVE"
        print(f"[CMD] Motors: {status}")
        tts.speak("Mute on" if is_muted else "Mute off")

    return False, is_muted


# ═══════════════════════════════════════════════════════════════
#  Webcam Loop (--sim)
# ═══════════════════════════════════════════════════════════════

class _WebcamGrabber:
    """Background thread that continuously grabs webcam frames."""

    def __init__(self, source):
        self.source = source
        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._running = False
        self._cap = None
        self._thread = None

    def start(self) -> bool:
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not self._cap.isOpened():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                    self._ret = True
            else:
                self._running = False

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    @property
    def is_alive(self):
        return self._running

    def release(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()


def webcam_loop(vision, tts, gemini):
    """Simulation mode: laptop webcam + OpenCV display."""
    grabber = _WebcamGrabber(config.WEBCAM_INDEX)
    if not grabber.start():
        print("ERROR: Could not open webcam!")
        sys.exit(1)

    print(f"[SIM] Webcam {config.WEBCAM_INDEX} active")
    print("[SIM] Keys: S=Scene  R=Read/OCR  M=Mute  Q=Quit\n")

    is_muted = False
    latest_frame = None

    try:
        while grabber.is_alive:
            ret, frame = grabber.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame, result, fps = _process_frame(
                frame, vision, tts, is_muted
            )
            latest_frame = frame

            display = render_display(frame, result, fps, is_muted, tts.last_message)
            cv2.imshow("NeoNetra V2.0", display)

            key = cv2.waitKey(1) & 0xFF
            should_quit, is_muted = _handle_keys(
                key, gemini, latest_frame, tts, is_muted, "SIM"
            )
            if should_quit:
                break

    except KeyboardInterrupt:
        print("\n[SIM] Interrupted.")
    finally:
        grabber.release()
        cv2.destroyAllWindows()
        print("[SIM] Shutdown complete.")


# ═══════════════════════════════════════════════════════════════
#  ESP32 Pipelined Capture Thread
# ═══════════════════════════════════════════════════════════════

class _ESP32CaptureThread:
    """Background thread that continuously captures single frames from /capture.

    While SegFormer processes frame N (~300ms), this thread is already
    fetching frame N+1 from the ESP32 (~100ms). This overlaps network
    latency with inference, nearly doubling effective FPS.
    """

    def __init__(self, capture_url, timeout=3.0):
        self.capture_url = capture_url
        self.timeout = timeout
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._running = False
        self._thread = None
        self._capture_ms = 0
        self._failures = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            t0 = time.time()
            ret, frame = _capture_esp32_frame(self.capture_url, self.timeout)
            elapsed_ms = (time.time() - t0) * 1000

            if ret:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._capture_ms = elapsed_ms
                    self._failures = 0
            else:
                with self._lock:
                    self._failures += 1
                time.sleep(0.3 if self._failures > 3 else 0.1)

    def get_latest(self):
        with self._lock:
            if self._frame is None:
                return None, 0, 0, self._failures
            return self._frame.copy(), self._frame_id, self._capture_ms, self._failures

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)


# ═══════════════════════════════════════════════════════════════
#  ESP32 Loop (--esp32) — Pipelined Capture
# ═══════════════════════════════════════════════════════════════

def esp32_loop(vision, tts, gemini, capture_url=None, ws_bridge=None):
    """ESP32 mode: pipelined capture + SegFormer inference.

    Background thread: GET /capture → decode → store (loops continuously)
    Main thread:       grab latest frame → SegFormer → display → repeat

    The overlap means effective FPS = max(capture, inference) instead of sum.
    """
    if capture_url is None:
        capture_url = config.ESP32_CAPTURE_URL

    # Check device health
    print("[ESP32] Checking device status...")
    if _check_esp32_status(config.ESP32_STATUS_URL):
        print("[ESP32] Device is online!")
    else:
        print("[ESP32] WARNING: Could not reach device.")
        print("[ESP32] Will try to capture frames anyway...")

    # Test first capture
    print(f"[ESP32] Capture URL: {capture_url}")
    print("[ESP32] Fetching first frame...", end="", flush=True)

    ret, frame = _capture_esp32_frame(capture_url, timeout=5.0)
    if ret:
        h, w = frame.shape[:2]
        print(f" OK! ({w}x{h})")
    else:
        print(" FAILED!")
        print("  -> Is the ESP32 powered on and connected to WiFi?")
        print(f"  -> Does {capture_url} respond in a browser?")
        print("  -> Check the ESP32 Serial Monitor for its IP address.")
        sys.exit(1)

    # Start pipelined capture thread
    capturer = _ESP32CaptureThread(capture_url, timeout=config.ESP32_CAPTURE_TIMEOUT)
    capturer.start()

    print("[ESP32] Mode: Pipelined capture (bg fetch + fg process)")
    print("[ESP32] Keys: S=Scene  R=Read/OCR  M=Mute  D=Debug-save  Q=Quit\n")

    is_muted = False
    latest_frame = None
    frame_count = 0
    last_frame_id = 0
    no_floor_last_warn = 0  # Cooldown timer for NO_FLOOR voice warnings

    try:
        while True:
            # Grab the latest frame (non-blocking)
            frame, frame_id, capture_ms, failures = capturer.get_latest()

            if frame is None or frame_id == last_frame_id:
                time.sleep(0.005)
                cv2.waitKey(1)
                continue

            last_frame_id = frame_id

            # Resize if needed
            if frame.shape[1] != config.FRAME_WIDTH or frame.shape[0] != config.FRAME_HEIGHT:
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

            latest_frame = frame.copy()
            frame_count += 1

            # Run SegFormer
            t0 = time.time()
            result = vision.process(frame)
            infer_ms = (time.time() - t0) * 1000
            fps = 1000.0 / max(infer_ms, 1)

            # Log non-trivial commands with timing
            if result.command not in ("CLEAR_0", "NO_FLOOR_0"):
                print(
                    f"[NAV] {result.command} | Floor: {result.floor_area_pct:.0f}% "
                    f"| Cap:{capture_ms:.0f}ms Inf:{infer_ms:.0f}ms"
                )

            # ⚠️ NO FLOOR WARNING — Voice alert with 15-second cooldown
            if result.command.startswith("NO_FLOOR"):
                if getattr(config, "ENABLE_NO_FLOOR_WARNING", True):
                    now_t = time.time()
                    if now_t - no_floor_last_warn > 15.0:
                        tts.speak_priority("Ruko! Zameen nahi dikh rahi. Sambhal ke chalo.")
                        no_floor_last_warn = now_t
                        print("⚠️ [SAFETY] No floor detected — voice warning sent!")
                # Force motors silent (person feels guidance stop)
                if ws_bridge:
                    ws_bridge.set_command("CLEAR_0")
            else:
                # Normal AI steering
                if ws_bridge:
                    ws_bridge.set_command(result.command if not is_muted else "CLEAR_0")

            # Render display
            display = render_display(frame, result, fps, is_muted, tts.last_message)

            # Extra HUD: timing breakdown (top-right)
            timing_text = f"Cap:{capture_ms:.0f}ms Inf:{infer_ms:.0f}ms"
            cv2.putText(
                display, timing_text, (config.FRAME_WIDTH - 280, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1
            )

            cv2.imshow("NeoNetra V2.0 [ESP32]", display)

            # Keyboard (GUI interaction)
            key = cv2.waitKey(1) & 0xFF

            # Hardware Switches (ESP32 Button interaction)
            if ws_bridge:
                if ws_bridge.scene_requested:
                    key = config.KEY_SCENE
                    ws_bridge.scene_requested = False
                elif ws_bridge.ocr_requested:
                    key = config.KEY_OCR
                    ws_bridge.ocr_requested = False
                elif ws_bridge.mute_toggled:
                    key = config.KEY_MUTE
                    ws_bridge.mute_toggled = False

            # Debug save: press D to save current frame
            if key == ord('d'):
                debug_path = f"esp32_debug_{frame_count}.jpg"
                cv2.imwrite(debug_path, latest_frame)
                print(f"[DEBUG] Saved frame to {debug_path}")

            should_quit, is_muted = _handle_keys(
                key, gemini, latest_frame, tts, is_muted, "ESP32"
            )
            if should_quit:
                break

    except KeyboardInterrupt:
        print("\n[ESP32] Interrupted.")
    finally:
        capturer.stop()
        if ws_bridge:
            ws_bridge.stop()
        cv2.destroyAllWindows()
        print(f"[ESP32] Processed {frame_count} frames. Shutdown complete.")


# ═══════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NeoNetra V2.0 Edge Hub")
    parser.add_argument(
        "--sim", action="store_true", help="Simulation mode (laptop webcam)"
    )
    parser.add_argument(
        "--esp32", action="store_true",
        help="ESP32-S3 Sense mode (pipelined capture over WiFi)"
    )
    parser.add_argument(
        "--url", type=str, default=None,
        help="Override ESP32 capture URL (e.g. http://192.168.1.42:81/capture)"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  NeoNetra V2.0 — Edge Hub")
    print("  Team Cytrix | SUIIT")
    print("=" * 50)

    # Initialize Vision Pipeline
    vision = VisionPipeline()

    # Initialize TTS
    tts = TTSEngine()

    # Initialize Gemini (optional — runs without it)
    gemini = None
    try:
        from gemini_engine import GeminiEngine

        gemini = GeminiEngine()
    except Exception as e:
        print(f"[Gemini] Unavailable: {e}")
        print("[Gemini] Scene/OCR disabled. Navigation still works offline.")

    print("\n[System] All systems initialized!")

    if args.esp32:
        capture_url = args.url if args.url else config.ESP32_CAPTURE_URL
        print(f"[System] Mode: ESP32-S3 Sense (Pipelined Capture)")
        print(f"[System] Capture URL: {capture_url}\n")
        
        # Start WebSocket Bridge
        print("[System] Connecting WebSocket bridge...")
        bridge = ESP32Bridge(config.ESP32_CAM_IP, port=82, tts_engine=tts)
        bridge.start()
        
        esp32_loop(vision, tts, gemini, capture_url=capture_url, ws_bridge=bridge)

    elif args.sim:
        print("[System] Mode: SIMULATION (Webcam)\n")
        webcam_loop(vision, tts, gemini)

    else:
        print("[System] Mode: PRODUCTION")
        print("[System] Production mode with FastAPI will be added in Phase 2.")
        print("[System] Use --sim for webcam or --esp32 for ESP32 camera.")
        sys.exit(0)


if __name__ == "__main__":
    main()

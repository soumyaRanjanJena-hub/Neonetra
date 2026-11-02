# --- NeoNetra V2.0 Configuration ---

# Server
FASTAPI_HOST = "0.0.0.0"
FASTAPI_PORT = 8000

# Camera
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WEBCAM_INDEX = 0

# ESP32-S3 Sense Camera
# Update this IP after flashing the ESP32 — check Serial Monitor output
ESP32_CAM_IP = "10.119.160.232"
ESP32_CAM_PORT = 81
ESP32_CAPTURE_URL = f"http://{ESP32_CAM_IP}:{ESP32_CAM_PORT}/capture"
ESP32_STREAM_URL  = f"http://{ESP32_CAM_IP}:{ESP32_CAM_PORT}/stream"
ESP32_STATUS_URL  = f"http://{ESP32_CAM_IP}:{ESP32_CAM_PORT}/status"
ESP32_CAPTURE_TIMEOUT = 3    # Seconds to wait for a single frame
ESP32_RECONNECT_DELAY = 2    # Seconds to wait before reconnection attempt
ESP32_MAX_RECONNECT = 15     # Max consecutive reconnection attempts

# SegFormer (ADE20K Floor Detection)
SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
SEGFORMER_INPUT_SIZE = (512, 512)

# ADE20K walkable-surface class IDs
FLOOR_CLASSES = {
    3: "floor",
    6: "road",
    11: "sidewalk",
    13: "earth",
    28: "carpet",
    52: "path",
    128: "rug",
}

# Navigation (Centroid Steering)
FRAME_CENTER_X = FRAME_WIDTH // 2   # 320
DEADBAND = 40                        # ±40px from center = perfectly aligned
PWM_MIN = 60                         # Minimum motor PWM
PWM_MAX = 200                        # Maximum steering PWM
MIN_FLOOR_AREA = 25000               # Min pixel count for valid floor polygon (~8% of frame)
ENABLE_NO_FLOOR_WARNING = False      # Toggle for "No Floor Detected" voice warning

# CLAHE (Low-Light Enhancement)
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# Gemini VLM
GEMINI_MODEL = "gemini-2.5-flash-lite"

SCENE_PROMPT = (
    "Aap ek visually impaired insaan ke navigation assistant ho. "
    "Unke saamne ka scene 2-3 chhoti sentences mein describe karo. "
    "Obstacles, logon, vehicles, doors, aur raaste pe focus karo. "
    "Sirf Hindi mein jawab do."
)

OCR_PROMPT = (
    "Is image mein jo bhi text dikh raha hai use dhyan se padho. "
    "Phir us text ke baare mein 1-2 chhoti Hindi sentences mein batao ki yeh kya likha hai aur kiske baare mein hai. "
    "Sirf Hindi mein jawab do. Agar koi text nahi dikh raha toh bolo 'Koi text nahi dikh raha'."
)

# gTTS
TTS_LANG = "hi"  # Hindi

# Keyboard Simulation Keys (OpenCV waitKey codes)
KEY_SCENE = ord('s')
KEY_OCR = ord('r')
KEY_MUTE = ord('m')
KEY_QUIT = ord('q')

# Dashboard / Rendering Colors (BGR)
COLOR_FLOOR = (0, 255, 0)       # Green overlay
COLOR_CENTROID = (0, 0, 255)    # Red dot
COLOR_ARROW = (0, 165, 255)     # Orange steering arrow
COLOR_CLEAR = (0, 255, 0)       # Green arrow when aligned
COLOR_TEXT = (255, 255, 255)     # White HUD text

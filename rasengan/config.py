# ── Paths ────────────────────────────────────────────────────────────────────
GIF_PATH   = "rasengan-green-white.gif"
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# ── Hand reference measurements ──────────────────────────────────────────────
HAND_REF_CM = 9.0    # palm width (knuckle-to-knuckle) ≈ 9 cm on adult hand
ORB_DIAM_CM = 14.0   # orb diameter in cm

# ── Palm landmark IDs ────────────────────────────────────────────────────────
PALM_IDS = [0, 5, 9, 13, 17]  # wrist + 4 MCP knuckles

# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTH_ALPHA = 0.35   # lower = smoother but more lag, higher = more responsive

# ── Detection thresholds ─────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE  = 0.5
NUM_HANDS                = 1

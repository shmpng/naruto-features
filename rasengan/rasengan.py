import urllib.request
import os
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ── Config ───────────────────────────────────────────────────────────────────
GIF_PATH    = "rasengan-green-white.gif"
MODEL_PATH  = "hand_landmarker.task"
MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_REF_CM = 9.0   # palm width (knuckle-to-knuckle) ≈ 9 cm on adult hand
ORB_DIAM_CM = 16.0  # orb diameter in cm
FLOAT_CM    = 5.0   # how many cm above the palm surface the orb floats
PALM_IDS    = [0, 5, 9, 13, 17]  # wrist + 4 MCP knuckles

# ── Auto-download model ──────────────────────────────────────────────────────
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~8 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")

# ── Load GIF frames (green screen removed, raw RGBA) ────────────────────────
def load_frames_raw(path: str) -> list:
    gif, frames = Image.open(path), []
    for i in range(gif.n_frames):
        gif.seek(i)
        arr = np.array(gif.convert("RGBA"), dtype=np.uint8)
        r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)
        arr[(g - r > 40) & (g - b > 40), 3] = 0
        h, w = arr.shape[:2]; s = min(h, w)
        arr = arr[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2]
        frames.append(arr)
    return frames

_cache: dict = {}
def get_orb(raw_frames, idx, diam):
    bucket = max((diam // 10) * 10, 10)
    if bucket not in _cache:
        _cache.clear()
        _cache[bucket] = [
            cv2.cvtColor(
                np.array(Image.fromarray(a, "RGBA").resize((bucket, bucket), Image.LANCZOS)),
                cv2.COLOR_RGBA2BGRA
            )
            for a in raw_frames
        ]
    return _cache[bucket][idx % len(_cache[bucket])]

# ── Blend BGRA orb onto BGR frame ────────────────────────────────────────────
def overlay(bg, orb, cx, cy):
    oh, ow = orb.shape[:2]
    x1, y1 = cx - ow // 2, cy - oh // 2
    x2, y2 = x1 + ow, y1 + oh
    fx1, fy1 = max(x1, 0), max(y1, 0)
    fx2, fy2 = min(x2, bg.shape[1]), min(y2, bg.shape[0])
    if fx1 >= fx2 or fy1 >= fy2:
        return
    src   = orb[fy1-y1:fy2-y1, fx1-x1:fx2-x1]
    alpha = src[:,:,3:4].astype(np.float32) / 255.0
    bg[fy1:fy2, fx1:fx2] = (
        src[:,:,:3] * alpha + bg[fy1:fy2, fx1:fx2] * (1 - alpha)
    ).astype(np.uint8)

# ── Hand helpers ─────────────────────────────────────────────────────────────
def palm_centre(lm, w, h):
    return (
        int(np.mean([lm[i].x for i in PALM_IDS]) * w),
        int(np.mean([lm[i].y for i in PALM_IDS]) * h)
    )

def palm_normal_2d(lm, w, h):
    """
    Estimate which direction the palm is FACING in 2D screen space.
    Use wrist(0) → middle-MCP(9) as the 'up' axis of the palm,
    then rotate 90° to get the normal pointing away from the palm surface.
    Returns a unit vector (nx, ny).
    """
    # Vector from wrist to middle knuckle
    dx = (lm[9].x - lm[0].x) * w
    dy = (lm[9].y - lm[0].y) * h
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return (0.0, -1.0)
    # Normalize then rotate -90° (perpendicular, pointing away from palm)
    ux, uy = dx / length, dy / length
    # Rotate -90°: (x,y) -> (y, -x)  — points "above" the palm in screen space
    return (uy, -ux)

# ── Add this near the top with other globals ─────────────────────────────────
_smoothed_pos: dict = {}   # key: hand index → (x, y)
SMOOTH_ALPHA = 0.35        # lower = smoother but more lag, higher = more responsive

def palm_centre_floating(lm, w, h, float_cm, px_per_cm, handedness_label):
    # Raw palm center
    cx = np.mean([lm[i].x for i in PALM_IDS]) * w
    cy = np.mean([lm[i].y for i in PALM_IDS]) * h

    # Wrist → middle-MCP vector
    dx = (lm[9].x - lm[0].x) * w
    dy = (lm[9].y - lm[0].y) * h
    length = np.hypot(dx, dy)

    verticality = abs(dy) / (length + 1e-6)

    # Smooth blend: 0 = full float, 1 = dead center
    blend = np.clip((verticality - 0.70) / (0.95 - 0.70), 0.0, 1.0)

    # Float direction (perpendicular to wrist→MCP, pointing away from palm)
    nx, ny = palm_normal_2d(lm, w, h)
    offset_px = float_cm * px_per_cm

    fx = cx + nx * offset_px
    fy = cy + ny * offset_px

    # Lerp toward palm center as hand faces camera
    final_x = fx * (1 - blend) + cx * blend
    final_y = fy * (1 - blend) + cy * blend

    # ── Lateral correction for sideways hand ────────────────────────────────
    # When hand is mostly horizontal (low verticality), the orb drifts away
    # from the thumb side. Pull it back toward index-MCP (landmark 5).
    horizontality = 1.0 - verticality  # 1.0 = fully sideways
    lat_correction = np.clip((horizontality - 0.4) / (0.6 - 0.4), 0.0, 1.0)

    # Vector from palm center toward index-MCP
    ix = lm[5].x * w - cx
    iy = lm[5].y * h - cy
    ilen = np.hypot(ix, iy) + 1e-6

    # Pull amount: up to 30% of palm scale toward index side
    pull_px = lat_correction * px_per_cm * 2.5
    final_x += (ix / ilen) * pull_px
    final_y += (iy / ilen) * pull_px

    return (int(final_x), int(final_y))

def palm_scale_px(lm, w, h):
    """
    Use MAX of palm length and palm width so the orb stays large
    regardless of hand angle / tilt toward the camera.
    - length: wrist(0) → middle-MCP(9)  — big when hand is flat
    - width:  index-MCP(5) → pinky-MCP(17) — big when hand is edge-on
    """
    length = np.hypot((lm[9].x  - lm[0].x)  * w, (lm[9].y  - lm[0].y)  * h)
    width  = np.hypot((lm[5].x  - lm[17].x) * w, (lm[5].y  - lm[17].y) * h)
    return max(length, width)

def is_palm_facing_camera(lm, handedness_label):
    """
    Returns True only when the PALM faces the camera (not the back of hand).
    Right hand facing cam → index-MCP(5) appears to the RIGHT of pinky-MCP(17).
    Left  hand facing cam → index-MCP(5) appears to the LEFT  of pinky-MCP(17).
    """
    index_x = lm[5].x
    pinky_x = lm[17].x
    if handedness_label == "Right":
        return index_x > pinky_x
    else:
        return index_x < pinky_x

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ensure_model()
    print("Loading GIF frames…")
    raw = load_frames_raw(GIF_PATH)
    print(f"{len(raw)} frames loaded. Show your PALM to the camera. Press Q to quit.")

    detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    # frame_idx now advances once per video frame, not once per hand
    frame_idx = 0
    ts_ms     = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        bgr = cv2.flip(bgr, 1)
        h, w = bgr.shape[:2]

        result = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            ts_ms
        )
        ts_ms += 33

        if result.hand_landmarks:
            for hand_lm, handedness in zip(result.hand_landmarks, result.handedness):
                label = handedness[0].category_name  # "Left" or "Right"

                # Only show orb when palm faces the camera
                if not is_palm_facing_camera(hand_lm, label):
                    continue

                # Scale orb using MAX of length/width — stays big at any angle
                ruler_px   = palm_scale_px(hand_lm, w, h)
                px_per_cm  = ruler_px / HAND_REF_CM
                diam       = int(px_per_cm * ORB_DIAM_CM)

                cx, cy = palm_centre_floating(
                    hand_lm, w, h, FLOAT_CM, px_per_cm, label
                )

                # Exponential smoothing — kills jitter and drift between frames
                hand_idx = list(result.handedness).index(handedness)
                if hand_idx in _smoothed_pos:
                    px, py = _smoothed_pos[hand_idx]
                    cx = int(px * (1 - SMOOTH_ALPHA) + cx * SMOOTH_ALPHA)
                    cy = int(py * (1 - SMOOTH_ALPHA) + cy * SMOOTH_ALPHA)
                _smoothed_pos[hand_idx] = (cx, cy)

                overlay(bgr, get_orb(raw, frame_idx, diam), cx, cy)
        # Advance animation once per frame (not once per hand)
        frame_idx += 1

        cv2.imshow("Rasengan – Q to quit", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
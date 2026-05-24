import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from config import (
    MODEL_PATH, MODEL_URL,
    NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


def ensure_model() -> None:
    """Download the hand landmark model if not already present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~8 MB)…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def create_detector() -> HandLandmarker:
    """Create and return a HandLandmarker configured for video mode."""
    ensure_model()
    return HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ))

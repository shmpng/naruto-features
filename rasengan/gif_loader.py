import numpy as np
import cv2
from PIL import Image


def load_frames(path: str) -> list:
    """Load GIF frames, remove green screen, crop to square. Returns list of RGBA numpy arrays."""
    gif, frames = Image.open(path), []
    for i in range(gif.n_frames):
        gif.seek(i)
        arr = np.array(gif.convert("RGBA"), dtype=np.uint8)
        r = arr[:, :, 0].astype(int)
        g = arr[:, :, 1].astype(int)
        b = arr[:, :, 2].astype(int)
        # Remove green screen pixels
        arr[(g - r > 40) & (g - b > 40), 3] = 0
        # Crop to square
        h, w = arr.shape[:2]
        s = min(h, w)
        arr = arr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        frames.append(arr)
    return frames


_cache: dict = {}

def get_orb(frames: list, frame_idx: int, diam: int):
    """Resize and cache orb frame at given diameter. Returns BGRA numpy array."""
    bucket = max((diam // 10) * 10, 10)
    if bucket not in _cache:
        _cache.clear()
        _cache[bucket] = [
            cv2.cvtColor(
                np.array(Image.fromarray(f, "RGBA").resize((bucket, bucket), Image.LANCZOS)),
                cv2.COLOR_RGBA2BGRA
            )
            for f in frames
        ]
    return _cache[bucket][frame_idx % len(_cache[bucket])]

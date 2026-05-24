import numpy as np
from config import PALM_IDS


def palm_center(lm, w: int, h: int) -> tuple:
    """Return (cx, cy) pixel coords of palm center (mean of wrist + 4 knuckles)."""
    return (
        int(np.mean([lm[i].x for i in PALM_IDS]) * w),
        int(np.mean([lm[i].y for i in PALM_IDS]) * h),
    )


def palm_scale(lm, w: int, h: int) -> float:
    """
    Return palm size in pixels as MAX of:
    - length: wrist(0) → middle-MCP(9)
    - width:  index-MCP(5) → pinky-MCP(17)
    Stays large regardless of hand tilt angle.
    """
    length = np.hypot((lm[9].x  - lm[0].x)  * w, (lm[9].y  - lm[0].y)  * h)
    width  = np.hypot((lm[5].x  - lm[17].x) * w, (lm[5].y  - lm[17].y) * h)
    return max(length, width)


def palm_normal_2d(lm, w: int, h: int) -> tuple:
    """
    Return a 2D unit vector perpendicular to the wrist→middle-MCP axis,
    pointing away from the palm surface.
    """
    dx = (lm[9].x - lm[0].x) * w
    dy = (lm[9].y - lm[0].y) * h
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return (0.0, -1.0)
    ux, uy = dx / length, dy / length
    # Rotate -90°: points "above" the palm in screen space
    return (uy, -ux)

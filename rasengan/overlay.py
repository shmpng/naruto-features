import numpy as np


def overlay(bg, orb, cx: int, cy: int) -> None:
    """Alpha-blend a BGRA orb onto a BGR background frame at center (cx, cy)."""
    oh, ow = orb.shape[:2]
    x1, y1 = cx - ow // 2, cy - oh // 2
    x2, y2 = x1 + ow, y1 + oh

    fx1, fy1 = max(x1, 0), max(y1, 0)
    fx2, fy2 = min(x2, bg.shape[1]), min(y2, bg.shape[0])

    if fx1 >= fx2 or fy1 >= fy2:
        return

    src   = orb[fy1 - y1:fy2 - y1, fx1 - x1:fx2 - x1]
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0

    bg[fy1:fy2, fx1:fx2] = (
        src[:, :, :3] * alpha + bg[fy1:fy2, fx1:fx2] * (1 - alpha)
    ).astype(np.uint8)

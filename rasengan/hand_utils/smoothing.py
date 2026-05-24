from config import SMOOTH_ALPHA

_state: dict = {}


def smooth(hand_idx: int, cx: int, cy: int) -> tuple:
    """
    Apply exponential smoothing to (cx, cy) for the given hand index.
    Returns smoothed (cx, cy).
    """
    if hand_idx in _state:
        px, py = _state[hand_idx]
        cx = int(px * (1 - SMOOTH_ALPHA) + cx * SMOOTH_ALPHA)
        cy = int(py * (1 - SMOOTH_ALPHA) + cy * SMOOTH_ALPHA)
    _state[hand_idx] = (cx, cy)
    return cx, cy


def reset() -> None:
    """Clear all smoothing state (call when hand is lost)."""
    _state.clear()

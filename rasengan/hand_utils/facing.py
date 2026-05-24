def is_palm_facing_camera(lm, handedness_label: str) -> bool:
    """
    Returns True when the PALM surface faces the camera.
    Right hand facing cam → index-MCP(5) appears to the RIGHT of pinky-MCP(17).
    Left  hand facing cam → index-MCP(5) appears to the LEFT  of pinky-MCP(17).
    """
    index_x = lm[5].x
    pinky_x = lm[17].x
    if handedness_label == "Right":
        return index_x > pinky_x
    else:
        return index_x < pinky_x


def is_back_facing_camera(lm, handedness_label: str) -> bool:
    """Returns True when the BACK/DORSAL side of the hand faces the camera."""
    return not is_palm_facing_camera(lm, handedness_label)

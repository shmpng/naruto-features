import cv2
import mediapipe as mp

from config      import GIF_PATH, HAND_REF_CM, ORB_DIAM_CM
from gif_loader  import load_frames, get_orb
from overlay     import overlay
from hand_utils  import create_detector, palm_center, palm_scale, smooth


def main():
    print("Loading GIF frames…")
    frames = load_frames(GIF_PATH)
    print(f"{len(frames)} frames loaded. Show palm to camera. Press Q to quit.")

    detector  = create_detector()
    frame_idx = 0
    ts_ms     = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    while True:
        ok, bgr = cap.read()
        if not ok:
            break

        bgr    = cv2.flip(bgr, 1)
        h, w   = bgr.shape[:2]

        result = detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
            ts_ms,
        )
        ts_ms += 33

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]          # first hand only → one orb

            ruler      = palm_scale(lm, w, h)
            px_per_cm  = ruler / HAND_REF_CM
            diam       = int(px_per_cm * ORB_DIAM_CM)

            cx, cy     = palm_center(lm, w, h)
            cx, cy     = smooth(0, cx, cy)          # stabilise jitter

            overlay(bgr, get_orb(frames, frame_idx, diam), cx, cy)

        frame_idx += 1

        cv2.imshow("Rasengan – Q to quit", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

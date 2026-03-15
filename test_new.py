"""
Hand Tracking Visualization from POV Helmet Camera
===================================================
Input : any MP4/AVI video (e.g. 20-min helmet-cam footage)
Output: hand_keypoints.csv  – per-frame landmark data
        hand_motion.mp4     – skeleton visualization video

Usage:
    python hand_tracking_pov.py --input factory.mp4

Install:
    pip install mediapipe opencv-python tqdm
"""

import argparse
import csv
import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions

# ── Landmark connections (21 keypoints) ─────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── Visual style ─────────────────────────────────────────────────────────────
CANVAS_BG   = (10, 10, 18)
BONE_COLOR  = (80, 220, 160)
JOINT_COLOR = (255, 255, 255)
TIP_COLOR   = (255, 100, 80)
TIPS        = {4, 8, 12, 16, 20}
BONE_THICK  = 2
JOINT_R     = 4
TIP_R       = 6
TRAIL_LEN   = 45
TRAIL_COLOR = (160, 100, 255)


def draw_hand(canvas, landmarks, h, w):
    pts = [(int(x * w), int(y * h)) for x, y in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], BONE_COLOR, BONE_THICK, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        color  = TIP_COLOR if i in TIPS else JOINT_COLOR
        radius = TIP_R     if i in TIPS else JOINT_R
        cv2.circle(canvas, pt, radius, color, -1, cv2.LINE_AA)


def draw_trail(canvas, trail, h, w):
    valid = [(i, p) for i, p in enumerate(trail) if p is not None]
    n = len(valid)
    for rank, (_, (rx, ry)) in enumerate(valid):
        fade  = (rank + 1) / n
        color = tuple(int(c * fade) for c in TRAIL_COLOR)
        cv2.circle(canvas, (int(rx * w), int(ry * h)),
                   max(1, int(3 * fade)), color, -1, cv2.LINE_AA)


def download_model():
    """Download hand_landmarker.task if not present."""
    import urllib.request, os
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading hand_landmarker.task model (~8 MB)...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded.")
    return model_path


def process_video(input_path, csv_path="hand_keypoints.csv",
                  out_path="hand_motion.mp4", canvas_size=720,
                  skip_frames=1, max_hands=2, conf=0.6):

    model_path = download_model()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps      = max(1.0, src_fps / skip_frames)
    W = H        = canvas_size

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H)
    )

    # Build HandLandmarker (new Tasks API)
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options   = HandLandmarkerOptions(
        base_options                       = base_opts,
        running_mode                       = mp_vision.RunningMode.VIDEO,
        num_hands                          = max_hands,
        min_hand_detection_confidence      = conf,
        min_hand_presence_confidence       = 0.5,
        min_tracking_confidence            = 0.5,
    )

    trails   = [[] for _ in range(max_hands)]
    csv_rows = []
    csv_hdr  = (["frame", "hand"] +
                [f"x{i}" for i in range(21)] +
                [f"y{i}" for i in range(21)])

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        frame_idx = 0
        pbar = tqdm(total=total_frames, unit="frame", desc="Processing")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            # Convert to MediaPipe Image
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / src_fps) * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)
            canvas = np.full((H, W, 3), CANVAS_BG, dtype=np.uint8)

            if result.hand_landmarks:
                for slot, (hlm, hinfo) in enumerate(
                    zip(result.hand_landmarks, result.handedness)
                ):
                    landmarks  = [(lm.x, lm.y) for lm in hlm]
                    hand_label = hinfo[0].display_name  # "Left" / "Right"

                    row = [frame_idx, hand_label]
                    row += [lm.x for lm in hlm]
                    row += [lm.y for lm in hlm]
                    csv_rows.append(row)

                    # Wrist trail
                    trails[slot].append(landmarks[0])
                    if len(trails[slot]) > TRAIL_LEN:
                        trails[slot].pop(0)

                    draw_trail(canvas, trails[slot], H, W)
                    draw_hand(canvas, landmarks, H, W)

                    wx = int(landmarks[0][0] * W)
                    wy = max(12, int(landmarks[0][1] * H) - 12)
                    cv2.putText(canvas, hand_label, (wx, wy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (180, 180, 180), 1, cv2.LINE_AA)
            else:
                # Fade trails when no hands detected
                for slot in range(max_hands):
                    if trails[slot]:
                        trails[slot].append(None)
                        if len(trails[slot]) > TRAIL_LEN:
                            trails[slot].pop(0)
                        draw_trail(canvas, trails[slot], H, W)

            cv2.putText(canvas, f"frame {frame_idx:06d}",
                        (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (60, 60, 60), 1, cv2.LINE_AA)

            writer.write(canvas)
            frame_idx += 1

        pbar.close()

    cap.release()
    writer.release()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_hdr)
        w.writerows(csv_rows)

    print(f"\nDone.")
    print(f"  Keypoints CSV : {csv_path}  ({len(csv_rows):,} rows)")
    print(f"  Output video  : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--csv",    default="hand_keypoints.csv")
    parser.add_argument("--output", default="hand_motion.mp4")
    parser.add_argument("--size",   type=int,   default=720)
    parser.add_argument("--skip",   type=int,   default=1)
    parser.add_argument("--hands",  type=int,   default=2)
    parser.add_argument("--conf",   type=float, default=0.6)
    args = parser.parse_args()

    process_video(
        input_path  = args.input,
        csv_path    = args.csv,
        out_path    = args.output,
        canvas_size = args.size,
        skip_frames = args.skip,
        max_hands   = args.hands,
        conf        = args.conf,
    )
"""
3D Hand Tracking Visualizer — Pure OpenCV Renderer (v4 — Clean & Clear)
========================================================================
Fixes from v3:
  - Two hands plotted SIDE BY SIDE (left hand left half, right hand right half)
    so they never overlap into one messy blob
  - Trail dots are SMALL and FAINT (max radius 4px) — purely background history
  - Current skeleton always rendered LAST so it sits cleanly on top
  - Bone thickness and joint sizes tuned for clarity
  - Trail length reduced so only recent motion is visible
  - Each finger's 4 joints connected with per-finger colour bones (not grey)

Usage:
    python hand_tracking_3d.py --input factory.mp4

Install:
    pip install mediapipe opencv-python tqdm numpy
"""

import argparse
import csv
import math
import os
import urllib.request

import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions

# ── Colours (BGR) ────────────────────────────────────────────────────────────
FINGER_COLORS = {
    "thumb":  (0,   150, 255),   # orange
    "index":  (30,  200,  50),   # green
    "middle": (220,  80,  30),   # blue
    "ring":   (30,   30, 210),   # red
    "pinky":  (190,  40, 190),   # purple
    "palm":   (80,   80,  80),   # dark grey
}

# Lighter tint of each finger colour for bones
BONE_COLORS = {
    "thumb":  (100, 200, 255),
    "index":  (120, 230, 140),
    "middle": (240, 160, 120),
    "ring":   (120, 120, 230),
    "pinky":  (220, 130, 220),
    "palm":   (160, 160, 160),
}

LM_FINGER = {}
for _name, _idxs in [
    ("palm",   [0]),
    ("thumb",  [1, 2, 3, 4]),
    ("index",  [5, 6, 7, 8]),
    ("middle", [9, 10, 11, 12]),
    ("ring",   [13, 14, 15, 16]),
    ("pinky",  [17, 18, 19, 20]),
]:
    for _i in _idxs:
        LM_FINGER[_i] = _name

FINGERTIPS = [4, 8, 12, 16, 20]

# Connections grouped by finger for coloured bones
FINGER_CONNECTIONS = {
    "palm":   [(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)],
    "thumb":  [(1, 2), (2, 3), (3, 4)],
    "index":  [(5, 6), (6, 7), (7, 8)],
    "middle": [(9, 10), (10, 11), (11, 12)],
    "ring":   [(13, 14), (14, 15), (15, 16)],
    "pinky":  [(17, 18), (18, 19), (19, 20)],
}

ALL_CONNECTIONS = [(a, b) for conns in FINGER_CONNECTIONS.values() for a, b in conns]

BG_COLOR   = (245, 245, 245)
CUBE_COLOR = (175, 175, 175)
GRID_COLOR = (215, 215, 215)
TEXT_COLOR = (50,  50,  50)
TRAIL_LEN  = 30   # keep trails short and clean


# ═══════════════════════════════════════════════════════════════════════════════
#  Projection
# ═══════════════════════════════════════════════════════════════════════════════

def make_R(elev=22, azim=-50):
    el = math.radians(elev)
    az = math.radians(azim)
    Ry = np.array([[ math.cos(az), 0, math.sin(az)],
                   [ 0,            1, 0            ],
                   [-math.sin(az), 0, math.cos(az)]])
    Rx = np.array([[1, 0,             0            ],
                   [0,  math.cos(el), -math.sin(el)],
                   [0,  math.sin(el),  math.cos(el)]])
    return (Rx @ Ry).astype(np.float64)

R_GLOBAL = make_R()


def proj(pts, cx, cy, scale):
    pts = np.atleast_2d(pts).astype(np.float64)
    r   = (R_GLOBAL @ pts.T).T
    return np.stack([r[:, 0] * scale + cx,
                    -r[:, 1] * scale + cy], axis=1)


def zdepth(pt3):
    return float((R_GLOBAL @ np.asarray(pt3, dtype=np.float64))[2])


# ═══════════════════════════════════════════════════════════════════════════════
#  MediaPipe → our 3-D space  (flip Y so +Y = up)
# ═══════════════════════════════════════════════════════════════════════════════

def mp_to_world(lm_list):
    arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float64)
    arr[:, 1] *= -1
    return arr


def centre_and_scale(lms, cube_half, fill=0.58):
    """Centre on wrist, scale so hand fills `fill` × cube_half."""
    centred = lms - lms[0]
    extent  = np.abs(centred).max()
    if extent < 1e-9:
        extent = 1.0
    return centred * (cube_half * fill / extent)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cube + grid
# ═══════════════════════════════════════════════════════════════════════════════

_CORNERS = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                     [ 1,-1,-1],[ 1,-1,1],[ 1,1,-1],[ 1,1,1]], dtype=np.float64)

CUBE_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
              (4,5),(4,6),(5,7),(6,7)]

CUBE_FACES = [([0,2,3,1], np.array([-1., 0., 0.])),
              ([4,5,7,6], np.array([ 1., 0., 0.])),
              ([0,1,5,4], np.array([ 0.,-1., 0.])),
              ([2,6,7,3], np.array([ 0., 1., 0.])),
              ([0,4,6,2], np.array([ 0., 0.,-1.])),
              ([1,3,7,5], np.array([ 0., 0., 1.]))]


def _grid_lines(n=4):
    lines = []
    for i in range(1, n):
        t = -1 + i * 2 / n
        lines += [
            (np.array([t,-1,-1]), np.array([t,-1, 1])),
            (np.array([-1,-1,t]), np.array([ 1,-1, t])),
            (np.array([t,-1,-1]), np.array([t, 1,-1])),
            (np.array([-1, t,-1]), np.array([ 1, t,-1])),
            (np.array([-1,-1,t]), np.array([-1, 1, t])),
            (np.array([-1, t,-1]), np.array([-1, t, 1])),
        ]
    return lines

GRID_LINES = _grid_lines(4)


def draw_cube_grid(canvas, cx, cy, scale, cube_half):
    C  = _CORNERS * cube_half
    c2 = proj(C, cx, cy, scale)

    # Faces back → front
    for fi in np.argsort([zdepth(C[idxs].mean(0)) for idxs, _ in CUBE_FACES]):
        idxs, _ = CUBE_FACES[fi]
        cv2.fillPoly(canvas, [c2[idxs].astype(np.int32)], (248, 248, 248))

    # Grid lines
    for p1, p2 in GRID_LINES:
        a = proj(p1[None] * cube_half, cx, cy, scale)[0].astype(int)
        b = proj(p2[None] * cube_half, cx, cy, scale)[0].astype(int)
        cv2.line(canvas, tuple(a), tuple(b), GRID_COLOR, 1, cv2.LINE_AA)

    # Cube edges
    for a, b in CUBE_EDGES:
        cv2.line(canvas, tuple(c2[a].astype(int)),
                 tuple(c2[b].astype(int)), CUBE_COLOR, 1, cv2.LINE_AA)


def draw_axes(canvas, cx, cy, scale, cube_half):
    h  = cube_half * 0.85
    o2 = proj(np.zeros((1, 3)), cx, cy, scale)[0].astype(int)
    for label, end3, color in [
        ("X", np.array([[h, 0, 0]]), (30,  50, 200)),
        ("Y", np.array([[0, h, 0]]), (30, 170,  30)),
        ("Z", np.array([[0, 0, h]]), (30, 110, 210)),
    ]:
        e2 = proj(end3, cx, cy, scale)[0].astype(int)
        cv2.arrowedLine(canvas, tuple(o2), tuple(e2),
                        color, 2, cv2.LINE_AA, tipLength=0.18)
        d = e2 - o2
        n = np.linalg.norm(d)
        if n > 0:
            lp = e2 + (d / n * 16).astype(int)
            cv2.putText(canvas, label, tuple(lp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  Trail rendering  — small faint dots only, drawn BEFORE skeleton
# ═══════════════════════════════════════════════════════════════════════════════

def draw_trails(canvas, raw_trail, wrist_raw, hand_scale, cx, cy, scale):
    """
    Render past fingertip positions as small fading dots.
    All coordinates transformed relative to CURRENT wrist so they
    stay anchored to the current hand position.
    """
    for tip_idx in FINGERTIPS:
        trail = raw_trail.get(tip_idx, [])
        n = len(trail)
        if n < 2:
            continue
        tip_color = FINGER_COLORS[LM_FINGER[tip_idx]]

        for rank, raw_pt in enumerate(trail[:-1]):   # skip last = current pos
            fade  = rank / max(n - 2, 1)             # 0 (oldest) → 1 (newest)
            alpha = 0.08 + 0.35 * fade               # very faint: 0.08 → 0.43
            radius = max(1, int(1 + 2 * fade))        # 1px → 3px max

            rel = (np.asarray(raw_pt) - wrist_raw) * hand_scale
            pt2 = proj(rel[None], cx, cy, scale)[0]
            x, y = int(pt2[0]), int(pt2[1])
            r = radius

            if r < x < canvas.shape[1]-r and r < y < canvas.shape[0]-r:
                roi     = canvas[y-r:y+r+1, x-r:x+r+1].copy()
                overlay = roi.copy()
                cv2.circle(overlay, (r, r), r, tip_color, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
                canvas[y-r:y+r+1, x-r:x+r+1] = roi


# ═══════════════════════════════════════════════════════════════════════════════
#  Skeleton rendering  — coloured bones + joints, drawn AFTER trails
# ═══════════════════════════════════════════════════════════════════════════════

def draw_skeleton(canvas, lms_scaled, cx, cy, scale):
    """
    Draw the 21-point hand skeleton cleanly:
    - Bones coloured by finger
    - Knuckle joints: white filled circle with finger-colour border
    - Fingertips: larger with white ring
    - All depth-sorted back→front
    """
    lms_2d = proj(lms_scaled, cx, cy, scale)

    # ── Floor shadow (faint grey dots) ────────────────────────────────────
    floor_y = lms_scaled[:, 1].min() - 0.008
    s3 = lms_scaled.copy()
    s3[:, 1] = floor_y
    s2 = proj(s3, cx, cy, scale)
    for i in range(21):
        cv2.circle(canvas, tuple(s2[i].astype(int)), 2, (195,195,195), -1, cv2.LINE_AA)

    # ── Bones depth-sorted ────────────────────────────────────────────────
    bone_list = []
    for fname, conns in FINGER_CONNECTIONS.items():
        for a, b in conns:
            d = (zdepth(lms_scaled[a]) + zdepth(lms_scaled[b])) / 2
            bone_list.append((d, fname, a, b))
    bone_list.sort()   # back → front

    for _, fname, a, b in bone_list:
        pa = tuple(lms_2d[a].astype(int))
        pb = tuple(lms_2d[b].astype(int))
        bone_col = BONE_COLORS[fname]
        cv2.line(canvas, pa, pb, bone_col, 3, cv2.LINE_AA)

    # ── Joints depth-sorted back→front ───────────────────────────────────
    joint_order = sorted(range(21), key=lambda i: zdepth(lms_scaled[i]))

    for i in joint_order:
        color   = FINGER_COLORS[LM_FINGER[i]]
        pt      = tuple(lms_2d[i].astype(int))
        is_tip  = i in FINGERTIPS

        if is_tip:
            # Large fingertip: white outer ring → colour fill → dark outline
            cv2.circle(canvas, pt, 11, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt,  9, color,           -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 11, (130, 130, 130),  1, cv2.LINE_AA)
        elif i == 0:
            # Wrist: slightly larger neutral dot
            cv2.circle(canvas, pt, 8, (220, 220, 220), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 6, (80, 80, 80),    -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 8, (150, 150, 150),  1, cv2.LINE_AA)
        else:
            # Regular knuckle
            cv2.circle(canvas, pt, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 5, color,           -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 7, (170, 170, 170),  1, cv2.LINE_AA)

    # ── Landmark index labels (optional debug) ────────────────────────────
    # Uncomment to show landmark numbers:
    # for i in range(21):
    #     pt = tuple(lms_2d[i].astype(int))
    #     cv2.putText(canvas, str(i), (pt[0]+6, pt[1]-4),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.28, TEXT_COLOR, 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Two-hand layout: split the cube into left/right halves
#
#  When two hands are detected we project each hand into its own half-width
#  sub-canvas, then composite them side-by-side. This prevents the two hand
#  skeletons from ever overlapping.
# ═══════════════════════════════════════════════════════════════════════════════

def make_hand_canvas(SZ, lms_scaled, raw_trail, wrist_raw, hand_scale,
                     hand_label, cube_half):
    """
    Render ONE hand into a SZ×SZ canvas and return it.
    """
    canvas = np.full((SZ, SZ, 3), BG_COLOR, dtype=np.uint8)
    cx     = SZ // 2
    cy     = int(SZ * 0.46)
    scale  = SZ / (cube_half * 2.6)

    draw_cube_grid(canvas, cx, cy, scale, cube_half)
    draw_axes(canvas, cx, cy, scale, cube_half)

    # Trails first (background layer)
    draw_trails(canvas, raw_trail, wrist_raw, hand_scale, cx, cy, scale)

    # Skeleton on top
    draw_skeleton(canvas, lms_scaled, cx, cy, scale)

    # Hand label (Left / Right)
    cv2.putText(canvas, hand_label, (10, SZ - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100),
                1, cv2.LINE_AA)
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
#  2-D overlay on original frame
# ═══════════════════════════════════════════════════════════════════════════════

def draw_2d_overlay(frame, norm_lms_list):
    out = frame.copy()
    h, w = out.shape[:2]
    for lms in norm_lms_list:
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in lms]
        for fname, conns in FINGER_CONNECTIONS.items():
            bc = BONE_COLORS[fname]
            for a, b in conns:
                cv2.line(out, pts[a], pts[b], bc, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            col = FINGER_COLORS[LM_FINGER[i]]
            r   = 7 if i in FINGERTIPS else 4
            cv2.circle(out, pt, r+2, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(out, pt, r,   col,           -1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Legend
# ═══════════════════════════════════════════════════════════════════════════════

def draw_legend(canvas, x0, y0):
    for i, (name, color) in enumerate([
        ("Thumb",  FINGER_COLORS["thumb"]),
        ("Index",  FINGER_COLORS["index"]),
        ("Middle", FINGER_COLORS["middle"]),
        ("Ring",   FINGER_COLORS["ring"]),
        ("Pinky",  FINGER_COLORS["pinky"]),
    ]):
        y = y0 + i * 24
        cv2.circle(canvas, (x0+8, y), 7, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x0+8, y), 7, (150,150,150), 1, cv2.LINE_AA)
        cv2.putText(canvas, name, (x0+20, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_COLOR, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model download
# ═══════════════════════════════════════════════════════════════════════════════

def download_model():
    path = "hand_landmarker.task"
    if not os.path.exists(path):
        print("Downloading hand_landmarker.task (~8 MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            path)
        print("Done.")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main processing loop
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(input_path, csv_path="hand_keypoints_3d.csv",
                  out_path="hand_3d_scatter.mp4", canvas_size=720,
                  skip_frames=1, max_hands=2, conf=0.55):

    model_path = download_model()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = max(1.0, fps / skip_frames)
    SZ      = canvas_size
    CUBE_HALF = 0.09

    # Output layout:
    # If 1 hand:  [3D view (SZ) | original frame (SZ)]  → total SZ×2
    # If 2 hands: [hand0 3D (SZ) | hand1 3D (SZ) | original frame (SZ)] → total SZ×3
    # We always write SZ×3 wide and fill unused 3D panels with blank
    OUT_W = SZ * 3
    OUT_H = SZ

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (OUT_W, OUT_H))

    # Per-slot trail storage: raw Y-flipped world coords
    raw_trails = [{tip: [] for tip in FINGERTIPS} for _ in range(max_hands)]

    csv_rows = []
    csv_hdr  = (["frame", "hand"] +
                [f"x{i}" for i in range(21)] +
                [f"y{i}" for i in range(21)] +
                [f"z{i}" for i in range(21)])

    opts = HandLandmarkerOptions(
        base_options                   = mp_python.BaseOptions(model_asset_path=model_path),
        running_mode                   = mp_vision.RunningMode.VIDEO,
        num_hands                      = max_hands,
        min_hand_detection_confidence  = conf,
        min_hand_presence_confidence   = 0.5,
        min_tracking_confidence        = 0.5,
    )

    with mp_vision.HandLandmarker.create_from_options(opts) as detector:
        frame_idx = 0
        pbar = tqdm(total=total, unit="frame", desc="Rendering")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((frame_idx / fps) * 1000)
            result = detector.detect_for_video(mp_img, ts_ms)

            # Blank 3D canvases (one per slot)
            hand_canvases   = [np.full((SZ, SZ, 3), BG_COLOR, dtype=np.uint8)
                               for _ in range(max_hands)]
            norm_lms_2d     = []
            active_hands    = 0

            if result.hand_world_landmarks:
                for slot, (wlm, nlm, hinfo) in enumerate(
                    zip(result.hand_world_landmarks,
                        result.hand_landmarks,
                        result.handedness)):

                    if slot >= max_hands:
                        break

                    label     = hinfo[0].display_name
                    lms_w     = mp_to_world(wlm)          # Y-flipped
                    wrist_raw = lms_w[0].copy()
                    centred   = lms_w - wrist_raw

                    extent     = np.abs(centred).max()
                    hand_scale = (CUBE_HALF * 0.58 / extent) if extent > 1e-9 else 1.0
                    lms_scaled = centred * hand_scale

                    norm_lms_2d.append([(lm.x, lm.y) for lm in nlm])
                    active_hands += 1

                    # Update raw trail (store Y-flipped, NOT centred)
                    for tip in FINGERTIPS:
                        raw_trails[slot][tip].append(tuple(lms_w[tip]))
                        if len(raw_trails[slot][tip]) > TRAIL_LEN:
                            raw_trails[slot][tip].pop(0)

                    # Render this hand into its own canvas
                    hand_canvases[slot] = make_hand_canvas(
                        SZ, lms_scaled, raw_trails[slot],
                        wrist_raw, hand_scale, label, CUBE_HALF)

                    # CSV
                    row  = [frame_idx, label]
                    row += list(lms_w[:, 0])
                    row += list(lms_w[:, 1])
                    row += list(lms_w[:, 2])
                    csv_rows.append(row)

            else:
                for slot in range(max_hands):
                    for tip in FINGERTIPS:
                        if raw_trails[slot][tip]:
                            raw_trails[slot][tip].pop(0)
                    # Draw empty cube in blank slot
                    cx = SZ // 2; cy = int(SZ * 0.46)
                    sc = SZ / (CUBE_HALF * 2.6)
                    draw_cube_grid(hand_canvases[slot], cx, cy, sc, CUBE_HALF)
                    draw_axes(hand_canvases[slot], cx, cy, sc, CUBE_HALF)

            # Info bar on hand0 canvas
            cv2.putText(hand_canvases[0], f"hands: {active_hands}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, TEXT_COLOR, 1, cv2.LINE_AA)
            draw_legend(hand_canvases[0], 12, SZ - 148)
            cv2.putText(hand_canvases[0], f"f:{frame_idx:06d}",
                        (10, SZ - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (150, 150, 150), 1, cv2.LINE_AA)

            # Right panel: original frame + 2D overlay
            panel = draw_2d_overlay(cv2.resize(frame, (SZ, SZ)), norm_lms_2d)

            # Dividers
            div = np.full((SZ, 2, 3), 190, dtype=np.uint8)

            combined = np.hstack([hand_canvases[0], div,
                                   hand_canvases[1], div,
                                   panel])[:, :OUT_W]
            writer.write(combined)
            frame_idx += 1

        pbar.close()

    cap.release()
    writer.release()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_hdr)
        w.writerows(csv_rows)

    print(f"\nDone.  CSV: {csv_path} ({len(csv_rows):,} rows)  Video: {out_path}")
    print(f"       Frame size: {OUT_W}x{OUT_H}  "
          f"[Hand0 | Hand1 | Camera]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--csv",    default="hand_keypoints_3d.csv")
    p.add_argument("--output", default="hand_3d_scatter.mp4")
    p.add_argument("--size",   type=int,   default=720,
                   help="Height and per-panel width in px (default 720)")
    p.add_argument("--skip",   type=int,   default=1)
    p.add_argument("--hands",  type=int,   default=2)
    p.add_argument("--conf",   type=float, default=0.55)
    a = p.parse_args()
    process_video(a.input, a.csv, a.output, a.size, a.skip, a.hands, a.conf)
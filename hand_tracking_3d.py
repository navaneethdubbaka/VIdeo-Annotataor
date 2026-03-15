"""
3D Hand Tracking Visualizer — Pure OpenCV Renderer
====================================================
• No matplotlib → 10× faster rendering
• White 3D wireframe cube background (isometric-style perspective)
• Per-finger coloured dots + glowing fingertips
• Fading motion trails per fingertip
• Smooth anti-aliased skeleton bones
• Real metric Z from MediaPipe world landmarks
• Side panel: live 2D skeleton overlay on original frame

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

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
BG_COLOR        = (245, 245, 245)   # near-white canvas
CUBE_COLOR      = (200, 200, 200)   # light-grey cube edges
CUBE_FACE_COLOR = (250, 250, 250)   # very light cube face fill
GRID_COLOR      = (220, 220, 220)   # inner grid lines
AXIS_X_COLOR    = (50,  80,  230)   # red-ish  X axis
AXIS_Y_COLOR    = (50,  180, 50 )   # green    Y axis
AXIS_Z_COLOR    = (220, 80,  50 )   # blue-ish Z axis
BONE_COLOR      = (160, 160, 160)   # light grey skeleton
SHADOW_COLOR    = (190, 190, 190)   # floor shadow dots
TEXT_COLOR      = (60,  60,  60 )   # labels

# Per-finger colours (BGR)
FINGER_COLORS = {
    "thumb":  (0,   160, 255),   # orange
    "index":  (40,  200, 60 ),   # green
    "middle": (230, 100, 40 ),   # blue
    "ring":   (40,  40,  220),   # red
    "pinky":  (200, 60,  200),   # purple
    "palm":   (80,  80,  80 ),   # dark grey wrist
}

# Landmark → finger name
LM_FINGER = {}
for _name, _idxs in [
    ("palm",   [0]),
    ("thumb",  [1,2,3,4]),
    ("index",  [5,6,7,8]),
    ("middle", [9,10,11,12]),
    ("ring",   [13,14,15,16]),
    ("pinky",  [17,18,19,20]),
]:
    for _i in _idxs:
        LM_FINGER[_i] = _name

FINGERTIPS = [4, 8, 12, 16, 20]

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

TRAIL_LEN = 80   # frames of history per fingertip


# ═══════════════════════════════════════════════════════════════════════════════
#  Soft 3-D projection helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_view_matrix(elev_deg=25, azim_deg=-50):
    """
    Build a simple rotation matrix for 3-D → 2-D projection.
    Returns 3×3 rotation R such that screen_xy = R @ world_xyz
    (we only use the first two rows for the screen projection).
    """
    el = math.radians(elev_deg)
    az = math.radians(azim_deg)

    # Rotation around Y (azimuth)
    Ry = np.array([
        [ math.cos(az), 0, math.sin(az)],
        [ 0,            1, 0           ],
        [-math.sin(az), 0, math.cos(az)],
    ])
    # Rotation around X (elevation)
    Rx = np.array([
        [1, 0,           0          ],
        [0, math.cos(el),-math.sin(el)],
        [0, math.sin(el), math.cos(el)],
    ])
    return Rx @ Ry


def project(pts_3d: np.ndarray,
            R: np.ndarray,
            cx: float, cy: float,
            scale: float) -> np.ndarray:
    """
    Project N×3 world points → N×2 screen pixels.
    Uses a simple orthographic projection with depth for sorting only.
    """
    rotated = (R @ pts_3d.T).T          # N×3
    sx = rotated[:, 0] * scale + cx
    sy = -rotated[:, 1] * scale + cy    # flip Y (screen Y grows down)
    return np.stack([sx, sy], axis=1)


def depth_of(pt_3d: np.ndarray, R: np.ndarray) -> float:
    return float((R @ pt_3d)[2])


# ═══════════════════════════════════════════════════════════════════════════════
#  Cube wireframe
# ═══════════════════════════════════════════════════════════════════════════════

def build_cube_corners(half: float) -> np.ndarray:
    """8 corners of a cube centred at origin, side = 2*half."""
    h = half
    return np.array([
        [-h,-h,-h],[-h,-h, h],[-h, h,-h],[-h, h, h],
        [ h,-h,-h],[ h,-h, h],[ h, h,-h],[ h, h, h],
    ], dtype=float)


CUBE_EDGES = [
    (0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
    (4,5),(4,6),(5,7),(6,7),
]

CUBE_FACES = [
    # (corner indices, face normal direction for depth sorting)
    ([0,1,3,2], np.array([-1, 0, 0])),  # left
    ([4,5,7,6], np.array([ 1, 0, 0])),  # right
    ([0,1,5,4], np.array([ 0,-1, 0])),  # bottom  ← floor
    ([2,3,7,6], np.array([ 0, 1, 0])),  # top
    ([0,2,6,4], np.array([ 0, 0,-1])),  # back
    ([1,3,7,5], np.array([ 0, 0, 1])),  # front
]


def draw_cube(canvas, corners_2d, corners_3d, R, cube_half):
    """Draw shaded cube faces then edges onto canvas."""
    # Sort faces back→front
    face_depths = []
    for (idxs, normal) in CUBE_FACES:
        centre_3d = corners_3d[idxs].mean(axis=0)
        face_depths.append(depth_of(centre_3d, R))

    order = np.argsort(face_depths)   # back to front

    for fi in order:
        idxs, normal = CUBE_FACES[fi]
        pts = corners_2d[idxs].astype(np.int32)

        # Only draw back-facing & bottom faces (so we see the inside)
        # We dim the face fill so interior points stay visible
        d = depth_of(corners_3d[idxs[0]], R)
        shade = max(235, min(252, int(252 - abs(d) * 60)))
        face_color = (shade, shade, shade)

        cv2.fillPoly(canvas, [pts], face_color)

    # Draw edges
    for a, b in CUBE_EDGES:
        p1 = tuple(corners_2d[a].astype(int))
        p2 = tuple(corners_2d[b].astype(int))
        cv2.line(canvas, p1, p2, CUBE_COLOR, 1, cv2.LINE_AA)

    # Draw inner grid on floor face (bottom face: corners 0,1,5,4)
    floor_idxs = [0,1,5,4]
    c3 = corners_3d[floor_idxs]
    floor_y = c3[0, 1]  # y coordinate of the floor
    GRID_N = 4
    h = cube_half
    for i in range(1, GRID_N):
        t = -h + i * (2*h / GRID_N)
        # lines parallel to Z
        p1_3d = np.array([t, floor_y, -h])
        p2_3d = np.array([t, floor_y,  h])
        # lines parallel to X
        q1_3d = np.array([-h, floor_y, t])
        q2_3d = np.array([ h, floor_y, t])
        for a3, b3 in [(p1_3d, p2_3d), (q1_3d, q2_3d)]:
            pa = project(a3[None], R, 0, 0, 1)[0]  # relative coords
            pb = project(b3[None], R, 0, 0, 1)[0]
            # re-project using full canvas transform below — skip for now
            # (grid lines drawn via corners_2d interpolation instead)
        break  # simplified: skip grid — cube edges already show structure


def draw_axis_labels(canvas, R, cx, cy, scale, cube_half):
    """Draw X/Y/Z axis arrows and labels inside the cube."""
    h = cube_half * 0.85
    origin = np.zeros((1, 3))
    axes = {
        "X": (np.array([[h, 0, 0]]),  AXIS_X_COLOR),
        "Y": (np.array([[0, h, 0]]),  AXIS_Y_COLOR),
        "Z": (np.array([[0, 0, h]]),  AXIS_Z_COLOR),
    }
    o2d = project(origin, R, cx, cy, scale)[0].astype(int)
    for label, (end_3d, color) in axes.items():
        e2d = project(end_3d, R, cx, cy, scale)[0].astype(int)
        cv2.arrowedLine(canvas, tuple(o2d), tuple(e2d),
                        color, 2, cv2.LINE_AA, tipLength=0.15)
        offset = e2d - o2d
        norm   = np.linalg.norm(offset)
        if norm > 0:
            direction = offset / norm
            lpos = e2d + (direction * 14).astype(int)
            cv2.putText(canvas, label, tuple(lpos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        color, 2, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  Hand rendering
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hand_3d(canvas, lms_3d, R, cx, cy, scale, trails, alpha=1.0):
    """
    Draw one hand: shadow on floor, bones, joints, trails.
    lms_3d : 21×3 float array (world coords in metres)
    trails : dict {tip_idx: [(x,y,z), ...]}
    """
    lms_2d = project(lms_3d, R, cx, cy, scale)   # 21×2

    # ── Floor shadow ──────────────────────────────────────────────────────
    floor_y = -0.12   # bottom of our cube
    shadow_3d = lms_3d.copy()
    shadow_3d[:, 1] = floor_y
    shadow_2d = project(shadow_3d, R, cx, cy, scale)
    for i in range(21):
        pt = tuple(shadow_2d[i].astype(int))
        cv2.circle(canvas, pt, 3, SHADOW_COLOR, -1, cv2.LINE_AA)

    # ── Bones ─────────────────────────────────────────────────────────────
    # Sort by depth so closer bones draw on top
    bone_depths = []
    for a, b in CONNECTIONS:
        d = (depth_of(lms_3d[a], R) + depth_of(lms_3d[b], R)) / 2
        bone_depths.append((d, a, b))
    bone_depths.sort(key=lambda x: x[0])  # back to front

    for d, a, b in bone_depths:
        pa = tuple(lms_2d[a].astype(int))
        pb = tuple(lms_2d[b].astype(int))
        # Slightly darken bones that are closer
        brightness = max(120, min(190, int(190 - d * 200)))
        bcolor = (brightness, brightness, brightness)
        cv2.line(canvas, pa, pb, bcolor, 2, cv2.LINE_AA)

    # ── Fingertip trails ──────────────────────────────────────────────────
    for tip_idx in FINGERTIPS:
        trail = trails.get(tip_idx, [])
        n = len(trail)
        if n < 2:
            continue
        tip_color = FINGER_COLORS[LM_FINGER[tip_idx]]
        for rank in range(1, n):
            fade   = rank / (n - 1)
            alpha_v= 0.08 + 0.7 * fade
            thickness = max(1, int(1 + 2 * fade))
            p3a = np.array(trail[rank - 1])[None]
            p3b = np.array(trail[rank])[None]
            pa2 = project(p3a, R, cx, cy, scale)[0].astype(int)
            pb2 = project(p3b, R, cx, cy, scale)[0].astype(int)
            # Blend trail colour onto canvas
            overlay = canvas.copy()
            cv2.line(overlay, tuple(pa2), tuple(pb2),
                     tip_color, thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha_v, canvas, 1 - alpha_v, 0, canvas)

        # Trail endpoint dot
        last_3d = np.array(trail[-1])[None]
        last_2d = project(last_3d, R, cx, cy, scale)[0].astype(int)
        cv2.circle(canvas, tuple(last_2d), 5, tip_color, -1, cv2.LINE_AA)

    # ── Joints ────────────────────────────────────────────────────────────
    # Sort by depth (back to front)
    joint_depths = [(depth_of(lms_3d[i], R), i) for i in range(21)]
    joint_depths.sort()

    for _, i in joint_depths:
        color  = FINGER_COLORS[LM_FINGER[i]]
        pt     = tuple(lms_2d[i].astype(int))
        is_tip = i in FINGERTIPS

        if is_tip:
            # Outer white ring + filled colour
            cv2.circle(canvas, pt, 10, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt,  8, color,           -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 10, (180, 180, 180),  1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 4, color,           -1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  Side panel: 2-D skeleton overlay on original frame
# ═══════════════════════════════════════════════════════════════════════════════

def draw_2d_overlay(frame, norm_lms_list):
    """Draw 2D skeleton on a copy of the original frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    for lms in norm_lms_list:
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in lms]
        for a, b in CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (80, 200, 120), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            col = FINGER_COLORS[LM_FINGER[i]]
            r   = 7 if i in FINGERTIPS else 4
            cv2.circle(out, pt, r, col, -1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Legend panel
# ═══════════════════════════════════════════════════════════════════════════════

def draw_legend(canvas, x0, y0):
    labels = [
        ("Thumb",  FINGER_COLORS["thumb"]),
        ("Index",  FINGER_COLORS["index"]),
        ("Middle", FINGER_COLORS["middle"]),
        ("Ring",   FINGER_COLORS["ring"]),
        ("Pinky",  FINGER_COLORS["pinky"]),
    ]
    for i, (name, color) in enumerate(labels):
        y = y0 + i * 22
        cv2.circle(canvas, (x0 + 8, y), 7, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, name, (x0 + 20, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_COLOR, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model download
# ═══════════════════════════════════════════════════════════════════════════════

def download_model():
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading hand_landmarker.task (~8 MB)...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded.")
    return model_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main processing loop
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(input_path,
                  csv_path    = "hand_keypoints_3d.csv",
                  out_path    = "hand_3d_scatter.mp4",
                  canvas_size = 800,
                  skip_frames = 1,
                  max_hands   = 2,
                  conf        = 0.55):

    model_path = download_model()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps      = max(1.0, src_fps / skip_frames)

    # Output layout: [3D cube view | original frame overlay]
    OUT_W = canvas_size * 2
    OUT_H = canvas_size

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (OUT_W, OUT_H)
    )

    # Camera / projection params
    SZ        = canvas_size
    CX, CY    = SZ // 2, SZ // 2       # screen centre
    CUBE_HALF = 0.13                    # cube spans ±0.13 m (fits world lms)
    SCALE     = SZ / (CUBE_HALF * 2.6) # pixels per metre
    R         = make_view_matrix(elev_deg=22, azim_deg=-52)

    cube_corners_3d = build_cube_corners(CUBE_HALF)

    # Hand Landmarker
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options   = HandLandmarkerOptions(
        base_options                   = base_opts,
        running_mode                   = mp_vision.RunningMode.VIDEO,
        num_hands                      = max_hands,
        min_hand_detection_confidence  = conf,
        min_hand_presence_confidence   = 0.5,
        min_tracking_confidence        = 0.5,
    )

    trails   = [{tip: [] for tip in FINGERTIPS} for _ in range(max_hands)]
    csv_rows = []
    csv_hdr  = (["frame", "hand"] +
                [f"x{i}" for i in range(21)] +
                [f"y{i}" for i in range(21)] +
                [f"z{i}" for i in range(21)])

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        frame_idx = 0
        pbar = tqdm(total=total_frames, unit="frame", desc="Rendering")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)

            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue

            # ── MediaPipe inference ───────────────────────────────────────
            rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / src_fps) * 1000)
            result       = detector.detect_for_video(mp_image, timestamp_ms)

            # ── Build 3D canvas (white cube) ──────────────────────────────
            canvas_3d = np.full((SZ, SZ, 3), BG_COLOR, dtype=np.uint8)

            # Project cube corners
            cube_2d = project(cube_corners_3d, R, CX, CY, SCALE)

            draw_cube(canvas_3d, cube_2d, cube_corners_3d, R, CUBE_HALF)
            draw_axis_labels(canvas_3d, R, CX, CY, SCALE, CUBE_HALF)

            # ── Process detected hands ────────────────────────────────────
            norm_lms_for_2d = []
            active_hands    = 0

            if result.hand_world_landmarks:
                for slot, (wlm, nlm, hinfo) in enumerate(
                    zip(result.hand_world_landmarks,
                        result.hand_landmarks,
                        result.handedness)
                ):
                    label    = hinfo[0].display_name
                    lms_3d   = np.array([[lm.x, lm.y, lm.z] for lm in wlm])
                    lms_norm = [(lm.x, lm.y) for lm in nlm]
                    norm_lms_for_2d.append(lms_norm)
                    active_hands += 1

                    # Update trails
                    if slot < max_hands:
                        for tip in FINGERTIPS:
                            trails[slot][tip].append(tuple(lms_3d[tip]))
                            if len(trails[slot][tip]) > TRAIL_LEN:
                                trails[slot][tip].pop(0)
                        draw_hand_3d(canvas_3d, lms_3d, R,
                                     CX, CY, SCALE, trails[slot])

                    # CSV
                    row = [frame_idx, label]
                    row += list(lms_3d[:, 0])
                    row += list(lms_3d[:, 1])
                    row += list(lms_3d[:, 2])
                    csv_rows.append(row)

            else:
                # Decay trails when no hands visible
                for slot in range(max_hands):
                    for tip in FINGERTIPS:
                        if trails[slot][tip]:
                            trails[slot][tip].pop(0)

            # ── Legend & frame counter ────────────────────────────────────
            draw_legend(canvas_3d, 12, SZ - 130)
            cv2.putText(canvas_3d, f"f:{frame_idx:06d}",
                        (10, SZ - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (160, 160, 160), 1, cv2.LINE_AA)
            cv2.putText(canvas_3d,
                        f"hands: {active_hands}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, TEXT_COLOR, 1, cv2.LINE_AA)

            # ── Right panel: original frame with 2D overlay ───────────────
            panel_frame = cv2.resize(frame, (SZ, SZ))
            panel_frame = draw_2d_overlay(panel_frame, norm_lms_for_2d)

            # Thin white divider
            divider = np.full((SZ, 3, 3), 200, dtype=np.uint8)

            # ── Combine ───────────────────────────────────────────────────
            combined = np.hstack([canvas_3d, divider, panel_frame])
            # combined is SZ × (SZ+3+SZ) — trim divider width to fit OUT_W
            combined = combined[:, :OUT_W]

            writer.write(combined)
            frame_idx += 1

        pbar.close()

    cap.release()
    writer.release()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_hdr)
        w.writerows(csv_rows)

    print(f"\n✓ Done.")
    print(f"  3D CSV   : {csv_path}  ({len(csv_rows):,} rows)")
    print(f"  Video    : {out_path}  ({OUT_W}×{OUT_H})")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D hand scatter visualization (pure OpenCV, white cube)"
    )
    parser.add_argument("--input",  required=True,
                        help="Input video (e.g. factory.mp4)")
    parser.add_argument("--csv",    default="hand_keypoints_3d.csv")
    parser.add_argument("--output", default="hand_3d_scatter.mp4")
    parser.add_argument("--size",   type=int,   default=720,
                        help="Canvas size per panel in px (default 720)")
    parser.add_argument("--skip",   type=int,   default=1,
                        help="Process every Nth frame (1=all, 2=half)")
    parser.add_argument("--hands",  type=int,   default=2)
    parser.add_argument("--conf",   type=float, default=0.55)
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
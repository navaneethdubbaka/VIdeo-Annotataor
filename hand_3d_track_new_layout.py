"""
3D Hand Tracking Visualizer — Reference-Matched Layout
=======================================================
Matches the exact visual style from the reference images:

LAYOUT
  ┌─────────────────────────────────┬──────────────┐
  │                                 │  Hand 0 3D   │
  │   Camera frame (large, left)    │  (dashed cube)│
  │   + task label top-left         ├──────────────┤
  │   + metadata bar bottom         │  Hand 1 3D   │
  └─────────────────────────────────┴──────────────┘

3D PLOT STYLE (matches reference exactly)
  - White background
  - Dashed grey grid lines (not solid cube wireframe)
  - Numeric axis tick labels (0, 0.2, 0.4, 0.6)
  - Thin axis lines with arrow tips coloured (blue X, teal Y, red Z)
  - Small solid dots per landmark coloured by finger
  - Thin GREY connecting lines (not coloured bones)
  - Blue dot for wrist (landmark 0)
  - Top-down-ish viewing angle (elev=30, azim=-60)
  - Only ONE hand per plot panel

Usage:
    python hand_tracking_3d.py --input factory.mp4 [--task "Grind metal piece"]

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

# ── Finger colours  (BGR, matching reference: orange/green/red/teal/purple) ──
FINGER_COLORS = {
    "thumb":  (0,   140, 255),   # orange
    "index":  (30,  180,  30),   # green
    "middle": (200,  60,  20),   # blue-ish
    "ring":   (20,   20, 200),   # red
    "pinky":  (170,  30, 170),   # purple
    "palm":   (180,  80,  20),   # blue (wrist)
}
WRIST_COLOR = (200, 120,  20)    # blue dot for landmark 0 (matches reference)

LM_FINGER = {}
for _n, _ids in [("palm",[0]),("thumb",[1,2,3,4]),("index",[5,6,7,8]),
                 ("middle",[9,10,11,12]),("ring",[13,14,15,16]),("pinky",[17,18,19,20])]:
    for _i in _ids: LM_FINGER[_i] = _n

FINGERTIPS = [4, 8, 12, 16, 20]

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── Layout constants ──────────────────────────────────────────────────────────
CARD_BG     = (255, 255, 255)
PLOT_BG     = (255, 255, 255)
META_BG     = (248, 248, 248)
BORDER_COL  = (210, 210, 210)
TEXT_DARK   = (40,  40,  40)
TEXT_GREY   = (120, 120, 120)
GRID_COL    = (195, 195, 195)
BONE_COL    = (155, 155, 155)   # thin grey bones matching reference
AXIS_X_COL  = (30,  40, 200)    # blue X
AXIS_Y_COL  = (30, 160,  30)    # green Y
AXIS_Z_COL  = (20, 100, 200)    # teal-blue Z
SHADOW_COL  = (210, 210, 210)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD   = cv2.FONT_HERSHEY_DUPLEX

TRAIL_LEN   = 25    # short, subtle trails


# ═══════════════════════════════════════════════════════════════════════════════
#  Projection  (orthographic)
#  Reference images show a more top-down angle: elev≈30, azim≈-60
# ═══════════════════════════════════════════════════════════════════════════════

def make_R(elev=30, azim=-60):
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
    return np.stack([r[:,0]*scale + cx, -r[:,1]*scale + cy], axis=1)


def zdepth(pt3):
    return float((R_GLOBAL @ np.asarray(pt3, np.float64))[2])


# ═══════════════════════════════════════════════════════════════════════════════
#  MediaPipe → right-handed 3-D (flip Y so +Y=up)
# ═══════════════════════════════════════════════════════════════════════════════

def mp_to_world(lm_list):
    a = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float64)
    a[:, 1] *= -1
    return a


def centre_scale(lms, half, fill=0.55):
    c = lms - lms[0]
    e = np.abs(c).max()
    return c * (half * fill / e) if e > 1e-9 else c


# ═══════════════════════════════════════════════════════════════════════════════
#  3-D plot panel  — matches reference style exactly
#
#  Reference observations:
#    • Dashed grid lines on three back walls (floor, back, left)
#    • Numeric labels on axes: 0, 0.2, 0.4, 0.6
#    • Thin coloured axis lines with arrowheads
#    • Solid small dots (r≈4-6) per landmark
#    • Thin grey connecting lines between joints
#    • No cube wireframe edges — only the dashed grid
# ═══════════════════════════════════════════════════════════════════════════════

def draw_3d_plot(W, H, lms_scaled, raw_trail, wrist_raw, hand_scale, cube_half):
    """
    Render one hand's 3D plot into a W×H canvas (pure white bg, dashed grid).
    Returns the canvas as a BGR numpy array.
    """
    canvas = np.full((H, W, 3), PLOT_BG, dtype=np.uint8)

    # Projection centre — offset up-left slightly (matching reference framing)
    pad_l = int(W * 0.18)   # left margin for Z axis labels
    pad_b = int(H * 0.18)   # bottom margin for X axis labels
    CX    = pad_l + (W - pad_l) // 2
    CY    = (H - pad_b) // 2
    SCALE = min(W - pad_l, H - pad_b) / (cube_half * 2.8)

    # ── Dashed grid walls ─────────────────────────────────────────────────
    N     = 4
    ticks = [i / N for i in range(N+1)]   # 0, 0.25, 0.5, 0.75, 1.0 of cube
    h     = cube_half

    def grid_line(p1, p2, dashed=True):
        a = proj(p1[None], CX, CY, SCALE)[0].astype(int)
        b = proj(p2[None], CX, CY, SCALE)[0].astype(int)
        if dashed:
            # Draw dashed line manually
            total = np.linalg.norm(b - a)
            if total < 1:
                return
            steps = max(2, int(total / 6))
            for s in range(steps):
                if s % 2 == 0:
                    t0 = s / steps
                    t1 = min(1.0, (s+1) / steps)
                    pa = (a + (b-a)*t0).astype(int)
                    pb = (a + (b-a)*t1).astype(int)
                    cv2.line(canvas, tuple(pa), tuple(pb), GRID_COL, 1, cv2.LINE_AA)
        else:
            cv2.line(canvas, tuple(a), tuple(b), GRID_COL, 1, cv2.LINE_AA)

    # Floor (y=-h): grid lines in X and Z
    for t in ticks:
        x = -h + t*2*h
        grid_line(np.array([x, -h, -h]), np.array([x, -h,  h]))
        grid_line(np.array([-h, -h, x]), np.array([ h, -h, x]))

    # Back wall (z=-h): grid lines in X and Y
    for t in ticks:
        x = -h + t*2*h
        grid_line(np.array([x, -h, -h]), np.array([x,  h, -h]))
        grid_line(np.array([-h, x, -h]), np.array([ h,  x, -h]))

    # Left wall (x=-h): grid lines in Y and Z
    for t in ticks:
        x = -h + t*2*h
        grid_line(np.array([-h, x, -h]), np.array([-h,  x,  h]))
        grid_line(np.array([-h, -h, x]), np.array([-h,  h,  x]))

    # ── Axis lines with tick labels ───────────────────────────────────────
    origin3 = np.zeros((1, 3))
    o2      = proj(origin3, CX, CY, SCALE)[0].astype(int)

    axis_len = h * 1.15   # extend slightly past cube edge

    def draw_axis(end3, color, label, tick_vals):
        e3  = np.array(end3)[None] * axis_len / h
        e2  = proj(e3, CX, CY, SCALE)[0].astype(int)
        cv2.arrowedLine(canvas, tuple(o2), tuple(e2),
                        color, 1, cv2.LINE_AA, tipLength=0.12)
        # Label at tip
        d = e2 - o2
        n = np.linalg.norm(d)
        if n > 0:
            lp = e2 + (d/n*14).astype(int)
            cv2.putText(canvas, label, tuple(lp),
                        FONT, 0.38, color, 1, cv2.LINE_AA)
        # Tick marks + numeric labels along axis
        for tv in tick_vals:
            if abs(tv) < 1e-9:
                continue
            frac = tv / (h * 2)   # 0→1 along the half-axis direction
            tp3  = np.array(end3)[None] * frac
            tp2  = proj(tp3, CX, CY, SCALE)[0].astype(int)
            # Small tick perpendicular mark
            perp = np.array([-d[1], d[0]], dtype=float)
            pn   = np.linalg.norm(perp)
            if pn > 0:
                perp = (perp/pn * 3).astype(int)
            cv2.line(canvas,
                     tuple(tp2 - perp), tuple(tp2 + perp),
                     color, 1, cv2.LINE_AA)
            # Numeric label
            lbl = f"{tv:.1f}"
            lsz = cv2.getTextSize(lbl, FONT, 0.32, 1)[0]
            offset = (perp * 3 + np.array([-lsz[0]//2, 4])).astype(int)
            cv2.putText(canvas, lbl,
                        tuple(tp2 + np.array([-lsz[0]//2, 12])),
                        FONT, 0.30, TEXT_GREY, 1, cv2.LINE_AA)

    tv = [0.2, 0.4, 0.6]
    draw_axis([ h, 0, 0], AXIS_X_COL, "X", tv)
    draw_axis([ 0, h, 0], AXIS_Y_COL, "Y", tv)
    draw_axis([ 0, 0, h], AXIS_Z_COL, "Z", tv)

    # ── Trail dots (very subtle, small) ───────────────────────────────────
    for tip_idx in FINGERTIPS:
        trail = raw_trail.get(tip_idx, [])
        n     = len(trail)
        if n < 2:
            continue
        tip_col = FINGER_COLORS[LM_FINGER[tip_idx]]
        for rank, raw_pt in enumerate(trail[:-1]):
            fade   = rank / max(n-2, 1)
            alpha  = 0.06 + 0.30 * fade
            radius = 2
            rel    = (np.asarray(raw_pt) - wrist_raw) * hand_scale
            pt2    = proj(rel[None], CX, CY, SCALE)[0]
            x, y   = int(pt2[0]), int(pt2[1])
            r = radius
            if r < x < canvas.shape[1]-r and r < y < canvas.shape[0]-r:
                roi = canvas[y-r:y+r+1, x-r:x+r+1].copy()
                ov  = roi.copy()
                cv2.circle(ov, (r,r), r, tip_col, -1, cv2.LINE_AA)
                cv2.addWeighted(ov, alpha, roi, 1-alpha, 0, roi)
                canvas[y-r:y+r+1, x-r:x+r+1] = roi

    if lms_scaled is None:
        return canvas

    lms_2d = proj(lms_scaled, CX, CY, SCALE)

    # ── Thin grey connecting lines (bones) ────────────────────────────────
    # Sort back→front so nearer bones draw over farther ones
    bone_list = sorted(
        [(( zdepth(lms_scaled[a]) + zdepth(lms_scaled[b])) / 2, a, b)
         for a, b in CONNECTIONS]
    )
    for _, a, b in bone_list:
        pa = tuple(lms_2d[a].astype(int))
        pb = tuple(lms_2d[b].astype(int))
        cv2.line(canvas, pa, pb, BONE_COL, 1, cv2.LINE_AA)

    # ── Landmark dots (depth-sorted, back→front) ──────────────────────────
    for _, i in sorted((zdepth(lms_scaled[i]), i) for i in range(21)):
        color  = FINGER_COLORS[LM_FINGER[i]]
        pt     = tuple(lms_2d[i].astype(int))
        is_tip = i in FINGERTIPS

        if i == 0:
            # Wrist: solid blue dot (reference shows blue/teal)
            cv2.circle(canvas, pt, 7, WRIST_COLOR, -1, cv2.LINE_AA)
        elif is_tip:
            # Fingertip: solid coloured, slightly larger
            cv2.circle(canvas, pt, 6, color, -1, cv2.LINE_AA)
        else:
            # Knuckle: solid, smaller
            cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
#  Camera panel with 2-D skeleton overlay
# ═══════════════════════════════════════════════════════════════════════════════

def draw_camera_panel(frame, norm_lms_list, task_text, environment,
                      scene, op_height, panel_w, panel_h):
    """
    Build the left camera panel exactly like reference:
      - Resized frame fills the panel
      - White rounded-corner card feel (achieved by borders)
      - Task label box top-left (white bg, bold text)
      - Metadata bar at bottom
      - 2D skeleton with coloured dots + grey bones
    """
    # Resize frame to panel dimensions (leave room for metadata bar)
    META_H = 48
    img_h  = panel_h - META_H
    panel  = np.full((panel_h, panel_w, 3), CARD_BG, dtype=np.uint8)

    # Fit frame into image area preserving aspect ratio
    fh, fw = frame.shape[:2]
    scale  = min(panel_w / fw, img_h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh))

    # Centre frame in image area
    x0 = (panel_w - nw) // 2
    y0 = (img_h  - nh) // 2
    panel[y0:y0+nh, x0:x0+nw] = resized

    # ── 2D skeleton overlay on the frame region ───────────────────────────
    for lms in norm_lms_list:
        pts = [(x0 + int(lm[0]*nw), y0 + int(lm[1]*nh)) for lm in lms]
        for a, b in CONNECTIONS:
            cv2.line(panel, pts[a], pts[b], (150,150,150), 1, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            col = FINGER_COLORS[LM_FINGER[i]]
            r   = 5 if i in FINGERTIPS else 3
            cv2.circle(panel, pt, r, col, -1, cv2.LINE_AA)

    # ── Task label box (top-left of frame area, matching reference) ───────
    if task_text:
        box_w = min(panel_w - 20, 300)
        box_h = 52
        bx, by = x0 + 10, y0 + 10
        # White semi-transparent bg
        overlay = panel.copy()
        cv2.rectangle(overlay, (bx, by), (bx+box_w, by+box_h),
                      (255,255,255), -1)
        cv2.addWeighted(overlay, 0.88, panel, 0.12, 0, panel)
        cv2.rectangle(panel, (bx, by), (bx+box_w, by+box_h),
                      BORDER_COL, 1)
        # "Skill:" label in grey
        cv2.putText(panel, "Skill:", (bx+8, by+16),
                    FONT, 0.38, TEXT_GREY, 1, cv2.LINE_AA)
        # Task name in bold dark
        cv2.putText(panel, task_text, (bx+50, by+16),
                    FONT_BOLD, 0.42, TEXT_DARK, 1, cv2.LINE_AA)
        # Subtitle (truncate if needed)
        sub = task_text[:48] + ("..." if len(task_text) > 48 else "")
        cv2.putText(panel, sub, (bx+8, by+34),
                    FONT, 0.32, TEXT_GREY, 1, cv2.LINE_AA)

    # ── Caption below frame ───────────────────────────────────────────────
    caption_y = y0 + nh + 10
    if caption_y < panel_h - META_H - 4:
        cv2.putText(panel, task_text or "",
                    (x0, caption_y),
                    FONT, 0.42, TEXT_DARK, 1, cv2.LINE_AA)

    # ── Metadata bar at bottom ────────────────────────────────────────────
    bar_y = panel_h - META_H
    cv2.rectangle(panel, (0, bar_y), (panel_w, panel_h), META_BG, -1)
    cv2.line(panel, (0, bar_y), (panel_w, bar_y), BORDER_COL, 1)

    # Three metadata columns
    def meta_col(label, value, x):
        # Pin icon (small circle)
        cv2.circle(panel, (x+6, bar_y+18), 4, TEXT_GREY, 1, cv2.LINE_AA)
        cv2.putText(panel, label, (x+14, bar_y+20),
                    FONT, 0.30, TEXT_GREY, 1, cv2.LINE_AA)
        cv2.putText(panel, value, (x+14, bar_y+36),
                    FONT_BOLD, 0.38, TEXT_DARK, 1, cv2.LINE_AA)

    col_w = panel_w // 3
    meta_col("ENVIRONMENT", environment or "Factory",   10)
    meta_col("SCENE",       scene       or "Workstation", col_w)
    meta_col("OPERATOR HEIGHT", op_height or "—",       col_w*2)

    return panel


# ═══════════════════════════════════════════════════════════════════════════════
#  Full card compositor
# ═══════════════════════════════════════════════════════════════════════════════

def build_card(cam_panel, plot_panels):
    """
    Combine camera panel (left, large) + stacked 3D plot panels (right).
    Returns a single H×W card as BGR array.
    """
    H      = cam_panel.shape[0]
    cam_w  = cam_panel.shape[1]
    plot_w = plot_panels[0].shape[1]
    # Each plot panel gets equal share of H
    ph     = H // max(len(plot_panels), 1)

    right_col = np.full((H, plot_w, 3), CARD_BG, dtype=np.uint8)
    for i, pp in enumerate(plot_panels):
        y0 = i * ph
        y1 = y0 + ph
        resized = cv2.resize(pp, (plot_w, ph))
        right_col[y0:y1] = resized
        if i > 0:
            cv2.line(right_col, (0, y0), (plot_w, y0), BORDER_COL, 1)

    # Vertical divider
    div = np.full((H, 1, 3), BORDER_COL, dtype=np.uint8)

    card = np.hstack([cam_panel, div, right_col])

    # Outer card border
    cv2.rectangle(card, (0,0), (card.shape[1]-1, card.shape[0]-1),
                  BORDER_COL, 2)
    return card


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
#  Main loop
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(input_path, csv_path="hand_keypoints_3d.csv",
                  out_path="hand_3d_card.mp4",
                  card_height=720, plot_ratio=0.38,
                  skip_frames=1, max_hands=2, conf=0.55,
                  task="", environment="Factory",
                  scene="Workstation", op_height="—"):

    model_path = download_model()
    cap        = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = max(1.0, fps / skip_frames)

    H       = card_height
    # plot panel is plot_ratio of total width; cam panel gets the rest
    # Total width = cam_w + 1 (divider) + plot_w
    # plot_w / (cam_w + plot_w) = plot_ratio
    # Let total = 1600 for 16:9-ish landscape
    TOTAL_W  = int(H * 2.05)
    PLOT_W   = int(TOTAL_W * plot_ratio)
    CAM_W    = TOTAL_W - PLOT_W - 1
    CUBE_HALF = 0.09

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps, (TOTAL_W, H))

    # Per-slot trail: raw Y-flipped world coords
    raw_trails = [{tip: [] for tip in FINGERTIPS} for _ in range(max_hands)]

    csv_rows = []
    csv_hdr  = (["frame","hand"] +
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
        frame_idx    = 0
        pbar         = tqdm(total=total, unit="frame", desc="Rendering")

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

            # Per-hand data
            hand_data    = []   # list of (lms_scaled, raw_trail, wrist_raw, hand_scale, label)
            norm_lms_2d  = []

            if result.hand_world_landmarks:
                for slot, (wlm, nlm, hinfo) in enumerate(
                    zip(result.hand_world_landmarks,
                        result.hand_landmarks,
                        result.handedness)):

                    if slot >= max_hands:
                        break

                    label      = hinfo[0].display_name
                    lms_w      = mp_to_world(wlm)
                    wrist_raw  = lms_w[0].copy()
                    centred    = lms_w - wrist_raw
                    extent     = np.abs(centred).max()
                    hand_scale = (CUBE_HALF * 0.55 / extent) if extent>1e-9 else 1.0
                    lms_scaled = centred * hand_scale

                    norm_lms_2d.append([(lm.x, lm.y) for lm in nlm])

                    for tip in FINGERTIPS:
                        raw_trails[slot][tip].append(tuple(lms_w[tip]))
                        if len(raw_trails[slot][tip]) > TRAIL_LEN:
                            raw_trails[slot][tip].pop(0)

                    hand_data.append((lms_scaled, raw_trails[slot],
                                      wrist_raw, hand_scale, label))

                    row  = [frame_idx, label]
                    row += list(lms_w[:,0])
                    row += list(lms_w[:,1])
                    row += list(lms_w[:,2])
                    csv_rows.append(row)

            else:
                for slot in range(max_hands):
                    for tip in FINGERTIPS:
                        if raw_trails[slot][tip]:
                            raw_trails[slot][tip].pop(0)

            # ── Build plot panels (one per hand, stacked) ─────────────────
            plot_panels = []
            plot_h_each = H // max(max_hands, 1)

            for slot in range(max_hands):
                if slot < len(hand_data):
                    lms_sc, r_trail, w_raw, h_sc, lbl = hand_data[slot]
                    pp = draw_3d_plot(PLOT_W, plot_h_each,
                                     lms_sc, r_trail, w_raw, h_sc, CUBE_HALF)
                    # Hand label inside plot
                    cv2.putText(pp, lbl, (6, plot_h_each-8),
                                FONT, 0.40, TEXT_GREY, 1, cv2.LINE_AA)
                else:
                    # Empty plot (no hand detected in this slot)
                    pp = draw_3d_plot(PLOT_W, plot_h_each,
                                     None,
                                     {tip: [] for tip in FINGERTIPS},
                                     np.zeros(3), 1.0, CUBE_HALF)
                plot_panels.append(pp)

            # ── Camera panel ──────────────────────────────────────────────
            cam_panel = draw_camera_panel(
                frame, norm_lms_2d,
                task, environment, scene, op_height,
                CAM_W, H)

            # ── Compose card ──────────────────────────────────────────────
            card = build_card(cam_panel, plot_panels)
            writer.write(card)
            frame_idx += 1

        pbar.close()

    cap.release()
    writer.release()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_hdr)
        w.writerows(csv_rows)

    print(f"\nDone.")
    print(f"  Output : {out_path}  ({TOTAL_W}×{H})")
    print(f"  CSV    : {csv_path}  ({len(csv_rows):,} rows)")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Hand tracking card visualizer — matches reference layout")
    p.add_argument("--input",       required=True,
                   help="Input video (e.g. factory.mp4)")
    p.add_argument("--csv",         default="hand_keypoints_3d.csv")
    p.add_argument("--output",      default="hand_3d_card.mp4")
    p.add_argument("--height",      type=int,   default=720,
                   help="Card height in pixels (default 720)")
    p.add_argument("--skip",        type=int,   default=1,
                   help="Process every Nth frame")
    p.add_argument("--hands",       type=int,   default=2)
    p.add_argument("--conf",        type=float, default=0.55)
    p.add_argument("--task",        default="",
                   help='Task label shown top-left, e.g. "Grind a metal piece"')
    p.add_argument("--environment", default="Factory")
    p.add_argument("--scene",       default="Workstation")
    p.add_argument("--op_height",   default="—",
                   help='Operator height e.g. "170cm"')
    a = p.parse_args()
    process_video(
        a.input, a.csv, a.output,
        card_height = a.height,
        skip_frames = a.skip,
        max_hands   = a.hands,
        conf        = a.conf,
        task        = a.task,
        environment = a.environment,
        scene       = a.scene,
        op_height   = a.op_height,
    )
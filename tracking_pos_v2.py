"""
3-Panel Hand + Pose Tracking Visualizer  (v5 — complete rewrite)
=================================================================
Architecture:
  • MediaPipe HandLandmarker  → 21 finger joints per hand (world coords)
  • MediaPipe PoseLandmarker  → 33 body joints (world coords)
  • Hand wrist OVERWRITES pose wrist → seamless attachment

Layout  (single matplotlib figure, GridSpec):
  ┌────────────────┬───────────────┐
  │                │  Block 2      │
  │   Block 1      │  3D hands     │
  │   Camera +     │  scatter      │
  │   2D overlay   ├───────────────┤
  │                │  Block 3      │
  │                │  Locked pose  │
  │                │  + live hands │
  └────────────────┴───────────────┘

Block 3 key behaviour:
  - Pose skeleton is LOCKED vertically (always upright, never rotates)
  - Only the hand/finger joints move, attached at the pose wrist coordinates
  - Rendered as a pure 2D front-projection (X=left/right, Y=up/down)
  - Arms drawn as: Shoulder → Elbow → Wrist (pose) → [hand fingers from wrist]

Install:
    pip install mediapipe opencv-python matplotlib tqdm numpy

Usage:
    python tracking_pos.py --input factory.mp4 --task "Press operation"
"""

import argparse, csv, os, urllib.request
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, PoseLandmarkerOptions

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# Matplotlib colours
C_LEFT   = "#2ECC71"   # green  – left  hand  (Block 2)
C_RIGHT  = "#E74C3C"   # red    – right hand  (Block 2)
C_BONE   = "#95A5A6"   # grey   – pose bones  (Block 3)
C_JOINT  = "#7F8C8D"   # dark   – pose joints (Block 3)
C_LHAND  = "#27AE60"   # dark green – left  hand  (Block 3)
C_RHAND  = "#C0392B"   # dark red   – right hand  (Block 3)

# OpenCV BGR colours
LEFT_BGR  = ( 46, 204, 113)   # green
RIGHT_BGR = ( 52,  73, 235)   # red

FINGER_BGR = {
    "palm":   ( 80, 80,  80),
    "thumb":  (  0,140, 255),
    "index":  ( 30,180,  30),
    "middle": (200, 60,  20),
    "ring":   ( 20, 20, 200),
    "pinky":  (170, 30, 170),
}
LM_FINGER = {}
for _n, _ids in [("palm",[0]),("thumb",[1,2,3,4]),("index",[5,6,7,8]),
                 ("middle",[9,10,11,12]),("ring",[13,14,15,16]),
                 ("pinky",[17,18,19,20])]:
    for _i in _ids: LM_FINGER[_i] = _n

FINGERTIPS = [4, 8, 12, 16, 20]

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# Pose landmark indices
POSE_NOSE        = 0
POSE_L_SHOULDER  = 11
POSE_R_SHOULDER  = 12
POSE_L_ELBOW     = 13
POSE_R_ELBOW     = 14
POSE_L_WRIST     = 15
POSE_R_WRIST     = 16
POSE_L_HIP       = 23
POSE_R_HIP       = 24

# Bones to draw for the "Human Presence" view (Block 3)
# We only need upper body — shoulders, arms, torso
POSE_UPPER_CONN = [
    (POSE_L_SHOULDER, POSE_R_SHOULDER),              # shoulder bar
    (POSE_L_SHOULDER, POSE_L_HIP),                   # left torso
    (POSE_R_SHOULDER, POSE_R_HIP),                   # right torso
    (POSE_L_HIP,      POSE_R_HIP),                   # hip bar
    (POSE_L_SHOULDER, POSE_L_ELBOW),                 # left upper arm
    (POSE_L_ELBOW,    POSE_L_WRIST),                 # left forearm
    (POSE_R_SHOULDER, POSE_R_ELBOW),                 # right upper arm
    (POSE_R_ELBOW,    POSE_R_WRIST),                 # right forearm
    (POSE_NOSE,       POSE_L_SHOULDER),              # neck left
    (POSE_NOSE,       POSE_R_SHOULDER),              # neck right
]

TRAIL_LEN  = 25
CUBE_HALF  = 0.10
DPI        = 100

# Model URLs
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")


# ─────────────────────────────────────────────────────────────────────────────
#  Model download
# ─────────────────────────────────────────────────────────────────────────────

def download(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} …")
        urllib.request.urlretrieve(url, path)
        print("  Done.")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def hand_world(lm_list):
    """MediaPipe hand world lms → (21,3), Y flipped (+Y=up)."""
    a = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float64)
    a[:, 1] *= -1
    return a

def hand_norm(lm_list):
    """Normalised image-space lms → list of (x,y) in [0,1]."""
    return [(lm.x, lm.y) for lm in lm_list]

def centre_scale(lms, half=CUBE_HALF, fill=0.55):
    """Centre on wrist, scale to fill the cube."""
    c = lms - lms[0]
    e = np.abs(c).max()
    return c * (half * fill / e) if e > 1e-9 else c

def pose_world(lm_list):
    """PoseLandmarker world lms → (33,3).
    MediaPipe Pose: X=right, Y=down, Z=depth (away from camera).
    We flip Y so +Y=up (head at positive Y).
    """
    a = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float64)
    a[:, 1] *= -1   # now +Y = head, -Y = feet
    return a


# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 1  —  Camera panel (OpenCV → numpy array)
# ─────────────────────────────────────────────────────────────────────────────

def build_block1(frame, norm_lms_list, handedness_list, W, H):
    fh, fw = frame.shape[:2]
    sc = min(W/fw, H/fh)
    nw, nh = int(fw*sc), int(fh*sc)
    canvas = np.full((H, W, 3), 245, dtype=np.uint8)
    x0, y0 = (W-nw)//2, (H-nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = cv2.resize(frame, (nw, nh))

    for lms, label in zip(norm_lms_list, handedness_list):
        pts  = [(x0 + int(lm[0]*nw), y0 + int(lm[1]*nh)) for lm in lms]
        hbgr = LEFT_BGR if label == "Left" else RIGHT_BGR
        for a, b in HAND_CONN:
            cv2.line(canvas, pts[a], pts[b], hbgr, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            col = FINGER_BGR[LM_FINGER[i]]
            r   = 6 if i in FINGERTIPS else 4
            cv2.circle(canvas, pt, r+1, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, r,   col,           -1, cv2.LINE_AA)

    # Labels
    for j, (label, col) in enumerate(
        [("Left", LEFT_BGR), ("Right", RIGHT_BGR)]
    ):
        cv2.circle(canvas, (12, H-28+j*16), 5, col, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"{label} hand", (22, H-24+j*16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60,60,60), 1, cv2.LINE_AA)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 2  —  3D hand scatter  (matplotlib axes object, drawn by caller)
# ─────────────────────────────────────────────────────────────────────────────

def draw_block2(ax, hands_data, trails):
    """
    hands_data : list of (lms_sc_21x3, label)   label='Left'|'Right'
    trails     : dict {label: [(x,y,z), ...]}
    Draws into an existing matplotlib 3D Axes.
    """
    ax.cla()
    ax.set_facecolor('#FAFAFA')

    for lms_sc, label in hands_data:
        color = C_LEFT if label == "Left" else C_RIGHT

        # Trail (wrist path, faint dots)
        tr = trails.get(label, [])
        if len(tr) > 1:
            trx = [p[0] for p in tr]
            try_ = [p[1] for p in tr]
            trz = [p[2] for p in tr]
            alphas = np.linspace(0.05, 0.35, len(tr))
            for k in range(len(tr)):
                ax.scatter(trx[k], try_[k], trz[k], s=6,
                           c=color, alpha=float(alphas[k]),
                           depthshade=False, zorder=1)

        # Bones
        xs, ys, zs = lms_sc[:,0], lms_sc[:,1], lms_sc[:,2]
        for a, b in HAND_CONN:
            ax.plot([xs[a],xs[b]], [ys[a],ys[b]], [zs[a],zs[b]],
                    color=color, lw=1.4, alpha=0.75, zorder=2)

        # All 21 joints — single scatter call per hand
        colors_per_lm = [color] * 21
        sizes = [60 if i in FINGERTIPS else (80 if i==0 else 30) for i in range(21)]
        ax.scatter(xs, ys, zs,
                   c=colors_per_lm, s=sizes,
                   edgecolors='white', linewidths=0.6,
                   alpha=0.95, depthshade=True, zorder=3)

    # Styling
    lim = CUBE_HALF * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('X', fontsize=7, color='#555', labelpad=1)
    ax.set_ylabel('Y', fontsize=7, color='#555', labelpad=1)
    ax.set_zlabel('Z', fontsize=7, color='#555', labelpad=1)
    ax.tick_params(labelsize=5, colors='#999', pad=0.5)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#DDDDDD')
    ax.xaxis.line.set_color('#CCCCCC')
    ax.yaxis.line.set_color('#CCCCCC')
    ax.zaxis.line.set_color('#CCCCCC')
    ax.grid(True, color='#EEEEEE', linewidth=0.4)
    ax.view_init(elev=22, azim=-55)
    ax.set_title('3D hand tracking', fontsize=9, color='#333', pad=6,
                 fontweight='semibold')

    patches = [mpatches.Patch(color=C_LEFT,  label='Left hand'),
               mpatches.Patch(color=C_RIGHT, label='Right hand')]
    ax.legend(handles=patches, loc='upper left', fontsize=7,
              framealpha=0.7, edgecolor='#CCCCCC')


# ─────────────────────────────────────────────────────────────────────────────
#  BLOCK 3 — Locked body + live hand clusters
#
#  Matches reference image exactly:
#  • 3D axes, white bg, dashed grey grid (same style as Block 2)
#  • Body skeleton is a FIXED T-pose (hardcoded world coords) — never moves
#  • Only hand finger joints animate each frame, attached at fixed wrist pts
#  • Grey thin lines + grey small dots for body
#  • Green/red larger dots + thin lines for hands
#  • Slight isometric view: elev=18, azim=-60 (matches reference)
# ─────────────────────────────────────────────────────────────────────────────

# Fixed T-pose skeleton (world-space metres, person ~1.7 m tall centred at origin)
# Landmarks: nose, l_shoulder, r_shoulder, l_elbow, r_elbow,
#            l_wrist, r_wrist, l_hip, r_hip
_S = 0.22   # shoulder half-width
_T = 0.18   # torso half-height from shoulder to hip
_A = 0.26   # upper arm length
_F = 0.22   # forearm length
_H = 0.10   # hip half-width

FIXED_POSE = {
    "nose":       np.array([ 0.0,   0.50,  0.0]),
    "l_shoulder": np.array([-_S,    0.30,  0.0]),
    "r_shoulder": np.array([ _S,    0.30,  0.0]),
    "l_elbow":    np.array([-_S-_A, 0.30,  0.0]),
    "r_elbow":    np.array([ _S+_A, 0.30,  0.0]),
    "l_wrist":    np.array([-_S-_A-_F, 0.30, 0.0]),
    "r_wrist":    np.array([ _S+_A+_F, 0.30, 0.0]),
    "l_hip":      np.array([-_H,  -_T,   0.0]),
    "r_hip":      np.array([ _H,  -_T,   0.0]),
}

FIXED_BONES = [
    ("nose",       "l_shoulder"),
    ("nose",       "r_shoulder"),
    ("l_shoulder", "r_shoulder"),
    ("l_shoulder", "l_hip"),
    ("r_shoulder", "r_hip"),
    ("l_hip",      "r_hip"),
    ("l_shoulder", "l_elbow"),
    ("l_elbow",    "l_wrist"),
    ("r_shoulder", "r_elbow"),
    ("r_elbow",    "r_wrist"),
]


def draw_block3(ax, pose_arr, hands_at_wrist):
    """
    pose_arr       : (33,3) or None  — used ONLY to get elbow angles (optional)
    hands_at_wrist : {'Left': lms_21x3, 'Right': lms_21x3}
                     lms centred on wrist: col0=X, col1=Y_up, col2=Z_depth
    """
    ax.cla()
    ax.set_facecolor('white')

    # ── Style to match reference exactly ─────────────────────────────────
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#DDDDDD')
        pane.set_alpha(0.4)
    ax.xaxis.line.set_color('#BBBBBB')
    ax.yaxis.line.set_color('#BBBBBB')
    ax.zaxis.line.set_color('#BBBBBB')
    ax.tick_params(labelsize=5, colors='#AAAAAA', pad=0.5)
    ax.grid(True, color='#EEEEEE', linewidth=0.5, linestyle='--')
    ax.set_xlabel('X', fontsize=6, color='#999', labelpad=0)
    ax.set_ylabel('Z', fontsize=6, color='#999', labelpad=0)
    ax.set_zlabel('Y', fontsize=6, color='#999', labelpad=0)

    # ── Use live elbow positions if pose available, else fixed T-pose ─────
    # We update elbow and wrist positions from pose model if available,
    # keeping shoulders/hips fixed. This makes arms move naturally.
    fp = {k: v.copy() for k, v in FIXED_POSE.items()}

    if pose_arr is not None:
        # pose_arr cols: X=right, Y=up(flipped), Z=depth
        # Extract key joints and rescale to match fixed skeleton scale
        p_ls = pose_arr[POSE_L_SHOULDER]
        p_rs = pose_arr[POSE_R_SHOULDER]
        p_le = pose_arr[POSE_L_ELBOW]
        p_re = pose_arr[POSE_R_ELBOW]
        p_lw = pose_arr[POSE_L_WRIST]
        p_rw = pose_arr[POSE_R_WRIST]

        # Compute pose arm vectors relative to shoulder, scaled to fixed lengths
        def scale_vec(start, end, fixed_len):
            v = end - start
            n = np.linalg.norm(v)
            return (v / n * fixed_len) if n > 1e-6 else v

        ls = fp["l_shoulder"]
        rs = fp["r_shoulder"]

        # Upper arm direction from pose, length from fixed skeleton
        fp["l_elbow"] = ls + scale_vec(p_ls, p_le, _A)
        fp["r_elbow"] = rs + scale_vec(p_rs, p_re, _A)

        # Forearm direction from pose
        fp["l_wrist"] = fp["l_elbow"] + scale_vec(p_le, p_lw, _F)
        fp["r_wrist"] = fp["r_elbow"] + scale_vec(p_re, p_rw, _F)

    # ── Draw fixed body bones (thin grey lines) ───────────────────────────
    for a_name, b_name in FIXED_BONES:
        a3 = fp[a_name]
        b3 = fp[b_name]
        ax.plot([a3[0], b3[0]], [a3[2], b3[2]], [a3[1], b3[1]],
                color='#AAAAAA', lw=1.5, alpha=0.9, zorder=2)

    # ── Draw body joints (small grey dots) ────────────────────────────────
    for name, pt in fp.items():
        sz = 40 if name == "nose" else 25
        ax.scatter([pt[0]], [pt[2]], [pt[1]],
                   s=sz, c='#888888', zorder=3,
                   edgecolors='white', linewidths=0.6,
                   depthshade=False)

    # ── Attach hands at wrist positions ──────────────────────────────────
    # Hand scale: make hand ≈ 0.22 units in the fixed skeleton space
    hand_target = _F * 0.85          # hand ≈ 85% of forearm length
    hand_world_span = CUBE_HALF      # lms_sc spans ±CUBE_HALF
    hand_sc = hand_target / hand_world_span

    wrist_pts = {'Left':  fp["l_wrist"], 'Right': fp["r_wrist"]}
    hcol_map  = {'Left':  C_LHAND,       'Right': C_RHAND}

    for side, lms_sc in hands_at_wrist.items():
        w3  = wrist_pts[side]       # wrist position in fixed skeleton space
        col = hcol_map[side]

        # Each hand landmark → 3D position attached to wrist
        # lms_sc: col0=X(right), col1=Y(up), col2=Z(depth)
        # Map to skeleton axes: X→X, Y→Y(Z in mpl), Z→Z(Y in mpl)
        hx = lms_sc[:, 0] * hand_sc + w3[0]
        hy = lms_sc[:, 1] * hand_sc + w3[1]
        hz = lms_sc[:, 2] * hand_sc + w3[2]

        # Bones (thin coloured lines)
        for a, b in HAND_CONN:
            ax.plot([hx[a], hx[b]], [hz[a], hz[b]], [hy[a], hy[b]],
                    color=col, lw=1.0, alpha=0.80, zorder=4)

        # All 21 joints — single scatter matching reference dot style
        sizes = np.array([
            50 if i in FINGERTIPS else (60 if i == 0 else 28)
            for i in range(21)
        ], dtype=float)
        ax.scatter(hx, hz, hy,
                   s=sizes, c=col,
                   edgecolors='white', linewidths=0.5,
                   alpha=0.95, depthshade=True, zorder=5)

    # ── Axis limits centred on skeleton ───────────────────────────────────
    all_pts = np.array(list(fp.values()))
    cx = all_pts[:, 0].mean()
    cy = all_pts[:, 1].mean()
    span = 0.75
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(-0.3, 0.3)          # depth axis: shallow
    ax.set_zlim(cy - span, cy + span)

    # View angle matching reference image: slight isometric tilt
    ax.view_init(elev=18, azim=-60)

    ax.set_title('Human pose + hands', fontsize=9, color='#333', pad=4,
                 fontweight='semibold')

    patches = [mpatches.Patch(color='#AAAAAA', label='Body'),
               mpatches.Patch(color=C_LHAND,   label='Left hand'),
               mpatches.Patch(color=C_RHAND,   label='Right hand')]
    ax.legend(handles=patches, loc='upper left', fontsize=6,
              framealpha=0.6, edgecolor='#CCCCCC')


# ─────────────────────────────────────────────────────────────────────────────
#  Composite frame builder  (single matplotlib figure, GridSpec)
# ─────────────────────────────────────────────────────────────────────────────

def build_frame(camera_img, norm_lms, handedness, hands_data, trails,
                pose_arr, hands_at_wrist, fig, axes, task):
    """
    fig, axes : persistent figure and (ax_cam, ax_3d, ax_pose) — reused each frame
    Returns BGR numpy array.
    """
    ax_cam, ax_3d, ax_pose = axes

    # ── Block 1: camera (blit as image into axes) ─────────────────────────
    ax_cam.cla()
    ax_cam.axis('off')
    rgb_canvas = build_block1(
        camera_img, norm_lms, handedness,
        int(fig.get_figwidth() * DPI * 0.45),
        int(fig.get_figheight() * DPI)
    )
    ax_cam.imshow(cv2.cvtColor(rgb_canvas, cv2.COLOR_BGR2RGB),
                  aspect='auto')
    title = f"Camera overlay" + (f"  |  Task: {task}" if task else "")
    ax_cam.set_title(title, fontsize=9, color='#333', pad=6,
                     fontweight='semibold')

    # ── Block 2: 3D hand scatter ──────────────────────────────────────────
    draw_block2(ax_3d, hands_data, trails)

    # ── Block 3: locked pose + live hands ─────────────────────────────────
    draw_block3(ax_pose, pose_arr, hands_at_wrist)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def process_video(input_path, csv_path="hand_keypoints_3d.csv",
                  out_path="hand_3panel.mp4",
                  fig_w=18, fig_h=7, skip_frames=1,
                  max_hands=2, conf=0.55, task="",
                  environment="Factory", scene="Workstation", op_height="—"):

    hand_model = download(HAND_URL, "hand_landmarker.task")
    pose_model = download(POSE_URL, "pose_landmarker_lite.task")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = max(1.0, fps / skip_frames)

    # ── Persistent matplotlib figure with GridSpec ────────────────────────
    # Layout:  [Block1 (left, wide)] | [Block2 top-right]
    #                                | [Block3 bottom-right]
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    fig.patch.set_facecolor('white')

    gs  = gridspec.GridSpec(
        2, 2,
        figure      = fig,
        width_ratios= [1.05, 0.95],   # left slightly wider
        height_ratios= [1, 1],
        hspace      = 0.08,
        wspace      = 0.06,
        left=0.01, right=0.99, top=0.96, bottom=0.03,
    )

    ax_cam  = fig.add_subplot(gs[:, 0])                    # full left column
    ax_3d   = fig.add_subplot(gs[0, 1], projection='3d')  # top right 3D
    ax_pose = fig.add_subplot(gs[1, 1], projection='3d')  # bottom right 3D

    axes = (ax_cam, ax_3d, ax_pose)

    # Output video dimensions (from figure)
    OUT_W = int(fig_w * DPI)
    OUT_H = int(fig_h * DPI)

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (OUT_W, OUT_H))

    # ── Detector options ──────────────────────────────────────────────────
    hand_opts = HandLandmarkerOptions(
        base_options                   = mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode                   = mp_vision.RunningMode.VIDEO,
        num_hands                      = max_hands,
        min_hand_detection_confidence  = conf,
        min_hand_presence_confidence   = 0.5,
        min_tracking_confidence        = 0.5,
    )
    pose_opts = PoseLandmarkerOptions(
        base_options                   = mp_python.BaseOptions(model_asset_path=pose_model),
        running_mode                   = mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence  = 0.5,
        min_pose_presence_confidence   = 0.5,
        min_tracking_confidence        = 0.5,
    )

    # Trail storage  {label: [(x,y,z), ...]}
    trails = {'Left': [], 'Right': []}

    csv_rows = []
    csv_hdr  = (["frame","hand"] +
                [f"x{i}" for i in range(21)] +
                [f"y{i}" for i in range(21)] +
                [f"z{i}" for i in range(21)])

    with (mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_det,
          mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_det):

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

            # ── Run both models in parallel ───────────────────────────────
            h_res = hand_det.detect_for_video(mp_img, ts_ms)
            p_res = pose_det.detect_for_video(mp_img, ts_ms)

            # ── Parse hands ───────────────────────────────────────────────
            norm_lms_2d   = []
            handedness_2d = []
            hands_data_b2 = []   # for Block 2: (lms_sc, label)
            hands_at_wrist= {}   # for Block 3: {label: lms_sc}

            if h_res.hand_world_landmarks:
                for slot, (wlm, nlm, hinfo) in enumerate(
                    zip(h_res.hand_world_landmarks,
                        h_res.hand_landmarks,
                        h_res.handedness)):
                    if slot >= max_hands: break

                    label  = hinfo[0].display_name
                    lms_w  = hand_world(wlm)
                    lms_sc = centre_scale(lms_w)

                    norm_lms_2d.append(hand_norm(nlm))
                    handedness_2d.append(label)
                    hands_data_b2.append((lms_sc, label))
                    hands_at_wrist[label] = lms_sc

                    # Trail: store wrist (centred = origin, but we track it anyway)
                    trails[label].append(tuple(lms_sc[0]))
                    if len(trails[label]) > TRAIL_LEN:
                        trails[label].pop(0)

                    row  = [frame_idx, label]
                    row += list(lms_w[:,0])
                    row += list(lms_w[:,1])
                    row += list(lms_w[:,2])
                    csv_rows.append(row)
            else:
                for lbl in trails:
                    if trails[lbl]: trails[lbl].pop(0)

            # ── Parse pose ────────────────────────────────────────────────
            pose_arr = None
            if p_res.pose_world_landmarks:
                pose_arr = pose_world(p_res.pose_world_landmarks[0])

                # ── KEY: overwrite pose wrist with hand wrist ─────────────
                # This creates a seamless arm→hand skeleton in Block 3.
                wrist_idx = {'Left': POSE_L_WRIST, 'Right': POSE_R_WRIST}
                for side, lms_sc in hands_at_wrist.items():
                    # lms_sc[0] is the hand wrist in hand-centred space.
                    # We want the POSE wrist position (already in world space)
                    # to remain as the anchor — the hand fingers are drawn
                    # RELATIVE to the pose wrist. So no overwrite needed here;
                    # draw_block3 reads pose wrist directly and offsets hand from it.
                    pass   # intentional — see draw_block3 hp() function

            # ── Compose frame ─────────────────────────────────────────────
            out_frame = build_frame(
                frame, norm_lms_2d, handedness_2d,
                hands_data_b2, trails,
                pose_arr, hands_at_wrist,
                fig, axes, task
            )

            # Ensure exact output dimensions
            out_frame = cv2.resize(out_frame, (OUT_W, OUT_H))
            writer.write(out_frame)
            frame_idx += 1

        pbar.close()

    cap.release()
    writer.release()
    plt.close(fig)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_hdr)
        w.writerows(csv_rows)

    print(f"\nDone.  {out_path}  ({OUT_W}×{OUT_H})")
    print(f"       CSV: {csv_path}  ({len(csv_rows):,} rows)")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="3-panel hand+pose tracking visualizer")
    p.add_argument("--input",        required=True)
    p.add_argument("--csv",          default="hand_keypoints_3d.csv")
    p.add_argument("--output",       default="hand_3panel.mp4")
    p.add_argument("--fig_w",        type=int,   default=18,
                   help="Figure width  in inches (default 18 → 1800px at 100dpi)")
    p.add_argument("--fig_h",        type=int,   default=7,
                   help="Figure height in inches (default 7  → 700px  at 100dpi)")
    p.add_argument("--skip",         type=int,   default=1)
    p.add_argument("--hands",        type=int,   default=2)
    p.add_argument("--conf",         type=float, default=0.55)
    p.add_argument("--task",         default="")
    p.add_argument("--environment",  default="Factory")
    p.add_argument("--scene",        default="Workstation")
    p.add_argument("--op_height",    default="—")
    a = p.parse_args()
    process_video(
        a.input, a.csv, a.output,
        a.fig_w, a.fig_h,
        a.skip, a.hands, a.conf,
        a.task, a.environment, a.scene, a.op_height
    )
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
#  BLOCK 3  —  Locked vertical pose + live hands  (matplotlib 2D axes)
#
#  STRATEGY:
#  1. Pose skeleton is projected as pure 2D front view (X,Y only — depth ignored)
#     This LOCKS the person upright. The 3D camera angle cannot tilt them.
#  2. Only the arm endpoint (wrist) moves — pulled from the Pose model each frame.
#  3. Hand fingers are attached starting from that wrist pixel coordinate.
#  4. Hand wrist (lm[0]) overwrites pose wrist → seamless joint.
# ─────────────────────────────────────────────────────────────────────────────

def draw_block3(ax, pose_arr, hands_at_wrist):
    """
    pose_arr       : (33,3) or None  — world coords, Y flipped
    hands_at_wrist : {'Left': lms_21x3, 'Right': lms_21x3}
                     lms are centred on wrist (col0=X, col1=Y_up, col2=Z_depth)
    """
    ax.cla()
    ax.set_facecolor('#FAFAFA')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Human pose + hands', fontsize=9, color='#333', pad=6,
                 fontweight='semibold')

    if pose_arr is None:
        ax.text(0.5, 0.5, 'No person detected',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=11, color='#BBBBBB')
        return

    # ── 2-D projection: use only X (left/right) and Y (up/down) ──────────
    # pose_arr[:, 0] = X  (right is positive)
    # pose_arr[:, 1] = Y  (up is positive after flip)
    # We IGNORE Z (depth) entirely — this is what locks the person vertical.
    wx = pose_arr[:, 0]
    wy = pose_arr[:, 1]

    # Normalise to [0,1] range so we can draw on the axes
    xmin, xmax = wx.min(), wx.max()
    ymin, ymax = wy.min(), wy.max()
    xr = max(xmax - xmin, 1e-6)
    yr = max(ymax - ymin, 1e-6)
    # Keep aspect ratio — pad the narrower dimension
    span = max(xr, yr) * 1.25
    xc   = (xmin + xmax) / 2
    yc   = (ymin + ymax) / 2

    def wp(i):
        """Pose world coord i → (ax_x, ax_y)."""
        return ((wx[i] - xc) / span + 0.5,
                (wy[i] - yc) / span + 0.5)

    # ── Draw pose upper-body bones ────────────────────────────────────────
    for a, b in POSE_UPPER_CONN:
        x0, y0 = wp(a)
        x1, y1 = wp(b)
        ax.plot([x0, x1], [y0, y1],
                color=C_BONE, lw=3.5, solid_capstyle='round',
                alpha=0.85, zorder=2)

    # ── Draw pose upper-body joints ───────────────────────────────────────
    upper_joints = {POSE_NOSE, POSE_L_SHOULDER, POSE_R_SHOULDER,
                    POSE_L_ELBOW, POSE_R_ELBOW,
                    POSE_L_WRIST, POSE_R_WRIST,
                    POSE_L_HIP, POSE_R_HIP}
    for i in upper_joints:
        x, y = wp(i)
        ax.scatter(x, y, s=80, c=C_JOINT, zorder=4,
                   edgecolors='white', linewidths=1.0)

    # ── Hand scale: fit hand to be ~20% of body height on the axes ────────
    body_h_world = yr          # vertical span of pose in world metres
    hand_target  = body_h_world * 0.22   # hand = 22% of body height
    # lms_sc spans approximately ±CUBE_HALF in each axis
    hand_sc      = hand_target / CUBE_HALF

    # ── Attach hands at pose wrist positions ──────────────────────────────
    wrist_map = {'Left':  POSE_L_WRIST,
                 'Right': POSE_R_WRIST}
    hcol_map  = {'Left':  C_LHAND,
                 'Right': C_RHAND}

    for side, lms_sc in hands_at_wrist.items():
        wi = wrist_map[side]
        # Pose wrist in axes coordinates
        wax, way = wp(wi)
        hcol = hcol_map[side]

        # Each hand landmark in axes coordinates:
        # lms_sc[i] is relative to wrist (lms_sc[0] ≈ 0,0,0)
        # We use col0 (X left/right) and col1 (Y up/down, already flipped)
        # col2 (depth) is ignored — same as pose: locked 2D front view
        def hp(i):
            dx = lms_sc[i, 0] * hand_sc / span
            dy = lms_sc[i, 1] * hand_sc / span
            return (wax + dx, way + dy)

        hand_pts = [hp(i) for i in range(21)]

        # Bones
        for a, b in HAND_CONN:
            ax.plot([hand_pts[a][0], hand_pts[b][0]],
                    [hand_pts[a][1], hand_pts[b][1]],
                    color=hcol, lw=2.0, solid_capstyle='round',
                    alpha=0.85, zorder=5)

        # Joints
        for i, (hx, hy) in enumerate(hand_pts):
            s  = 55 if i in FINGERTIPS else (70 if i == 0 else 30)
            ax.scatter(hx, hy, s=s, c=hcol, zorder=6,
                       edgecolors='white', linewidths=0.8)

    # ── Axis limits — a little padding around the figure ──────────────────
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    # ── Legend ────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=C_BONE,  label='Pose'),
               mpatches.Patch(color=C_LHAND, label='Left hand'),
               mpatches.Patch(color=C_RHAND, label='Right hand')]
    ax.legend(handles=patches, loc='lower left', fontsize=7,
              framealpha=0.7, edgecolor='#CCCCCC')


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

    ax_cam  = fig.add_subplot(gs[:, 0])           # full left column
    ax_3d   = fig.add_subplot(gs[0, 1], projection='3d')  # top right
    ax_pose = fig.add_subplot(gs[1, 1])           # bottom right (2D)

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
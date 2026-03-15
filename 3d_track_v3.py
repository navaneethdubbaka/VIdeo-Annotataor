"""
6-DoF Embodied AI Data Collection Visualizer
=============================================
Implements the full system described in the brief:

  1. Egocentric RGB-D Vision Capture
     - First-person POV video feed (helmet/head-mounted camera)
     - Annotation card: Skill label + NL description + timestamps
     - Third-person NL caption below frame
     - Environmental metadata: ENVIRONMENT / SCENE / OPERATOR HEIGHT

  2. 6-DoF Kinematic Pose Estimation
     - Top-right: Cumulative 3D hand trajectory point cloud
                  Green=Left hand, Red=Right hand
                  Tri-axial orientation arrows at each joint (X=red,Y=green,Z=blue)
     - Bottom-right: Full-body stick skeleton (locked upright) with live
                     hand skeletons attached at wrist terminators
                     Orientation arrows at every body + hand joint

  3. Hierarchical Temporal Segmentation
     - Macro label: bottom caption (full episode description)
     - Micro label: top-left card (short-horizon action + timestamps)

  4. Environmental State Logging
     - Operator height logged for spatial normalisation
     - Scene + environment tags for context embeddings

Layout (single matplotlib figure):
  ┌────────────────────────┬──────────────────┐
  │   Egocentric video     │  3D hand cloud   │
  │   + annotation card    │  (trajectory)    │
  │   + metadata bar       ├──────────────────┤
  │                        │  Full-body pose  │
  │                        │  + live hands    │
  └────────────────────────┴──────────────────┘

Install:
    pip install mediapipe opencv-python matplotlib tqdm numpy

Usage:
    python vla_visualizer.py \\
        --input factory.mp4 \\
        --skill "Inflate tire" \\
        --description "Inflate the car tire using the pressure gauge." \\
        --nl_caption "The person inflates a car tire using a pressure gauge." \\
        --environment "Car Workshop" \\
        --scene "Car service" \\
        --op_height "162cm" \\
        --skip 2
"""

import argparse
import csv
import math
import os
import textwrap
import urllib.request

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
#  Visual constants
# ─────────────────────────────────────────────────────────────────────────────

# Matplotlib hex colours
C_LEFT  = "#2ECC71"   # green – left  hand cloud + skeleton
C_RIGHT = "#E74C3C"   # red   – right hand cloud + skeleton
C_BODY  = "#95A5A6"   # grey  – body skeleton lines
C_JOINT = "#7F8C8D"   # dark grey – body joints
C_AX_X  = "#E74C3C"   # red   – X orientation arrow
C_AX_Y  = "#2ECC71"   # green – Y orientation arrow
C_AX_Z  = "#3498DB"   # blue  – Z orientation arrow

# OpenCV BGR colours for 2-D overlay
L_BGR = ( 46, 204, 113)   # green
R_BGR = ( 52,  73, 235)   # red

FINGER_BGR = {
    "palm":   ( 60, 60,  60),
    "thumb":  (  0,140, 255),
    "index":  ( 30,180,  30),
    "middle": (200, 60,  20),
    "ring":   ( 20, 20, 200),
    "pinky":  (170, 30, 170),
}

LM_FINGER: dict[int, str] = {}
for _nm, _ids in [("palm",[0]),("thumb",[1,2,3,4]),("index",[5,6,7,8]),
                  ("middle",[9,10,11,12]),("ring",[13,14,15,16]),
                  ("pinky",[17,18,19,20])]:
    for _i in _ids:
        LM_FINGER[_i] = _nm

FINGERTIPS = [4, 8, 12, 16, 20]

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

# ── MediaPipe Pose landmark indices ──────────────────────────────────────────
P_NOSE  = 0
P_LS    = 11   # left  shoulder
P_RS    = 12   # right shoulder
P_LE    = 13   # left  elbow
P_RE    = 14   # right elbow
P_LW    = 15   # left  wrist
P_RW    = 16   # right wrist
P_LH    = 23   # left  hip
P_RH    = 24   # right hip

# Upper-body bone pairs for the pose panel
BODY_BONES = [
    (P_NOSE, P_LS), (P_NOSE, P_RS),
    (P_LS,   P_RS),
    (P_LS,   P_LH), (P_RS,   P_RH),
    (P_LH,   P_RH),
    (P_LS,   P_LE), (P_LE,   P_LW),
    (P_RS,   P_RE), (P_RE,   P_RW),
]

BODY_JOINTS = [P_NOSE, P_LS, P_RS, P_LE, P_RE, P_LW, P_RW, P_LH, P_RH]

DPI        = 100
CUBE_HALF  = 0.10   # hand landmarks normalised to ±CUBE_HALF

# ── Fixed T-pose geometry (skeleton space, metres) ────────────────────────────
_SW = 0.22   # shoulder half-width
_TH = 0.18   # torso half-height (shoulder→hip)
_UA = 0.26   # upper arm length
_FA = 0.22   # forearm length
_HW = 0.10   # hip half-width

TPOSE: dict[str, np.ndarray] = {
    "nose":       np.array([ 0.0,        0.52,  0.0]),
    "l_shoulder": np.array([-_SW,        0.30,  0.0]),
    "r_shoulder": np.array([ _SW,        0.30,  0.0]),
    "l_elbow":    np.array([-_SW-_UA,    0.30,  0.0]),
    "r_elbow":    np.array([ _SW+_UA,    0.30,  0.0]),
    "l_wrist":    np.array([-_SW-_UA-_FA,0.30,  0.0]),
    "r_wrist":    np.array([ _SW+_UA+_FA,0.30,  0.0]),
    "l_hip":      np.array([-_HW,       -_TH,   0.0]),
    "r_hip":      np.array([ _HW,       -_TH,   0.0]),
}

SKEL_BONES = [
    ("nose","l_shoulder"),("nose","r_shoulder"),
    ("l_shoulder","r_shoulder"),
    ("l_shoulder","l_hip"),("r_shoulder","r_hip"),
    ("l_hip","r_hip"),
    ("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
    ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
]

# Model download URLs
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def download_model(url: str, path: str) -> str:
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)} …")
        urllib.request.urlretrieve(url, path)
        print("  Done.")
    return path


def fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS.mmm"""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"


def hand_to_world(lm_list) -> np.ndarray:
    """MediaPipe hand world landmarks → (21,3) float64. Y-flipped so +Y=up."""
    a = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float64)
    a[:, 1] *= -1
    return a


def hand_to_norm(lm_list) -> list[tuple[float,float]]:
    """Normalised image-space hand landmarks → list of (x,y) in [0,1]."""
    return [(lm.x, lm.y) for lm in lm_list]


def pose_to_world(lm_list) -> np.ndarray:
    """MediaPipe pose world landmarks → (33,3). Y-flipped so +Y=up."""
    a = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float64)
    a[:, 1] *= -1
    return a


def centre_scale(lms: np.ndarray, half: float = CUBE_HALF,
                 fill: float = 0.55) -> np.ndarray:
    """Centre hand on wrist, scale so max extent = half*fill."""
    c = lms - lms[0]
    e = np.abs(c).max()
    return c * (half * fill / e) if e > 1e-9 else c


def scale_vec(start: np.ndarray, end: np.ndarray,
              length: float) -> np.ndarray:
    """Direction vector from start→end, rescaled to `length`."""
    v = end - start
    n = np.linalg.norm(v)
    return (v / n * length) if n > 1e-6 else np.zeros(3)


def local_frame(a: np.ndarray, b: np.ndarray):
    """Right-handed local frame along bone a→b.
    Returns (x_axis, y_axis, z_axis) all unit vectors."""
    fwd = b - a
    fn  = np.linalg.norm(fwd)
    if fn < 1e-9:
        return np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])
    fwd /= fn
    up   = np.array([0., 1., 0.])
    if abs(np.dot(fwd, up)) > 0.95:
        up = np.array([0., 0., 1.])
    right = np.cross(fwd, up);  right /= np.linalg.norm(right)
    up2   = np.cross(right, fwd)
    return right, up2, fwd  # x=red, y=green, z=blue


def draw_axes_at(ax, origin: np.ndarray, frame, length: float,
                 lw: float = 1.0, alpha: float = 0.85) -> None:
    """Draw tri-axial orientation arrows (red X, green Y, blue Z) at origin.
    Matplotlib 3D axes are used with (X, Z, Y) coordinate order."""
    ox, oy, oz = origin[0], origin[2], origin[1]   # mpl order: X, Z, Y
    for vec, col in zip(frame, [C_AX_X, C_AX_Y, C_AX_Z]):
        dx, dz, dy = vec[0]*length, vec[2]*length, vec[1]*length
        ax.quiver(ox, oy, oz, dx, dz, dy,
                  color=col, length=1.0, normalize=False,
                  arrow_length_ratio=0.35,
                  linewidth=lw, alpha=alpha)


# ─────────────────────────────────────────────────────────────────────────────
#  PANEL LEFT  —  Egocentric video + annotation overlay + metadata
# ─────────────────────────────────────────────────────────────────────────────

def render_left_panel(
        frame: np.ndarray,
        norm_lms: list, handedness: list,
        W: int, H: int,
        skill: str, description: str,
        ts_start: float, ts_cur: float,
        nl_caption: str,
        environment: str, scene: str, op_height: str,
) -> np.ndarray:
    """
    Build the left panel as a BGR numpy array of shape (H, W, 3).

    Vertical sections:
      ┌─────────────────────────┐  ← 0
      │   Video frame (fills)   │
      │   + annotation card     │
      │   + 2D hand skeleton    │
      ├─────────────────────────┤  ← VIDEO_H
      │   NL caption            │  ← CAP_H = 36px
      ├─────────────────────────┤
      │   Metadata bar          │  ← META_H = 44px
      └─────────────────────────┘  ← H
    """
    CAP_H   = 36
    META_H  = 44
    VIDEO_H = H - CAP_H - META_H

    canvas = np.full((H, W, 3), 250, dtype=np.uint8)

    # ── 1. Egocentric video frame (fill, no letterbox) ────────────────────
    fh, fw = frame.shape[:2]
    sc  = max(W / fw, VIDEO_H / fh)
    nw  = int(fw * sc);  nh = int(fh * sc)
    res = cv2.resize(frame, (nw, nh))
    # Centre-crop
    x0c = max(0, (nw - W) // 2)
    y0c = max(0, (nh - VIDEO_H) // 2)
    crop = res[y0c: y0c + VIDEO_H, x0c: x0c + W]
    h_crop, w_crop = crop.shape[:2]
    canvas[0:h_crop, 0:w_crop] = crop

    # ── 2. 2-D hand skeleton overlay ─────────────────────────────────────
    # Landmarks are in [0,1] normalised coords relative to original frame
    for lms, label in zip(norm_lms, handedness):
        pts = [(int(lm[0] * W), int(lm[1] * VIDEO_H)) for lm in lms]
        hbgr = L_BGR if label == "Left" else R_BGR
        for a, b in HAND_CONN:
            cv2.line(canvas, pts[a], pts[b], hbgr, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            col = FINGER_BGR[LM_FINGER[i]]
            r   = 6 if i in FINGERTIPS else 4
            cv2.circle(canvas, pt, r + 1, (255,255,255), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, r,     col,           -1, cv2.LINE_AA)

    # ── 3. Annotation overlay card (top-left, Gap 1) ─────────────────────
    if skill or description:
        crd_w = min(W - 20, 320)
        crd_h = 68
        cx0, cy0 = 10, 10

        # White semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx0, cy0), (cx0+crd_w, cy0+crd_h),
                      (255,255,255), -1)
        cv2.addWeighted(overlay, 0.87, canvas, 0.13, 0, canvas)
        cv2.rectangle(canvas, (cx0, cy0), (cx0+crd_w, cy0+crd_h),
                      (200,200,200), 1)

        # "Skill:" label + value
        cv2.putText(canvas, "Skill:", (cx0+8, cy0+17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130,130,130),
                    1, cv2.LINE_AA)
        cv2.putText(canvas, skill or "", (cx0+52, cy0+17),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (20,20,20),
                    1, cv2.LINE_AA)

        # Description (wrapped)
        desc = (description or "")[:55]
        cv2.putText(canvas, desc, (cx0+8, cy0+35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (70,70,70),
                    1, cv2.LINE_AA)

        # Timestamps — ASCII only (OpenCV cannot render Unicode)
        ts_str = f"{fmt_ts(ts_start)}  ->  {fmt_ts(ts_cur)}"
        cv2.putText(canvas, ts_str, (cx0+8, cy0+54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100,100,100),
                    1, cv2.LINE_AA)

    # ── 4. NL caption row ────────────────────────────────────────────────
    cap_y0 = VIDEO_H
    cv2.rectangle(canvas, (0, cap_y0), (W, cap_y0+CAP_H), (238,238,238), -1)
    cv2.line(canvas, (0, cap_y0), (W, cap_y0), (210,210,210), 1)
    caption_lines = textwrap.wrap(nl_caption or "", width=W // 8) or [""]
    for li, line in enumerate(caption_lines[:2]):
        cv2.putText(canvas, line,
                    (12, cap_y0 + 14 + li * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (45,45,45),
                    1, cv2.LINE_AA)

    # ── 5. Metadata bar ───────────────────────────────────────────────────
    meta_y0 = VIDEO_H + CAP_H
    cv2.rectangle(canvas, (0, meta_y0), (W, H), (248,248,248), -1)
    cv2.line(canvas, (0, meta_y0), (W, meta_y0), (210,210,210), 1)

    col_w = W // 3
    for ci, (lbl, val) in enumerate([
        ("ENVIRONMENT", environment or "—"),
        ("SCENE",       scene        or "—"),
        ("OPERATOR HEIGHT", op_height or "—"),
    ]):
        lx = ci * col_w + 10
        cv2.putText(canvas, lbl,
                    (lx, meta_y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (140,140,140),
                    1, cv2.LINE_AA)
        cv2.putText(canvas, val,
                    (lx, meta_y0 + 33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.40, (30,30,30),
                    1, cv2.LINE_AA)
        if ci > 0:
            cv2.line(canvas,
                     (ci*col_w, meta_y0+5),
                     (ci*col_w, H-5),
                     (210,210,210), 1)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  PANEL TOP-RIGHT  —  6-DoF hand trajectory cloud + current skeleton
# ─────────────────────────────────────────────────────────────────────────────

def render_3d_hands(
        ax,
        current_hands: list,   # [(lms_sc_21x3, label), ...]
        cloud: dict,           # {'Left': [(x,y,z),...], 'Right': ...}
        show_arrows: bool = True,
) -> None:
    """
    Draw cumulative point cloud + current-frame skeleton + orientation arrows.
    Uses matplotlib 3D axes with coordinate order (X, Z, Y) so Y=up is screen-up.
    """
    ax.cla()
    ax.set_facecolor("white")

    # ── Cumulative trajectory cloud ───────────────────────────────────────
    # Each hand stored as raw world coords (not centred) so L and R are
    # spatially separated and show real movement range.
    for label, pts in cloud.items():
        if not pts:
            continue
        arr   = np.array(pts)
        color = C_LEFT if label == "Left" else C_RIGHT
        # Subsample for speed when cloud is huge
        step = max(1, len(arr) // 2000)
        sub  = arr[::step]
        ax.scatter(sub[:,0], sub[:,2], sub[:,1],   # (X, Z, Y)
                   s=5, c=color, alpha=0.18,
                   depthshade=False, zorder=1)

    # ── Current frame skeleton ────────────────────────────────────────────
    offsets = {"Left": np.array([-0.05, 0., 0.]),
               "Right": np.array([ 0.05, 0., 0.])}

    for lms_sc, label in current_hands:
        color = C_LEFT if label == "Left" else C_RIGHT
        off   = offsets.get(label, np.zeros(3))
        lm    = lms_sc + off   # slight X offset so L/R don't stack at origin

        # Bones
        for a, b in HAND_CONN:
            ax.plot([lm[a,0], lm[b,0]],
                    [lm[a,2], lm[b,2]],    # Z
                    [lm[a,1], lm[b,1]],    # Y (up)
                    color=color, lw=1.2, alpha=0.80, zorder=2)

        # Joint dots
        sizes = [55 if i in FINGERTIPS else (65 if i==0 else 26) for i in range(21)]
        ax.scatter(lm[:,0], lm[:,2], lm[:,1],
                   c=[color]*21, s=sizes,
                   edgecolors="white", linewidths=0.5,
                   alpha=0.95, depthshade=True, zorder=3)

        # ── 6-DoF orientation arrows at every finger joint ────────────────
        if show_arrows:
            segs = [(0,1),(1,2),(2,3),(3,4),
                    (5,6),(6,7),(7,8),
                    (9,10),(10,11),(11,12),
                    (13,14),(14,15),(15,16),
                    (17,18),(18,19),(19,20)]
            for a, b in segs:
                ja = lm[a];  jb = lm[b]
                mid = (ja + jb) / 2
                frm = local_frame(ja, jb)
                draw_axes_at(ax, mid, frm, length=0.018, lw=0.7, alpha=0.75)

    # ── Axis limits and styling ───────────────────────────────────────────
    lim = 0.14
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("X", fontsize=6, color="#666", labelpad=1)
    ax.set_ylabel("Z", fontsize=6, color="#666", labelpad=1)
    ax.set_zlabel("Y", fontsize=6, color="#666", labelpad=1)
    ax.tick_params(labelsize=5, colors="#AAA", pad=0.5)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#DDDDDD")
    ax.xaxis.line.set_color("#CCCCCC")
    ax.yaxis.line.set_color("#CCCCCC")
    ax.zaxis.line.set_color("#CCCCCC")
    ax.grid(True, color="#EEEEEE", linewidth=0.4, linestyle="--")
    ax.view_init(elev=22, azim=-55)
    ax.set_title("6-DoF hand trajectory", fontsize=8, color="#333",
                 pad=4, fontweight="semibold")
    patches = [mpatches.Patch(color=C_LEFT,  label="Left"),
               mpatches.Patch(color=C_RIGHT, label="Right"),
               mpatches.Patch(color=C_AX_X,  label="X"),
               mpatches.Patch(color=C_AX_Y,  label="Y"),
               mpatches.Patch(color=C_AX_Z,  label="Z")]
    ax.legend(handles=patches, loc="upper left", fontsize=5,
              framealpha=0.75, edgecolor="#CCCCCC", ncol=2)


# ─────────────────────────────────────────────────────────────────────────────
#  PANEL BOTTOM-RIGHT  —  Full-body skeleton + live hand terminator
# ─────────────────────────────────────────────────────────────────────────────

def render_pose_panel(
        ax,
        pose_arr: np.ndarray | None,
        hands_at_wrist: dict,
) -> None:
    """
    Draw locked upright body skeleton with live arm directions from pose model.
    Hand skeletons attach at wrist terminators.
    Tri-axial orientation arrows at every body and hand joint.

    Coordinate convention in this axes:
      mpl X  ←  skeleton X  (left/right)
      mpl Y  ←  skeleton Z  (depth)
      mpl Z  ←  skeleton Y  (up/down)    ← this is what keeps person upright
    """
    ax.cla()
    ax.set_facecolor("white")

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#DDDDDD")
        pane.set_alpha(0.4)
    ax.xaxis.line.set_color("#BBBBBB")
    ax.yaxis.line.set_color("#BBBBBB")
    ax.zaxis.line.set_color("#BBBBBB")
    ax.tick_params(labelsize=5, colors="#AAAAAA", pad=0.5)
    ax.grid(True, color="#EEEEEE", linewidth=0.4, linestyle="--")
    ax.set_xlabel("X", fontsize=6, color="#999", labelpad=0)
    ax.set_ylabel("Z", fontsize=6, color="#999", labelpad=0)
    ax.set_zlabel("Y", fontsize=6, color="#999", labelpad=0)

    # ── Build live skeleton (locked torso, live arms) ─────────────────────
    sk = {k: v.copy() for k, v in TPOSE.items()}

    if pose_arr is not None:
        ls = sk["l_shoulder"];  rs = sk["r_shoulder"]
        # Update elbows from pose direction, fixed upper-arm length
        sk["l_elbow"] = ls + scale_vec(pose_arr[P_LS], pose_arr[P_LE], _UA)
        sk["r_elbow"] = rs + scale_vec(pose_arr[P_RS], pose_arr[P_RE], _UA)
        # Update wrists from pose direction, fixed forearm length
        sk["l_wrist"] = sk["l_elbow"] + scale_vec(pose_arr[P_LE], pose_arr[P_LW], _FA)
        sk["r_wrist"] = sk["r_elbow"] + scale_vec(pose_arr[P_RE], pose_arr[P_RW], _FA)

    # ── Body bones (thin grey) ────────────────────────────────────────────
    for a_nm, b_nm in SKEL_BONES:
        a3 = sk[a_nm];  b3 = sk[b_nm]
        ax.plot([a3[0], b3[0]],
                [a3[2], b3[2]],    # Z→mpl Y
                [a3[1], b3[1]],    # Y→mpl Z
                color="#AAAAAA", lw=1.8, alpha=0.90, zorder=2,
                solid_capstyle="round")

    # ── Body joint orientation arrows ─────────────────────────────────────
    body_oriented = [
        ("l_shoulder","l_elbow"),("l_elbow","l_wrist"),
        ("r_shoulder","r_elbow"),("r_elbow","r_wrist"),
    ]
    for a_nm, b_nm in body_oriented:
        a3 = sk[a_nm];  b3 = sk[b_nm]
        mid = (a3 + b3) / 2
        frm = local_frame(a3, b3)
        draw_axes_at(ax, mid, frm, length=0.048, lw=1.1, alpha=0.88)

    # ── Body joint dots ───────────────────────────────────────────────────
    for nm, pt in sk.items():
        sz = 42 if nm == "nose" else 26
        ax.scatter([pt[0]], [pt[2]], [pt[1]],
                   s=sz, c="#666666", zorder=4,
                   edgecolors="white", linewidths=0.7, depthshade=False)

    # ── Hand terminators attached at wrist positions ──────────────────────
    # Scale hand to ~80% of forearm length in skeleton space
    hand_px_sc = (_FA * 0.80) / CUBE_HALF
    wrist_map  = {"Left": sk["l_wrist"], "Right": sk["r_wrist"]}
    hcol_map   = {"Left": C_LEFT,        "Right": C_RIGHT}

    for side, lms_sc in hands_at_wrist.items():
        w3  = wrist_map[side]   # (X, Y, Z) in skeleton space
        col = hcol_map[side]

        # lms_sc: col0=X(right), col1=Y(up), col2=Z(depth)
        # Skeleton space is same: X=right, Y=up, Z=depth
        hx = lms_sc[:,0] * hand_px_sc + w3[0]
        hy = lms_sc[:,1] * hand_px_sc + w3[1]
        hz = lms_sc[:,2] * hand_px_sc + w3[2]

        # Bones  (mpl order: X, Z→mplY, Y→mplZ)
        for a, b in HAND_CONN:
            ax.plot([hx[a], hx[b]],
                    [hz[a], hz[b]],    # skeleton Z → mpl Y axis
                    [hy[a], hy[b]],    # skeleton Y → mpl Z axis
                    color=col, lw=1.0, alpha=0.80, zorder=5)

        # Joint dots
        sizes = np.array([50 if i in FINGERTIPS else (55 if i==0 else 22)
                          for i in range(21)], dtype=float)
        ax.scatter(hx, hz, hy,
                   s=sizes, c=col,
                   edgecolors="white", linewidths=0.4,
                   alpha=0.92, depthshade=True, zorder=6)

        # ── Finger orientation arrows ──────────────────────────────────────
        # Drawn AFTER all scatter/plot to stay within axis limits
        finger_segs = [(0,1),(1,2),(2,3),(3,4),
                       (5,6),(6,7),(7,8),
                       (9,10),(10,11),(11,12),
                       (13,14),(14,15),(15,16),
                       (17,18),(18,19),(19,20)]
        for a, b in finger_segs:
            ja = np.array([hx[a], hy[a], hz[a]])
            jb = np.array([hx[b], hy[b], hz[b]])
            mid_world = (ja + jb) / 2
            frm = local_frame(ja, jb)
            # mid in (X,Y,Z) skeleton → pass as array to draw_axes_at
            draw_axes_at(ax, mid_world, frm, length=0.022, lw=0.65, alpha=0.75)

    # ── Set axis limits AFTER all drawing (quiver must be within limits) ──
    all_sk = np.array(list(sk.values()))
    cx     = all_sk[:,0].mean()
    cy     = all_sk[:,1].mean()
    span   = 0.82
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(-0.35, 0.35)        # depth: narrow
    ax.set_zlim(cy - span, cy + span)
    ax.view_init(elev=18, azim=-62)

    ax.set_title("Full-body 6-DoF skeleton", fontsize=8, color="#333",
                 pad=4, fontweight="semibold")
    patches = [
        mpatches.Patch(color="#AAAAAA", label="Body"),
        mpatches.Patch(color=C_LEFT,   label="Left hand"),
        mpatches.Patch(color=C_RIGHT,  label="Right hand"),
        mpatches.Patch(color=C_AX_X,   label="X axis"),
        mpatches.Patch(color=C_AX_Y,   label="Y axis"),
        mpatches.Patch(color=C_AX_Z,   label="Z axis"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=5,
              framealpha=0.75, edgecolor="#CCCCCC", ncol=2)


# ─────────────────────────────────────────────────────────────────────────────
#  Composite frame assembly (GridSpec)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_frame(
        frame: np.ndarray,
        norm_lms: list, handedness: list,
        current_hands: list, cloud: dict,
        pose_arr, hands_at_wrist: dict,
        fig, ax_cam, ax_3d, ax_pose,
        skill: str, description: str,
        ts_start: float, ts_cur: float,
        nl_caption: str,
        environment: str, scene: str, op_height: str,
) -> np.ndarray:
    """Build one output frame and return as BGR numpy array."""

    # Left panel: rendered in OpenCV, blitted into matplotlib axes
    pw = int(fig.get_figwidth() * DPI * 0.52)
    ph = int(fig.get_figheight() * DPI)
    left_img = render_left_panel(
        frame, norm_lms, handedness, pw, ph,
        skill, description, ts_start, ts_cur,
        nl_caption, environment, scene, op_height)

    ax_cam.cla()
    ax_cam.axis("off")
    ax_cam.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB), aspect="auto")

    # Top-right: 3D hand cloud
    render_3d_hands(ax_3d, current_hands, cloud, show_arrows=True)

    # Bottom-right: body pose + hand terminators
    render_pose_panel(ax_pose, pose_arr, hands_at_wrist)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
#  Main processing loop
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
        input_path: str,
        csv_path:   str   = "hand_6dof.csv",
        out_path:   str   = "vla_output.mp4",
        fig_w:      int   = 18,
        fig_h:      int   = 7,
        skip:       int   = 1,
        max_hands:  int   = 2,
        conf:       float = 0.55,
        skill:      str   = "",
        description:str   = "",
        nl_caption: str   = "",
        ts_start:   float = 0.0,
        ts_end:     float = 0.0,
        environment:str   = "Factory",
        scene:      str   = "Workstation",
        op_height:  str   = "—",
) -> None:

    hand_model = download_model(HAND_URL, "hand_landmarker.task")
    pose_model = download_model(POSE_URL, "pose_landmarker_lite.task")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps  = max(1.0, src_fps / skip)
    if ts_end <= ts_start:
        ts_end = total / src_fps   # auto: full duration

    # ── Persistent matplotlib figure ─────────────────────────────────────
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.15, 0.85],
        height_ratios=[1, 1],
        hspace=0.07, wspace=0.04,
        left=0.005, right=0.995, top=0.98, bottom=0.02,
    )
    ax_cam  = fig.add_subplot(gs[:, 0])              # full left column
    ax_3d   = fig.add_subplot(gs[0, 1], projection="3d")  # top-right 3D
    ax_pose = fig.add_subplot(gs[1, 1], projection="3d")  # bottom-right 3D

    OUT_W = int(fig_w * DPI)
    OUT_H = int(fig_h * DPI)

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (OUT_W, OUT_H))

    # ── MediaPipe detectors ───────────────────────────────────────────────
    hand_opts = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_opts = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Cumulative trajectory cloud (raw world coords, not centred)
    cloud: dict[str, list] = {"Left": [], "Right": []}

    csv_rows: list[list] = []
    csv_hdr = (
        ["frame", "hand", "ts_sec",
         "op_height", "environment", "scene"] +
        [f"x{i}" for i in range(21)] +
        [f"y{i}" for i in range(21)] +
        [f"z{i}" for i in range(21)]
    )

    with (mp_vision.HandLandmarker.create_from_options(hand_opts) as h_det,
          mp_vision.PoseLandmarker.create_from_options(pose_opts) as p_det):

        frame_idx = 0
        pbar = tqdm(total=total, unit="frame", desc="Processing")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)

            if frame_idx % skip != 0:
                frame_idx += 1
                continue

            ts_cur = frame_idx / src_fps
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int(ts_cur * 1000)

            # ── Run detectors ─────────────────────────────────────────────
            h_res = h_det.detect_for_video(mp_img, ts_ms)
            p_res = p_det.detect_for_video(mp_img, ts_ms)

            # ── Parse hand results ────────────────────────────────────────
            norm_lms_2d:   list = []
            handedness_2d: list = []
            current_hands: list = []   # (lms_sc, label)
            hands_at_wrist: dict = {}  # label → lms_sc

            if h_res.hand_world_landmarks:
                for slot, (wlm, nlm, hinfo) in enumerate(
                    zip(h_res.hand_world_landmarks,
                        h_res.hand_landmarks,
                        h_res.handedness)):
                    if slot >= max_hands:
                        break

                    label  = hinfo[0].display_name   # "Left" | "Right"
                    lms_w  = hand_to_world(wlm)      # (21,3) world, Y-flipped
                    lms_sc = centre_scale(lms_w)     # centred + scaled

                    norm_lms_2d.append(hand_to_norm(nlm))
                    handedness_2d.append(label)
                    current_hands.append((lms_sc, label))
                    hands_at_wrist[label] = lms_sc

                    # Accumulate raw world coords for trajectory cloud
                    # Raw coords preserve spatial separation of L/R hands
                    cloud[label].extend(lms_w.tolist())

                    # CSV row — operator height + env metadata logged per frame
                    row = [frame_idx, label, f"{ts_cur:.3f}",
                           op_height, environment, scene]
                    row += lms_w[:,0].tolist()
                    row += lms_w[:,1].tolist()
                    row += lms_w[:,2].tolist()
                    csv_rows.append(row)

            # ── Parse pose results ────────────────────────────────────────
            pose_arr = None
            if p_res.pose_world_landmarks:
                pose_arr = pose_to_world(p_res.pose_world_landmarks[0])

            # ── Compose output frame ──────────────────────────────────────
            out_frame = assemble_frame(
                frame, norm_lms_2d, handedness_2d,
                current_hands, cloud,
                pose_arr, hands_at_wrist,
                fig, ax_cam, ax_3d, ax_pose,
                skill, description,
                ts_start, ts_cur,
                nl_caption, environment, scene, op_height,
            )

            out_frame = cv2.resize(out_frame, (OUT_W, OUT_H))
            writer.write(out_frame)
            frame_idx += 1

        pbar.close()

    cap.release()
    writer.release()
    plt.close(fig)

    # ── Save CSV ──────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_hdr)
        w.writerows(csv_rows)

    print(f"\nDone.")
    print(f"  Video : {out_path}  ({OUT_W}x{OUT_H} @ {out_fps:.1f}fps)")
    print(f"  CSV   : {csv_path}  ({len(csv_rows):,} rows)")
    print(f"  Cloud : Left={len(cloud['Left'])//21} frames, "
          f"Right={len(cloud['Right'])//21} frames")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="6-DoF Embodied AI Data Collection Visualizer")
    ap.add_argument("--input",       required=True,
                    help="Input video path (e.g. factory.mp4)")
    ap.add_argument("--csv",         default="hand_6dof.csv")
    ap.add_argument("--output",      default="vla_output.mp4")
    ap.add_argument("--fig_w",       type=int,   default=18,
                    help="Figure width in inches (default 18 → 1800px)")
    ap.add_argument("--fig_h",       type=int,   default=7,
                    help="Figure height in inches (default 7 → 700px)")
    ap.add_argument("--skip",        type=int,   default=1,
                    help="Process every Nth frame (2=half speed, faster)")
    ap.add_argument("--hands",       type=int,   default=2)
    ap.add_argument("--conf",        type=float, default=0.55)
    # Annotation card (micro-level action label)
    ap.add_argument("--skill",       default="",
                    help='Short-horizon action label  e.g. "Inflate tire"')
    ap.add_argument("--description", default="",
                    help='One-line skill description for the card')
    # Caption (macro-level episode label)
    ap.add_argument("--nl_caption",  default="",
                    help='Third-person NL caption below the video')
    # Temporal segmentation
    ap.add_argument("--ts_start",    type=float, default=0.0,
                    help="Clip start time in seconds")
    ap.add_argument("--ts_end",      type=float, default=0.0,
                    help="Clip end time in seconds (0 = auto from duration)")
    # Metadata for operator height normalisation + context embeddings
    ap.add_argument("--environment", default="Factory",
                    help="Macro environment tag  e.g. Car Workshop")
    ap.add_argument("--scene",       default="Workstation",
                    help="Micro scene tag  e.g. Press Station")
    ap.add_argument("--op_height",   default="—",
                    help="Operator height for spatial normalisation  e.g. 170cm")

    a = ap.parse_args()
    process_video(
        input_path  = a.input,
        csv_path    = a.csv,
        out_path    = a.output,
        fig_w       = a.fig_w,
        fig_h       = a.fig_h,
        skip        = a.skip,
        max_hands   = a.hands,
        conf        = a.conf,
        skill       = a.skill,
        description = a.description,
        nl_caption  = a.nl_caption,
        ts_start    = a.ts_start,
        ts_end      = a.ts_end,
        environment = a.environment,
        scene       = a.scene,
        op_height   = a.op_height,
    )
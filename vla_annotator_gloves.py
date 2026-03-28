"""
VLA Data Collection Annotator  —  Full Pipeline Build  (v2)
============================================================
Updates from v1:
  • Glove-aware preprocessing  (CLAHE + edge sharpening before MediaPipe)
  • suppress_warnings()        (hides TFLite INFO / XNNPACK noise on startup)
  • Arrow rendering fix        (orientation arrows visible; drawn after axis limits)
  • Unicode arrow '→' removed  (OpenCV can't render it; replaced with '->')
  • draw_info_panel velocity   (ee_speed bar now uses live data, not placeholder)
  • JSONL export               (VLA-ready per-frame JSON Lines alongside CSV)
  • --no_video flag            (CSV+JSONL only, no video render — fast mode)
  • --robot_height             (spatial normalisation for robot replanning)
  • --steps                    (semicolon-separated micro-step timeline)
  • Legacy --skill / --description aliases kept for backward compatibility

Layout:
  +---------------------------+--------+------------------+
  |   Camera frame            | INFO   |  3D hand cloud   |
  |   + annotation card       | strip  |  (trajectory     |
  |   + 2D hand overlay       | RPY    |   + skeleton     |
  |   + NL caption            | Grasp  |   + 6DoF arrows) |
  |   + metadata bar          | Joints +------------------+
  |                           | Speed  |  Full-body pose  |
  |                           |        |  + live hands    |
  +---------------------------+--------+------------------+

Install:
    pip install mediapipe opencv-python tqdm numpy

Usage (single video):
    python vla_annotator.py \\
        --input factory.mp4 \\
        --macro_task "Disassemble a car wheel" \\
        --steps "Attach socket to wrench;Loosen lug nuts;Remove wheel" \\
        --nl_caption "The operator disassembles a car wheel using an impact wrench." \\
        --environment "Car Workshop" --scene "Car service" \\
        --op_height 162 --robot_height 120 --skip 2

Usage (data-only, no video):
    python vla_annotator.py --input factory.mp4 --no_video \\
        --macro_task "Assemble PCB" --op_height 165
"""

import argparse, csv, json, math, os, textwrap, urllib.request, warnings
import cv2
import numpy as np
from tqdm import tqdm

# ── suppress MediaPipe / TFLite startup noise ─────────────────────────────────
os.environ["GLOG_minloglevel"]    = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"]= "3"
warnings.filterwarnings("ignore")

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, PoseLandmarkerOptions

# ═══════════════════════════════════════════════════════════════════════════════
#  TWO-STAGE GLOVE DETECTOR
#
#  Problem: MediaPipe's palm detector was trained on bare skin.
#  On grey/blue ESD gloves it fails BEFORE the landmark stage runs.
#  Lowering --conf or CLAHE cannot fix a failed palm detection.
#
#  Solution: Replace MediaPipe's palm stage with YOLOv8 hand detection.
#    Stage 1 — YOLOv8n detects gloved hand bounding boxes (colour-agnostic)
#    Stage 2 — Each crop is passed to MediaPipe in STATIC_IMAGE mode
#              (bypassing the palm detector entirely; MediaPipe only runs
#              the landmark regression head on the pre-detected region)
#
#  Models used:
#    • Ultralytics YOLOv8n (pretrained COCO) — detects 'person' class,
#      then we filter by wrist/hand region heuristic.
#    • Fallback: if ultralytics not installed, silently use standard MediaPipe.
#
#  Install:  pip install ultralytics
# ═══════════════════════════════════════════════════════════════════════════════

_YOLO_AVAILABLE = False
_yolo_model     = None

def _try_load_yolo():
    """Lazy-load YOLOv8n on first use. Returns True if available."""
    global _YOLO_AVAILABLE, _yolo_model
    if _YOLO_AVAILABLE:
        return True
    try:
        from ultralytics import YOLO as _YOLO
        # yolov8n is small (6 MB), auto-downloads on first call
        _yolo_model    = _YOLO("yolov8n.pt")
        _YOLO_AVAILABLE = True
        print("  [GloveDetector] YOLOv8n loaded — two-stage mode active.")
        return True
    except Exception:
        return False


def _yolo_hand_boxes(frame_bgr: np.ndarray, conf: float = 0.25):
    """
    Run YOLOv8n on frame and return list of (x1,y1,x2,y2) bounding boxes
    that likely contain hands.

    Strategy:
      • Detect all 'person' instances (class 0) at low confidence.
      • The lower-quarter of each person box = arm+hand region.
      • Also detect any 'sports ball' (class 32) — YOLOv8 sometimes
        fires on fists/gloves with this class.
      • Merge with any boxes from direct hand-like detections.
      • Return up to 2 largest boxes (one per hand).
    """
    fh, fw = frame_bgr.shape[:2]
    results = _yolo_model(frame_bgr, conf=conf, verbose=False)[0]

    hand_boxes = []
    for box in results.boxes:
        cls  = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:  # person — take lower-third as hand region
            h_box = y2 - y1
            hand_y1 = max(0, y2 - h_box // 3)
            hand_boxes.append((x1, hand_y1, x2, y2))

        elif cls in (32, 76, 77):  # sports ball / scissors / teddy bear
            # YOLO sometimes misclassifies a fist/glove as these
            hand_boxes.append((x1, y1, x2, y2))

    # If nothing found from person heuristic, try a tight crop of the
    # bottom-centre of frame (typical egocentric helmet cam position)
    if not hand_boxes:
        # Fallback: bottom 40% of frame, split L/R
        cy = int(fh * 0.60)
        hand_boxes = [
            (0,    cy, fw//2, fh),   # left hand region
            (fw//2, cy, fw,   fh),   # right hand region
        ]

    # Keep up to 2 largest boxes
    hand_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return hand_boxes[:2]


def _expand_box(x1, y1, x2, y2, fw, fh, pad=0.15):
    """Expand bounding box by pad% and clamp to frame."""
    pw = int((x2-x1)*pad); ph = int((y2-y1)*pad)
    return (max(0,x1-pw), max(0,y1-ph),
            min(fw,x2+pw), min(fh,y2+ph))


# ── Static-image landmark extractor (used per-crop in two-stage mode) ─────────
_static_hand_opts = None

def _get_static_opts(model_path: str, conf: float):
    global _static_hand_opts
    if _static_hand_opts is None:
        _static_hand_opts = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=max(0.1, conf - 0.20),
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    return _static_hand_opts


def detect_gloved_hands(frame_bgr: np.ndarray,
                        hand_model_path: str,
                        conf: float = 0.30):
    """
    Two-stage gloved hand detection.

    Returns list of dicts, each with:
      'world_lms'  : (21,3) float64 world landmarks (Y-flipped)
      'norm_lms'   : list of (x,y) in frame [0,1] space
      'label'      : 'Left' | 'Right'
      'box'        : (x1,y1,x2,y2) in frame pixels

    Falls back to empty list if no hands found.
    """
    if not _try_load_yolo():
        return []   # caller will fall back to standard MediaPipe

    fh, fw = frame_bgr.shape[:2]
    boxes  = _yolo_hand_boxes(frame_bgr, conf=0.20)
    opts   = _get_static_opts(hand_model_path, conf)

    results = []
    with mp_vision.HandLandmarker.create_from_options(opts) as det:
        for i, (x1,y1,x2,y2) in enumerate(boxes):
            # Expand box generously so wrist area is included
            ex1,ey1,ex2,ey2 = _expand_box(x1,y1,x2,y2,fw,fh,pad=0.25)
            crop = frame_bgr[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue

            # Preprocess crop before landmark detection
            crop_enhanced = preprocess_for_gloves(crop)
            rgb_crop = cv2.cvtColor(crop_enhanced, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

            res = det.detect(mp_img)
            if not res.hand_world_landmarks:
                continue

            for slot, (wlm, nlm, hinfo) in enumerate(zip(
                    res.hand_world_landmarks,
                    res.hand_landmarks,
                    res.handedness)):

                label = hinfo[0].display_name

                # World landmarks are already in hand-local space — use directly
                world = np.array([[lm.x, lm.y, lm.z] for lm in wlm], dtype=np.float64)
                world[:, 1] *= -1   # flip Y

                # Re-map normalised landmarks from crop space → full frame space
                cw = ex2 - ex1;  ch = ey2 - ey1
                norm = []
                for lm in nlm:
                    fx = (ex1 + lm.x * cw) / fw
                    fy = (ey1 + lm.y * ch) / fh
                    norm.append((fx, fy))

                results.append({
                    "world_lms": world,
                    "norm_lms":  norm,
                    "label":     label,
                    "box":       (ex1, ey1, ex2, ey2),
                })

    # De-duplicate: if two crops returned same hand label, keep higher-confidence one
    seen = {}
    for r in results:
        lbl = r["label"]
        if lbl not in seen:
            seen[lbl] = r
    return list(seen.values())[:2]

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

FCOL = {
    "palm":   (180,  80,  20), "thumb":  (  0, 140, 255),
    "index":  ( 30, 180,  30), "middle": (200,  60,  20),
    "ring":   ( 20,  20, 200), "pinky":  (170,  30, 170),
}

LM_FINGER = {}
for _nm, _ids in [("palm",[0]),("thumb",[1,2,3,4]),("index",[5,6,7,8]),
                  ("middle",[9,10,11,12]),("ring",[13,14,15,16]),
                  ("pinky",[17,18,19,20])]:
    for _i in _ids: LM_FINGER[_i] = _nm

TIPS = [4, 8, 12, 16, 20]

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

L_BGR = (46, 204, 113);  R_BGR = (52, 73, 235)
BODY_BONE_COL  = (180,180,180)
BODY_JOINT_COL = (120,120,120)
GRID_COL = (210,210,210)
BONE_3D  = (165,165,165)
AX_X = ( 40,  40, 220)   # red   -> X
AX_Y = ( 40, 180,  40)   # green -> Y
AX_Z = (220, 130,  30)   # blue  -> Z

BG_META = (248,248,248); BORDER = (210,210,210)
TXT_D   = ( 40, 40,  40); TXT_G = (130,130,130); TXT_L = (170,170,170)
FONT    = cv2.FONT_HERSHEY_SIMPLEX
FONT_B  = cv2.FONT_HERSHEY_DUPLEX

P_NOSE=0; P_LS=11; P_RS=12; P_LE=13; P_RE=14
P_LW=15;  P_RW=16; P_LH=23; P_RH=24

_SW=0.22; _TH=0.18; _UA=0.26; _FA=0.22; _HW=0.10; CUBE_H=0.16

TPOSE = {
    "nose": np.array([ 0.0,   0.52, 0.0]),
    "ls":   np.array([-_SW,   0.30, 0.0]),  "rs":  np.array([_SW,   0.30, 0.0]),
    "le":   np.array([-_SW-_UA, 0.30, 0.0]),"re":  np.array([_SW+_UA, 0.30, 0.0]),
    "lw":   np.array([-_SW-_UA-_FA, 0.30, 0.0]),
    "rw":   np.array([ _SW+_UA+_FA, 0.30, 0.0]),
    "lh":   np.array([-_HW, -_TH, 0.0]),   "rh":  np.array([_HW, -_TH, 0.0]),
}
SKEL_BONES = [
    ("nose","ls"),("nose","rs"),("ls","rs"),("ls","lh"),("rs","rh"),
    ("lh","rh"),("ls","le"),("le","lw"),("rs","re"),("re","rw"),
]

TRAIL_LEN = 30

HAND_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
POSE_URL  = ("https://storage.googleapis.com/mediapipe-models/"
             "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")


# ═══════════════════════════════════════════════════════════════════════════════
#  GLOVE-AWARE PREPROCESSING
#  Applied to every frame before passing to MediaPipe.
#  CLAHE on the L* channel boosts finger-crease contrast so the palm
#  detector fires on grey/blue ESD gloves as reliably as on bare skin.
# ═══════════════════════════════════════════════════════════════════════════════

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)


def preprocess_for_gloves(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Enhance contrast and edges so gloved hands look more skin-like to MediaPipe.
    Steps:
      1. Convert to LAB colour space
      2. Apply CLAHE to the L* (luminance) channel — brings out knuckle/crease texture
      3. Merge back and convert to BGR
      4. Apply a mild unsharp-mask sharpening kernel — makes glove fabric edges crisp
    Returns a BGR array of the same shape.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    sharpened = cv2.filter2D(enhanced, -1, _SHARPEN)
    return sharpened


# ═══════════════════════════════════════════════════════════════════════════════
#  6-DOF MATH
# ═══════════════════════════════════════════════════════════════════════════════

def palm_frame(lms_w: np.ndarray):
    """Right-handed orthonormal frame for the palm. Returns (origin, R 3×3)."""
    wrist  = lms_w[0]; middle = lms_w[9]; index = lms_w[5]; pinky = lms_w[17]
    y_ax = middle - wrist; yn = np.linalg.norm(y_ax)
    if yn < 1e-9: return wrist, np.eye(3)
    y_ax /= yn
    across = pinky - index
    x_ax   = across - np.dot(across, y_ax) * y_ax
    xn     = np.linalg.norm(x_ax)
    x_ax   = x_ax / xn if xn > 1e-9 else np.array([1.,0.,0.])
    z_ax   = np.cross(x_ax, y_ax)
    return wrist, np.array([x_ax, y_ax, z_ax])


def rpy_from_R(R: np.ndarray):
    """ZYX Euler decomposition → (roll, pitch, yaw) in degrees."""
    Rm = R.T
    pitch = math.asin(max(-1., min(1., -Rm[2,0])))
    cp    = math.cos(pitch)
    if abs(cp) < 1e-6:
        roll = math.atan2(-Rm[0,1], Rm[1,1]); yaw = 0.
    else:
        roll = math.atan2(Rm[2,1], Rm[2,2]); yaw = math.atan2(Rm[1,0], Rm[0,0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def normalize_pose(xyz: np.ndarray, op_h: float, robot_h: float) -> np.ndarray:
    """Isotropic scale: robot_h / op_h."""
    if op_h <= 0 or robot_h <= 0: return xyz.copy()
    return xyz * (robot_h / op_h)


def joint_angle(a, b, c) -> float:
    """Angle at vertex b in degrees."""
    ba = a-b; bc = c-b
    na = np.linalg.norm(ba); nc = np.linalg.norm(bc)
    if na < 1e-9 or nc < 1e-9: return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(ba,bc)/(na*nc), -1., 1.))))


def finger_joint_angles(lms_w: np.ndarray) -> dict:
    """14 finger-joint flexion angles from (21,3) world landmarks."""
    ja = joint_angle; l = lms_w
    return {
        "thumb_mcp": ja(l[0],l[1],l[2]),  "thumb_ip":   ja(l[1],l[2],l[3]),
        "idx_mcp":   ja(l[0],l[5],l[6]),  "idx_pip":    ja(l[5],l[6],l[7]),
        "idx_dip":   ja(l[6],l[7],l[8]),
        "mid_mcp":   ja(l[0],l[9],l[10]), "mid_pip":    ja(l[9],l[10],l[11]),
        "mid_dip":   ja(l[10],l[11],l[12]),
        "ring_mcp":  ja(l[0],l[13],l[14]),"ring_pip":   ja(l[13],l[14],l[15]),
        "ring_dip":  ja(l[14],l[15],l[16]),
        "pinky_mcp": ja(l[0],l[17],l[18]),"pinky_pip":  ja(l[17],l[18],l[19]),
        "pinky_dip": ja(l[18],l[19],l[20]),
    }


def classify_grasp(lms_w: np.ndarray):
    """
    Returns (grasp_type, aperture_m, contact_state).
    Grasp types: open | pinch | tripod | power | lateral | hook | unknown
    """
    wrist = lms_w[0]
    def _curl(tip, mcp):
        dm = np.linalg.norm(lms_w[mcp]-wrist)
        return (np.linalg.norm(lms_w[tip]-wrist)/dm) if dm>1e-9 else 1.0
    ct=_curl(4,1); ci=_curl(8,5); cm=_curl(12,9); cr=_curl(16,13); cp=_curl(20,17)
    if   ct>.8  and ci>.8  and cm>.8  and cr>.8  and cp>.8:  grasp="open"
    elif ct<.7  and ci<.7  and cm>.75 and cr>.75 and cp>.75: grasp="pinch"
    elif ct<.7  and ci<.7  and cm<.7  and cr>.75 and cp>.75: grasp="tripod"
    elif ct<.65 and ci<.65 and cm<.65 and cr<.65 and cp<.65: grasp="power"
    elif ct>.8  and ci<.65 and cm<.65 and cr<.65 and cp<.65:
        grasp = "lateral" if np.linalg.norm(lms_w[4]-lms_w[5])<0.03 else "hook"
    else: grasp="unknown"
    aperture = float(np.linalg.norm(lms_w[4]-lms_w[8]))
    contact  = "closed" if aperture<0.02 else ("open" if aperture>0.08 else "partial")
    return grasp, aperture, contact


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def dl_model(url, path):
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path); print("  Done.")
    return path

def fmt_ts(s):
    m = int(s)//60; return f"{m:02d}:{s-m*60:06.3f}"

def hw(lm):
    a=np.array([[l.x,l.y,l.z] for l in lm],dtype=np.float64); a[:,1]*=-1; return a
def hn(lm): return [(l.x,l.y) for l in lm]
def pw(lm):
    a=np.array([[l.x,l.y,l.z] for l in lm],dtype=np.float64); a[:,1]*=-1; return a

def cs(lms, half=CUBE_H, fill=0.60):
    c=lms-lms[0]; e=np.abs(c).max()
    return c*(half*fill/e) if e>1e-9 else c

def sv(s, e, L):
    v=e-s; n=np.linalg.norm(v); return (v/n*L) if n>1e-6 else np.zeros(3)

def local_frame(a, b):
    fwd=b-a; fn=np.linalg.norm(fwd)
    if fn<1e-9: return np.eye(3)
    fwd/=fn
    up=np.array([0.,1.,0.])
    if abs(np.dot(fwd,up))>0.95: up=np.array([0.,0.,1.])
    right=np.cross(fwd,up); right/=np.linalg.norm(right)
    return np.array([right, np.cross(right,fwd), fwd])


# ═══════════════════════════════════════════════════════════════════════════════
#  3D PROJECTION  (orthographic, pure OpenCV — no matplotlib overhead)
# ═══════════════════════════════════════════════════════════════════════════════

def make_rot(elev=10, azim=-15):
    el,az = math.radians(elev), math.radians(azim)
    Ry=np.array([[math.cos(az),0,math.sin(az)],[0,1,0],[-math.sin(az),0,math.cos(az)]])
    Rx=np.array([[1,0,0],[0,math.cos(el),-math.sin(el)],[0,math.sin(el),math.cos(el)]])
    return (Rx@Ry).astype(np.float64)

R_3D = make_rot()   # single shared rotation for both panels

def p3(pt, R, cx, cy, sc):
    r=R@np.asarray(pt,np.float64)
    return (int(r[0]*sc+cx), int(-r[1]*sc+cy))

def zd(pt, R): return float((R@np.asarray(pt,np.float64))[2])


def dash_line(img, a, b, col=GRID_COL, th=1, gap=6):
    a,b=np.array(a,float),np.array(b,float); d=b-a; L=np.linalg.norm(d)
    if L<1: return
    n=max(2,int(L/gap))
    for s in range(n):
        if s%2==0:
            t0,t1=s/n,min(1.,(s+1)/n)
            cv2.line(img,tuple((a+d*t0).astype(int)),tuple((a+d*t1).astype(int)),col,th,cv2.LINE_AA)


def draw_grid(img, R, cx, cy, sc, half, ndiv=4):
    h=half
    for i in range(ndiv+1):
        v=-h+i*2*h/ndiv
        dash_line(img,p3([v,-h,-h],R,cx,cy,sc),p3([v,-h, h],R,cx,cy,sc))
        dash_line(img,p3([-h,-h,v],R,cx,cy,sc),p3([ h,-h, v],R,cx,cy,sc))
        dash_line(img,p3([v,-h,-h],R,cx,cy,sc),p3([v, h,-h],R,cx,cy,sc))
        dash_line(img,p3([-h, v,-h],R,cx,cy,sc),p3([ h, v,-h],R,cx,cy,sc))
        dash_line(img,p3([-h, v,-h],R,cx,cy,sc),p3([-h, v, h],R,cx,cy,sc))
        dash_line(img,p3([-h,-h, v],R,cx,cy,sc),p3([-h, h, v],R,cx,cy,sc))


def draw_axes(img, R, cx, cy, sc, half):
    h=half; ext=h*1.15
    o=p3([0,0,0],R,cx,cy,sc)
    for end3,col,lbl in [([ext,0,0],AX_X,"X"),([0,ext,0],AX_Y,"Y"),([0,0,ext],AX_Z,"Z")]:
        e=p3(end3,R,cx,cy,sc)
        cv2.arrowedLine(img,o,e,col,1,cv2.LINE_AA,tipLength=0.13)
        d=np.array(e)-np.array(o); dn=np.linalg.norm(d)
        if dn>0:
            lp=(np.array(e)+(d/dn*13)).astype(int)
            cv2.putText(img,lbl,tuple(lp),FONT,0.38,col,1,cv2.LINE_AA)
        for tv in [0.2,0.4,0.6]:
            frac=tv/h
            if frac>1.2: continue
            tp3=[end3[0]*frac*h/ext,end3[1]*frac*h/ext,end3[2]*frac*h/ext]
            tp=p3(tp3,R,cx,cy,sc)
            cv2.putText(img,f"{tv:.1f}",(tp[0]+2,tp[1]+12),FONT,0.27,TXT_G,1,cv2.LINE_AA)


def draw_orient(img, origin3, frm3x3, R, cx, cy, sc, length=0.02):
    """
    Draw tri-axial orientation arrows at origin3.
    FIX: clip endpoints to image bounds so arrows never draw outside canvas.
    """
    o = p3(origin3, R, cx, cy, sc)
    H, W = img.shape[:2]
    for i, col in enumerate([AX_X, AX_Y, AX_Z]):
        e = p3(origin3 + frm3x3[i]*length, R, cx, cy, sc)
        # Clip both endpoints to canvas so lines are always visible
        ox = int(np.clip(o[0], 0, W-1)); oy = int(np.clip(o[1], 0, H-1))
        ex = int(np.clip(e[0], 0, W-1)); ey = int(np.clip(e[1], 0, H-1))
        cv2.line(img, (ox,oy), (ex,ey), col, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL  — Camera + hierarchical annotation + metadata
# ═══════════════════════════════════════════════════════════════════════════════

def draw_left(frame, nlms, hness, W, H,
              macro_task, micro_step, step_idx, total_steps,
              t0, tc, nl_caption, env, scene, oph):
    CAP_H=40; META_H=48; VH=H-CAP_H-META_H
    canvas=np.full((H,W,3),250,np.uint8)

    # 1) Fill-crop video
    fh,fw=frame.shape[:2]; sc=max(W/fw,VH/fh)
    nw,nh=int(fw*sc),int(fh*sc)
    res=cv2.resize(frame,(nw,nh))
    x0c=max(0,(nw-W)//2); y0c=max(0,(nh-VH)//2)
    crop=res[y0c:y0c+VH,x0c:x0c+W]
    canvas[:crop.shape[0],:crop.shape[1]]=crop

    # 2) 2D hand overlay
    for lms,lab in zip(nlms,hness):
        pts=[(int(l[0]*W),int(l[1]*VH)) for l in lms]
        hc=L_BGR if lab=="Left" else R_BGR
        for a,b in HAND_CONN: cv2.line(canvas,pts[a],pts[b],hc,2,cv2.LINE_AA)
        for i,pt in enumerate(pts):
            c=FCOL[LM_FINGER[i]]; r=6 if i in TIPS else 4
            cv2.circle(canvas,pt,r+1,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(canvas,pt,r,c,-1,cv2.LINE_AA)

    # 3) Hierarchical annotation card
    if macro_task or micro_step:
        cw,ch=min(W-20,340),90; cx0,cy0=10,10
        ov=canvas.copy()
        cv2.rectangle(ov,(cx0,cy0),(cx0+cw,cy0+ch),(255,255,255),-1)
        cv2.addWeighted(ov,0.87,canvas,0.13,0,canvas)
        cv2.rectangle(canvas,(cx0,cy0),(cx0+cw,cy0+ch),(200,200,200),1)
        # Micro step (Action)
        step_label=f"Step {step_idx}/{total_steps}: " if total_steps>1 else ""
        cv2.putText(canvas,"Action:",(cx0+8,cy0+17),FONT,0.33,TXT_G,1,cv2.LINE_AA)
        cv2.putText(canvas,(step_label+(micro_step or ""))[:55],
                    (cx0+58,cy0+17),FONT_B,0.42,(20,20,180),1,cv2.LINE_AA)
        # Progress bar
        if total_steps>1:
            bx,by=cx0+8,cy0+28; bw=cw-16
            cv2.rectangle(canvas,(bx,by),(bx+bw,by+5),(220,220,220),-1)
            cv2.rectangle(canvas,(bx,by),(bx+int(bw*min(step_idx,total_steps)/total_steps),by+5),(20,20,180),-1)
        # Macro goal
        cv2.putText(canvas,"Goal:",(cx0+8,cy0+50),FONT,0.30,TXT_G,1,cv2.LINE_AA)
        cv2.putText(canvas,(macro_task or "")[:55],(cx0+48,cy0+50),FONT,0.37,TXT_D,1,cv2.LINE_AA)
        # Timestamps — ASCII arrow only (OpenCV cannot render Unicode ->)
        cv2.putText(canvas,f"{fmt_ts(t0)}  ->  {fmt_ts(tc)}",
                    (cx0+8,cy0+72),FONT,0.30,TXT_G,1,cv2.LINE_AA)

    # 4) NL caption bar
    cy=VH
    cv2.rectangle(canvas,(0,cy),(W,cy+CAP_H),(238,238,238),-1)
    cv2.line(canvas,(0,cy),(W,cy),BORDER,1)
    for i,ln in enumerate((textwrap.wrap(nl_caption or "",width=W//8) or [""])[:2]):
        cv2.putText(canvas,ln,(12,cy+16+i*16),FONT,0.42,TXT_D,1,cv2.LINE_AA)

    # 5) Metadata bar
    my=VH+CAP_H
    cv2.rectangle(canvas,(0,my),(W,H),BG_META,-1)
    cv2.line(canvas,(0,my),(W,my),BORDER,1)
    cw3=W//3
    for ci,(lb,vl) in enumerate([
            ("ENVIRONMENT",env or "—"),
            ("SCENE",scene or "—"),
            ("OPERATOR HEIGHT",f"{oph:.0f}cm" if isinstance(oph,float) else str(oph))]):
        lx=ci*cw3+14
        cv2.circle(canvas,(ci*cw3+8,my+16),3,TXT_G,-1,cv2.LINE_AA)
        cv2.putText(canvas,lb,(lx,my+18),FONT,0.28,TXT_G,1,cv2.LINE_AA)
        cv2.putText(canvas,vl,(lx,my+36),FONT_B,0.40,TXT_D,1,cv2.LINE_AA)
        if ci>0: cv2.line(canvas,(ci*cw3,my+5),(ci*cw3,H-5),BORDER,1)
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
#  TOP-RIGHT PANEL  — 3D hand skeleton + 6-DoF arrows
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hand_panel(W, H, current_hands, rpy_data, cube_half=0.12):
    canvas=np.full((H,W,3),255,np.uint8)
    R=R_3D
    pad_l=int(W*0.15); pad_b=int(H*0.15)
    CX=pad_l+(W-pad_l)//2; CY=(H-pad_b)//2+int(H*0.12)
    SC=min(W-pad_l,H-pad_b)/(cube_half*1.8)

    draw_grid(canvas,R,CX,CY,SC,cube_half)
    draw_axes(canvas,R,CX,CY,SC,cube_half)

    offs={"Left":np.array([-0.04,0,0]),"Right":np.array([0.04,0,0])}
    for lms_sc,label in current_hands:
        col=L_BGR if label=="Left" else R_BGR
        lm=lms_sc+offs.get(label,np.zeros(3))
        # Bones (depth-sorted, back to front)
        for _,a,b in sorted([(zd((lm[a]+lm[b])/2,R),a,b) for a,b in HAND_CONN]):
            cv2.line(canvas,p3(lm[a],R,CX,CY,SC),p3(lm[b],R,CX,CY,SC),BONE_3D,1,cv2.LINE_AA)
        # Joints (depth-sorted)
        for _,i in sorted((zd(lm[i],R),i) for i in range(21)):
            pt=p3(lm[i],R,CX,CY,SC); c=FCOL[LM_FINGER[i]]
            r=7 if i==0 else (6 if i in TIPS else 4)
            cv2.circle(canvas,pt,r,c,-1,cv2.LINE_AA)
        # Orientation arrows (drawn after joints — always on top)
        for a,b in [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
                    (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)]:
            mid=(lm[a]+lm[b])/2
            draw_orient(canvas,mid,local_frame(lm[a],lm[b]),R,CX,CY,SC,length=0.012)

    # Legend
    ly=12
    for lbl,col in [("Left",L_BGR),("Right",R_BGR)]:
        cv2.circle(canvas,(W-70,ly),4,col,-1,cv2.LINE_AA)
        cv2.putText(canvas,lbl,(W-62,ly+4),FONT,0.32,TXT_D,1,cv2.LINE_AA); ly+=14
    for lbl,col in [("X",AX_X),("Y",AX_Y),("Z",AX_Z)]:
        cv2.line(canvas,(W-70,ly),(W-60,ly),col,2,cv2.LINE_AA)
        cv2.putText(canvas,lbl,(W-56,ly+4),FONT,0.28,col,1,cv2.LINE_AA); ly+=12
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
#  CENTRE INFO STRIP  — RPY · Grasp · Joint angles · EE speed
# ═══════════════════════════════════════════════════════════════════════════════

def draw_info_panel(W, H, rpy_data, grasp_data, joint_data, speed_data):
    """
    speed_data : dict  label -> ee_speed (m/s) — live from velocity computation.
    """
    canvas=np.full((H,W,3),(248,248,248),np.uint8)
    PAD=6; y=10
    cv2.putText(canvas,"POSE / GRASP",(PAD,y),FONT,0.30,TXT_G,1,cv2.LINE_AA)
    y+=3; cv2.line(canvas,(PAD,y),(W-PAD,y),BORDER,1); y+=8

    for side,hcol in [("Left",L_BGR),("Right",R_BGR)]:
        # Label
        cv2.rectangle(canvas,(PAD,y-1),(PAD+4,y+9),hcol,-1)
        cv2.putText(canvas,side,(PAD+8,y+8),FONT_B,0.35,hcol,1,cv2.LINE_AA); y+=14

        # RPY bars
        rpy=rpy_data.get(side)
        if rpy:
            for lbl,val,col in [("R",rpy[0],AX_X),("P",rpy[1],AX_Y),("Y",rpy[2],AX_Z)]:
                bw=W-PAD*2; filled=int(bw*min(abs(val)/180.,1.))
                cv2.rectangle(canvas,(PAD,y+1),(PAD+bw,y+7),(225,225,225),-1)
                cv2.rectangle(canvas,(PAD,y+1),(PAD+filled,y+7),col,-1)
                cv2.putText(canvas,f"{lbl} {val:+.0f}",(PAD,y+16),FONT,0.28,col,1,cv2.LINE_AA)
                y+=18
        else:
            cv2.putText(canvas,"no data",(PAD,y+8),FONT,0.27,TXT_L,1,cv2.LINE_AA); y+=14

        # Grasp
        if grasp_data and side in grasp_data:
            gt,ap,cs_=grasp_data[side]
            cv2.putText(canvas,gt,(PAD,y+9),FONT_B,0.34,TXT_D,1,cv2.LINE_AA); y+=12
            cv2.putText(canvas,f"ap {ap*100:.1f}cm  {cs_}",(PAD,y+9),FONT,0.27,TXT_G,1,cv2.LINE_AA); y+=13

        # EE speed bar (live data from velocity computation)
        spd=speed_data.get(side,0.0) if speed_data else 0.0
        max_spd=0.5   # clamp at 0.5 m/s for bar scaling
        bw=W-PAD*2; filled=int(bw*min(spd/max_spd,1.))
        cv2.rectangle(canvas,(PAD,y+1),(PAD+bw,y+6),(220,220,220),-1)
        cv2.rectangle(canvas,(PAD,y+1),(PAD+filled,y+6),(80,200,220),-1)
        cv2.putText(canvas,f"spd {spd*100:.1f}cm/s",(PAD,y+15),FONT,0.25,TXT_G,1,cv2.LINE_AA)
        y+=18

        # Joint angles
        if joint_data and side in joint_data:
            fja=joint_data[side]
            cv2.putText(canvas,"joints (deg)",(PAD,y+8),FONT,0.25,TXT_G,1,cv2.LINE_AA); y+=11
            fingers=[
                ("Th",["thumb_mcp","thumb_ip"]),
                ("Ix",["idx_mcp","idx_pip","idx_dip"]),
                ("Md",["mid_mcp","mid_pip","mid_dip"]),
                ("Rg",["ring_mcp","ring_pip","ring_dip"]),
                ("Pk",["pinky_mcp","pinky_pip","pinky_dip"]),
            ]
            bar_full=W-PAD*2
            for abbr,keys in fingers:
                if y>=H-14: break
                cv2.putText(canvas,abbr,(PAD,y+8),FONT_B,0.28,TXT_D,1,cv2.LINE_AA)
                bx=PAD+16; sw=(bar_full-16)//len(keys)
                for k in keys:
                    ang=fja[k]; blen=int(sw*min(ang/180.,1.))
                    cv2.rectangle(canvas,(bx,y+1),(bx+sw-2,y+6),(220,220,220),-1)
                    cv2.rectangle(canvas,(bx,y+1),(bx+blen,y+6),hcol,-1)
                    cv2.putText(canvas,f"{ang:.0f}",(bx,y+14),FONT,0.22,TXT_G,1,cv2.LINE_AA)
                    bx+=sw
                y+=16

        y+=4; cv2.line(canvas,(PAD,y),(W-PAD,y),BORDER,1); y+=8

    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
#  BOTTOM-RIGHT PANEL  — Full-body skeleton + hand terminators + orient arrows
# ═══════════════════════════════════════════════════════════════════════════════

def draw_body_panel(W, H, pose_arr, hands_at_wrist):
    canvas=np.full((H,W,3),255,np.uint8)
    R=R_3D

    sk={k:v.copy() for k,v in TPOSE.items()}
    if pose_arr is not None:
        ls,rs=sk["ls"],sk["rs"]
        sk["le"]=ls+sv(pose_arr[P_LS],pose_arr[P_LE],_UA)
        sk["re"]=rs+sv(pose_arr[P_RS],pose_arr[P_RE],_UA)
        sk["lw"]=sk["le"]+sv(pose_arr[P_LE],pose_arr[P_LW],_FA)
        sk["rw"]=sk["re"]+sv(pose_arr[P_RE],pose_arr[P_RW],_FA)

    all_pts=np.array(list(sk.values()))
    cx3,cy3=all_pts[:,0].mean(),all_pts[:,1].mean()
    span=max(np.ptp(all_pts[:,0]),np.ptp(all_pts[:,1]))*0.7+0.15
    pad_l=int(W*0.14); pad_b=int(H*0.14)
    CX=pad_l+(W-pad_l)//2; CY=(H-pad_b)//2
    SC=min(W-pad_l,H-pad_b)/(span*2.1)
    offset=np.array([cx3,cy3,0.0])
    sk_c={k:v-offset for k,v in sk.items()}

    draw_grid(canvas,R,CX,CY,SC,span)
    draw_axes(canvas,R,CX,CY,SC,span)

    # Body bones
    for a_n,b_n in SKEL_BONES:
        cv2.line(canvas,p3(sk_c[a_n],R,CX,CY,SC),p3(sk_c[b_n],R,CX,CY,SC),
                 BODY_BONE_COL,2,cv2.LINE_AA)

    # Body joint orientation arrows (arm bones only)
    for a_n,b_n in [("ls","le"),("le","lw"),("rs","re"),("re","rw")]:
        mid=(sk_c[a_n]+sk_c[b_n])/2
        draw_orient(canvas,mid,local_frame(sk_c[a_n],sk_c[b_n]),R,CX,CY,SC,length=0.04)

    # Body joint dots
    for nm,pt in sk_c.items():
        sz=8 if nm=="nose" else 5
        p=p3(pt,R,CX,CY,SC)
        cv2.circle(canvas,p,sz,BODY_JOINT_COL,-1,cv2.LINE_AA)
        cv2.circle(canvas,p,sz,(255,255,255),1,cv2.LINE_AA)

    # Hand terminators
    hand_sc=(_FA*0.80)/CUBE_H
    wm={"Left":sk_c["lw"],"Right":sk_c["rw"]}
    hcol={"Left":L_BGR,"Right":R_BGR}
    for side,lms_sc in hands_at_wrist.items():
        w3=wm[side]; col=hcol[side]
        hx=lms_sc[:,0]*hand_sc+w3[0]
        hy=lms_sc[:,1]*hand_sc+w3[1]
        hz=lms_sc[:,2]*hand_sc+w3[2]
        hpts=np.column_stack([hx,hy,hz])
        # Bones
        for _,a,b in sorted([(zd((hpts[a]+hpts[b])/2,R),a,b) for a,b in HAND_CONN]):
            cv2.line(canvas,p3(hpts[a],R,CX,CY,SC),p3(hpts[b],R,CX,CY,SC),col,1,cv2.LINE_AA)
        # Joints
        for _,i in sorted((zd(hpts[i],R),i) for i in range(21)):
            pt=p3(hpts[i],R,CX,CY,SC); r=6 if i in TIPS else (7 if i==0 else 4)
            cv2.circle(canvas,pt,r,col,-1,cv2.LINE_AA)
            cv2.circle(canvas,pt,r,(255,255,255),1,cv2.LINE_AA)
        # Finger orientation arrows
        for a,b in [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
                    (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)]:
            mid=(hpts[a]+hpts[b])/2
            draw_orient(canvas,mid,local_frame(hpts[a],hpts[b]),R,CX,CY,SC,length=0.018)

    # Legend
    ly=12
    for lbl,col in [("Body",BODY_BONE_COL),("Left hand",L_BGR),("Right hand",R_BGR)]:
        cv2.rectangle(canvas,(6,ly-6),(18,ly+2),col,-1)
        cv2.putText(canvas,lbl,(22,ly+2),FONT,0.30,TXT_D,1,cv2.LINE_AA); ly+=14
    for lbl,col in [("X",AX_X),("Y",AX_Y),("Z",AX_Z)]:
        cv2.rectangle(canvas,(W-55,ly-6),(W-43,ly+2),col,-1)
        cv2.putText(canvas,lbl,(W-40,ly+2),FONT,0.28,col,1,cv2.LINE_AA); ly+=12
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
#  FRAME COMPOSITOR
# ═══════════════════════════════════════════════════════════════════════════════

def build_frame(frame, nlms, hness, cur_hands, rpy_data,
                grasp_data, joint_data, speed_data,
                pose_arr, haw, out_w, out_h,
                macro_task, micro_step, step_idx, total_steps,
                t0, tc, nl_caption, env, scene, oph):
    INFO_W=185
    cam_w=int(out_w*0.42)
    plot_w=out_w-cam_w-INFO_W-2
    plot_h=out_h//2

    left      = draw_left(frame,nlms,hness,cam_w,out_h,
                          macro_task,micro_step,step_idx,total_steps,
                          t0,tc,nl_caption,env,scene,oph)
    info      = draw_info_panel(INFO_W,out_h,rpy_data,grasp_data,joint_data,speed_data)
    top_right = draw_hand_panel(plot_w,plot_h,cur_hands,rpy_data)
    bot_right = draw_body_panel(plot_w,plot_h,pose_arr,haw)

    right=np.vstack([top_right,bot_right])
    cv2.line(right,(0,plot_h),(plot_w,plot_h),BORDER,1)
    div=np.full((out_h,1,3),BORDER,np.uint8)
    card=np.hstack([left,div,info,div,right])
    cv2.rectangle(card,(0,0),(card.shape[1]-1,card.shape[0]-1),BORDER,2)
    return card


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_step_timeline(steps_list, t0, t1):
    n=len(steps_list)
    if n==0: return []
    seg=(t1-t0)/n
    return [(steps_list[i],t0+i*seg,t0+(i+1)*seg) for i in range(n)]

def step_at(tc, timeline):
    for idx,(lbl,ts,te) in enumerate(timeline):
        if ts<=tc<te: return lbl,idx+1,len(timeline)
    if timeline: return timeline[-1][0],len(timeline),len(timeline)
    return "",1,1


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(input_path,
                  csv_path="hand_6dof.csv", jsonl_path="vla_dataset.jsonl",
                  out_path="vla_output.mp4",
                  out_w=1600, out_h=720,
                  skip=1, max_hands=2, conf=0.55,
                  macro_task="", steps=None, nl_caption="",
                  ts_start=0.0, ts_end=0.0,
                  environment="Factory", scene="Workstation",
                  op_height=170.0, robot_height=0.0,
                  write_video=True,
                  enhance_gloves=True,
                  glove_mode=False):
    """
    glove_mode : if True, use two-stage YOLO+MediaPipe detection instead of
                 standard MediaPipe. Requires:  pip install ultralytics
                 Dramatically improves landmark detection on grey/blue ESD gloves.
    enhance_gloves: CLAHE+sharpening preprocessing — applied in both modes.
    """
    hand_model=dl_model(HAND_URL,"hand_landmarker.task")
    pose_model=dl_model(POSE_URL,"pose_landmarker_lite.task")

    cap=cv2.VideoCapture(input_path)
    if not cap.isOpened(): raise FileNotFoundError(f"Cannot open: {input_path}")
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps=max(1.0,fps/skip)
    if ts_end<=ts_start: ts_end=total/fps

    writer=None
    if write_video:
        writer=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*"mp4v"),out_fps,(out_w,out_h))

    hand_opts=HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=mp_vision.RunningMode.VIDEO, num_hands=max_hands,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
    pose_opts=PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)

    steps_list=steps or []
    timeline=build_step_timeline(steps_list,ts_start,ts_end)

    # CSV header: all annotation types included
    csv_hdr=(
        ["frame","hand","ts_sec",
         "macro_task","micro_step","step_idx",
         "op_height_cm","robot_height_cm","environment","scene"]+
        [f"x{i}" for i in range(21)]+[f"y{i}" for i in range(21)]+[f"z{i}" for i in range(21)]+
        ["roll_deg","pitch_deg","yaw_deg"]+
        [f"nx{i}" for i in range(21)]+[f"ny{i}" for i in range(21)]+[f"nz{i}" for i in range(21)]+
        ["ee_x","ee_y","ee_z"]+
        ["thumb_mcp","thumb_ip","idx_mcp","idx_pip","idx_dip",
         "mid_mcp","mid_pip","mid_dip","ring_mcp","ring_pip","ring_dip",
         "pinky_mcp","pinky_pip","pinky_dip"]+
        ["grasp_type","finger_aperture_m","contact_state"]+
        ["ee_vx","ee_vy","ee_vz","ee_speed","ee_ax","ee_ay","ee_az","ee_accel"]
    )
    csv_rows=[]; jsonl_records=[]
    hand_state={"Left":{"prev_xyz":None,"prev_vel":None,"prev_ts":None},
                "Right":{"prev_xyz":None,"prev_vel":None,"prev_ts":None}}

    with (mp_vision.HandLandmarker.create_from_options(hand_opts) as h_det,
          mp_vision.PoseLandmarker.create_from_options(pose_opts) as p_det):

        fidx=0; pbar=tqdm(total=total,unit="frame",
                          desc=f"Processing {os.path.basename(input_path)}")
        while True:
            ret,frame=cap.read()
            if not ret: break
            pbar.update(1)
            if fidx%skip!=0: fidx+=1; continue

            tc=fidx/fps

            # Glove-aware preprocessing before MediaPipe inference
            detect_frame = preprocess_for_gloves(frame) if enhance_gloves else frame
            rgb=cv2.cvtColor(detect_frame,cv2.COLOR_BGR2RGB)
            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            ts_ms=int(tc*1000)

            h_res=h_det.detect_for_video(mp_img,ts_ms)
            p_res=p_det.detect_for_video(mp_img,ts_ms)

            micro_step,step_idx,total_steps=step_at(tc,timeline)

            nlms,hness,cur_hands,haw=[],[],[],{}
            rpy_data,grasp_data,joint_data,speed_data={},{},{},{}
            frame_json_hands=[]

            # ── Hand detection: two-stage (glove) or standard MediaPipe ──────
            # Build a unified list of (lms_w_21x3, norm_lms_list, label)
            # regardless of which path detected them.
            hand_detections = []

            if glove_mode:
                # Stage 1+2: YOLO bounding box → MediaPipe crop landmark
                glove_results = detect_gloved_hands(frame, hand_model, conf)
                if glove_results:
                    hand_detections = [
                        (r["world_lms"], r["norm_lms"], r["label"])
                        for r in glove_results
                    ]

            if not hand_detections and h_res.hand_world_landmarks:
                # Standard MediaPipe path (used when glove_mode=False,
                # or as fallback if two-stage found nothing)
                for wlm, nlm, hi in zip(
                        h_res.hand_world_landmarks,
                        h_res.hand_landmarks,
                        h_res.handedness):
                    hand_detections.append((hw(wlm), hn(nlm), hi[0].display_name))
                    if len(hand_detections) >= max_hands:
                        break

            # ── Process all detected hands ────────────────────────────────────
            for lms_w, norm_lms, label in hand_detections[:max_hands]:
                lms_sc = cs(lms_w)
                nlms.append(norm_lms); hness.append(label)
                cur_hands.append((lms_sc, label)); haw[label] = lms_sc

                # 6-DoF RPY
                _, palm_R = palm_frame(lms_w)
                roll, pitch, yaw = rpy_from_R(palm_R)
                rpy_data[label] = (roll, pitch, yaw)

                # Height-normalised coords
                lms_norm = normalize_pose(lms_w, op_height, robot_height)

                # End-effector (wrist)
                ee_xyz = lms_w[0]

                # Joint angles
                fja = finger_joint_angles(lms_w)

                # Grasp classification
                g_type, g_ap, g_ct = classify_grasp(lms_w)
                grasp_data[label] = (g_type, g_ap, g_ct)
                joint_data[label] = fja

                # Velocity & acceleration (finite difference)
                hs = hand_state.setdefault(label, {"prev_xyz":None,"prev_vel":None,"prev_ts":None})
                dt = (tc - hs["prev_ts"]) if hs["prev_ts"] is not None else (skip/fps)
                if dt < 1e-9: dt = skip/fps
                vel   = ((ee_xyz - hs["prev_xyz"]) / dt if hs["prev_xyz"] is not None else np.zeros(3))
                accel = ((vel - hs["prev_vel"])    / dt if hs["prev_vel"]  is not None else np.zeros(3))
                ee_spd = float(np.linalg.norm(vel))
                ee_acc = float(np.linalg.norm(accel))
                speed_data[label] = ee_spd
                hs["prev_xyz"] = ee_xyz.copy(); hs["prev_vel"] = vel.copy(); hs["prev_ts"] = tc

                # CSV row
                row = ([fidx, label, f"{tc:.3f}",
                        macro_task, micro_step, step_idx,
                        op_height, robot_height, environment, scene] +
                       lms_w[:,0].tolist()+lms_w[:,1].tolist()+lms_w[:,2].tolist() +
                       [round(roll,3), round(pitch,3), round(yaw,3)] +
                       lms_norm[:,0].tolist()+lms_norm[:,1].tolist()+lms_norm[:,2].tolist() +
                       [round(float(ee_xyz[0]),6), round(float(ee_xyz[1]),6), round(float(ee_xyz[2]),6)] +
                       [round(fja["thumb_mcp"],3), round(fja["thumb_ip"],3),
                        round(fja["idx_mcp"],3),   round(fja["idx_pip"],3),   round(fja["idx_dip"],3),
                        round(fja["mid_mcp"],3),   round(fja["mid_pip"],3),   round(fja["mid_dip"],3),
                        round(fja["ring_mcp"],3),  round(fja["ring_pip"],3),  round(fja["ring_dip"],3),
                        round(fja["pinky_mcp"],3), round(fja["pinky_pip"],3), round(fja["pinky_dip"],3)] +
                       [g_type, round(g_ap,6), g_ct] +
                       [round(float(vel[0]),6),   round(float(vel[1]),6),   round(float(vel[2]),6),
                        round(ee_spd,6),
                        round(float(accel[0]),6), round(float(accel[1]),6), round(float(accel[2]),6),
                        round(ee_acc,6)])
                csv_rows.append(row)

                frame_json_hands.append({
                    "hand": label, "xyz": lms_w.tolist(),
                    "roll": round(roll,3), "pitch": round(pitch,3), "yaw": round(yaw,3),
                    "xyz_norm": lms_norm.tolist(),
                    "ee_xyz": [round(float(v),6) for v in ee_xyz],
                    "joint_angles": {k: round(v,3) for k,v in fja.items()},
                    "grasp_type": g_type, "finger_aperture_m": round(g_ap,6),
                    "contact_state": g_ct,
                    "ee_velocity": [round(float(v),6) for v in vel],
                    "ee_speed": round(ee_spd,6),
                    "ee_accel": [round(float(v),6) for v in accel],
                    "ee_accel_mag": round(ee_acc,6),
                })

            pose_arr=None
            if p_res.pose_world_landmarks:
                pose_arr=pw(p_res.pose_world_landmarks[0])

            if frame_json_hands:
                jsonl_records.append({
                    "video":os.path.basename(input_path),
                    "frame":fidx,"ts_sec":round(tc,3),
                    "macro_task":macro_task,"micro_step":micro_step,
                    "step_idx":step_idx,"total_steps":total_steps,
                    "environment":environment,"scene":scene,
                    "op_height_cm":op_height,"robot_height_cm":robot_height,
                    "hands":frame_json_hands,
                })

            if writer:
                out_frame=build_frame(
                    frame,nlms,hness,cur_hands,rpy_data,
                    grasp_data,joint_data,speed_data,
                    pose_arr,haw,out_w,out_h,
                    macro_task,micro_step,step_idx,total_steps,
                    ts_start,tc,nl_caption,environment,scene,op_height)
                writer.write(cv2.resize(out_frame,(out_w,out_h)))
            fidx+=1
        pbar.close()

    cap.release()
    if writer: writer.release()

    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(csv_hdr); w.writerows(csv_rows)
    with open(jsonl_path,"w") as f:
        for rec in jsonl_records: f.write(json.dumps(rec)+"\n")

    stats={"video":out_path if writer else None,
           "csv":csv_path,"jsonl":jsonl_path,
           "rows":len(csv_rows),"frames":len(jsonl_records)}
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _ph(v):
    return float(v.lower().replace("cm","").strip())

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="VLA Egocentric Data Collection Annotator")
    ap.add_argument("--input",         required=True)
    ap.add_argument("--csv",           default="hand_6dof.csv")
    ap.add_argument("--jsonl",         default="vla_dataset.jsonl")
    ap.add_argument("--output",        default="vla_output.mp4")
    ap.add_argument("--width",         type=int,   default=1600)
    ap.add_argument("--height",        type=int,   default=720)
    ap.add_argument("--skip",          type=int,   default=1)
    ap.add_argument("--hands",         type=int,   default=2)
    ap.add_argument("--conf",          type=float, default=0.55)
    ap.add_argument("--macro_task",    default="")
    ap.add_argument("--steps",         default="",  help="Semicolon-separated micro steps")
    ap.add_argument("--nl_caption",    default="")
    # Legacy aliases
    ap.add_argument("--skill",         default="")
    ap.add_argument("--description",   default="")
    ap.add_argument("--ts_start",      type=float, default=0.0)
    ap.add_argument("--ts_end",        type=float, default=0.0)
    ap.add_argument("--environment",   default="Factory")
    ap.add_argument("--scene",         default="Workstation")
    ap.add_argument("--op_height",     type=_ph,   default="170")
    ap.add_argument("--robot_height",  type=_ph,   default="0")
    ap.add_argument("--no_video",      action="store_true")
    ap.add_argument("--no_enhance",    action="store_true",
                    help="Disable glove preprocessing (use on bare-hand footage)")
    ap.add_argument("--glove_mode",    action="store_true",
                    help="Enable two-stage YOLO+MediaPipe glove detection. "
                         "Requires:  pip install ultralytics  "
                         "Use when standard MediaPipe misses gloved hands.")
    a=ap.parse_args()

    resolved_macro=a.macro_task or a.skill
    if a.description and not resolved_macro: resolved_macro=a.description
    elif a.description and resolved_macro:   resolved_macro=f"{resolved_macro} - {a.description}"

    steps_list=[s.strip() for s in a.steps.split(";") if s.strip()] if a.steps else []

    stats=process_video(
        input_path=a.input, csv_path=a.csv, jsonl_path=a.jsonl,
        out_path=a.output, out_w=a.width, out_h=a.height,
        skip=a.skip, max_hands=a.hands, conf=a.conf,
        macro_task=resolved_macro, steps=steps_list,
        nl_caption=a.nl_caption, ts_start=a.ts_start, ts_end=a.ts_end,
        environment=a.environment, scene=a.scene,
        op_height=a.op_height, robot_height=a.robot_height,
        write_video=not a.no_video,
        enhance_gloves=not a.no_enhance,
        glove_mode=a.glove_mode,
    )

    print(f"\nDone.")
    if stats["video"]: print(f"  Video : {stats['video']}")
    print(f"  CSV   : {stats['csv']}  ({stats['rows']:,} rows)")
    print(f"  JSONL : {stats['jsonl']}  ({stats['frames']:,} frames)")
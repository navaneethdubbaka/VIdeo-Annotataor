"""
6-DoF Embodied AI Data Collection Visualizer  (pure-OpenCV build)
=================================================================
All rendering is done with OpenCV + orthographic 3D projection.
No matplotlib dependency — matches reference visual style exactly.

Layout:
  +---------------------------+------------------+
  |   Camera frame            |  3D hand cloud   |
  |   + annotation card       |  (trajectory     |
  |   + 2D hand overlay       |   + skeleton)    |
  |   + NL caption            +------------------+
  |   + metadata bar          |  Full-body pose  |
  |                           |  + live hands    |
  +---------------------------+------------------+

Install:
    pip install mediapipe opencv-python tqdm numpy

Usage:
    python vla_annotator.py \
        --input factory.mp4 \
        --skill "Inflate tire" \
        --description "Inflate the car tire using the pressure gauge." \
        --nl_caption "The person inflates a car tire using a pressure gauge." \
        --environment "Car Workshop" --scene "Car service" --op_height "162cm" \
        --skip 2
"""

import argparse, csv, math, os, textwrap, urllib.request
import cv2
import numpy as np
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, PoseLandmarkerOptions

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Finger colours (BGR) ─────────────────────────────────────────────────────
FCOL = {
    "palm":   (180, 80,  20),  "thumb":  (0, 140, 255),
    "index":  (30, 180,  30),  "middle": (200, 60,  20),
    "ring":   (20,  20, 200),  "pinky":  (170, 30, 170),
}
WRIST_COL = (200, 120, 20)

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

# ── Hand / body colours (BGR) ────────────────────────────────────────────────
L_BGR = (46, 204, 113);   R_BGR = (52, 73, 235)
BODY_BONE_COL = (180, 180, 180);  BODY_JOINT_COL = (120, 120, 120)

# ── 3D-plot styling (BGR) ────────────────────────────────────────────────────
GRID_COL   = (210, 210, 210)
BONE_3D    = (165, 165, 165)
AX_X = (40,  40, 220)   # red
AX_Y = (40, 180,  40)   # green
AX_Z = (220, 130, 30)   # blue

# ── Layout colours (BGR) ─────────────────────────────────────────────────────
BG_W   = (255,255,255);  BG_META = (248,248,248);  BORDER = (210,210,210)
TXT_D  = (40,40,40);     TXT_G   = (130,130,130);  TXT_L  = (170,170,170)
FONT   = cv2.FONT_HERSHEY_SIMPLEX
FONT_B = cv2.FONT_HERSHEY_DUPLEX

# ── Pose indices ──────────────────────────────────────────────────────────────
P_NOSE=0; P_LS=11; P_RS=12; P_LE=13; P_RE=14; P_LW=15; P_RW=16; P_LH=23; P_RH=24

# ── Fixed T-pose ─────────────────────────────────────────────────────────────
_SW=0.22; _TH=0.18; _UA=0.26; _FA=0.22; _HW=0.10; CUBE_H=0.16

TPOSE = {
    "nose":np.array([0,.52,0]),
    "ls":np.array([-_SW,.30,0]),  "rs":np.array([_SW,.30,0]),
    "le":np.array([-_SW-_UA,.30,0]), "re":np.array([_SW+_UA,.30,0]),
    "lw":np.array([-_SW-_UA-_FA,.30,0]),  "rw":np.array([_SW+_UA+_FA,.30,0]),
    "lh":np.array([-_HW,-_TH,0]),  "rh":np.array([_HW,-_TH,0]),
}
SKEL_BONES = [
    ("nose","ls"),("nose","rs"),("ls","rs"),("ls","lh"),("rs","rh"),
    ("lh","rh"),("ls","le"),("le","lw"),("rs","re"),("re","rw"),
]

TRAIL_LEN = 30

HAND_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def dl_model(url, path):
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)}…")
        urllib.request.urlretrieve(url, path); print("  Done.")
    return path

def fmt_ts(s):
    m=int(s)//60; return f"{m:02d}:{s-m*60:06.3f}"

def hw(lm_list):
    a=np.array([[l.x,l.y,l.z] for l in lm_list],dtype=np.float64); a[:,1]*=-1; return a

def hn(lm_list):
    return [(l.x,l.y) for l in lm_list]

def pw(lm_list):
    a=np.array([[l.x,l.y,l.z] for l in lm_list],dtype=np.float64); a[:,1]*=-1; return a

def cs(lms, half=CUBE_H, fill=0.60):
    c=lms-lms[0]; e=np.abs(c).max()
    return c*(half*fill/e) if e>1e-9 else c

def sv(start, end, length):
    v=end-start; n=np.linalg.norm(v)
    return (v/n*length) if n>1e-6 else np.zeros(3)

def local_frame(a, b):
    fwd=b-a; fn=np.linalg.norm(fwd)
    if fn<1e-9: return np.eye(3)
    fwd/=fn; up=np.array([0.,1.,0.])
    if abs(np.dot(fwd,up))>0.95: up=np.array([0.,0.,1.])
    right=np.cross(fwd,up); right/=np.linalg.norm(right)
    up2=np.cross(right,fwd)
    return np.array([right,up2,fwd])

# ═══════════════════════════════════════════════════════════════════════════════
#  3D PROJECTION  (orthographic)
# ═══════════════════════════════════════════════════════════════════════════════

def make_rot(elev=30, azim=-60):
    el,az = math.radians(elev), math.radians(azim)
    Ry=np.array([[math.cos(az),0,math.sin(az)],[0,1,0],[-math.sin(az),0,math.cos(az)]])
    Rx=np.array([[1,0,0],[0,math.cos(el),-math.sin(el)],[0,math.sin(el),math.cos(el)]])
    return (Rx@Ry).astype(np.float64)

R_HAND = make_rot(elev=10, azim=-15)
R_BODY = make_rot(elev=10, azim=-15)

def p3(pt, R, cx, cy, sc):
    """Project one 3D point → (x,y) int tuple."""
    r=R@np.asarray(pt,np.float64)
    return (int(r[0]*sc+cx), int(-r[1]*sc+cy))

def p3n(pts, R, cx, cy, sc):
    """Project Nx3 array → Nx2 int array."""
    pts=np.atleast_2d(pts).astype(np.float64)
    r=(R@pts.T).T
    return np.column_stack([(r[:,0]*sc+cx).astype(int), (-r[:,1]*sc+cy).astype(int)])

def zd(pt, R):
    return float((R@np.asarray(pt,np.float64))[2])

# ═══════════════════════════════════════════════════════════════════════════════
#  DASHED LINES / GRID / AXIS ARROWS
# ═══════════════════════════════════════════════════════════════════════════════

def dash_line(img, a, b, col=GRID_COL, th=1, gap=6):
    a,b=np.array(a,float),np.array(b,float)
    d=b-a; L=np.linalg.norm(d)
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
        # Floor y=-h
        dash_line(img,p3([v,-h,-h],R,cx,cy,sc),p3([v,-h,h],R,cx,cy,sc))
        dash_line(img,p3([-h,-h,v],R,cx,cy,sc),p3([h,-h,v],R,cx,cy,sc))
        # Back z=-h
        dash_line(img,p3([v,-h,-h],R,cx,cy,sc),p3([v,h,-h],R,cx,cy,sc))
        dash_line(img,p3([-h,v,-h],R,cx,cy,sc),p3([h,v,-h],R,cx,cy,sc))
        # Left x=-h
        dash_line(img,p3([-h,v,-h],R,cx,cy,sc),p3([-h,v,h],R,cx,cy,sc))
        dash_line(img,p3([-h,-h,v],R,cx,cy,sc),p3([-h,h,v],R,cx,cy,sc))

def draw_axes(img, R, cx, cy, sc, half):
    h=half; ext=h*1.15; o=p3([0,0,0],R,cx,cy,sc)
    for end3,col,lbl in [([ext,0,0],AX_X,"X"),([0,ext,0],AX_Y,"Y"),([0,0,ext],AX_Z,"Z")]:
        e=p3(end3,R,cx,cy,sc)
        cv2.arrowedLine(img,o,e,col,1,cv2.LINE_AA,tipLength=0.13)
        d=np.array(e)-np.array(o); dn=np.linalg.norm(d)
        if dn>0:
            lp=(np.array(e)+(d/dn*13).astype(int))
            cv2.putText(img,lbl,tuple(lp.astype(int)),FONT,0.38,col,1,cv2.LINE_AA)
        for tv in [0.2,0.4,0.6]:
            frac=tv/h
            if frac>1.2: continue
            tp3=[end3[0]*frac*h/ext, end3[1]*frac*h/ext, end3[2]*frac*h/ext]
            tp=p3(tp3,R,cx,cy,sc)
            cv2.putText(img,f"{tv:.1f}",(tp[0]+2,tp[1]+12),FONT,0.27,TXT_G,1,cv2.LINE_AA)

def draw_orient(img, origin3, frame3x3, R, cx, cy, sc, length=0.02):
    """Draw small RGB orientation arrows at a 3D point."""
    o=p3(origin3,R,cx,cy,sc)
    for i,col in enumerate([AX_X,AX_Y,AX_Z]):
        e=p3(origin3+frame3x3[i]*length, R,cx,cy,sc)
        cv2.line(img,o,e,col,1,cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL  — Camera + overlay + metadata
# ═══════════════════════════════════════════════════════════════════════════════

def draw_left(frame, nlms, hness, W, H, skill, desc, t0, tc, caption, env, scene, oph):
    CAP_H=40; META_H=48; VH=H-CAP_H-META_H
    canvas=np.full((H,W,3),250,np.uint8)

    # 1) Video (fill-crop)
    fh,fw=frame.shape[:2]; sc=max(W/fw,VH/fh)
    nw,nh=int(fw*sc),int(fh*sc); res=cv2.resize(frame,(nw,nh))
    x0c,y0c=max(0,(nw-W)//2),max(0,(nh-VH)//2)
    crop=res[y0c:y0c+VH, x0c:x0c+W]
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

    # 3) Annotation card
    if skill or desc:
        cw,ch=min(W-20,320),72; cx0,cy0=10,10
        ov=canvas.copy()
        cv2.rectangle(ov,(cx0,cy0),(cx0+cw,cy0+ch),(255,255,255),-1)
        cv2.addWeighted(ov,0.87,canvas,0.13,0,canvas)
        cv2.rectangle(canvas,(cx0,cy0),(cx0+cw,cy0+ch),(200,200,200),1)
        cv2.putText(canvas,"Skill:",(cx0+8,cy0+18),FONT,0.38,TXT_G,1,cv2.LINE_AA)
        cv2.putText(canvas,skill or "",(cx0+52,cy0+18),FONT_B,0.45,TXT_D,1,cv2.LINE_AA)
        cv2.putText(canvas,(desc or "")[:60],(cx0+8,cy0+38),FONT,0.32,(70,70,70),1,cv2.LINE_AA)
        cv2.putText(canvas,f"{fmt_ts(t0)}  ->  {fmt_ts(tc)}",(cx0+8,cy0+58),FONT,0.33,(100,100,100),1,cv2.LINE_AA)

    # 4) NL caption
    cy=VH
    cv2.rectangle(canvas,(0,cy),(W,cy+CAP_H),(238,238,238),-1)
    cv2.line(canvas,(0,cy),(W,cy),BORDER,1)
    for i,ln in enumerate((textwrap.wrap(caption or "",width=W//8) or [""])[:2]):
        cv2.putText(canvas,ln,(12,cy+16+i*16),FONT,0.42,TXT_D,1,cv2.LINE_AA)

    # 5) Metadata bar
    my=VH+CAP_H
    cv2.rectangle(canvas,(0,my),(W,H),BG_META,-1); cv2.line(canvas,(0,my),(W,my),BORDER,1)
    cw3=W//3
    for ci,(lb,vl) in enumerate([("ENVIRONMENT",env or"—"),("SCENE",scene or"—"),("OPERATOR HEIGHT",oph or"—")]):
        lx=ci*cw3+14
        cv2.circle(canvas,(ci*cw3+8,my+16),3,TXT_G,-1,cv2.LINE_AA)
        cv2.putText(canvas,lb,(lx,my+18),FONT,0.28,TXT_G,1,cv2.LINE_AA)
        cv2.putText(canvas,vl,(lx,my+36),FONT_B,0.40,TXT_D,1,cv2.LINE_AA)
        if ci>0: cv2.line(canvas,(ci*cw3,my+5),(ci*cw3,H-5),BORDER,1)
    return canvas

# ═══════════════════════════════════════════════════════════════════════════════
#  TOP-RIGHT PANEL  — 3D hand trajectory cloud + current skeleton
# ═══════════════════════════════════════════════════════════════════════════════

def draw_hand_panel(W, H, current_hands, cloud, cube_half=0.12):
    """
    current_hands: list of (lms_sc_21x3, label)
    cloud: {'Left': list_of_xyz, 'Right': list_of_xyz}
    """
    canvas = np.full((H,W,3),255,np.uint8)
    R = R_HAND

    # Projection centre and scale
    pad_l, pad_b = int(W*0.15), int(H*0.15)
    CX = pad_l + (W-pad_l)//2
    CY = (H-pad_b)//2
    SC = min(W-pad_l, H-pad_b) / (cube_half*1.8)

    # Grid + axes
    draw_grid(canvas, R, CX, CY, SC, cube_half)
    draw_axes(canvas, R, CX, CY, SC, cube_half)

    # Cloud scatter removed — only show current-frame hand skeleton

    # Current frame hand skeletons
    offsets = {"Left": np.array([-0.04,0,0]), "Right": np.array([0.04,0,0])}
    for lms_sc, label in current_hands:
        col = L_BGR if label=="Left" else R_BGR
        off = offsets.get(label, np.zeros(3))
        lm = lms_sc + off

        # Sort bones back-to-front
        bone_z = sorted([(zd((lm[a]+lm[b])/2,R),a,b) for a,b in HAND_CONN])
        for _,a,b in bone_z:
            pa,pb = p3(lm[a],R,CX,CY,SC), p3(lm[b],R,CX,CY,SC)
            cv2.line(canvas,pa,pb,BONE_3D,1,cv2.LINE_AA)

        # Joint dots (depth-sorted)
        for _,i in sorted((zd(lm[i],R),i) for i in range(21)):
            pt = p3(lm[i],R,CX,CY,SC)
            c = FCOL[LM_FINGER[i]]
            r = 7 if i==0 else (6 if i in TIPS else 4)
            cv2.circle(canvas,pt,r,c,-1,cv2.LINE_AA)

        # 6-DoF orientation arrows at bone midpoints
        segs=[(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
              (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)]
        for a,b in segs:
            mid=(lm[a]+lm[b])/2
            frm=local_frame(lm[a],lm[b])
            draw_orient(canvas,mid,frm,R,CX,CY,SC,length=0.012)

    # Legend
    ly=12
    for lbl,col in [("Left",L_BGR),("Right",R_BGR)]:
        cv2.circle(canvas,(W-70,ly),4,col,-1,cv2.LINE_AA)
        cv2.putText(canvas,lbl,(W-62,ly+4),FONT,0.32,TXT_D,1,cv2.LINE_AA)
        ly+=14
    for lbl,col in [("X",AX_X),("Y",AX_Y),("Z",AX_Z)]:
        cv2.line(canvas,(W-70,ly),(W-60,ly),col,2,cv2.LINE_AA)
        cv2.putText(canvas,lbl,(W-56,ly+4),FONT,0.28,col,1,cv2.LINE_AA)
        ly+=12

    return canvas

# ═══════════════════════════════════════════════════════════════════════════════
#  BOTTOM-RIGHT PANEL  — Full-body skeleton + hand terminators
# ═══════════════════════════════════════════════════════════════════════════════

def draw_body_panel(W, H, pose_arr, hands_at_wrist):
    """
    pose_arr: (33,3) or None
    hands_at_wrist: {'Left': lms_sc_21x3, 'Right': lms_sc_21x3}
    """
    canvas = np.full((H,W,3),255,np.uint8)
    R = R_BODY

    # Build live skeleton
    sk = {k:v.copy() for k,v in TPOSE.items()}
    if pose_arr is not None:
        ls,rs=sk["ls"],sk["rs"]
        sk["le"]=ls+sv(pose_arr[P_LS],pose_arr[P_LE],_UA)
        sk["re"]=rs+sv(pose_arr[P_RS],pose_arr[P_RE],_UA)
        sk["lw"]=sk["le"]+sv(pose_arr[P_LE],pose_arr[P_LW],_FA)
        sk["rw"]=sk["re"]+sv(pose_arr[P_RE],pose_arr[P_RW],_FA)

    # Compute bounding box → set grid
    all_pts = np.array(list(sk.values()))
    cx3, cy3 = all_pts[:,0].mean(), all_pts[:,1].mean()
    span = max(np.ptp(all_pts[:,0]), np.ptp(all_pts[:,1]))*0.7 + 0.15
    half = span

    # Projection centre & scale
    pad_l, pad_b = int(W*0.14), int(H*0.14)
    CX = pad_l + (W-pad_l)//2
    CY = (H-pad_b)//2
    SC = min(W-pad_l, H-pad_b) / (half*2.8)

    # Shift skeleton to centre the view
    offset = np.array([cx3, cy3, 0.0])
    sk_c = {k: v-offset for k,v in sk.items()}

    draw_grid(canvas, R, CX, CY, SC, half)
    draw_axes(canvas, R, CX, CY, SC, half)

    # Body bones
    for a_nm,b_nm in SKEL_BONES:
        a3,b3=sk_c[a_nm],sk_c[b_nm]
        cv2.line(canvas,p3(a3,R,CX,CY,SC),p3(b3,R,CX,CY,SC),BODY_BONE_COL,2,cv2.LINE_AA)

    # Body orientation arrows at arm joints
    for a_nm,b_nm in [("ls","le"),("le","lw"),("rs","re"),("re","rw")]:
        mid=(sk_c[a_nm]+sk_c[b_nm])/2
        frm=local_frame(sk_c[a_nm],sk_c[b_nm])
        draw_orient(canvas,mid,frm,R,CX,CY,SC,length=0.04)

    # Body joint dots
    for nm,pt in sk_c.items():
        sz = 8 if nm=="nose" else 5
        cv2.circle(canvas,p3(pt,R,CX,CY,SC),sz,BODY_JOINT_COL,-1,cv2.LINE_AA)
        cv2.circle(canvas,p3(pt,R,CX,CY,SC),sz,(255,255,255),1,cv2.LINE_AA)

    # Hand terminators at wrists
    hand_sc = (_FA*0.80)/CUBE_H
    wrist_map = {"Left":sk_c["lw"], "Right":sk_c["rw"]}
    hcol = {"Left":L_BGR, "Right":R_BGR}

    for side, lms_sc in hands_at_wrist.items():
        w3 = wrist_map[side]
        col = hcol[side]
        hx = lms_sc[:,0]*hand_sc + w3[0]
        hy = lms_sc[:,1]*hand_sc + w3[1]
        hz = lms_sc[:,2]*hand_sc + w3[2]

        # Bones (depth-sorted)
        hpts = np.column_stack([hx,hy,hz])
        bone_z = sorted([(zd((hpts[a]+hpts[b])/2,R),a,b) for a,b in HAND_CONN])
        for _,a,b in bone_z:
            cv2.line(canvas,p3(hpts[a],R,CX,CY,SC),p3(hpts[b],R,CX,CY,SC),col,1,cv2.LINE_AA)

        # Joint dots
        for _,i in sorted((zd(hpts[i],R),i) for i in range(21)):
            pt=p3(hpts[i],R,CX,CY,SC)
            r=6 if i in TIPS else (7 if i==0 else 4)
            cv2.circle(canvas,pt,r,col,-1,cv2.LINE_AA)
            cv2.circle(canvas,pt,r,(255,255,255),1,cv2.LINE_AA)

        # Finger orientation arrows
        for a,b in [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
                     (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)]:
            mid=(hpts[a]+hpts[b])/2
            frm=local_frame(hpts[a],hpts[b])
            draw_orient(canvas,mid,frm,R,CX,CY,SC,length=0.018)

    # Legend
    ly=12
    for lbl,col in [("Body",BODY_BONE_COL),("Left hand",L_BGR),("Right hand",R_BGR)]:
        cv2.rectangle(canvas,(6,ly-6),(18,ly+2),col,-1)
        cv2.putText(canvas,lbl,(22,ly+2),FONT,0.30,TXT_D,1,cv2.LINE_AA)
        ly+=14
    for lbl,col in [("X",AX_X),("Y",AX_Y),("Z",AX_Z)]:
        cv2.rectangle(canvas,(W-55,ly-6),(W-43,ly+2),col,-1)
        cv2.putText(canvas,lbl,(W-40,ly+2),FONT,0.28,col,1,cv2.LINE_AA)
        ly+=12

    return canvas

# ═══════════════════════════════════════════════════════════════════════════════
#  FRAME COMPOSITOR
# ═══════════════════════════════════════════════════════════════════════════════

def build_frame(frame, nlms, hness, cur_hands, cloud, pose_arr, haw,
                out_w, out_h, skill, desc, t0, tc, caption, env, scene, oph):
    # Panel dimensions
    cam_w = int(out_w * 0.42)
    plot_w = out_w - cam_w - 1   # -1 for divider
    plot_h = out_h // 2

    # Render panels
    left = draw_left(frame, nlms, hness, cam_w, out_h, skill, desc, t0, tc, caption, env, scene, oph)
    top_right = draw_hand_panel(plot_w, plot_h, cur_hands, cloud)
    bot_right = draw_body_panel(plot_w, plot_h, pose_arr, haw)

    # Stack right panels
    right = np.vstack([top_right, bot_right])
    # Add divider between top and bottom
    cv2.line(right, (0, plot_h), (plot_w, plot_h), BORDER, 1)

    # Vertical divider
    div = np.full((out_h, 1, 3), BORDER, np.uint8)

    # Combine
    card = np.hstack([left, div, right])
    # Outer border
    cv2.rectangle(card, (0,0), (card.shape[1]-1,card.shape[0]-1), BORDER, 2)
    return card

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(input_path, csv_path="hand_6dof.csv", out_path="vla_output.mp4",
                  out_w=1600, out_h=720, skip=1, max_hands=2, conf=0.55,
                  skill="", description="", nl_caption="",
                  ts_start=0.0, ts_end=0.0,
                  environment="Factory", scene="Workstation", op_height="—"):

    hand_model = dl_model(HAND_URL, "hand_landmarker.task")
    pose_model = dl_model(POSE_URL, "pose_landmarker_lite.task")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = max(1.0, fps/skip)
    if ts_end<=ts_start: ts_end=total/fps

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w,out_h))

    hand_opts = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=mp_vision.RunningMode.VIDEO, num_hands=max_hands,
        min_hand_detection_confidence=conf, min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    pose_opts = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)

    cloud = {"Left":[], "Right":[]}
    csv_rows, csv_hdr = [], (
        ["frame","hand","ts_sec","op_height","environment","scene"] +
        [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)])

    with (mp_vision.HandLandmarker.create_from_options(hand_opts) as h_det,
          mp_vision.PoseLandmarker.create_from_options(pose_opts) as p_det):

        fidx, pbar = 0, tqdm(total=total, unit="frame", desc="Processing")

        while True:
            ret, frame = cap.read()
            if not ret: break
            pbar.update(1)
            if fidx%skip!=0: fidx+=1; continue

            tc = fidx/fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(tc*1000)

            h_res = h_det.detect_for_video(mp_img, ts_ms)
            p_res = p_det.detect_for_video(mp_img, ts_ms)

            nlms, hness, cur_hands, haw = [], [], [], {}

            if h_res.hand_world_landmarks:
                for slot,(wlm,nlm,hi) in enumerate(
                    zip(h_res.hand_world_landmarks,h_res.hand_landmarks,h_res.handedness)):
                    if slot>=max_hands: break
                    label = hi[0].display_name
                    lms_w = hw(wlm)
                    lms_sc = cs(lms_w)
                    nlms.append(hn(nlm))
                    hness.append(label)
                    cur_hands.append((lms_sc, label))
                    haw[label] = lms_sc
                    # Cloud: accumulate only fingertips + wrist (6 pts per frame)
                    for ti in [0]+TIPS:
                        cloud[label].append(lms_sc[ti].tolist())
                    # CSV
                    row = [fidx,label,f"{tc:.3f}",op_height,environment,scene]
                    row += lms_w[:,0].tolist()+lms_w[:,1].tolist()+lms_w[:,2].tolist()
                    csv_rows.append(row)

            pose_arr = None
            if p_res.pose_world_landmarks:
                pose_arr = pw(p_res.pose_world_landmarks[0])

            out_frame = build_frame(
                frame, nlms, hness, cur_hands, cloud, pose_arr, haw,
                out_w, out_h, skill, description, ts_start, tc, nl_caption,
                environment, scene, op_height)

            out_frame = cv2.resize(out_frame, (out_w,out_h))
            writer.write(out_frame)
            fidx += 1

        pbar.close()

    cap.release(); writer.release()

    with open(csv_path,"w",newline="") as f:
        w=csv.writer(f); w.writerow(csv_hdr); w.writerows(csv_rows)

    print(f"\nDone.")
    print(f"  Video : {out_path}  ({out_w}x{out_h} @ {out_fps:.1f}fps)")
    print(f"  CSV   : {csv_path}  ({len(csv_rows):,} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="6-DoF Embodied AI Data Collection Visualizer")
    ap.add_argument("--input",       required=True, help="Input video path")
    ap.add_argument("--csv",         default="hand_6dof.csv")
    ap.add_argument("--output",      default="vla_output.mp4")
    ap.add_argument("--width",       type=int, default=1600, help="Output width (px)")
    ap.add_argument("--height",      type=int, default=720,  help="Output height (px)")
    ap.add_argument("--skip",        type=int, default=1,    help="Process every Nth frame")
    ap.add_argument("--hands",       type=int, default=2)
    ap.add_argument("--conf",        type=float, default=0.55)
    ap.add_argument("--skill",       default="",  help='Short action label, e.g. "Inflate tire"')
    ap.add_argument("--description", default="",  help="One-line skill description")
    ap.add_argument("--nl_caption",  default="",  help="Third-person NL caption below video")
    ap.add_argument("--ts_start",    type=float, default=0.0)
    ap.add_argument("--ts_end",      type=float, default=0.0)
    ap.add_argument("--environment", default="Factory")
    ap.add_argument("--scene",       default="Workstation")
    ap.add_argument("--op_height",   default="—", help='Operator height e.g. "170cm"')
    a = ap.parse_args()
    process_video(
        input_path=a.input, csv_path=a.csv, out_path=a.output,
        out_w=a.width, out_h=a.height, skip=a.skip,
        max_hands=a.hands, conf=a.conf,
        skill=a.skill, description=a.description, nl_caption=a.nl_caption,
        ts_start=a.ts_start, ts_end=a.ts_end,
        environment=a.environment, scene=a.scene, op_height=a.op_height)

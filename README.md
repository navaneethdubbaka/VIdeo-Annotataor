# VLA Data Collection Annotator

A complete egocentric video annotation pipeline for building **Vision-Language-Action (VLA)** robot training datasets. It processes first-person video of a human operator, runs real-time hand and body pose estimation, computes full 6-DoF kinematics, classifies grasps, measures joint angles, tracks end-effector velocity — and exports everything as an annotated visualisation video, a 167-column CSV, and a JSONL file ready for robot imitation learning.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Why It Matters for Robotics](#why-it-matters-for-robotics)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [CLI Arguments](#cli-arguments)
6. [Output Files](#output-files)
7. [Visual Layout](#visual-layout)
8. [Architecture & Code Walkthrough](#architecture--code-walkthrough)
9. [6-DoF Mathematics](#6-dof-mathematics)
10. [Grasp Classification](#grasp-classification)
11. [Finger Joint Angles](#finger-joint-angles)
12. [Velocity & Acceleration](#velocity--acceleration)
13. [Hierarchical Task Labelling](#hierarchical-task-labelling)
14. [Height Normalisation](#height-normalisation)
15. [CSV Column Reference](#csv-column-reference)
16. [JSONL Record Reference](#jsonl-record-reference)
17. [Batch Pipeline](#batch-pipeline)
18. [Backward Compatibility](#backward-compatibility)

---

## What It Does

Given any first-person (egocentric) video of a human performing a manipulation task, `vla_annotator.py` produces:

| Output | Description |
|---|---|
| Annotated video `.mp4` | 4-panel visualisation with 2D overlay, 3D pose plots, RPY/grasp/joint readouts |
| `hand_6dof.csv` | 167 columns per detected hand per frame — raw XYZ, RPY, joint angles, grasp, velocity |
| `vla_dataset.jsonl` | One JSON record per frame, structured for VLA training pipelines |

---

## Why It Matters for Robotics

A robot learning from human demonstration needs more than pixels. It needs:

- **Where** the hands are in 3D space (XYZ translation)
- **How** the hands are oriented (Roll/Pitch/Yaw rotation) → together = **6-DoF pose**
- **What** grasp the operator is using (power, pinch, lateral…) so the robot can reproduce it
- **How bent** each finger joint is (θ_MCP, θ_PIP, θ_DIP) to replicate the exact hand shape
- **How fast** the hand is moving and in which direction (velocity, acceleration) so the robot can match timing, not just trajectory
- **What task** is being performed at both long-horizon (macro) and short-horizon (micro) levels so a Vision-Language model can condition on text instructions

This tool captures all of the above in a single pass over any video.

---

## Installation

```bash
pip install mediapipe opencv-python tqdm numpy
```

MediaPipe model files are downloaded automatically on first run:
- `hand_landmarker.task` (~8 MB)
- `pose_landmarker_lite.task` (~5 MB)

**Python:** 3.8 or later
**OpenCV:** 4.5+
**MediaPipe:** 0.10+

---

## Quick Start

```bash
# Single video — car wheel disassembly with 3 micro steps
python vla_annotator.py \
  --input factory.mp4 \
  --macro_task "Disassemble a car wheel" \
  --steps "Attach socket to impact wrench;Loosen lug nuts;Remove wheel" \
  --nl_caption "The operator disassembles a car wheel using an impact wrench." \
  --environment "Car Workshop" \
  --scene "Car service" \
  --op_height 162 \
  --robot_height 120 \
  --skip 2
```

```bash
# Data-only mode (no video render, fastest)
python vla_annotator.py \
  --input sample1.mp4 \
  --macro_task "Load metal blank into press die" \
  --op_height 170 \
  --skip 3 \
  --no_video
```

```bash
# Legacy command style (fully backward-compatible)
python vla_annotator.py \
  --input sample1.mp4 \
  --skill "Load metal blank into press die" \
  --description "Operator loads metal blank into hydraulic punch press die cavity." \
  --op_height "162cm" \
  --skip 2
```

---

## CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--input` | str | **required** | Path to the input video file |
| `--output` | str | `vla_output.mp4` | Path for the annotated output video |
| `--csv` | str | `hand_6dof.csv` | Path for the 6-DoF CSV export |
| `--jsonl` | str | `vla_dataset.jsonl` | Path for the JSONL training data export |
| `--width` | int | `1600` | Output video width in pixels |
| `--height` | int | `720` | Output video height in pixels |
| `--skip` | int | `1` | Process every N-th frame (2 = half frame rate) |
| `--hands` | int | `2` | Maximum number of hands to track simultaneously |
| `--conf` | float | `0.55` | Minimum hand detection confidence threshold |
| `--macro_task` | str | `""` | Long-horizon goal label (episode level) |
| `--steps` | str | `""` | Semicolon-separated micro-step labels, divided evenly across the video duration |
| `--nl_caption` | str | `""` | Third-person natural language description shown below video (for VLA language conditioning) |
| `--ts_start` | float | `0.0` | Clip start time in seconds |
| `--ts_end` | float | `0.0` | Clip end time in seconds (0 = full video) |
| `--environment` | str | `"Factory"` | Environment tag (e.g. "Car Workshop", "Kitchen") |
| `--scene` | str | `"Workstation"` | Scene tag (e.g. "Hydraulic Press Station") |
| `--op_height` | str/float | `170` | Operator height in cm — accepts `"162cm"` or `162` |
| `--robot_height` | str/float | `0` | Target robot height in cm for coordinate normalisation (0 = disabled) |
| `--no_video` | flag | off | Skip video rendering — only produce CSV and JSONL |
| `--skill` | str | `""` | **Legacy alias** for `--macro_task` |
| `--description` | str | `""` | **Legacy alias** — appended to macro_task with `—` separator |

---

## Output Files

### Annotated Video

A 4-panel composite video at your specified resolution. Each frame shows the raw camera view, the pose strip, and two 3D projections synchronised to the current timestamp.

### `hand_6dof.csv`

One row per detected hand per processed frame. **167 columns** covering:
- Frame metadata, task labels, timestamps
- Raw 3D world coordinates for all 21 hand landmarks
- Palm orientation (Roll, Pitch, Yaw)
- Height-normalised coordinates
- Dedicated end-effector (wrist) columns
- 14 finger joint angles
- Grasp type, aperture, contact state
- Wrist velocity and acceleration (X, Y, Z + magnitude)

Full column reference: [CSV Column Reference](#csv-column-reference)

### `vla_dataset.jsonl`

One JSON object per frame (only frames where hands are detected). Each record contains nested structures for hand pose, joint angles, grasp state, and kinematics — suitable for direct ingestion by VLA training frameworks such as LeRobot, OpenVLA, or Diffusion Policy.

Full record reference: [JSONL Record Reference](#jsonl-record-reference)

---

## Visual Layout

```
┌──────────────────────┬──────────────┬──────────────────────────┐
│                      │  POSE/GRASP  │   Panel 2: 3D Hand Plot  │
│  Panel 1: Camera     │    Strip     │                          │
│                      │              │  • Orthographic 3D view  │
│  • Raw video feed    │  • R/P/Y     │  • Both hands coloured   │
│  • 2D hand skeleton  │    bars      │  • RGB orientation axes  │
│  • Annotation card   │  • Grasp     │    at every bone segment │
│    (micro step +     │    type      │                          │
│     macro goal +     │  • Aperture  ├──────────────────────────┤
│     progress bar)    │  • Contact   │   Panel 3: Body Pose     │
│  • NL caption        │  • Per-      │                          │
│  • Metadata bar      │    finger    │  • T-pose skeleton       │
│    (environment,     │    joint     │  • Live arm angles       │
│     scene, height)   │    angle     │  • Hand terminators      │
│                      │    bars      │    at wrists             │
└──────────────────────┴──────────────┴──────────────────────────┘
```

**Panel 1 — Camera + Overlay**
The raw video is fill-cropped to the panel size. MediaPipe hand landmarks are projected back onto the 2D image as a coloured skeleton (green = left hand, blue = right hand, finger segments colour-coded individually). An annotation card in the top-left shows the current micro step, a progress bar across all steps, the macro task goal, and a timestamp range. A natural-language caption sits below the video. The metadata bar at the bottom shows Environment, Scene, and Operator Height.

**Centre Strip — POSE / GRASP**
A 185 px-wide vertical panel between the camera and the 3D plots. For each detected hand:
- Three horizontal progress bars for Roll, Pitch, Yaw (coloured red/green/blue, proportional to ±180°)
- Grasp type badge (e.g. `power`, `pinch`, `open`)
- Aperture in centimetres and contact state
- A compact bar-chart table showing the flexion angle of all 5 fingers' joints (Th/Ix/Md/Rg/Pk)

**Panel 2 — 3D Hand Skeleton**
Orthographic 3D projection of both hands in world space. The coordinate system is a right-handed frame with Y-up. A dashed grid cube provides spatial reference. RGB tri-axial orientation arrows are drawn at every bone midpoint showing the local 6-DoF frame. Left hand is rendered in green, right hand in blue.

**Panel 3 — Full Body Pose**
A stylised stick-figure body skeleton using a T-pose base deformed by live MediaPipe pose landmarks (shoulder → elbow → wrist angles). Scaled hand skeletons are attached at both wrists showing the actual hand orientation in body context.

---

## Architecture & Code Walkthrough

```
vla_annotator.py
│
├── CONSTANTS               Colours, landmark indices, T-pose geometry
│
├── 6-DOF MATH
│   ├── palm_frame()        Build right-handed orthonormal frame from palm
│   ├── rpy_from_R()        ZYX Euler decomposition → Roll/Pitch/Yaw
│   ├── normalize_pose()    Isotropic height-scale operator→robot frame
│   ├── joint_angle()       Dot-product angle at a skeleton vertex
│   ├── finger_joint_angles() 14 MCP/PIP/DIP angles across all fingers
│   └── classify_grasp()    Curl-ratio classifier → 7 grasp types
│
├── UTILITIES               Model downloader, timestamp formatter,
│                           landmark array converters (hw, hn, pw, cs)
│
├── 3D PROJECTION           Orthographic project(), depth-sort helpers,
│                           dash_line(), draw_grid(), draw_axes(),
│                           draw_orient() (RGB axes at a 3D point)
│
├── PANELS
│   ├── draw_left()         Camera frame + 2D overlay + annotation card
│   │                       + NL caption + metadata bar
│   ├── draw_info_panel()   Centre strip: RPY bars, grasp, joint angle bars
│   ├── draw_hand_panel()   3D hand skeleton with orientation arrows
│   └── draw_body_panel()   Full-body pose with hand terminators
│
├── build_frame()           Composites all panels into one output frame
│
├── TEMPORAL LOGIC
│   ├── build_step_timeline() Divides [ts_start,ts_end] evenly among steps
│   └── step_at()           Returns current micro step for a timestamp
│
└── process_video()         Main loop: reads frames, runs MediaPipe,
                            computes all metrics, writes CSV/JSONL/video
```

---

## 6-DoF Mathematics

### Palm Frame (`palm_frame`)

A right-handed orthonormal coordinate frame is built from three anatomical reference points:

| Axis | Definition |
|---|---|
| **Y** (finger direction) | Unit vector from wrist (lm 0) to middle-finger MCP (lm 9) |
| **X** (across knuckles) | Index MCP (lm 5) → Pinky MCP (lm 17), Gram–Schmidt orthogonalised against Y |
| **Z** (palm normal) | Cross product X × Y, points away from the palm surface |

```
         lm9 (middle MCP)
          ↑  Y-axis
          |
lm17 ────── lm5     Z-axis out of palm
(pinky)   lm0     X-axis →
         (wrist)
```

### Roll / Pitch / Yaw (`rpy_from_R`)

The ZYX Euler decomposition is applied to the palm rotation matrix:

```
Pitch = arcsin(-R[2,0])
Roll  = atan2(R[2,1], R[2,2])
Yaw   = atan2(R[1,0], R[0,0])
```

All angles are in degrees. Range: Roll ±180°, Pitch ±90°, Yaw ±180°. Gimbal lock is handled when |cos(pitch)| < 1e-6.

---

## Grasp Classification

`classify_grasp(lms_w)` returns `(grasp_type, aperture_m, contact_state)`.

### Finger Curl Ratio

For each finger, a dimensionless curl ratio is computed:

```
curl = dist(fingertip, wrist) / dist(MCP, wrist)
```

- `curl ≈ 1.0` → finger fully extended (straight)
- `curl < 0.65` → finger fully curled (closed fist)

| Landmark indices used | Finger |
|---|---|
| tip=4, mcp=1 | Thumb |
| tip=8, mcp=5 | Index |
| tip=12, mcp=9 | Middle |
| tip=16, mcp=13 | Ring |
| tip=20, mcp=17 | Pinky |

### Grasp Type Rules (checked in priority order)

| Type | Rule |
|---|---|
| `open` | All 5 curl ratios > 0.8 |
| `pinch` | Thumb + index curled (< 0.7), middle/ring/pinky extended (> 0.75) |
| `tripod` | Thumb + index + middle curled (< 0.7), ring + pinky extended (> 0.75) |
| `power` | All 5 curled (< 0.65) |
| `lateral` | Thumb extended (> 0.8), rest curled (< 0.65), thumb near index MCP (< 3 cm) |
| `hook` | Thumb extended (> 0.8), rest curled (< 0.65), thumb far from index MCP |
| `unknown` | None of the above matched |

### Aperture & Contact State

```
aperture = dist(thumb_tip lm4, index_tip lm8)   [metres]

contact_state:
  "closed"  if aperture < 0.02 m  (2 cm)
  "partial" if aperture 0.02–0.08 m
  "open"    if aperture > 0.08 m  (8 cm)
```

---

## Finger Joint Angles

`finger_joint_angles(lms_w)` returns a dict of 14 flexion angles in degrees.

The angle at a joint vertex **B** between segments **A→B** and **B→C** is:

```
θ = arccos( dot(B→A, B→C) / (|B→A| × |B→C|) )
```

- `θ = 180°` → fully straight (extended)
- `θ = 0°` → fully folded back on itself

### Landmark Triplets Used

```
Finger   Joint     Landmarks (A, B, C)
───────────────────────────────────────────
Thumb    MCP       lm0, lm1, lm2
Thumb    IP        lm1, lm2, lm3
Index    MCP       lm0, lm5, lm6
Index    PIP       lm5, lm6, lm7
Index    DIP       lm6, lm7, lm8
Middle   MCP       lm0, lm9,  lm10
Middle   PIP       lm9,  lm10, lm11
Middle   DIP       lm10, lm11, lm12
Ring     MCP       lm0, lm13, lm14
Ring     PIP       lm13, lm14, lm15
Ring     DIP       lm14, lm15, lm16
Pinky    MCP       lm0, lm17, lm18
Pinky    PIP       lm17, lm18, lm19
Pinky    DIP       lm18, lm19, lm20
```

MCP = Metacarpophalangeal (base knuckle)
PIP = Proximal Interphalangeal (middle knuckle)
DIP = Distal Interphalangeal (top knuckle)
IP  = Interphalangeal (thumb only — no PIP/DIP distinction)

---

## Velocity & Acceleration

End-effector kinematics are computed from the wrist landmark (lm 0) using finite differences across consecutive detected frames.

### State Tracking

A `hand_state` dict is maintained per hand across the entire video:

```python
hand_state = {
    "Left":  {"prev_xyz": None, "prev_vel": None, "prev_ts": None},
    "Right": {"prev_xyz": None, "prev_vel": None, "prev_ts": None},
}
```

### Computation Per Frame

```
dt  = current_ts - prev_ts          (seconds; fallback: skip/fps on first frame)

velocity  = (ee_xyz - prev_xyz) / dt     [m/s, 3-vector]
speed     = |velocity|                   [m/s, scalar]

accel     = (velocity - prev_vel) / dt   [m/s², 3-vector]
accel_mag = |accel|                      [m/s², scalar]
```

**First frame for each hand:** velocity and acceleration are set to `(0, 0, 0)`.
**Zero-dt guard:** if two consecutive processed frames share the same timestamp (rare codec edge case), `dt` is forced to `skip/fps`.

The velocity is computed in MediaPipe world space (metres). Units are physically correct for a real-world scene.

---

## Hierarchical Task Labelling

The system uses a two-tier language hierarchy matching the VLA training paradigm:

```
Episode (macro_task):  "Disassemble a car wheel"
 │
 ├── Step 1 (micro):   "Attach socket to impact wrench"
 ├── Step 2 (micro):   "Loosen lug nuts"
 └── Step 3 (micro):   "Remove wheel"
```

Steps are passed as a semicolon-separated string via `--steps`. The video duration `[ts_start, ts_end]` is divided evenly among the steps. The correct step label is resolved per-frame and written to every CSV row and JSONL record.

The progress bar in the annotation card visually reflects how far through the step sequence the current timestamp is.

---

## Height Normalisation

A human operator of height H₁ views a workbench from a different eye level than a robot of height H₂. Raw MediaPipe world coordinates are in a camera-relative frame anchored to the operator's perspective. To make training data reusable across robot morphologies, all XYZ coordinates can be rescaled:

```
xyz_normalised = xyz_raw × (robot_height / operator_height)
```

This is a simple isotropic (uniform) scale — it compresses or expands all three axes equally. More sophisticated affine transforms (accounting for camera tilt, table height, etc.) can be applied downstream using the raw and normalised coordinates together.

- **Raw coordinates** (`x0-x20`, `y0-y20`, `z0-z20`): original MediaPipe world metres
- **Normalised coordinates** (`nx0-nx20`, `ny0-ny20`, `nz0-nz20`): robot-frame scaled metres
- Set `--robot_height 0` to disable normalisation (normalised columns equal raw)

---

## CSV Column Reference

**Total: 167 columns per row** (one row = one hand, one processed frame)

### Group 1 — Metadata (10 columns)

| Column | Type | Description |
|---|---|---|
| `frame` | int | Frame index in the input video |
| `hand` | str | `"Left"` or `"Right"` |
| `ts_sec` | float | Timestamp in seconds (`frame / fps`) |
| `macro_task` | str | Long-horizon episode goal |
| `micro_step` | str | Current short-horizon step label |
| `step_idx` | int | 1-based index of current step |
| `op_height_cm` | float | Operator height in centimetres |
| `robot_height_cm` | float | Target robot height (0 = normalisation disabled) |
| `environment` | str | Environment tag |
| `scene` | str | Scene tag |

### Group 2 — Raw 3D World Coordinates (63 columns)

| Columns | Description |
|---|---|
| `x0` … `x20` | X coordinate of each of the 21 hand landmarks (metres, world space) |
| `y0` … `y20` | Y coordinate (Y-up, flipped from MediaPipe convention) |
| `z0` … `z20` | Z coordinate (depth from camera) |

Landmark index map:

```
0=wrist   1-4=thumb   5-8=index   9-12=middle   13-16=ring   17-20=pinky
                      (MCP,PIP,DIP,tip per finger)
```

### Group 3 — Palm Orientation (3 columns)

| Column | Description |
|---|---|
| `roll_deg` | Palm roll angle in degrees (ZYX Euler, rotation around X) |
| `pitch_deg` | Palm pitch angle in degrees (rotation around Y) |
| `yaw_deg` | Palm yaw angle in degrees (rotation around Z) |

### Group 4 — Height-Normalised Coordinates (63 columns)

| Columns | Description |
|---|---|
| `nx0` … `nx20` | Height-normalised X for each landmark |
| `ny0` … `ny20` | Height-normalised Y for each landmark |
| `nz0` … `nz20` | Height-normalised Z for each landmark |

### Group 5 — End-Effector (3 columns)

| Column | Description |
|---|---|
| `ee_x` | Wrist X position, world metres (= `x0`, explicitly named for IK solvers) |
| `ee_y` | Wrist Y position, world metres |
| `ee_z` | Wrist Z position, world metres |

### Group 6 — Finger Joint Angles (14 columns)

| Column | Joint | Finger |
|---|---|---|
| `thumb_mcp` | Metacarpophalangeal | Thumb |
| `thumb_ip` | Interphalangeal | Thumb |
| `idx_mcp` | Metacarpophalangeal | Index |
| `idx_pip` | Proximal Interphalangeal | Index |
| `idx_dip` | Distal Interphalangeal | Index |
| `mid_mcp` | Metacarpophalangeal | Middle |
| `mid_pip` | Proximal Interphalangeal | Middle |
| `mid_dip` | Distal Interphalangeal | Middle |
| `ring_mcp` | Metacarpophalangeal | Ring |
| `ring_pip` | Proximal Interphalangeal | Ring |
| `ring_dip` | Distal Interphalangeal | Ring |
| `pinky_mcp` | Metacarpophalangeal | Pinky |
| `pinky_pip` | Proximal Interphalangeal | Pinky |
| `pinky_dip` | Distal Interphalangeal | Pinky |

All angles in degrees. `180°` = straight, lower = more flexed.

### Group 7 — Grasp Configuration (3 columns)

| Column | Type | Description |
|---|---|---|
| `grasp_type` | str | `open` / `pinch` / `tripod` / `power` / `lateral` / `hook` / `unknown` |
| `finger_aperture_m` | float | Euclidean distance between thumb tip and index tip, in metres |
| `contact_state` | str | `open` / `partial` / `closed` |

### Group 8 — Velocity & Acceleration (8 columns)

| Column | Unit | Description |
|---|---|---|
| `ee_vx` | m/s | Wrist velocity along X axis |
| `ee_vy` | m/s | Wrist velocity along Y axis |
| `ee_vz` | m/s | Wrist velocity along Z axis |
| `ee_speed` | m/s | Wrist speed (velocity magnitude) |
| `ee_ax` | m/s² | Wrist acceleration along X axis |
| `ee_ay` | m/s² | Wrist acceleration along Y axis |
| `ee_az` | m/s² | Wrist acceleration along Z axis |
| `ee_accel` | m/s² | Wrist acceleration magnitude |

---

## JSONL Record Reference

Each line in `vla_dataset.jsonl` is a self-contained JSON object for one frame:

```jsonc
{
  "video":          "factory.mp4",
  "frame":          142,
  "ts_sec":         4.733,
  "macro_task":     "Disassemble a car wheel",
  "micro_step":     "Loosen lug nuts",
  "step_idx":       2,
  "total_steps":    3,
  "environment":    "Car Workshop",
  "scene":          "Car service",
  "op_height_cm":   162.0,
  "robot_height_cm": 120.0,
  "hands": [
    {
      "hand":               "Right",
      "xyz":                [[x0,y0,z0], ..., [x20,y20,z20]],   // 21×3 raw world coords
      "roll":               -12.4,
      "pitch":              -5.1,
      "yaw":                54.3,
      "xyz_norm":           [[nx0,ny0,nz0], ..., [nx20,ny20,nz20]], // height-normalised
      "ee_xyz":             [0.031, -0.012, 0.008],              // wrist position
      "joint_angles": {
        "thumb_mcp": 157.3, "thumb_ip": 163.1,
        "idx_mcp":   122.4, "idx_pip":  140.0, "idx_dip": 166.2,
        "mid_mcp":   118.7, "mid_pip":  131.4, "mid_dip": 155.8,
        "ring_mcp":  127.3, "ring_pip":  125.0, "ring_dip": 180.0,
        "pinky_mcp": 122.1, "pinky_pip": 140.2, "pinky_dip": 149.3
      },
      "grasp_type":         "power",
      "finger_aperture_m":  0.037,
      "contact_state":      "partial",
      "ee_velocity":        [0.023, -0.008, 0.001],             // [vx, vy, vz] m/s
      "ee_speed":           0.0245,
      "ee_accel":           [0.004, -0.001, 0.0],               // [ax, ay, az] m/s²
      "ee_accel_mag":       0.0041
    }
  ]
}
```

---

## Batch Pipeline

`vla_pipeline.py` processes an entire dataset folder structure automatically.

```bash
python vla_pipeline.py \
  --dataset_root factory001_worker001_part01/ \
  --macro_task "Factory assembly task" \
  --steps "Pick component;Align part;Fasten bolt;Inspect result" \
  --environment "Factory" \
  --op_height 170 \
  --robot_height 120 \
  --skip 3 \
  --no_video \
  --max_clips 10
```

**Input structure expected:**
```
dataset_root/
  part_A/
    clip_001.mp4
    clip_001.json   ← {"factory_id":..., "worker_id":..., "duration_sec":...}
    clip_002.mp4
    clip_002.json
  part_B/
    ...
```

**Outputs in `vla_dataset/`:**
```
vla_dataset/
  clip_001.csv
  clip_001.jsonl
  clip_001_vis.mp4    (if --no_video not set)
  ...
  dataset_manifest.json   ← index of all clips with metadata and stats
  all_frames.jsonl        ← merged JSONL of all clips for direct training use
```

---

## Backward Compatibility

All commands written for the original version continue to work unchanged:

| Old argument | Handled as |
|---|---|
| `--skill "..."` | Treated as `--macro_task` |
| `--description "..."` | Appended to macro_task with `—` separator |
| `--op_height "162cm"` | `"cm"` suffix stripped and parsed to float |

---

## Coordinate System

MediaPipe provides world-space landmarks in metres, with Z pointing toward the camera and Y pointing downward. This tool flips Y so that **Y points upward**, giving a standard right-handed coordinate frame consistent with most robotics and 3D graphics conventions:

```
        Y (up)
        │
        │
        └──── X (right)
       /
      Z (toward viewer)
```

All exported coordinates use this Y-up convention.

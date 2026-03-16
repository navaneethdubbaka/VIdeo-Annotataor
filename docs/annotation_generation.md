# Annotation Generation Overview

This document explains, in plain terms, how each annotation produced by the VLA Data Collection Annotator is generated — from the raw video input through to the final values written to the CSV and JSONL outputs.

---

## Pipeline at a Glance

```
Video file
    │
    ▼
Frame extraction  (every N-th frame, configurable)
    │
    ├──► MediaPipe Hand Landmarker  →  21 3D hand landmarks per hand
    └──► MediaPipe Pose Landmarker  →  33 body landmarks (visualisation only)
              │
              ▼ (Y-axis corrected to world-up convention)
              │
    ┌─────────────────────────────────────────────────┐
    │  Per detected hand:                              │
    │   1. Raw 3D landmarks                            │
    │   2. Palm orientation (Roll / Pitch / Yaw)       │
    │   3. Height-normalised coordinates               │
    │   4. End-effector position                       │
    │   5. Finger joint angles                         │
    │   6. Grasp classification                        │
    │   7. Velocity and acceleration                   │
    │   8. Task labels (assigned from metadata)        │
    └─────────────────────────────────────────────────┘
              │
              ▼
    CSV row + JSONL record written
```

---

## Input Sources

### 1. Video frames

The pipeline reads the video frame by frame. A configurable **frame stride** (default: every frame) controls how frequently annotations are produced. The timestamp assigned to each frame is derived from the video's frame index and reported frame rate.

### 2. MediaPipe Hand Landmarker

Google's MediaPipe Hand Landmarker model detects and tracks up to two hands simultaneously. For each detected hand it returns 21 three-dimensional landmark positions in metric world coordinates (metres). The model runs in **VIDEO mode**, meaning it uses temporal information across frames to maintain stable tracking.

Key detection parameters (all configurable):

| Parameter | Default | Effect |
|-----------|---------|--------|
| Max hands | 2 | Maximum simultaneous hand tracks |
| Detection confidence | 0.55 | Minimum score to detect a new hand |
| Presence confidence | 0.50 | Minimum score to keep a tracked hand |
| Tracking confidence | 0.50 | Minimum score to continue tracking |

### 3. MediaPipe Pose Landmarker

A separate body-pose model detects 33 skeletal landmarks. This is used only to render the body visualisation panel in the annotated video output; it does not contribute any values to the CSV or JSONL files.

### 4. Operator metadata

Supplied as command-line arguments at run time: task description, micro-step labels, operator height, robot height, environment and scene tags.

---

## Stage 0 — Coordinate System Correction

MediaPipe world landmarks use a Y-down convention (Y increases downward). Immediately after extraction, the Y component of every landmark is negated to produce the Y-up convention used throughout the rest of the pipeline. This single step ensures that all downstream calculations interpret "up" correctly.

---

## Stage 1 — Raw 3D Landmarks

No computation is performed beyond Stage 0. The corrected 21 × 3 position array is written directly to:
- CSV columns `x0`–`x20`, `y0`–`y20`, `z0`–`z20`
- JSONL key `xyz`

These values are the unmodified metric positions of every hand landmark in the operator's local world space.

---

## Stage 2 — Palm Orientation Frame

To express how the palm is oriented in space, a local coordinate frame is constructed from four anatomical landmarks: the **wrist** (lm 0), the **index MCP** (lm 5), the **middle MCP** (lm 9), and the **pinky MCP** (lm 17).

The three axes are built as follows:

| Axis | Direction | Constructed from |
|------|-----------|-----------------|
| Y (finger-pointing) | Wrist toward middle MCP | Normalised vector lm0 → lm9 |
| X (across knuckles) | Index MCP toward pinky MCP | Vector lm5 → lm17, orthogonalised against Y using Gram-Schmidt |
| Z (palm normal) | Outward from palm dorsum | Cross product of X and Y |

The result is a 3 × 3 orthonormal rotation matrix **R** that encodes the complete orientation of the palm relative to the world frame.

**Gram-Schmidt orthogonalisation** ensures that the X axis is exactly perpendicular to Y even when the knuckle row is not perfectly horizontal relative to the finger direction. This gives stable, consistent results regardless of hand pose.

---

## Stage 3 — Roll / Pitch / Yaw Decomposition

The rotation matrix **R** from Stage 2 is decomposed into three Euler angles using the **ZYX convention** (first yaw around Z, then pitch around Y, then roll around X).

The formulas are:

```
pitch = arcsin(−R[2,0])                    range: −90° to +90°
roll  = atan2(R[2,1], R[2,2])             range: −180° to +180°
yaw   = atan2(R[1,0], R[0,0])             range: −180° to +180°
```

When the cosine of pitch is near zero (pitch approaching ±90°), the rotation enters **gimbal lock** — a mathematical singularity where roll and yaw become indistinguishable. In this case yaw is set to 0° and roll absorbs the full rotation. Frames where `|pitch_deg| > 80°` should be treated with caution.

---

## Stage 4 — Height Normalisation

To transfer operator hand movements to a robot with a different physical scale, every landmark position is multiplied by a single scale factor:

> **scale = robot height (cm) ÷ operator height (cm)**

Because the same factor is applied equally to X, Y, and Z, all angles and shape proportions are preserved — only absolute distances change. The result is stored in the `nx`, `ny`, `nz` columns.

When the robot height is not specified (default = 0), the normalised columns are identical to the raw columns and no scaling is applied.

---

## Stage 5 — End-Effector Position

The wrist (landmark 0) position is extracted directly from the corrected landmark array and written to `ee_x`, `ee_y`, `ee_z`. These values are numerically the same as `x0`, `y0`, `z0`; the duplication exists as a convenience for robot control and machine-learning frameworks that expect a named end-effector field.

---

## Stage 6 — Finger Joint Angles

Each joint angle is the geometric angle formed at a vertex B by the two bone segments that meet there (B→A and B→C):

> **θ = arccos( (B→A · B→C) / (|B→A| × |B→C|) )**

where · denotes the dot product. The result is in degrees. The formula is **scale-invariant** because both vectors are normalised before the dot product, so the value depends only on direction, not hand size.

The 14 joints are computed from the following landmark triplets:

| Column | Proximal (A) | Vertex (B) | Distal (C) |
|--------|-------------|-----------|-----------|
| `thumb_mcp` | Wrist (0) | lm 1 | lm 2 |
| `thumb_ip` | lm 1 | lm 2 | lm 3 |
| `idx_mcp` | Wrist (0) | lm 5 | lm 6 |
| `idx_pip` | lm 5 | lm 6 | lm 7 |
| `idx_dip` | lm 6 | lm 7 | lm 8 |
| `mid_mcp` | Wrist (0) | lm 9 | lm 10 |
| `mid_pip` | lm 9 | lm 10 | lm 11 |
| `mid_dip` | lm 10 | lm 11 | lm 12 |
| `ring_mcp` | Wrist (0) | lm 13 | lm 14 |
| `ring_pip` | lm 13 | lm 14 | lm 15 |
| `ring_dip` | lm 14 | lm 15 | lm 16 |
| `pinky_mcp` | Wrist (0) | lm 17 | lm 18 |
| `pinky_pip` | lm 17 | lm 18 | lm 19 |
| `pinky_dip` | lm 18 | lm 19 | lm 20 |

Note: MCP joints use the wrist (lm 0) as the proximal reference rather than the CMC joint. This provides a more stable anchor but slightly departs from strict anatomical convention.

---

## Stage 7 — Grasp Classification

### Curl ratio

For each finger, a **curl ratio** is computed:

> **curl = distance(fingertip, wrist) ÷ distance(MCP knuckle, wrist)**

A ratio greater than 1.0 means the fingertip is farther from the wrist than the knuckle — the finger is extended. A ratio well below 1.0 means the tip has moved much closer to the wrist — the finger is curled inward.

### Classification rules

The five curl ratios (one per finger) are combined using threshold rules evaluated in priority order:

| Class | Rule |
|-------|------|
| `open` | All five curl ratios > 0.80 |
| `pinch` | Thumb + index < 0.70; middle, ring, pinky > 0.75 |
| `tripod` | Thumb + index + middle < 0.70; ring + pinky > 0.75 |
| `power` | All five curl ratios < 0.65 |
| `lateral` | Thumb > 0.80; all others < 0.65; thumb tip within 3 cm of index MCP |
| `hook` | Thumb > 0.80; all others < 0.65; thumb tip more than 3 cm from index MCP |
| `unknown` | No rule matched |

The first matching rule in the table above is applied. Frames during transitions between stable grasps typically fall into `unknown`.

### Aperture

The **finger aperture** is the straight-line distance between the thumb tip (lm 4) and the index tip (lm 8), in metres. It is independent of the grasp classification.

### Contact state

Three contact states are derived from the aperture:

| State | Threshold |
|-------|-----------|
| `closed` | aperture < 0.02 m |
| `partial` | 0.02 m ≤ aperture ≤ 0.08 m |
| `open` | aperture > 0.08 m |

---

## Stage 8 — Velocity and Acceleration

Velocity and acceleration are computed by **first-order finite differencing** on the wrist position between consecutive processed frames.

> **velocity = (current wrist position − previous wrist position) ÷ Δt**

> **acceleration = (current velocity − previous velocity) ÷ Δt**

where **Δt** is the elapsed time between the two frames in seconds. The speed and acceleration magnitudes are the Euclidean norms of the respective 3D vectors.

A separate state (previous position, previous velocity, previous timestamp) is maintained for each hand independently. The first frame processed for any given hand always yields zero velocity and zero acceleration, since there is no prior position to compare against.

**Noise sensitivity:** Finite differencing amplifies high-frequency jitter in landmark positions. A single-frame position error of ~10 mm at 30 fps produces a velocity uncertainty of ~0.4 m/s. Post-processing with a short temporal smoothing filter (3–5 frames) is recommended before using these values as learning targets.

---

## Stage 9 — Task Labels

Micro-step labels are assigned by dividing the video duration evenly among the steps provided by the operator:

> **step duration = (end timestamp − start timestamp) ÷ number of steps**

Each frame receives the label of whichever step's time window contains its timestamp. The final step extends to the end of the clip. Macro-task text is identical for every frame in the video.

---

## Processing Order per Frame

For each video frame, the pipeline executes stages in the following order:

1. Extract frame and submit to MediaPipe
2. For each detected hand, apply Y-axis correction (Stage 0)
3. Write raw landmarks (Stage 1)
4. Construct palm frame (Stage 2) and decompose to RPY (Stage 3)
5. Compute normalised coordinates (Stage 4)
6. Extract end-effector position (Stage 5)
7. Compute all 14 joint angles (Stage 6)
8. Classify grasp, aperture, and contact state (Stage 7)
9. Compute velocity and acceleration (Stage 8)
10. Assign task labels (Stage 9)
11. Write CSV row and JSONL record
12. Render annotated video frame

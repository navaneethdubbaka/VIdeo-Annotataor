# Annotation Types Reference

This document describes every category of annotation produced by the VLA Data Collection Annotator. For each annotation type it explains the physical meaning, the units and expected value ranges, and how the output can be visualised or spot-checked.

---

## Coordinate System

All 3D coordinates use a **right-handed, Y-up, metric** space.

```
        +Y  (upward)
         |
         |
         +----------> +X  (to the operator's right)
        /
       /
     +Z  (toward the camera)
```

Distances are in **metres**. The origin sits approximately at the operator's wrist. This convention is consistent with most robot kinematics libraries and simulation environments.

---

## Hand Landmark Numbering

MediaPipe identifies 21 anatomical points on each hand. The annotator preserves this numbering throughout:

| Index | Anatomical point | Finger |
|-------|-----------------|--------|
| 0 | Wrist | — |
| 1–4 | CMC → Tip | Thumb |
| 5–8 | MCP → Tip | Index |
| 9–12 | MCP → Tip | Middle |
| 13–16 | MCP → Tip | Ring |
| 17–20 | MCP → Tip | Pinky |

MCP = metacarpophalangeal (knuckle); PIP = proximal interphalangeal; DIP = distal interphalangeal.

---

## Annotation Group 1 — Metadata

These fields label every row with its context. They appear in both the CSV and the JSONL record.

| Field | Description |
|-------|-------------|
| `frame` | Zero-based video frame index |
| `hand` | `Left` or `Right` |
| `ts_sec` | Timestamp from video start, in seconds |
| `macro_task` | Long-horizon task description (e.g. *"Guide fabric through sewing machine"*) |
| `micro_step` | Current short-horizon sub-task label; blank if no steps were specified |
| `step_idx` | 1-based index of the active micro step |
| `op_height_cm` | Operator standing height in centimetres |
| `robot_height_cm` | Target robot height in centimetres; `0` means height normalisation is inactive |
| `environment` | Broad scene tag (e.g. *"Garment Factory"*) |
| `scene` | Specific workstation tag (e.g. *"Sewing Station"*) |

**Visualisation:** Filter by `hand` and plot `ts_sec` on the time axis to align all other annotations with the video timeline.

---

## Annotation Group 2 — Raw 3D Landmarks

**Fields:** `x0`–`x20`, `y0`–`y20`, `z0`–`z20` (63 columns in CSV; `xyz` array in JSONL)

### What it is

The three-dimensional world position of each of the 21 hand landmarks, in metres. These are the foundational measurements from which all other annotations are derived.

### Value ranges

| Axis | Typical range | Notes |
|------|--------------|-------|
| X | −0.15 m to +0.15 m | Lateral spread |
| Y | −0.10 m to +0.10 m | Vertical extent |
| Z | 0.00 m to +0.12 m | Depth; near-zero at wrist, positive toward camera for fingertips |

The XY plane is substantially more accurate than Z. Z is estimated from a monocular camera and carries an expected error of 20–30 mm, compared with 8–12 mm in X and Y.

### How to spot anomalies

- Any coordinate with absolute value greater than 0.5 m indicates a detection failure (hand out of frame or heavily occluded).
- All 21 landmarks sharing identical values means MediaPipe lost tracking on that frame.

### Visualisation

A single frame can be rendered as a 3D scatter plot of 21 points, connected with the standard hand skeleton edges, to verify pose quality. A trajectory view plots landmark 0 (wrist) position over time.

---

## Annotation Group 3 — Palm Orientation (Roll / Pitch / Yaw)

**Fields:** `roll_deg`, `pitch_deg`, `yaw_deg`

### What it is

The orientation of the palm expressed as three rotation angles in degrees, following the ZYX Euler convention (also called aerospace angles). The palm coordinate frame is anchored to the wrist and built from the knuckle positions.

| Angle | Range | Physical meaning |
|-------|-------|-----------------|
| Roll | −180° to +180° | Palm rolling left or right around the finger-pointing axis |
| Pitch | −90° to +90° | Palm tilting up or down |
| Yaw | −180° to +180° | Wrist twisting (pronation / supination) |

### Anatomical reference

| Hand posture | Approx. roll | Approx. pitch | Approx. yaw |
|-------------|-------------|--------------|------------|
| Palm facing camera, fingers pointing up | ~0° | ~0° | ~0° |
| Palm facing the floor | ~±180° | ~0° | ~0° |
| Palm facing upward (supinated) | ~+90° | ~0° | ~0° |
| Reaching upward at 45° | ~0° | ~−45° | ~0° |

### Important limitation — gimbal lock

When `|pitch_deg|` approaches 90°, the roll and yaw axes become mathematically co-planar. In this region `yaw_deg` is forced to 0° and `roll_deg` absorbs the full rotation. Frames where `|pitch_deg| > 80°` should be flagged and excluded from orientation analyses.

### Observed values

In the sewing-machine task: roll ≈ −47°, pitch ≈ −29°, yaw ≈ +55°. In the packing task: roll ≈ −60°, pitch ≈ −51°, yaw ≈ −31°.

### Visualisation

A time-series line chart of all three angles on a shared axis (−180° to +180°) clearly shows wrist orientation changes throughout a task. A shaded band at ±80°–90° highlights the gimbal lock zone.

---

## Annotation Group 4 — Height-Normalised Coordinates

**Fields:** `nx0`–`nx20`, `ny0`–`ny20`, `nz0`–`nz20` (63 columns in CSV; `xyz_norm` in JSONL)

### What it is

A rescaled copy of the raw 3D landmarks that maps the operator's gesture space to a robot's physical scale. The scaling factor is:

> **scale = robot height ÷ operator height**

All three axes are scaled by the same factor, preserving orientation and proportions. When `robot_height_cm` is 0 (the default), these columns are numerically identical to the raw landmarks.

### When to use each

| Use case | Recommended coordinates |
|----------|------------------------|
| Training on a same-scale simulated robot | Raw (`x`, `y`, `z`) |
| Transferring to a smaller or larger physical robot | Normalised (`nx`, `ny`, `nz`) |

### Example

For an operator 165 cm tall and a robot 120 cm tall, every coordinate is multiplied by 0.727. A wrist X position of 0.030 m becomes 0.022 m in normalised space.

---

## Annotation Group 5 — End-Effector Position

**Fields:** `ee_x`, `ee_y`, `ee_z`

### What it is

The 3D world position of the wrist (landmark 0), provided as a named, standalone field. The values are numerically identical to `x0`, `y0`, `z0`; the separate column exists because robot inverse-kinematics solvers and imitation-learning frameworks typically expect an explicitly named `ee_xyz` vector.

| Field | Unit | Typical range |
|-------|------|--------------|
| `ee_x` | m | −0.15 to +0.15 |
| `ee_y` | m | −0.10 to +0.10 |
| `ee_z` | m | 0.00 to +0.12 |

### Visualisation

Plotting `ee_x`, `ee_y`, `ee_z` as a 3D line coloured by speed gives a clear picture of the operator's hand trajectory over the entire episode.

---

## Annotation Group 6 — Finger Joint Angles

**Fields:** `thumb_mcp`, `thumb_ip`, `idx_mcp`, `idx_pip`, `idx_dip`, `mid_mcp`, `mid_pip`, `mid_dip`, `ring_mcp`, `ring_pip`, `ring_dip`, `pinky_mcp`, `pinky_pip`, `pinky_dip` (14 columns)

### What it is

The flexion angle at each finger joint, in degrees. **180° means the joint is perfectly straight; lower values indicate increasing curl.** The scale runs from ~180° (fully extended) down to ~90° (maximum curl for most joints).

### Joint map

```
     Tip
      |  DIP joint
      |  PIP joint
      |  MCP joint
      |
    Knuckle
```

The thumb has an MCP and an IP joint (no PIP/DIP distinction).

### Interpretation guide

| Angle range | Meaning |
|-------------|---------|
| 160°–180° | Joint nearly straight — finger extended |
| 130°–160° | Moderate flexion |
| 100°–130° | Strong curl |
| Below 100° | Maximum grip (tight fist) |

### Grasp shape signatures

| Grasp posture | MCP range | PIP range |
|---------------|-----------|-----------|
| Open hand | 150°–180° | 160°–180° |
| Pinch (thumb + index) | index MCP: 130–150° | index PIP: 110–140° |
| Full power grip | All: 100°–130° | All: 90°–120° |

### Observed values

Sewing task (relaxed open hand): most joints in the range 127°–170°. Packing task: range 122°–179°.

### Visualisation

A **radar (spider) chart** with one spoke per joint gives an instant visual fingerprint of the hand shape for any frame. A **heatmap** of all 14 joints over time clearly shows when the hand opens and closes across the episode.

---

## Annotation Group 7 — Grasp Configuration

**Fields:** `grasp_type`, `finger_aperture_m`, `contact_state`

### 7a. Grasp Type

Seven discrete categories describe the hand's functional grip:

| Value | Description | Typical use |
|-------|-------------|-------------|
| `open` | All five fingers extended | Reaching, releasing an object |
| `pinch` | Thumb and index curled; remaining fingers open | Picking up small or delicate items |
| `tripod` | Thumb, index, and middle curled; ring and pinky open | Pen grip, precision tools |
| `power` | All five fingers tightly curled | Jar lids, handles, heavy tools |
| `lateral` | Thumb extended, all other fingers curled, thumb resting near the index knuckle | Key grip, turning dial |
| `hook` | Thumb extended, all other fingers curled, thumb clear of the index | Carrying a bag handle |
| `unknown` | No rule matched — typically a transition posture | Between-grasp transitions |

Categories are assigned in the order listed above (first match wins). The `unknown` class is expected on frames that fall between two stable grasps.

### 7b. Finger Aperture

The straight-line distance between the thumb tip (landmark 4) and the index tip (landmark 8), in metres.

| Approximate value | Context |
|------------------|---------|
| 0.00–0.02 m | Thumb and index making or very near contact |
| 0.02–0.08 m | Partially open — approaching or departing contact |
| 0.08–0.12 m | Fully open hand |

### 7c. Contact State

Derived from the aperture:

| Value | Aperture threshold | Meaning |
|-------|-------------------|---------|
| `closed` | < 0.02 m | Thumb and index tips within 2 cm — gripping or touching |
| `partial` | 0.02–0.08 m | Near-contact; pinch forming or releasing |
| `open` | > 0.08 m | Hand wide open |

Note: contact state reflects only the thumb–index relationship. Other finger combinations are not assessed.

### Observed values

Both sample tasks produced predominantly `open` grasp type with `partial` contact state (aperture 0.047–0.063 m), consistent with a relaxed working hand.

### Visualisation

- A **pie or bar chart** of grasp-type counts per video shows the distribution of hand postures across the task.
- A **scatter plot** of aperture over time, with points coloured by contact state, shows the rhythm of grip opening and closing.

---

## Annotation Group 8 — Velocity and Acceleration

**Fields:** `ee_vx`, `ee_vy`, `ee_vz`, `ee_speed`, `ee_ax`, `ee_ay`, `ee_az`, `ee_accel`

### What it is

The instantaneous velocity and acceleration of the wrist (end-effector), computed by comparing its position between consecutive processed frames.

| Field | Unit | Description |
|-------|------|-------------|
| `ee_vx/vy/vz` | m/s | Velocity components along X, Y, Z |
| `ee_speed` | m/s | Velocity magnitude — scalar hand speed |
| `ee_ax/ay/az` | m/s² | Acceleration components along X, Y, Z |
| `ee_accel` | m/s² | Acceleration magnitude |

### Value ranges

| Metric | Typical range | Notes |
|--------|--------------|-------|
| Speed | 0–0.5 m/s | Slow manipulation tasks; rapid actions up to ~1 m/s |
| Acceleration | 0–5 m/s² | Spikes up to ~15 m/s² at sharp direction reversals |

The first frame for each hand track always records zero velocity and zero acceleration — there is no previous position to difference against.

### Interpretation

| Speed | Acceleration | Meaning |
|-------|-------------|---------|
| Low (< 0.05 m/s) | Low | Steady hold or fine micro-adjustment |
| Medium (0.05–0.3 m/s) | Low | Smooth sweep or repositioning |
| Any | High (> 3 m/s²) | Direction reversal, abrupt start, or stop |
| Spike (> 1 m/s) | Very high | Single-frame landmark noise — treat as artifact |

Raw finite-difference velocity is sensitive to per-frame landmark jitter. A 3–5 frame smoothing filter is recommended before using velocity or acceleration as training targets.

### Observed values

Sewing task: speed 0–0.19 m/s, acceleration 0–3.3 m/s². Both consistent with slow, controlled fabric manipulation.

### Visualisation

A time-series of `ee_speed` with `grasp_type` overlaid as a colour or secondary axis clearly shows how movement speed varies with grip posture throughout the episode.

---

## Annotation Group 9 — Task Labels

**Fields:** `macro_task`, `micro_step`, `step_idx`

### What it is

Hierarchical text labels that describe what the operator is doing at two levels of granularity.

| Field | Level | Example |
|-------|-------|---------|
| `macro_task` | Long-horizon goal | *"Guide fabric through sewing machine — Operator guides fabric under presser foot for straight stitching"* |
| `micro_step` | Short-horizon sub-task for the current time window | *"Position fabric under presser foot"* |
| `step_idx` | 1-based index of the active step | `2` |

Labels are supplied by the operator at annotation time. Steps are distributed evenly across the video duration; a three-step task on a 30-second video assigns 10 seconds to each step. The final step extends to the end of the clip regardless of its calculated boundary.

---

## Quick-Reference Table — All 167 CSV Columns

| Columns | Count | Group | Unit |
|---------|-------|-------|------|
| `frame`, `hand`, `ts_sec`, `macro_task`, `micro_step`, `step_idx`, `op_height_cm`, `robot_height_cm`, `environment`, `scene` | 10 | Metadata | — / s / cm |
| `x0`–`x20` | 21 | Raw landmarks | m |
| `y0`–`y20` | 21 | Raw landmarks | m |
| `z0`–`z20` | 21 | Raw landmarks | m |
| `roll_deg`, `pitch_deg`, `yaw_deg` | 3 | Orientation | ° |
| `nx0`–`nx20` | 21 | Normalised landmarks | m |
| `ny0`–`ny20` | 21 | Normalised landmarks | m |
| `nz0`–`nz20` | 21 | Normalised landmarks | m |
| `ee_x`, `ee_y`, `ee_z` | 3 | End-effector position | m |
| `thumb_mcp`, `thumb_ip`, `idx_mcp`, `idx_pip`, `idx_dip`, `mid_mcp`, `mid_pip`, `mid_dip`, `ring_mcp`, `ring_pip`, `ring_dip`, `pinky_mcp`, `pinky_pip`, `pinky_dip` | 14 | Joint angles | ° |
| `grasp_type`, `finger_aperture_m`, `contact_state` | 3 | Grasp | — / m / — |
| `ee_vx`, `ee_vy`, `ee_vz`, `ee_speed` | 4 | Velocity | m/s |
| `ee_ax`, `ee_ay`, `ee_az`, `ee_accel` | 4 | Acceleration | m/s² |
| **Total** | **167** | | |
